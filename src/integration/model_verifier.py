"""Verification, validation, and zero-refit execution for frozen NeuroADHD ML model artifacts."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

# Authoritative cryptographic checksums for frozen benchmark artifacts
FROZEN_PREPROCESSING_PIPELINE_SHA256 = (
    "228d3f0180e4855d49fd732327ab1fc4f255beb7bea43dd4fd1db3064aa86249"
)
FROZEN_CLASSIFIER_SHA256 = (
    "d8eb23eb8c6d25ffc12b2bab515a2a7112daeeb0afa8f4e9eaa499dbe5b5f2ef"
)

EXPECTED_STEP_NAMES = ["imputer", "variance", "scaler", "select"]

# The exact 100 deterministic feature indices selected by SelectKBest(f_classif, k=100)
EXPECTED_SELECTED_FEATURE_INDICES: List[int] = [
    164, 209, 374, 399, 406, 410, 416, 427, 443, 449,
    457, 466, 486, 491, 508, 529, 544, 549, 573, 1017,
    1042, 1093, 1105, 1152, 1330, 2122, 2341, 3037, 3065, 3080,
    3443, 3459, 3502, 4181, 4226, 4228, 4261, 4312, 4532, 4801,
    4814, 4822, 5647, 5931, 6045, 6085, 6107, 6109, 6142, 6166,
    6763, 6830, 6837, 6847, 6935, 6947, 6999, 8437, 8474, 8698,
    9280, 9492, 9508, 9541, 9575, 9839, 10206, 10254, 10297, 10326,
    10693, 10893, 11463, 11480, 11722, 11760, 11872, 12834, 12858, 12939,
    12947, 13760, 13913, 14893, 14902, 15378, 15383, 16303, 17519, 17974,
    18256, 18277, 18432, 18596, 18603, 18613, 18676, 19318, 19327, 19554,
]


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute the SHA-256 hexadecimal digest of a file in 64 KiB chunks."""
    path = Path(file_path)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_and_load_frozen_model(
    model_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Any, Any]:
    """Verify cryptographic integrity and load the frozen preprocessing pipeline and classifier.

    Parameters
    ----------
    model_dir : Optional[Union[str, Path]]
        Directory containing 'final_preprocessing_pipeline.joblib' and 'final_model.joblib'.
        If None, reads from the NEUROADHD_MODEL_DIR environment variable.

    Returns
    -------
    Tuple[Any, Any]
        (pipeline, classifier) loaded and verified scikit-learn model objects.

    Raises
    ------
    FileNotFoundError
        If model_dir or either artifact is missing (fail closed).
    RuntimeError
        If symlink substitution or unauthorized alterations are detected.
    ValueError
        If SHA-256 checksum or internal model architecture does not match expected frozen contract.
    """
    if model_dir is None:
        env_dir = os.environ.get("NEUROADHD_MODEL_DIR")
        if not env_dir or not env_dir.strip():
            raise FileNotFoundError(
                "NEUROADHD_MODEL_DIR environment variable is not configured. "
                "Frozen model artifacts must be explicitly mounted via NEUROADHD_MODEL_DIR."
            )
        model_dir = env_dir.strip()

    raw_dir = Path(model_dir)
    if raw_dir.is_symlink():
        raise RuntimeError(f"Symlink substitution detected at model directory: {raw_dir}")
    dir_path = raw_dir.resolve()
    if dir_path.is_symlink():
        raise RuntimeError(f"Symlink substitution detected at model directory: {dir_path}")
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Configured model directory does not exist or is not a directory: {dir_path}")

    pipeline_path = dir_path / "final_preprocessing_pipeline.joblib"
    model_path = dir_path / "final_model.joblib"

    for p, name in [(pipeline_path, "pipeline"), (model_path, "classifier")]:
        if p.is_symlink():
            raise RuntimeError(f"Symlink substitution detected at {name} artifact: {p}")
        if not p.is_file():
            raise FileNotFoundError(f"Mandatory frozen {name} artifact missing at '{p}'.")

    # 1. SHA-256 Cryptographic Checksum Verification
    actual_pipeline_sha = compute_file_sha256(pipeline_path)
    if actual_pipeline_sha != FROZEN_PREPROCESSING_PIPELINE_SHA256:
        raise ValueError(
            f"Frozen preprocessing pipeline SHA-256 mismatch! "
            f"Expected {FROZEN_PREPROCESSING_PIPELINE_SHA256}, got {actual_pipeline_sha}. "
            "Artifact integrity verification failed; file is corrupted or modified."
        )

    actual_model_sha = compute_file_sha256(model_path)
    if actual_model_sha != FROZEN_CLASSIFIER_SHA256:
        raise ValueError(
            f"Frozen classifier SHA-256 mismatch! "
            f"Expected {FROZEN_CLASSIFIER_SHA256}, got {actual_model_sha}. "
            "Artifact integrity verification failed; file is corrupted or modified."
        )

    # 2. Deserialization
    pipeline = joblib.load(pipeline_path)
    model = joblib.load(model_path)

    # 3. Pipeline Architectural Contract Verification
    if not hasattr(pipeline, "named_steps"):
        raise ValueError("Loaded preprocessing pipeline lacks 'named_steps' attribute.")

    actual_steps = list(pipeline.named_steps.keys())
    if actual_steps != EXPECTED_STEP_NAMES:
        raise ValueError(
            f"Preprocessing pipeline steps mismatch! Expected {EXPECTED_STEP_NAMES}, got {actual_steps}."
        )

    # Step 1: Imputer
    imputer = pipeline.named_steps["imputer"]
    if not hasattr(imputer, "statistics_") or len(imputer.statistics_) != 19900:
        raise ValueError("Imputer statistics shape is invalid; expected exactly 19,900 features.")

    # Step 2: Variance
    variance = pipeline.named_steps["variance"]
    if getattr(variance, "threshold", None) != 0.0:
        raise ValueError("VarianceThreshold parameter mismatch; expected threshold=0.0.")
    if hasattr(variance, "get_support") and variance.get_support().sum() != 19900:
        raise ValueError("VarianceThreshold support mismatch; expected 19,900 surviving features.")

    # Step 3: Scaler
    scaler = pipeline.named_steps["scaler"]
    if not hasattr(scaler, "mean_") or len(scaler.mean_) != 19900:
        raise ValueError("StandardScaler mean_ statistics shape is invalid; expected 19,900 features.")
    if not hasattr(scaler, "scale_") or len(scaler.scale_) != 19900:
        raise ValueError("StandardScaler scale_ statistics shape is invalid; expected 19,900 features.")

    # Step 4: Feature Selection
    select = pipeline.named_steps["select"]
    if getattr(select, "k", None) != 100:
        raise ValueError("SelectKBest parameter mismatch; expected k=100.")
    selected_indices = list(select.get_support(indices=True))
    if [int(x) for x in selected_indices] != EXPECTED_SELECTED_FEATURE_INDICES:
        raise ValueError("SelectKBest indices do not match the frozen 100 historical feature indices.")

    # 4. Classifier Architectural Contract Verification
    if getattr(model, "n_estimators", None) != 50:
        raise ValueError("RandomForestClassifier n_estimators mismatch; expected 50.")
    if getattr(model, "max_depth", None) != 3:
        raise ValueError("RandomForestClassifier max_depth mismatch; expected 3.")
    if getattr(model, "class_weight", None) != "balanced":
        raise ValueError("RandomForestClassifier class_weight mismatch; expected 'balanced'.")
    if getattr(model, "random_state", None) != 42:
        raise ValueError("RandomForestClassifier random_state mismatch; expected 42.")
    if not hasattr(model, "classes_") or list(model.classes_) != [0, 1]:
        raise ValueError("RandomForestClassifier classes_ mismatch; expected [0, 1].")
    if getattr(model, "n_features_in_", None) != 100:
        raise ValueError("RandomForestClassifier n_features_in_ mismatch; expected 100.")

    return pipeline, model


def run_frozen_inference(
    z_vector: np.ndarray,
    pipeline: Any,
    model: Any,
) -> Dict[str, Any]:
    """Execute zero-refit inference on a single 19,900-element Fisher-z connectivity vector.

    Parameters
    ----------
    z_vector : np.ndarray
        1D float64 array of shape (19,900,) containing upper-triangle Fisher-z connectivity features.
    pipeline : Any
        Pre-fitted frozen scikit-learn preprocessing pipeline.
    model : Any
        Pre-fitted frozen scikit-learn RandomForestClassifier.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing prediction results, probabilities, and execution status.
    """
    if not isinstance(z_vector, np.ndarray) or z_vector.shape != (19900,):
        raise ValueError(f"Expected z_vector of shape (19900,), got {getattr(z_vector, 'shape', type(z_vector))}")
    if not np.all(np.isfinite(z_vector)):
        raise ValueError("Input z_vector contains non-finite values (NaN/Inf).")

    # Ingest as 2D NumPy array (1, 19900) without fitting
    X_raw = z_vector.reshape(1, -1)
    X_selected = pipeline.transform(X_raw)

    if X_selected.shape != (1, 100):
        raise RuntimeError(f"Expected pipeline output shape (1, 100), got {X_selected.shape}")

    probs = model.predict_proba(X_selected)[0]
    pred = int(model.predict(X_selected)[0])

    if len(probs) != 2 or not np.all(np.isfinite(probs)):
        raise RuntimeError(f"Model predict_proba returned non-finite or invalid probabilities: {probs}")
    if not np.isclose(float(np.sum(probs)), 1.0, atol=1e-5):
        raise RuntimeError(f"Probabilities do not sum to 1.0: sum={float(np.sum(probs))}")

    prob_control = float(probs[0])
    prob_adhd = float(probs[1])

    return {
        "status": "SUCCESS",
        "predicted_class": pred,
        "class_label": "ADHD Pattern" if pred == 1 else "Typical Control Pattern",
        "adhd_probability": round(prob_adhd, 4),
        "control_probability": round(prob_control, 4),
        "decision_threshold": 0.5,
        "features_in": 19900,
        "features_selected": 100,
        "refit_performed": False,
    }
