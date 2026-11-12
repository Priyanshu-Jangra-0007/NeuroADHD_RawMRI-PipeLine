"""MRI Defacing, Metadata De-Identification, and Privacy Export Gate (Step 6 / DEC-DEF-03).

Provides:
- Defacing tool discovery (PyDeface preferred; checks mri_deface, @afni_refacer_run, FSL)
  without fabricating facial-removal algorithms or substituting skull stripping.
- Structural MRI input validation (3D T1w prioritized; optional T2w/FLAIR; rejects BOLD/DWI/fmap).
- Immutability verification (pre/post SHA-256) and strict path containment.
- Quantitative defacing QC (dimensions, affine, qform/sform, finiteness, brain preservation,
  non-face voxel checks, and authentic-only facemask handling).
- Denylist-driven metadata privacy audit and sanitization for BIDS JSON sidecars, TSV tables
  (e.g., participants.tsv), NIfTI header text fields, provenance JSON, and manifests without
  stripping scientific acquisition parameters or logging raw personal identifier values.
- Export-readiness privacy gate using explicit non-misleading statuses
  (never ``ANONYMIZED`` or ``ANONYMOUS``).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# Status Constants
# ---------------------------------------------------------------------------

DEFACING_STATUS_APPLIED = "APPLIED"
DEFACING_STATUS_SKIPPED_TOOL_UNAVAILABLE = "SKIPPED_TOOL_UNAVAILABLE"
DEFACING_STATUS_SKIPPED_UNSUPPORTED_MODALITY = "SKIPPED_UNSUPPORTED_MODALITY"
DEFACING_STATUS_FAILED_INVALID_INPUT = "FAILED_INVALID_INPUT"
DEFACING_STATUS_FAILED_OUTPUT_VALIDATION = "FAILED_OUTPUT_VALIDATION"
DEFACING_STATUS_QC_PENDING_MANUAL_REVIEW = "QC_PENDING_MANUAL_REVIEW"
DEFACING_STATUS_REJECTED_BRAIN_TISSUE_LOSS = "REJECTED_BRAIN_TISSUE_LOSS"

ALL_DEFACING_STATUSES = frozenset({
    DEFACING_STATUS_APPLIED,
    DEFACING_STATUS_SKIPPED_TOOL_UNAVAILABLE,
    DEFACING_STATUS_SKIPPED_UNSUPPORTED_MODALITY,
    DEFACING_STATUS_FAILED_INVALID_INPUT,
    DEFACING_STATUS_FAILED_OUTPUT_VALIDATION,
    DEFACING_STATUS_QC_PENDING_MANUAL_REVIEW,
    DEFACING_STATUS_REJECTED_BRAIN_TISSUE_LOSS,
})

# Privacy & Export Gate overall statuses (never ANONYMIZED / ANONYMOUS)
PRIVACY_STATUS_READY_FOR_REVIEW = "PRIVACY_READY_FOR_REVIEW"
PRIVACY_STATUS_DEFACING_PENDING = "DEFACING_PENDING"
PRIVACY_STATUS_METADATA_SANITIZATION_PENDING = "METADATA_SANITIZATION_PENDING"
PRIVACY_STATUS_AUDIT_FAILED = "PRIVACY_AUDIT_FAILED"
PRIVACY_STATUS_SKIPPED_TOOL_UNAVAILABLE = "SKIPPED_TOOL_UNAVAILABLE"

ALL_PRIVACY_EXPORT_STATUSES = frozenset({
    PRIVACY_STATUS_READY_FOR_REVIEW,
    PRIVACY_STATUS_DEFACING_PENDING,
    PRIVACY_STATUS_METADATA_SANITIZATION_PENDING,
    PRIVACY_STATUS_AUDIT_FAILED,
    PRIVACY_STATUS_SKIPPED_TOOL_UNAVAILABLE,
})

SUPPORTED_ANATOMICAL_MODALITIES = ("T1w", "T2w", "FLAIR")
UNSUPPORTED_NON_ANAT_SUFFIXES = (
    "bold",
    "dwi",
    "sbref",
    "fieldmap",
    "phasediff",
    "phase1",
    "phase2",
    "magnitude",
    "magnitude1",
    "magnitude2",
    "epi",
    "fmap",
)

# Direct identifier metadata denylist (normalized lowercase alphanumeric tokens)
DEFAULT_IDENTIFIER_DENYLIST: Tuple[str, ...] = (
    "PatientName",
    "PatientsName",
    "PatientID",
    "PatientBirthDate",
    "PatientsBirthDate",
    "BirthDate",
    "DOB",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "MedicalRecordLocator",
    "MRN",
    "AccessionNumber",
    "StudyID",
    "InstitutionAddress",
    "InstitutionName",
    "InstitutionalDepartmentName",
    "DeviceSerialNumber",
    "StationName",
    "OperatorsName",
    "PerformingPhysicianName",
    "ReferringPhysicianName",
    "RequestingPhysician",
    "PhysiciansOfRecord",
    "NameOfPhysiciansReadingStudy",
    "EthnicGroup",
    "Occupation",
    "AdditionalPatientHistory",
    "PersonName",
    "FullName",
    "Email",
    "Phone",
    "HomeAddress",
)

# Scientific BIDS acquisition & structural metadata preserved unconditionally
PRESERVED_SCIENTIFIC_FIELDS: Tuple[str, ...] = (
    "RepetitionTime",
    "SliceTiming",
    "SliceEncodingDirection",
    "PhaseEncodingDirection",
    "TotalReadoutTime",
    "EchoTime",
    "EffectiveEchoSpacing",
    "ReconMatrixPE",
    "MagneticFieldStrength",
    "FlipAngle",
    "InversionTime",
    "TaskName",
    "Units",
    "IntendedFor",
    "B0FieldIdentifier",
    "B0FieldSource",
    "StartTime",
    "ManufacturersModelName",
    "Manufacturer",
    "Modality",
    "PulseSequenceType",
    "ReceiveCoilName",
)

# Text patterns inside NIfTI header fields (descrip, aux_file, db_name, intent_name)
_NIFTI_PHI_PATTERN = re.compile(
    r"(patient\s*name|patient\s*id|birth\s*date|\bdob\b|accession|\bmrn\b|"
    r"operator|physician|institution|serial\s*number|station\s*name|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
    re.IGNORECASE,
)

# Safety blocklist patterns mirroring scripts/run_poc_v2.py
_BLOCKED_CLINICAL_PATTERNS = (
    "down/outputs/adhd200",
    "sub-2344857",
    "peking",
    "kki",
    "nyu",
    "abide",
    "adhd-200",
    "adhd200",
    "neuroadhd_converted",
)

# Documented brain-tissue QC tolerance and limitations:
# Defacing removes facial voxels anterior/inferior to the brain; inside an explicit brain mask,
# >=99% of positive brain voxels must remain nonzero (allows <=1% edge interpolation tolerance)
# and thresholded mask Dice similarity must remain >=0.95.
MIN_BRAIN_VOXEL_RETENTION_FRACTION = 0.99
MIN_BRAIN_MASK_DICE = 0.95

QC_LIMITATIONS: Tuple[str, ...] = (
    "Automated defacing QC verifies spatial geometry, finiteness, and brain-mask preservation on "
    "synthetic or masked images, but cannot prove anatomical facial identifiability removal on real MRI scans.",
    "Non-zero exit or exit code 0 from external defacing software does not guarantee absence of subtle "
    "frontal/temporal cortical clipping; manual visual orthoslice review remains mandatory.",
    "NIfTI header and BIDS JSON sidecar sanitization does not sanitize upstream DICOM files; DICOM "
    "de-identification must be executed and validated separately.",
    "PyDeface requires FSL registration utilities (flirt) at runtime unless executed in an environment "
    "where FSL is installed alongside pydeface.",
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DefacingError(ValueError):
    """Base error for Step 6 anatomical MRI defacing and privacy checks."""


class InvalidDefacingInputError(DefacingError):
    """Raised when input NIfTI geometry, paths, or containment checks fail."""


class UnsupportedModalityError(DefacingError):
    """Raised when strict defacing is requested on BOLD, DWI, fieldmaps, or 4D volumes."""


class DefacingQCError(DefacingError):
    """Raised when strict defacing detects output corruption or brain-tissue loss."""


class PrivacyExportGateError(DefacingError):
    """Raised when export-readiness gate is violated in strict/export mode."""


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _is_within_root(target: Path, permitted_root: Path) -> bool:
    if ".." in target.parts:
        return False
    try:
        resolved_root = permitted_root.expanduser().resolve(strict=False)
        resolved_target = target.expanduser().resolve(strict=False)
        resolved_target.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def _assert_safe_path(
    path: Path,
    *,
    permitted_root: Optional[Path] = None,
    label: str = "path",
    allow_nonexistent: bool = False,
) -> Path:
    if ".." in path.parts:
        raise InvalidDefacingInputError(f"Unsafe {label}: path traversal ('..') is forbidden.")
    if path.is_symlink():
        raise InvalidDefacingInputError(f"Unsafe {label}: symlinks are forbidden.")
    # Check parent chain up to permitted_root (if provided) for symlinks
    resolved = path.expanduser().resolve(strict=False)
    for blocked in _BLOCKED_CLINICAL_PATTERNS:
        if blocked in str(path).lower() or blocked in str(resolved).lower():
            raise InvalidDefacingInputError(
                f"SAFETY VIOLATION: Access to protected clinical path blocked ({label})."
            )
    if permitted_root is not None:
        if not _is_within_root(path, permitted_root):
            raise InvalidDefacingInputError(
                f"Path containment violation for {label}: output must stay within permitted directory."
            )
        try:
            rel_parts = path.expanduser().relative_to(permitted_root.expanduser()).parts
        except ValueError:
            rel_parts = ()
        cur = permitted_root.expanduser().resolve(strict=False)
        for part in rel_parts:
            cur = cur / part
            if cur.is_symlink():
                raise InvalidDefacingInputError(
                    f"Unsafe {label}: symlinked directory or file inside permitted root is forbidden."
                )
    if not allow_nonexistent and not path.is_file():
        raise InvalidDefacingInputError(f"Required {label} does not exist or is not a regular file.")
    return resolved


def _infer_bids_modality(path: Path) -> str:
    name = path.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else path.stem
    tokens = [tok for tok in stem.split("_") if tok]
    if not tokens:
        return "UNKNOWN"
    suffix = tokens[-1]
    lower = suffix.lower()
    if lower == "t1w":
        return "T1w"
    if lower == "t2w":
        return "T2w"
    if lower == "flair":
        return "FLAIR"
    return suffix


def _effective_denylist_tokens(extra_denied_fields: Optional[Sequence[str]] = None) -> Dict[str, str]:
    tokens: Dict[str, str] = {}
    for item in DEFAULT_IDENTIFIER_DENYLIST:
        tokens[_normalize_key(item)] = item
    if extra_denied_fields:
        for item in extra_denied_fields:
            tokens[_normalize_key(item)] = str(item)
    return tokens


def _scientific_tokens() -> set[str]:
    return {_normalize_key(k) for k in PRESERVED_SCIENTIFIC_FIELDS}


# ---------------------------------------------------------------------------
# Dataclass Result
# ---------------------------------------------------------------------------


@dataclass
class DefacingResult:
    """Structured provenance record emitted by Stage 7 / ``run_defacing``."""

    status: str
    defaced: bool
    accepted_for_downstream_preprocessing: bool
    algorithm: Optional[str] = None
    algorithm_version: Optional[str] = None
    modality: Optional[str] = None
    input_filename: Optional[str] = None
    input_sha256: Optional[str] = None
    input_sha256_after: Optional[str] = None
    input_file_unmodified_verified: bool = False
    output_filename: Optional[str] = None
    output_relative_path: Optional[str] = None
    output_sha256: Optional[str] = None
    defacing_mask_available: bool = False
    defacing_mask_filename: Optional[str] = None
    defacing_mask_relative_path: Optional[str] = None
    defacing_mask_sha256: Optional[str] = None
    planned_command: Optional[List[str]] = None
    executed_command: Optional[List[str]] = None
    missing_dependencies: List[str] = field(default_factory=list)
    software: Dict[str, Any] = field(default_factory=dict)
    output_validation: Optional[Dict[str, Any]] = None
    qc_metrics: Optional[Dict[str, Any]] = None
    manual_visual_review_required: bool = True
    manual_review_status: str = DEFACING_STATUS_QC_PENDING_MANUAL_REVIEW
    warnings: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    limitations: List[str] = field(default_factory=lambda: list(QC_LIMITATIONS))
    defaced_image: Optional[nib.Nifti1Image] = None

    def to_provenance_dict(self) -> Dict[str, Any]:
        return build_defacing_provenance(self)


# ---------------------------------------------------------------------------
# 1. Software Discovery
# ---------------------------------------------------------------------------


def _probe_cli_version(executable: str, args: Sequence[str] = ("--version",)) -> Optional[str]:
    try:
        proc = subprocess.run(
            [executable, *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        for line in out.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]
    except Exception:
        pass
    return None


def detect_defacing_software(
    pydeface_executable: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Detect locally installed anatomical defacing tools and Python dependencies."""
    if pydeface_executable is not None:
        cand = Path(pydeface_executable)
        pydeface_path = str(cand) if cand.is_file() and os.access(cand, os.X_OK) else shutil.which(str(cand))
    else:
        pydeface_path = shutil.which("pydeface")

    pydeface_pkg_version: Optional[str] = None
    try:
        pydeface_pkg_version = importlib.metadata.version("pydeface")
    except importlib.metadata.PackageNotFoundError:
        pydeface_pkg_version = None

    pydeface_available = bool(pydeface_path)
    pydeface_version = (
        _probe_cli_version(pydeface_path, ("--version",)) if pydeface_path else pydeface_pkg_version
    )
    if pydeface_available and not pydeface_version:
        pydeface_version = pydeface_pkg_version or "installed"

    mri_deface_path = shutil.which("mri_deface")
    afni_refacer_path = shutil.which("@afni_refacer_run")
    mri_reface_path = shutil.which("mri_reface")
    quickshear_path = shutil.which("quickshear")
    flirt_path = shutil.which("flirt")
    fsl_available = bool(flirt_path or os.environ.get("FSLDIR"))

    missing_dependencies: List[str] = []
    if not pydeface_available:
        missing_dependencies.append("pydeface")
    if not flirt_path:
        missing_dependencies.append("flirt (FSL)")

    if pydeface_available:
        selected_algorithm = "pydeface"
        selected_version = pydeface_version
    elif mri_deface_path:
        selected_algorithm = "mri_deface"
        selected_version = "installed"
    elif afni_refacer_path:
        selected_algorithm = "@afni_refacer_run"
        selected_version = "installed"
    else:
        selected_algorithm = "none"
        selected_version = None

    return {
        "selected_algorithm": selected_algorithm,
        "selected_version": selected_version,
        "pydeface_available": pydeface_available,
        "pydeface_path": pydeface_path,
        "pydeface_version": pydeface_version,
        "pydeface_package_version": pydeface_pkg_version,
        "mri_deface_available": bool(mri_deface_path),
        "afni_refacer_available": bool(afni_refacer_path),
        "mri_reface_available": bool(mri_reface_path),
        "quickshear_available": bool(quickshear_path),
        "fsl_flirt_available": bool(flirt_path),
        "fsl_configured": fsl_available,
        "missing_dependencies": missing_dependencies,
        "nibabel_version": str(nib.__version__),
        "numpy_version": str(np.__version__),
    }


# ---------------------------------------------------------------------------
# 2. Input Validation
# ---------------------------------------------------------------------------


def validate_defacing_input(
    image_path: Union[str, Path],
    *,
    modality: Optional[str] = None,
    allow_additional_anatomical_modalities: bool = True,
) -> Dict[str, Any]:
    """Validate 3D structural anatomical NIfTI input before defacing.

    Rejects:
    - Non-existent files, symlinks, path traversal, or blocked clinical paths.
    - Non-anatomical MRI modalities (BOLD, DWI, fieldmaps, etc.).
    - Images that are not 3D (e.g., 4D fMRI time series), empty, non-finite, or have invalid affines.
    """
    raw_path = Path(image_path)
    resolved = _assert_safe_path(raw_path, label="anatomical_input")

    inferred_mod = modality or _infer_bids_modality(raw_path)
    inferred_lower = str(inferred_mod).strip().lower()

    if inferred_lower in UNSUPPORTED_NON_ANAT_SUFFIXES:
        raise UnsupportedModalityError(
            f"Refusing to apply anatomical defacing to non-anatomical modality '{inferred_mod}'."
        )
    if inferred_mod not in SUPPORTED_ANATOMICAL_MODALITIES:
        raise UnsupportedModalityError(
            f"Unsupported modality '{inferred_mod}' for anatomical defacing; "
            f"supported modalities: {SUPPORTED_ANATOMICAL_MODALITIES}."
        )
    if inferred_mod in ("T2w", "FLAIR") and not allow_additional_anatomical_modalities:
        raise UnsupportedModalityError(
            f"Additional anatomical modality '{inferred_mod}' disabled; only T1w is enabled."
        )

    try:
        img = nib.load(str(resolved))
        shape = tuple(int(v) for v in img.shape)
        affine = np.asarray(img.affine, dtype=np.float64)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        data = np.asanyarray(img.dataobj, dtype=np.float32)
    except Exception as exc:
        raise InvalidDefacingInputError(
            f"Failed to load valid NIfTI image from '{raw_path.name}': {type(exc).__name__}"
        ) from exc

    if len(shape) != 3:
        if len(shape) == 4 and shape[3] > 1:
            raise UnsupportedModalityError(
                f"Refusing to deface 4D time-series volume with shape {shape}; 3D structural MRI required."
            )
        raise InvalidDefacingInputError(f"Anatomical defacing requires a 3D volume; got shape {shape}.")

    if any(dim < 8 for dim in shape):
        raise InvalidDefacingInputError(f"Anatomical volume dimensions too small/invalid: {shape}.")
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise InvalidDefacingInputError("Anatomical NIfTI affine must be finite 4x4.")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-9:
        raise InvalidDefacingInputError("Anatomical NIfTI affine is singular (determinant ~ 0).")
    if len(zooms) < 3 or any((not np.isfinite(z) or z <= 0.0) for z in zooms[:3]):
        raise InvalidDefacingInputError(f"Anatomical NIfTI voxel zooms must be finite and > 0; got {zooms}.")
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise InvalidDefacingInputError("Anatomical NIfTI contains NaN or Inf voxel values.")
    if np.count_nonzero(data) == 0:
        raise InvalidDefacingInputError("Anatomical NIfTI volume is completely empty (all zeros).")

    qform, qcode = img.header.get_qform(coded=True)
    sform, scode = img.header.get_sform(coded=True)

    return {
        "valid": True,
        "resolved_path": resolved,
        "filename": raw_path.name,
        "modality": inferred_mod,
        "shape": list(shape),
        "zooms_mm": [round(float(z), 6) for z in zooms[:3]],
        "affine": affine,
        "qform_code": int(qcode),
        "sform_code": int(scode),
        "qform": np.asarray(qform, dtype=np.float64) if qform is not None else None,
        "sform": np.asarray(sform, dtype=np.float64) if sform is not None else None,
        "sha256": _sha256_file(resolved),
        "nonzero_voxels": int(np.count_nonzero(data)),
    }


# ---------------------------------------------------------------------------
# 3. Output Quality Control & Scientific Integrity Metrics
# ---------------------------------------------------------------------------


def _dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def _derive_foreground_proxy_mask(data: np.ndarray) -> np.ndarray:
    finite_pos = data[np.isfinite(data) & (data > 0)]
    if finite_pos.size == 0:
        return np.zeros(data.shape, dtype=bool)
    thresh = max(float(np.percentile(finite_pos, 20)), 1e-6)
    return data > thresh


def validate_defaced_output(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    expected_input_sha256: Optional[str] = None,
    brain_mask: Optional[Union[np.ndarray, nib.Nifti1Image]] = None,
    expected_face_mask: Optional[Union[np.ndarray, nib.Nifti1Image]] = None,
    defacing_mask_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Validate geometric integrity, finiteness, brain preservation, and input immutability."""
    in_file = Path(input_path)
    out_file = Path(output_path)

    if expected_input_sha256 is not None:
        actual_in_sha = _sha256_file(in_file)
        input_unmodified = actual_in_sha == expected_input_sha256
    else:
        actual_in_sha = _sha256_file(in_file)
        input_unmodified = True

    if not out_file.is_file() or out_file.is_symlink():
        return {
            "status": DEFACING_STATUS_FAILED_OUTPUT_VALIDATION,
            "passed": False,
            "input_file_unmodified_verified": input_unmodified,
            "reason": "Defaced output file is missing or unsafe symlink.",
            "warnings": [],
        }

    try:
        in_img = nib.load(str(in_file))
        out_img = nib.load(str(out_file))
        in_data = np.asanyarray(in_img.dataobj, dtype=np.float32)
        out_data = np.asanyarray(out_img.dataobj, dtype=np.float32)
    except Exception as exc:
        return {
            "status": DEFACING_STATUS_FAILED_OUTPUT_VALIDATION,
            "passed": False,
            "input_file_unmodified_verified": input_unmodified,
            "reason": f"Failed to read NIfTI output '{out_file.name}': {type(exc).__name__}",
            "warnings": [],
        }

    dimensions_match = tuple(in_img.shape) == tuple(out_img.shape) and len(out_img.shape) == 3
    if not dimensions_match:
        return {
            "status": DEFACING_STATUS_FAILED_OUTPUT_VALIDATION,
            "passed": False,
            "dimensions_match": False,
            "input_shape": [int(x) for x in in_img.shape],
            "output_shape": [int(x) for x in out_img.shape],
            "input_file_unmodified_verified": input_unmodified,
            "reason": f"Output shape {out_img.shape} does not match input shape {in_img.shape}.",
            "warnings": [],
        }

    finite_values = bool(np.all(np.isfinite(out_data)))
    non_empty_output = bool(np.count_nonzero(out_data) > 0)
    affine_max_abs_diff = float(np.max(np.abs(np.asarray(out_img.affine) - np.asarray(in_img.affine))))
    affine_preserved = bool(np.allclose(out_img.affine, in_img.affine, atol=1e-4))
    zooms_max_abs_diff = float(
        np.max(np.abs(np.asarray(out_img.header.get_zooms()[:3]) - np.asarray(in_img.header.get_zooms()[:3])))
    )

    in_qform, in_qcode = in_img.header.get_qform(coded=True)
    in_sform, in_scode = in_img.header.get_sform(coded=True)
    out_qform, out_qcode = out_img.header.get_qform(coded=True)
    out_sform, out_scode = out_img.header.get_sform(coded=True)

    qform_preserved = (
        in_qform is None
        or out_qform is None
        or bool(np.allclose(out_qform, in_qform, atol=1e-4))
    )
    sform_preserved = (
        in_sform is None
        or out_sform is None
        or bool(np.allclose(out_sform, in_sform, atol=1e-4))
    )
    qform_sform_consistent = True
    if out_qform is not None and out_sform is not None and int(out_qcode) > 0 and int(out_scode) > 0:
        qform_sform_consistent = bool(np.allclose(out_qform, out_sform, atol=1e-3))

    if (
        not finite_values
        or not non_empty_output
        or not affine_preserved
        or not qform_preserved
        or not sform_preserved
        or not qform_sform_consistent
        or not input_unmodified
    ):
        reasons = []
        if not finite_values:
            reasons.append("non-finite voxels")
        if not non_empty_output:
            reasons.append("empty output volume")
        if not affine_preserved:
            reasons.append(f"affine mismatch (max diff {affine_max_abs_diff:.2e})")
        if not qform_preserved or not sform_preserved or not qform_sform_consistent:
            reasons.append("qform/sform mismatch or inconsistency")
        if not input_unmodified:
            reasons.append("input SHA-256 changed")
        return {
            "status": DEFACING_STATUS_FAILED_OUTPUT_VALIDATION,
            "passed": False,
            "dimensions_match": dimensions_match,
            "finite_values": finite_values,
            "non_empty_output": non_empty_output,
            "affine_preserved": affine_preserved,
            "affine_max_abs_diff": affine_max_abs_diff,
            "zooms_max_abs_diff": zooms_max_abs_diff,
            "qform_preserved": qform_preserved,
            "sform_preserved": sform_preserved,
            "qform_sform_consistent": qform_sform_consistent,
            "input_file_unmodified_verified": input_unmodified,
            "reason": "Output validation failed: " + ", ".join(reasons),
            "warnings": [],
        }

    # Brain-mask preservation & intensity checks
    warnings: List[str] = []
    if brain_mask is not None:
        if isinstance(brain_mask, nib.Nifti1Image):
            bmask = np.asanyarray(brain_mask.dataobj) > 0
        else:
            bmask = np.asarray(brain_mask) > 0
        if bmask.shape != in_data.shape:
            raise InvalidDefacingInputError(
                f"Brain mask shape {bmask.shape} does not match volume shape {in_data.shape}."
            )
        brain_mask_source = "explicit_brain_mask"
    else:
        bmask = _derive_foreground_proxy_mask(in_data)
        brain_mask_source = "foreground_proxy_no_explicit_mask"
        warnings.append(
            "No explicit brain mask provided; foreground proxy mask includes non-brain head voxels."
        )

    bmask_active = bmask & (np.abs(in_data) > 1e-6)
    brain_voxels_before = int(np.count_nonzero(bmask_active))
    brain_after_mask = bmask_active & (np.abs(out_data) > 1e-6)
    brain_voxels_after = int(np.count_nonzero(brain_after_mask))
    brain_retention_fraction = (
        float(brain_voxels_after / brain_voxels_before) if brain_voxels_before > 0 else 1.0
    )
    brain_mask_dice = _dice_coefficient(bmask_active, brain_after_mask) if brain_voxels_before > 0 else 1.0

    if np.any(bmask):
        brain_diff = np.abs(out_data[bmask] - in_data[bmask])
        brain_intensity_max_abs_diff = float(np.max(brain_diff))
        brain_intensity_mean_abs_diff = float(np.mean(brain_diff))
    else:
        brain_intensity_max_abs_diff = 0.0
        brain_intensity_mean_abs_diff = 0.0

    # Voxel modification & face-region accounting
    changed_mask = np.abs(out_data - in_data) > 1e-6
    total_modified_voxels = int(np.count_nonzero(changed_mask))
    defaced_voxels_removed = int(np.count_nonzero((np.abs(in_data) > 1e-6) & (np.abs(out_data) <= 1e-6)))

    non_face_voxels_modified = 0
    non_face_voxels_modified_fraction = 0.0
    if expected_face_mask is not None:
        if isinstance(expected_face_mask, nib.Nifti1Image):
            fmask = np.asanyarray(expected_face_mask.dataobj) > 0
        else:
            fmask = np.asarray(expected_face_mask) > 0
        if fmask.shape != in_data.shape:
            raise InvalidDefacingInputError(
                f"Expected face mask shape {fmask.shape} does not match volume shape {in_data.shape}."
            )
        outside_face = ~fmask
        non_face_voxels_modified = int(np.count_nonzero(changed_mask & outside_face))
        outside_nonzero = int(np.count_nonzero(outside_face & (np.abs(in_data) > 1e-6)))
        non_face_voxels_modified_fraction = (
            float(non_face_voxels_modified / outside_nonzero) if outside_nonzero > 0 else 0.0
        )
        if non_face_voxels_modified > 0:
            warnings.append(
                f"{non_face_voxels_modified} voxels outside the expected facial removal region were modified."
            )

    # Validate authentic facemask if provided
    facemask_valid: Optional[bool] = None
    facemask_voxel_count: Optional[int] = None
    if defacing_mask_path is not None:
        fm_path = Path(defacing_mask_path)
        if not fm_path.is_file() or fm_path.is_symlink():
            return {
                "status": DEFACING_STATUS_FAILED_OUTPUT_VALIDATION,
                "passed": False,
                "reason": "Provided defacing mask path is missing or unsafe symlink.",
                "input_file_unmodified_verified": input_unmodified,
                "warnings": warnings,
            }
        fm_img = nib.load(str(fm_path))
        fm_arr = np.asanyarray(fm_img.dataobj)
        if fm_img.shape != in_img.shape or not np.allclose(fm_img.affine, in_img.affine, atol=1e-4):
            return {
                "status": DEFACING_STATUS_FAILED_OUTPUT_VALIDATION,
                "passed": False,
                "reason": "Defacing mask geometry does not match input anatomical image.",
                "input_file_unmodified_verified": input_unmodified,
                "warnings": warnings,
            }
        facemask_valid = True
        facemask_voxel_count = int(np.count_nonzero(fm_arr > 0))

    brain_non_empty = brain_voxels_after > 0
    brain_tissue_lost = False
    if brain_mask is not None:
        if not brain_non_empty or brain_retention_fraction < MIN_BRAIN_VOXEL_RETENTION_FRACTION or brain_mask_dice < MIN_BRAIN_MASK_DICE:
            brain_tissue_lost = True
    elif not brain_non_empty:
        brain_tissue_lost = True

    warnings.append(
        "Manual visual inspection (orthoslice review) is required before treating defaced anatomy as de-identified; "
        "executable exit code 0 and automated metrics alone never guarantee complete facial removal."
    )

    if brain_tissue_lost:
        return {
            "status": DEFACING_STATUS_REJECTED_BRAIN_TISSUE_LOSS,
            "passed": False,
            "dimensions_match": dimensions_match,
            "finite_values": finite_values,
            "non_empty_output": non_empty_output,
            "brain_region_non_empty": brain_non_empty,
            "affine_preserved": affine_preserved,
            "affine_max_abs_diff": affine_max_abs_diff,
            "zooms_max_abs_diff": zooms_max_abs_diff,
            "qform_preserved": qform_preserved,
            "sform_preserved": sform_preserved,
            "qform_sform_consistent": qform_sform_consistent,
            "qform_code_out": int(out_qcode),
            "sform_code_out": int(out_scode),
            "brain_mask_source": brain_mask_source,
            "brain_voxels_before": brain_voxels_before,
            "brain_voxels_after": brain_voxels_after,
            "brain_retention_fraction": round(brain_retention_fraction, 6),
            "brain_mask_dice": round(brain_mask_dice, 6),
            "brain_intensity_max_abs_diff": round(brain_intensity_max_abs_diff, 6),
            "brain_intensity_mean_abs_diff": round(brain_intensity_mean_abs_diff, 6),
            "total_modified_voxels": total_modified_voxels,
            "defaced_voxels_removed": defaced_voxels_removed,
            "non_face_voxels_modified": non_face_voxels_modified,
            "non_face_voxels_modified_fraction": round(non_face_voxels_modified_fraction, 6),
            "facemask_valid": facemask_valid,
            "facemask_voxel_count": facemask_voxel_count,
            "input_file_unmodified_verified": input_unmodified,
            "input_sha256_verified": actual_in_sha,
            "output_sha256": _sha256_file(out_file),
            "reason": (
                f"Brain tissue preservation check failed (retention={brain_retention_fraction:.4f}, "
                f"Dice={brain_mask_dice:.4f}, threshold={MIN_BRAIN_VOXEL_RETENTION_FRACTION:.2f}/{MIN_BRAIN_MASK_DICE:.2f})."
            ),
            "warnings": warnings,
        }

    return {
        "status": DEFACING_STATUS_APPLIED,
        "passed": True,
        "dimensions_match": dimensions_match,
        "finite_values": finite_values,
        "non_empty_output": non_empty_output,
        "brain_region_non_empty": brain_non_empty,
        "affine_preserved": affine_preserved,
        "affine_max_abs_diff": affine_max_abs_diff,
        "zooms_max_abs_diff": zooms_max_abs_diff,
        "qform_preserved": qform_preserved,
        "sform_preserved": sform_preserved,
        "qform_sform_consistent": qform_sform_consistent,
        "qform_code_out": int(out_qcode),
        "sform_code_out": int(out_scode),
        "brain_mask_source": brain_mask_source,
        "brain_voxels_before": brain_voxels_before,
        "brain_voxels_after": brain_voxels_after,
        "brain_retention_fraction": round(brain_retention_fraction, 6),
        "brain_mask_dice": round(brain_mask_dice, 6),
        "brain_intensity_max_abs_diff": round(brain_intensity_max_abs_diff, 6),
        "brain_intensity_mean_abs_diff": round(brain_intensity_mean_abs_diff, 6),
        "total_modified_voxels": total_modified_voxels,
        "defaced_voxels_removed": defaced_voxels_removed,
        "non_face_voxels_modified": non_face_voxels_modified,
        "non_face_voxels_modified_fraction": round(non_face_voxels_modified_fraction, 6),
        "facemask_valid": facemask_valid,
        "facemask_voxel_count": facemask_voxel_count,
        "input_file_unmodified_verified": input_unmodified,
        "input_sha256_verified": actual_in_sha,
        "output_sha256": _sha256_file(out_file),
        "reason": None,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 4. Defacing Adapter Execution
# ---------------------------------------------------------------------------


def _default_subprocess_runner(command: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def run_defacing(
    input_path: Union[str, Path],
    derivative_dir: Union[str, Path],
    *,
    subject_id: str = "sub-synthetic001",
    modality: Optional[str] = None,
    strict_defacing: bool = False,
    allow_additional_anatomical_modalities: bool = True,
    pydeface_executable: Optional[Union[str, Path]] = None,
    defacing_runner: Optional[Callable[[List[str]], Any]] = None,
    brain_mask: Optional[Union[np.ndarray, nib.Nifti1Image]] = None,
    expected_face_mask: Optional[Union[np.ndarray, nib.Nifti1Image]] = None,
    require_manual_qc_signoff: bool = False,
    manual_qc_approved: bool = False,
) -> DefacingResult:
    """Execute anatomical MRI defacing into BIDS derivatives while keeping input immutable.

    Never fabricates a homemade facial-removal filter or substitutes skull stripping when the
    supported external tool (PyDeface) is unavailable.
    """
    in_path = Path(input_path)
    deriv_root = Path(derivative_dir)
    _assert_safe_path(deriv_root, label="derivative_dir", allow_nonexistent=True)

    sw = detect_defacing_software(pydeface_executable=pydeface_executable)

    # Validate anatomical input & modality first
    try:
        in_meta = validate_defacing_input(
            in_path,
            modality=modality,
            allow_additional_anatomical_modalities=allow_additional_anatomical_modalities,
        )
    except UnsupportedModalityError as exc:
        if strict_defacing:
            raise
        return DefacingResult(
            status=DEFACING_STATUS_SKIPPED_UNSUPPORTED_MODALITY,
            defaced=False,
            accepted_for_downstream_preprocessing=False,
            algorithm=sw["selected_algorithm"],
            algorithm_version=sw["selected_version"],
            modality=modality or _infer_bids_modality(in_path),
            input_filename=in_path.name,
            input_sha256=_sha256_file(in_path) if in_path.is_file() else None,
            input_sha256_after=_sha256_file(in_path) if in_path.is_file() else None,
            input_file_unmodified_verified=in_path.is_file(),
            missing_dependencies=list(sw["missing_dependencies"]),
            software=sw,
            reason=str(exc),
            warnings=[str(exc)],
        )
    except InvalidDefacingInputError as exc:
        # Always fail closed on explicit safety/path traversal violations
        msg = str(exc)
        if strict_defacing or "SAFETY VIOLATION" in msg or "traversal" in msg or "symlink" in msg or "containment" in msg:
            raise
        return DefacingResult(
            status=DEFACING_STATUS_FAILED_INVALID_INPUT,
            defaced=False,
            accepted_for_downstream_preprocessing=False,
            algorithm=sw["selected_algorithm"],
            algorithm_version=sw["selected_version"],
            modality=modality or _infer_bids_modality(in_path),
            input_filename=in_path.name,
            input_sha256=_sha256_file(in_path) if in_path.is_file() else None,
            input_sha256_after=_sha256_file(in_path) if in_path.is_file() else None,
            input_file_unmodified_verified=in_path.is_file(),
            missing_dependencies=list(sw["missing_dependencies"]),
            software=sw,
            reason=msg,
            warnings=[msg],
        )

    in_sha256 = in_meta["sha256"]
    resolved_in = in_meta["resolved_path"]
    anat_modality = in_meta["modality"]

    # Construct strictly contained BIDS derivative output path
    out_anat_dir = deriv_root / "defaced" / subject_id / "anat"
    _assert_safe_path(out_anat_dir, permitted_root=deriv_root, label="defaced_anat_dir", allow_nonexistent=True)
    out_anat_dir.mkdir(parents=True, exist_ok=True)
    out_nii_path = out_anat_dir / f"{subject_id}_desc-defaced_{anat_modality}.nii.gz"
    _assert_safe_path(out_nii_path, permitted_root=deriv_root, label="defaced_output_path", allow_nonexistent=True)

    candidate_exe = sw.get("pydeface_path") or (str(pydeface_executable) if pydeface_executable else "pydeface")
    planned_cmd = [
        str(candidate_exe),
        str(resolved_in),
        "--outfile",
        str(out_nii_path),
        "--force",
    ]

    tool_ready = bool(sw["pydeface_available"] or defacing_runner is not None)
    if not tool_ready:
        post_sha = _sha256_file(resolved_in)
        msg = (
            "Supported defacing tool 'pydeface' is not installed; skipping anatomical defacing "
            "without substituting skull stripping or homemade masking. Original anatomical MRI preserved."
        )
        if strict_defacing:
            raise DefacingError(f"STRICT DEFACING FAILURE: {msg}")
        return DefacingResult(
            status=DEFACING_STATUS_SKIPPED_TOOL_UNAVAILABLE,
            defaced=False,
            accepted_for_downstream_preprocessing=False,
            algorithm="pydeface (unavailable)",
            algorithm_version=None,
            modality=anat_modality,
            input_filename=in_path.name,
            input_sha256=in_sha256,
            input_sha256_after=post_sha,
            input_file_unmodified_verified=(post_sha == in_sha256),
            planned_command=planned_cmd,
            missing_dependencies=list(sw["missing_dependencies"]),
            software=sw,
            reason=msg,
            warnings=[msg],
        )

    # Execute PyDeface (real binary or injected runner for command/error tests)
    runner = defacing_runner or _default_subprocess_runner
    warnings_list: List[str] = []
    if anat_modality in ("T2w", "FLAIR"):
        warnings_list.append(
            f"Non-T1w modality '{anat_modality}' defaced with {candidate_exe}; confirm contrast compatibility manually."
        )

    with tempfile.TemporaryDirectory(prefix="neuroadhd_deface_") as work_tmp:
        work_dir = Path(work_tmp)
        try:
            proc = runner(planned_cmd)
            retcode = getattr(proc, "returncode", 0 if proc is None else 1)
            stderr_txt = str(getattr(proc, "stderr", "") or "").strip()
        except Exception as exc:
            proc = None
            retcode = 1
            stderr_txt = f"{type(exc).__name__}: {exc}"
        if retcode != 0 or not out_nii_path.is_file():
            post_sha = _sha256_file(resolved_in)
            err_detail = stderr_txt[:240] if stderr_txt else f"returncode={retcode}"
            reason = f"PyDeface execution failed ({err_detail})."
            if out_nii_path.is_file():
                out_nii_path.unlink(missing_ok=True)
            if strict_defacing:
                raise DefacingQCError(f"STRICT DEFACING FAILURE: {reason}")
            return DefacingResult(
                status=DEFACING_STATUS_FAILED_OUTPUT_VALIDATION,
                defaced=False,
                accepted_for_downstream_preprocessing=False,
                algorithm="pydeface" if defacing_runner is None else "pydeface (test runner)",
                algorithm_version=sw.get("pydeface_version") or "unknown",
                modality=anat_modality,
                input_filename=in_path.name,
                input_sha256=in_sha256,
                input_sha256_after=post_sha,
                input_file_unmodified_verified=(post_sha == in_sha256),
                planned_command=planned_cmd,
                executed_command=planned_cmd,
                missing_dependencies=list(sw["missing_dependencies"]),
                software=sw,
                reason=reason,
                warnings=warnings_list + [reason],
            )
        _ = work_dir  # keep work_dir scoped cleanly

    # Detect authentic tool-provided binary face mask ONLY if the tool created one
    produced_mask_path: Optional[Path] = None
    final_mask_path: Optional[Path] = None
    mask_candidates = [
        out_nii_path.with_name(f"{subject_id}_desc-defaced_{anat_modality}_facemask.nii.gz"),
        out_nii_path.with_name(f"{subject_id}_desc-facemask_mask.nii.gz"),
        out_nii_path.with_name(in_path.name.replace(".nii.gz", "_defacemask.nii.gz")),
        out_nii_path.with_name(in_path.name.replace(".nii.gz", "_facemask.nii.gz")),
    ]
    for cand in mask_candidates:
        if cand.is_file() and not cand.is_symlink():
            produced_mask_path = cand
            break

    if produced_mask_path is not None:
        final_mask_path = out_anat_dir / f"{subject_id}_desc-facemask_mask.nii.gz"
        _assert_safe_path(final_mask_path, permitted_root=deriv_root, label="facemask_output", allow_nonexistent=True)
        if produced_mask_path != final_mask_path:
            shutil.move(str(produced_mask_path), str(final_mask_path))
    else:
        warnings_list.append(
            "Selected defacing execution did not emit a separate binary face mask; no mask was invented."
        )

    # Quantitative output validation & brain preservation QC
    qc_out = validate_defaced_output(
        resolved_in,
        out_nii_path,
        expected_input_sha256=in_sha256,
        brain_mask=brain_mask,
        expected_face_mask=expected_face_mask,
        defacing_mask_path=final_mask_path,
    )
    post_sha = _sha256_file(resolved_in)
    qc_status = qc_out["status"]
    warnings_list.extend(qc_out.get("warnings", []))

    if not qc_out["passed"]:
        # Remove rejected derivative output so it is never mistakenly consumed downstream
        out_nii_path.unlink(missing_ok=True)
        if final_mask_path is not None:
            final_mask_path.unlink(missing_ok=True)
        if strict_defacing:
            raise DefacingQCError(f"STRICT DEFACING FAILURE ({qc_status}): {qc_out.get('reason')}")
        return DefacingResult(
            status=qc_status,
            defaced=False,
            accepted_for_downstream_preprocessing=False,
            algorithm="pydeface" if defacing_runner is None else "pydeface (test runner)",
            algorithm_version=sw.get("pydeface_version") or "unknown",
            modality=anat_modality,
            input_filename=in_path.name,
            input_sha256=in_sha256,
            input_sha256_after=post_sha,
            input_file_unmodified_verified=(post_sha == in_sha256),
            planned_command=planned_cmd,
            executed_command=planned_cmd,
            missing_dependencies=list(sw["missing_dependencies"]),
            software=sw,
            output_validation=qc_out,
            qc_metrics=qc_out,
            reason=qc_out.get("reason"),
            warnings=warnings_list,
        )

    # Scrub any flagged header text inside the derivative NIfTI copy itself
    defaced_img_loaded = nib.load(str(out_nii_path))
    defaced_data = np.asanyarray(defaced_img_loaded.dataobj, dtype=np.float32)
    sanitized_img = nib.Nifti1Image(defaced_data, defaced_img_loaded.affine, defaced_img_loaded.header.copy())
    for hdr_field in ("descrip", "aux_file", "db_name", "intent_name"):
        if hdr_field in sanitized_img.header:
            sanitized_img.header[hdr_field] = b""
    nib.save(sanitized_img, str(out_nii_path))

    out_sha256 = _sha256_file(out_nii_path)
    qc_out["output_sha256"] = out_sha256

    final_status = DEFACING_STATUS_APPLIED
    manual_review_state = DEFACING_STATUS_APPLIED if manual_qc_approved else DEFACING_STATUS_QC_PENDING_MANUAL_REVIEW
    if require_manual_qc_signoff and not manual_qc_approved:
        final_status = DEFACING_STATUS_QC_PENDING_MANUAL_REVIEW

    mask_rel = str(final_mask_path.relative_to(deriv_root)) if final_mask_path is not None and final_mask_path.is_file() else None
    mask_sha = _sha256_file(final_mask_path) if final_mask_path is not None and final_mask_path.is_file() else None

    return DefacingResult(
        status=final_status,
        defaced=True,
        accepted_for_downstream_preprocessing=True,
        algorithm="pydeface" if defacing_runner is None else "pydeface (test runner)",
        algorithm_version=sw.get("pydeface_version") or "unknown",
        modality=anat_modality,
        input_filename=in_path.name,
        input_sha256=in_sha256,
        input_sha256_after=post_sha,
        input_file_unmodified_verified=(post_sha == in_sha256),
        output_filename=out_nii_path.name,
        output_relative_path=str(out_nii_path.relative_to(deriv_root)),
        output_sha256=out_sha256,
        defacing_mask_available=bool(final_mask_path is not None and final_mask_path.is_file()),
        defacing_mask_filename=final_mask_path.name if final_mask_path is not None and final_mask_path.is_file() else None,
        defacing_mask_relative_path=mask_rel,
        defacing_mask_sha256=mask_sha,
        planned_command=planned_cmd,
        executed_command=planned_cmd,
        missing_dependencies=list(sw["missing_dependencies"]),
        software=sw,
        output_validation=qc_out,
        qc_metrics=qc_out,
        manual_visual_review_required=(not manual_qc_approved),
        manual_review_status=manual_review_state,
        warnings=warnings_list,
        reason=None,
        defaced_image=sanitized_img,
    )


# ---------------------------------------------------------------------------
# 5. Metadata Privacy Audit & Sanitization
# ---------------------------------------------------------------------------


def _audit_json_mapping(
    obj: Any,
    denied_tokens: Dict[str, str],
    sci_tokens: set[str],
    prefix: str = "",
) -> Tuple[List[str], List[str]]:
    detected_fields: List[str] = []
    preserved_scientific: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else str(k)
            norm = _normalize_key(k)
            if norm in denied_tokens and v not in (None, "", []):
                detected_fields.append(full_key)
            elif norm in sci_tokens:
                preserved_scientific.append(full_key)
            if isinstance(v, (dict, list)):
                sub_det, sub_sci = _audit_json_mapping(v, denied_tokens, sci_tokens, full_key)
                detected_fields.extend(sub_det)
                preserved_scientific.extend(sub_sci)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                sub_det, sub_sci = _audit_json_mapping(item, denied_tokens, sci_tokens, f"{prefix}[{idx}]")
                detected_fields.extend(sub_det)
                preserved_scientific.extend(sub_sci)
    return detected_fields, preserved_scientific


def _sanitize_json_mapping(
    obj: Any,
    denied_tokens: Dict[str, str],
    sci_tokens: set[str],
    prefix: str = "",
) -> Tuple[Any, List[str], List[str]]:
    removed: List[str] = []
    preserved_sci: List[str] = []
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else str(k)
            norm = _normalize_key(k)
            if norm in denied_tokens and norm not in sci_tokens:
                removed.append(full_key)
                continue
            if norm in sci_tokens:
                preserved_sci.append(full_key)
            if isinstance(v, (dict, list)):
                sub_val, sub_rem, sub_sci = _sanitize_json_mapping(v, denied_tokens, sci_tokens, full_key)
                cleaned[k] = sub_val
                removed.extend(sub_rem)
                preserved_sci.extend(sub_sci)
            else:
                cleaned[k] = v
        return cleaned, removed, preserved_sci
    if isinstance(obj, list):
        cleaned_list = []
        for idx, item in enumerate(obj):
            sub_val, sub_rem, sub_sci = _sanitize_json_mapping(item, denied_tokens, sci_tokens, f"{prefix}[{idx}]")
            cleaned_list.append(sub_val)
            removed.extend(sub_rem)
            preserved_sci.extend(sub_sci)
        return cleaned_list, removed, preserved_sci
    return obj, removed, preserved_sci


def _audit_nifti_header(path: Path) -> List[str]:
    img = nib.load(str(path))
    flagged: List[str] = []
    for fld in ("descrip", "aux_file", "db_name", "intent_name"):
        if fld not in img.header:
            continue
        raw = img.header[fld].tobytes() if hasattr(img.header[fld], "tobytes") else bytes(img.header[fld])
        text = raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()
        if text and _NIFTI_PHI_PATTERN.search(text):
            flagged.append(f"nifti_header.{fld}")
    return flagged


def _audit_tsv_headers(path: Path) -> Tuple[List[str], List[str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    denied = _effective_denylist_tokens()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
    detected = [col for col in fieldnames if _normalize_key(col) in denied]
    preserved = [col for col in fieldnames if col == "participant_id"]
    return detected, preserved


def audit_metadata_identifiers(
    metadata_sources: Union[Dict[str, Any], Path, str, Sequence[Union[Path, str]]],
    *,
    extra_denied_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Audit JSON mappings, BIDS JSON/TSV files, and NIfTI headers for direct personal identifiers.

    Never returns, logs, or exposes raw personal identifier values—only field names and counts.
    """
    denied_tokens = _effective_denylist_tokens(extra_denied_fields)
    sci_tokens = _scientific_tokens()

    detected_fields: List[str] = []
    scientific_fields_present: List[str] = []
    audited_items: List[Dict[str, Any]] = []

    if isinstance(metadata_sources, dict):
        det, sci = _audit_json_mapping(metadata_sources, denied_tokens, sci_tokens)
        detected_fields.extend(det)
        scientific_fields_present.extend(sci)
        audited_items.append({
            "source": "<in_memory_dict>",
            "source_type": "json_mapping",
            "identifier_fields_detected": sorted(set(det)),
            "clean": len(det) == 0,
        })
    else:
        paths = [metadata_sources] if isinstance(metadata_sources, (str, Path)) else list(metadata_sources)
        for item in paths:
            p = Path(item)
            _assert_safe_path(p, label=f"metadata_source({p.name})")
            name_lower = p.name.lower()
            if name_lower.endswith(".json"):
                payload = json.loads(p.read_text(encoding="utf-8"))
                det, sci = _audit_json_mapping(payload, denied_tokens, sci_tokens)
                detected_fields.extend(det)
                scientific_fields_present.extend(sci)
                audited_items.append({
                    "source": p.name,
                    "source_type": "bids_json",
                    "identifier_fields_detected": sorted(set(det)),
                    "clean": len(det) == 0,
                })
            elif name_lower.endswith((".tsv", ".csv")):
                det, preserved = _audit_tsv_headers(p)
                detected_fields.extend(det)
                audited_items.append({
                    "source": p.name,
                    "source_type": "tabular_tsv",
                    "identifier_fields_detected": sorted(set(det)),
                    "pseudonymous_id_columns_preserved": preserved,
                    "clean": len(det) == 0,
                })
            elif name_lower.endswith((".nii", ".nii.gz")):
                det = _audit_nifti_header(p)
                detected_fields.extend(det)
                audited_items.append({
                    "source": p.name,
                    "source_type": "nifti_header",
                    "identifier_fields_detected": sorted(set(det)),
                    "clean": len(det) == 0,
                })
            else:
                raise InvalidDefacingInputError(f"Unsupported metadata file type for audit: '{p.name}'.")

    unique_detected = sorted(set(detected_fields))
    return {
        "audit_completed": True,
        "clean": len(unique_detected) == 0,
        "identifier_field_count": len(unique_detected),
        "detected_identifier_fields": unique_detected,
        "scientific_fields_detected": sorted(set(scientific_fields_present)),
        "audited_sources": audited_items,
        "raw_identifier_values_logged": False,
        "dicom_deidentification_separate_required": True,
        "dicom_notice": (
            "Metadata audit inspects NIfTI headers, BIDS JSON sidecars, and TSV tables only; "
            "it does not inspect or sanitize upstream DICOM header PHI."
        ),
    }


def sanitize_metadata(
    source: Union[Dict[str, Any], Path, str],
    output_path: Optional[Union[str, Path]] = None,
    *,
    permitted_root: Optional[Union[str, Path]] = None,
    extra_denied_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Sanitize a JSON sidecar, TSV metadata table, or NIfTI header into a derivative copy.

    Never modifies the original source file and preserves scientific BIDS fields and
    pseudonymous ``participant_id`` labels. Reports only field names removed.
    """
    denied_tokens = _effective_denylist_tokens(extra_denied_fields)
    sci_tokens = _scientific_tokens()

    if isinstance(source, dict):
        cleaned_dict, removed_fields, preserved_sci = _sanitize_json_mapping(
            source, denied_tokens, sci_tokens
        )
        if output_path is not None:
            out_p = Path(output_path)
            _assert_safe_path(
                out_p,
                permitted_root=Path(permitted_root) if permitted_root else None,
                label="sanitized_json_output",
                allow_nonexistent=True,
            )
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(cleaned_dict, indent=2), encoding="utf-8")
        return {
            "status": "SANITIZED",
            "source_type": "json_mapping",
            "sanitized_data": cleaned_dict,
            "removed_fields": sorted(set(removed_fields)),
            "scientific_fields_preserved": sorted(set(preserved_sci)),
            "raw_identifier_values_logged": False,
            "original_modified": False,
            "dicom_header_phi_removed": False,
        }

    src_path = Path(source)
    _assert_safe_path(src_path, label=f"metadata_source({src_path.name})")
    if output_path is None:
        raise InvalidDefacingInputError(
            "An explicit derivative output_path is required so original metadata files are never overwritten."
        )
    out_p = Path(output_path)
    if out_p.resolve(strict=False) == src_path.resolve(strict=False):
        raise InvalidDefacingInputError(
            "Refusing to overwrite original metadata file in place; write to a derivative path instead."
        )
    perm_root = Path(permitted_root) if permitted_root is not None else out_p.parent
    _assert_safe_path(out_p, permitted_root=perm_root, label="sanitized_metadata_output", allow_nonexistent=True)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    src_sha_before = _sha256_file(src_path)
    lower_name = src_path.name.lower()

    if lower_name.endswith(".json"):
        raw_payload = json.loads(src_path.read_text(encoding="utf-8"))
        cleaned_payload, removed_fields, preserved_sci = _sanitize_json_mapping(
            raw_payload, denied_tokens, sci_tokens
        )
        out_p.write_text(json.dumps(cleaned_payload, indent=2), encoding="utf-8")
        src_type = "bids_json"
    elif lower_name.endswith((".tsv", ".csv")):
        delimiter = "\t" if lower_name.endswith(".tsv") else ","
        with src_path.open("r", encoding="utf-8", newline="") as sf:
            reader = csv.DictReader(sf, delimiter=delimiter)
            in_fields = list(reader.fieldnames or [])
            rows = list(reader)
        kept_fields = [c for c in in_fields if _normalize_key(c) not in denied_tokens]
        removed_fields = [c for c in in_fields if _normalize_key(c) in denied_tokens]
        preserved_sci = [c for c in kept_fields if c == "participant_id"]
        with out_p.open("w", encoding="utf-8", newline="") as df:
            writer = csv.DictWriter(df, fieldnames=kept_fields, delimiter=delimiter)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in kept_fields})
        src_type = "tabular_tsv"
    elif lower_name.endswith((".nii", ".nii.gz")):
        flagged_hdr = _audit_nifti_header(src_path)
        img = nib.load(str(src_path))
        out_img = nib.Nifti1Image(
            np.asanyarray(img.dataobj), img.affine, img.header.copy()
        )
        removed_fields = []
        for hdr_field in ("descrip", "aux_file", "db_name", "intent_name"):
            if hdr_field in out_img.header:
                raw_val = out_img.header[hdr_field].tobytes() if hasattr(out_img.header[hdr_field], "tobytes") else bytes(out_img.header[hdr_field])
                if raw_val.split(b"\x00", 1)[0].strip():
                    out_img.header[hdr_field] = b""
                    if f"nifti_header.{hdr_field}" in flagged_hdr:
                        removed_fields.append(f"nifti_header.{hdr_field}")
        nib.save(out_img, str(out_p))
        preserved_sci = ["affine", "dim", "pixdim", "qform_code", "sform_code"]
        src_type = "nifti_header"
    else:
        raise InvalidDefacingInputError(f"Unsupported metadata file type: '{src_path.name}'.")

    src_sha_after = _sha256_file(src_path)
    return {
        "status": "SANITIZED",
        "source_type": src_type,
        "source_filename": src_path.name,
        "source_sha256": src_sha_before,
        "source_unmodified_verified": (src_sha_after == src_sha_before),
        "output_filename": out_p.name,
        "output_sha256": _sha256_file(out_p),
        "removed_fields": sorted(set(removed_fields)),
        "scientific_fields_preserved": sorted(set(preserved_sci)),
        "raw_identifier_values_logged": False,
        "original_modified": (src_sha_after != src_sha_before),
        "dicom_header_phi_removed": False,
        "dicom_notice": (
            "Sidecar/header sanitization only; DICOM header PHI must be de-identified separately."
        ),
    }


# ---------------------------------------------------------------------------
# 6. Data Privacy & Export Readiness Gate
# ---------------------------------------------------------------------------


def evaluate_privacy_export_gate(
    *,
    defacing_result: DefacingResult,
    metadata_audit: Optional[Dict[str, Any]],
    sanitized_export_sources: Optional[Sequence[Union[Path, str]]] = None,
    raw_file_checksums: Optional[Dict[Union[Path, str], str]] = None,
    export_paths: Optional[Sequence[Union[Path, str]]] = None,
    permitted_root: Optional[Union[str, Path]] = None,
    require_export_ready: bool = False,
) -> Dict[str, Any]:
    """Evaluate overall privacy and export-readiness gate across defacing + metadata + containment.

    Never returns ``ANONYMIZED`` or ``ANONYMOUS``.
    """
    failures: List[str] = []
    warnings: List[str] = list(defacing_result.warnings)

    # 1. Original raw data immutability check
    raw_immutable = bool(defacing_result.input_file_unmodified_verified)
    if raw_file_checksums:
        for raw_p, expected_sha in raw_file_checksums.items():
            rp = Path(raw_p)
            if not rp.is_file() or _sha256_file(rp) != expected_sha:
                raw_immutable = False
                failures.append(f"Original raw file '{rp.name}' hash mismatch or missing.")
    if not raw_immutable:
        failures.append("Original raw MRI or metadata immutability could not be verified.")

    # 2. Path containment check across all export paths
    paths_contained = True
    if permitted_root is not None and export_paths:
        perm = Path(permitted_root)
        for ep in export_paths:
            ep_path = Path(ep)
            if ".." in ep_path.parts or ep_path.is_symlink() or not _is_within_root(ep_path, perm):
                paths_contained = False
                failures.append(f"Export path '{ep_path.name}' escapes permitted processing directory.")

    # 3. Export metadata cleanliness re-audit
    export_audit_clean = bool(metadata_audit and metadata_audit.get("audit_completed"))
    export_re_audit: Optional[Dict[str, Any]] = None
    if sanitized_export_sources:
        export_re_audit = audit_metadata_identifiers(list(sanitized_export_sources))
        if not export_re_audit["clean"]:
            export_audit_clean = False
            failures.append(
                "Direct identifiers remain in export metadata fields: "
                + ", ".join(export_re_audit["detected_identifier_fields"])
            )
    elif metadata_audit is not None and not metadata_audit.get("clean", False):
        export_audit_clean = False
        failures.append(
            "Direct identifiers detected in audited metadata fields: "
            + ", ".join(metadata_audit.get("detected_identifier_fields", []))
        )

    # 4. Compute explicit overall privacy status
    if failures:
        overall_status = PRIVACY_STATUS_AUDIT_FAILED
    elif defacing_result.status == DEFACING_STATUS_SKIPPED_TOOL_UNAVAILABLE:
        overall_status = PRIVACY_STATUS_SKIPPED_TOOL_UNAVAILABLE
        warnings.append("Dataset cannot be marked export-ready because defacing tool is unavailable.")
    elif not defacing_result.defaced or defacing_result.status != DEFACING_STATUS_APPLIED:
        if defacing_result.status in (DEFACING_STATUS_FAILED_INVALID_INPUT, DEFACING_STATUS_FAILED_OUTPUT_VALIDATION, DEFACING_STATUS_REJECTED_BRAIN_TISSUE_LOSS):
            overall_status = PRIVACY_STATUS_AUDIT_FAILED
            failures.append(defacing_result.reason or f"Defacing failed with status {defacing_result.status}.")
        else:
            overall_status = PRIVACY_STATUS_DEFACING_PENDING
            warnings.append(
                f"Dataset cannot be marked export-ready while defacing status is {defacing_result.status}."
            )
    elif metadata_audit is None or not metadata_audit.get("audit_completed"):
        overall_status = PRIVACY_STATUS_METADATA_SANITIZATION_PENDING
        warnings.append("Metadata privacy audit has not been completed.")
    elif not export_audit_clean:
        overall_status = PRIVACY_STATUS_AUDIT_FAILED
    else:
        overall_status = PRIVACY_STATUS_READY_FOR_REVIEW

    export_ready_for_review = overall_status == PRIVACY_STATUS_READY_FOR_REVIEW
    warnings.append(
        "Defacing and metadata sanitization reduce identification risk for review, but do NOT "
        "guarantee anonymity or prevent re-identification; manual visual QC and DICOM audit remain separate."
    )

    report = {
        "overall_status": overall_status,
        "export_ready_for_review": export_ready_for_review,
        "defacing_status": defacing_result.status,
        "metadata_sanitization_completed": bool(metadata_audit and metadata_audit.get("audit_completed")),
        "direct_identifiers_absent_from_export": bool(export_audit_clean and overall_status != PRIVACY_STATUS_AUDIT_FAILED),
        "raw_data_immutable_verified": raw_immutable,
        "output_paths_contained": paths_contained,
        "manual_visual_review_required": bool(defacing_result.manual_visual_review_required),
        "unresolved_warnings": sorted(set(warnings)),
        "failures": failures,
        "export_metadata_reaudit": export_re_audit,
        "synthetic_only_validation": True,
    }

    if require_export_ready and not export_ready_for_review:
        raise PrivacyExportGateError(
            f"PRIVACY EXPORT GATE BLOCKED ({overall_status}): "
            + ("; ".join(failures) if failures else "; ".join(warnings))
        )

    return report


# ---------------------------------------------------------------------------
# 7. Provenance Builder & Pipeline Stage Helper
# ---------------------------------------------------------------------------


def build_defacing_provenance(result: DefacingResult) -> Dict[str, Any]:
    """Serialize structured anatomical defacing outcome for ``provenance.json``."""
    return {
        "status": result.status,
        "defaced": bool(result.defaced),
        "accepted_for_downstream_preprocessing": bool(result.accepted_for_downstream_preprocessing),
        "algorithm": result.algorithm,
        "algorithm_version": result.algorithm_version,
        "modality": result.modality,
        "input_filename": result.input_filename,
        "input_sha256": result.input_sha256,
        "input_sha256_after": result.input_sha256_after,
        "input_file_unmodified_verified": bool(result.input_file_unmodified_verified),
        "output_filename": result.output_filename,
        "output_relative_path": result.output_relative_path,
        "output_sha256": result.output_sha256,
        "defacing_mask_available": bool(result.defacing_mask_available),
        "defacing_mask_filename": result.defacing_mask_filename,
        "defacing_mask_relative_path": result.defacing_mask_relative_path,
        "defacing_mask_sha256": result.defacing_mask_sha256,
        "missing_dependencies": list(result.missing_dependencies),
        "software": dict(result.software),
        "output_validation": result.output_validation,
        "qc_metrics": result.qc_metrics,
        "manual_visual_review_required": bool(result.manual_visual_review_required),
        "manual_review_status": result.manual_review_status,
        "warnings": list(result.warnings),
        "reason": result.reason,
        "limitations": list(result.limitations),
    }


def run_anatomical_defacing_and_privacy(
    *,
    t1w_path: Union[str, Path],
    bold_json_path: Union[str, Path],
    derivative_dir: Union[str, Path],
    subject_id: str = "sub-synthetic001",
    session_id: str = "ses-1",
    strict_defacing: bool = False,
    require_export_ready: bool = False,
    pydeface_executable: Optional[Union[str, Path]] = None,
    defacing_runner: Optional[Callable[[List[str]], Any]] = None,
    participants_tsv_path: Optional[Union[str, Path]] = None,
    dataset_description_path: Optional[Union[str, Path]] = None,
) -> Tuple[DefacingResult, Dict[str, Any]]:
    """Run Stage 7: Anatomical MRI defacing, BIDS sidecar sanitization, and privacy export gate.

    Returns ``(DefacingResult, combined_provenance_dict)``. Original raw files are never modified.
    """
    t1w_file = Path(t1w_path)
    bold_json_file = Path(bold_json_path)
    deriv_root = Path(derivative_dir)
    _assert_safe_path(deriv_root, label="derivative_dir", allow_nonexistent=True)

    raw_checksums: Dict[Union[Path, str], str] = {
        t1w_file: _sha256_file(t1w_file),
        bold_json_file: _sha256_file(bold_json_file),
    }
    if participants_tsv_path is not None and Path(participants_tsv_path).is_file():
        raw_checksums[Path(participants_tsv_path)] = _sha256_file(Path(participants_tsv_path))
    if dataset_description_path is not None and Path(dataset_description_path).is_file():
        raw_checksums[Path(dataset_description_path)] = _sha256_file(Path(dataset_description_path))

    # 1. Deface structural T1w image (or cleanly skip if PyDeface unavailable)
    defacing_res = run_defacing(
        input_path=t1w_file,
        derivative_dir=deriv_root,
        subject_id=subject_id,
        modality="T1w",
        strict_defacing=strict_defacing,
        pydeface_executable=pydeface_executable,
        defacing_runner=defacing_runner,
    )

    # 2. Sanitize permitted metadata into derivative location (never modifying original files)
    sanitized_dir = deriv_root / "defaced" / subject_id / "metadata_sanitized"
    sanitized_dir.mkdir(parents=True, exist_ok=True)
    export_paths: List[Path] = []
    sanitized_sources: List[Path] = []
    sanitization_reports: List[Dict[str, Any]] = []

    sanitized_bold_json = sanitized_dir / bold_json_file.name
    bold_san_res = sanitize_metadata(
        bold_json_file,
        sanitized_bold_json,
        permitted_root=deriv_root,
    )
    sanitization_reports.append(bold_san_res)
    sanitized_sources.append(sanitized_bold_json)
    export_paths.append(sanitized_bold_json)

    # Optional T1w sidecar JSON next to t1w_path
    t1w_stem = t1w_file.name[:-7] if t1w_file.name.lower().endswith(".nii.gz") else t1w_file.stem
    t1w_json_candidate = t1w_file.with_name(f"{t1w_stem}.json")
    if t1w_json_candidate.is_file() and not t1w_json_candidate.is_symlink():
        raw_checksums[t1w_json_candidate] = _sha256_file(t1w_json_candidate)
        san_t1w_json = sanitized_dir / t1w_json_candidate.name
        t1w_san_res = sanitize_metadata(
            t1w_json_candidate,
            san_t1w_json,
            permitted_root=deriv_root,
        )
        sanitization_reports.append(t1w_san_res)
        sanitized_sources.append(san_t1w_json)
        export_paths.append(san_t1w_json)

    if participants_tsv_path is not None and Path(participants_tsv_path).is_file():
        p_tsv = Path(participants_tsv_path)
        san_tsv = sanitized_dir / p_tsv.name
        sanitization_reports.append(sanitize_metadata(p_tsv, san_tsv, permitted_root=deriv_root))
        sanitized_sources.append(san_tsv)
        export_paths.append(san_tsv)

    if dataset_description_path is not None and Path(dataset_description_path).is_file():
        d_json = Path(dataset_description_path)
        san_djson = sanitized_dir / d_json.name
        sanitization_reports.append(sanitize_metadata(d_json, san_djson, permitted_root=deriv_root))
        sanitized_sources.append(san_djson)
        export_paths.append(san_djson)

    # Include defaced derivative NIfTI in sanitized_sources when created
    if defacing_res.defaced and defacing_res.output_relative_path:
        defaced_nii = deriv_root / defacing_res.output_relative_path
        if defaced_nii.is_file():
            sanitized_sources.append(defaced_nii)
            export_paths.append(defaced_nii)
            # Also write BIDS derivative sidecar JSON for defaced T1w
            defaced_sidecar = defaced_nii.with_name(
                defaced_nii.name.replace(".nii.gz", ".json")
            )
            sidecar_meta = {
                "Sources": [t1w_file.name],
                "DefacingMethod": defacing_res.algorithm,
                "DefacingMethodVersion": defacing_res.algorithm_version,
                "SkullStripped": False,
                "ManualVisualReviewRequired": defacing_res.manual_visual_review_required,
                "Session": session_id,
            }
            defaced_sidecar.write_text(json.dumps(sidecar_meta, indent=2), encoding="utf-8")
            sanitized_sources.append(defaced_sidecar)
            export_paths.append(defaced_sidecar)

    metadata_audit = audit_metadata_identifiers(sanitized_sources)
    all_removed_fields = sorted(
        {fld for rep in sanitization_reports for fld in rep.get("removed_fields", [])}
    )
    all_preserved_sci = sorted(
        {fld for rep in sanitization_reports for fld in rep.get("scientific_fields_preserved", [])}
    )

    privacy_gate = evaluate_privacy_export_gate(
        defacing_result=defacing_res,
        metadata_audit=metadata_audit,
        sanitized_export_sources=sanitized_sources,
        raw_file_checksums=raw_checksums,
        export_paths=export_paths,
        permitted_root=deriv_root,
        require_export_ready=require_export_ready,
    )

    prov_dict = defacing_res.to_provenance_dict()
    prov_dict["metadata_sanitization"] = {
        "status": "SANITIZED" if metadata_audit["clean"] else "IDENTIFIER_FIELDS_DETECTED",
        "sources_sanitized_count": len(sanitization_reports),
        "removed_identifier_fields": all_removed_fields,
        "scientific_fields_preserved": all_preserved_sci,
        "raw_identifier_values_logged": False,
        "dicom_header_phi_removed": False,
        "audit": metadata_audit,
    }
    prov_dict["privacy_export_gate"] = privacy_gate

    # Write Stage 7 derivative report JSON
    report_dir = deriv_root / "defaced" / subject_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{subject_id}_{session_id}_desc-defacing_report.json"
    _assert_safe_path(report_path, permitted_root=deriv_root, label="defacing_report_json", allow_nonexistent=True)
    prov_dict["report_json_relative_path"] = str(report_path.relative_to(deriv_root))
    prov_dict["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(prov_dict, indent=2), encoding="utf-8")

    return defacing_res, prov_dict
