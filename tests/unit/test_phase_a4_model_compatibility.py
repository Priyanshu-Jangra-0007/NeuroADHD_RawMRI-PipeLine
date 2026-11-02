"""Phase A.4 / A.4.1 Model Compatibility and Frozen Artifact Verification Tests.

Separates:
1. Feature contract and ordering tests.
2. Synthetic model-interface smoke tests using disposable fixtures.
3. Frozen artifact discovery and integrity tests.
4. Actual frozen artifact integration tests.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.connectivity import (
    FISHER_EPSILON,
    N_ROIS,
    compute_connectivity_vector,
    get_feature_names,
)
from src.integration.model_verifier import (
    EXPECTED_SELECTED_FEATURE_INDICES,
    EXPECTED_STEP_NAMES,
    FROZEN_CLASSIFIER_SHA256,
    FROZEN_PREPROCESSING_PIPELINE_SHA256,
    compute_file_sha256,
    run_frozen_inference,
    verify_and_load_frozen_model,
)


class TestFeatureOrderingAndContract(unittest.TestCase):
    """1. Feature contract and ordering tests."""

    def setUp(self):
        np.random.seed(42)
        self.n_rois = N_ROIS
        self.n_edges = (N_ROIS * (N_ROIS - 1)) // 2  # 19,900
        self.t_points = 60

    def test_feature_names_dimensions_and_lexicographic_order(self):
        """Feature names must be exactly 19,900 and match itertools upper triangle order."""
        names = get_feature_names()
        self.assertEqual(len(names), self.n_edges)
        self.assertEqual(len(names), 19900)

        # Check boundary entries
        self.assertEqual(names[0], "ROI_001__ROI_002")
        self.assertEqual(names[1], "ROI_001__ROI_003")
        self.assertEqual(names[198], "ROI_001__ROI_200")
        self.assertEqual(names[199], "ROI_002__ROI_003")
        self.assertEqual(names[-1], "ROI_199__ROI_200")

        # Verify exact correspondence with itertools.combinations
        expected_pairs = list(itertools.combinations(range(1, 201), 2))
        expected_names = [f"ROI_{i:03d}__ROI_{j:03d}" for i, j in expected_pairs]
        self.assertEqual(names, expected_names)

    def test_upper_triangle_indices_order(self):
        """np.triu_indices(200, k=1) ordering must match itertools combinations row-by-row."""
        tri_i, tri_j = np.triu_indices(self.n_rois, k=1)
        self.assertEqual(len(tri_i), 19900)
        self.assertEqual(len(tri_j), 19900)

        expected_pairs = list(itertools.combinations(range(1, 201), 2))
        for idx, (i_exp, j_exp) in enumerate(expected_pairs):
            self.assertEqual(tri_i[idx] + 1, i_exp)
            self.assertEqual(tri_j[idx] + 1, j_exp)

    def test_compute_connectivity_vector_finite_and_symmetric(self):
        """Connectivity vector must be 1D (19900,), finite, and based on symmetric matrix."""
        signals = np.random.randn(self.t_points, self.n_rois)
        z, names = compute_connectivity_vector(signals)

        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (19900,))
        self.assertEqual(z.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(z)))
        self.assertEqual(names, get_feature_names())

    def test_compute_connectivity_vector_rejection_rules(self):
        """compute_connectivity_vector must strictly reject invalid shapes and constant ROIs."""
        # Wrong ROI count
        signals_199 = np.random.randn(self.t_points, 199)
        with self.assertRaises(ValueError) as ctx:
            compute_connectivity_vector(signals_199)
        self.assertIn("Expected 200 ROIs", str(ctx.exception))

        # Constant ROI signal (zero variance)
        signals_const = np.random.randn(self.t_points, 200)
        signals_const[:, 5] = 1.0
        with self.assertRaises(ValueError) as ctx:
            compute_connectivity_vector(signals_const)
        self.assertIn("Constant ROI signal detected", str(ctx.exception))

        # Non-finite values in signal
        signals_nan = np.random.randn(self.t_points, 200)
        signals_nan[10, 20] = np.nan
        with self.assertRaises(ValueError) as ctx:
            compute_connectivity_vector(signals_nan)

    def test_fisher_z_boundary_clipping(self):
        """Singularities at r = +/- 1.0 must be safely clipped by epsilon = 1e-7."""
        signals = np.random.randn(self.t_points, self.n_rois)
        signals[:, 1] = signals[:, 0]  # r = +1.0
        signals[:, 2] = -signals[:, 0]  # r = -1.0

        z, _ = compute_connectivity_vector(signals)
        self.assertTrue(np.all(np.isfinite(z)))

        expected_max_z = np.arctanh(1.0 - FISHER_EPSILON)
        self.assertAlmostEqual(z[0], expected_max_z, places=7)

        expected_min_z = np.arctanh(-1.0 + FISHER_EPSILON)
        self.assertAlmostEqual(z[1], expected_min_z, places=7)


class TestSyntheticModelInterfaceSmoke(unittest.TestCase):
    """2. Synthetic model-interface smoke tests using disposable test fixtures."""

    def test_synthetic_pipeline_interface_smoke_with_disposable_fixture(self):
        """Smoke test verifying feature transformation flow using a disposable toy pipeline."""
        n_train_subjects = 30
        X_train = np.random.randn(n_train_subjects, 19900)
        y_train = np.random.randint(0, 2, size=n_train_subjects)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", SelectKBest(score_func=f_classif, k=100)),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)

        test_signals = np.random.randn(60, 200)
        z, _ = compute_connectivity_vector(test_signals)

        X_new = z.reshape(1, -1)
        self.assertEqual(X_new.shape, (1, 19900))

        probs = pipeline.predict_proba(X_new)[0]
        pred = pipeline.predict(X_new)[0]

        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)
        self.assertIn(pred, [0, 1])


class TestFrozenArtifactIntegrityAndDiscovery(unittest.TestCase):
    """3. Frozen artifact discovery and integrity tests (Task 6 Cases A-G)."""

    def test_case_a_no_neuroadhd_model_dir_configured(self):
        """Case A: Calling verify_and_load_frozen_model without NEUROADHD_MODEL_DIR fails closed."""
        old_val = os.environ.pop("NEUROADHD_MODEL_DIR", None)
        try:
            with self.assertRaises(FileNotFoundError) as ctx:
                verify_and_load_frozen_model(None)
            self.assertIn("NEUROADHD_MODEL_DIR", str(ctx.exception))
        finally:
            if old_val is not None:
                os.environ["NEUROADHD_MODEL_DIR"] = old_val

    def test_case_b_model_directory_does_not_exist(self):
        """Case B: Model directory does not exist fails closed."""
        nonexistent = Path("/nonexistent/directory/for/neuro_adhd_models")
        with self.assertRaises(FileNotFoundError) as ctx:
            verify_and_load_frozen_model(nonexistent)
        self.assertIn("does not exist", str(ctx.exception))

    def test_case_c_pipeline_missing(self):
        """Case C: Model directory exists but preprocessing pipeline is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            model_file = tmp / "final_model.joblib"
            model_file.write_bytes(b"dummy_model")
            with self.assertRaises(FileNotFoundError) as ctx:
                verify_and_load_frozen_model(tmp)
            self.assertIn("pipeline artifact missing", str(ctx.exception))

    def test_case_d_classifier_missing(self):
        """Case D: Model directory exists but classifier is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pipe_file = tmp / "final_preprocessing_pipeline.joblib"
            pipe_file.write_bytes(b"dummy_pipe")
            with self.assertRaises(FileNotFoundError) as ctx:
                verify_and_load_frozen_model(tmp)
            self.assertIn("classifier artifact missing", str(ctx.exception))

    def test_case_e_pipeline_checksum_mismatch(self):
        """Case E: Pipeline checksum mismatch fails closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pipe_file = tmp / "final_preprocessing_pipeline.joblib"
            pipe_file.write_bytes(b"corrupted_pipeline_data")
            model_file = tmp / "final_model.joblib"
            model_file.write_bytes(b"dummy_model_data")

            with self.assertRaises(ValueError) as ctx:
                verify_and_load_frozen_model(tmp)
            self.assertIn("Frozen preprocessing pipeline SHA-256 mismatch", str(ctx.exception))

    def test_case_f_classifier_checksum_mismatch(self):
        """Case F: Classifier checksum mismatch fails closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            model_dir = os.environ.get("NEUROADHD_MODEL_DIR")
            pipe_file = tmp / "final_preprocessing_pipeline.joblib"
            model_file = tmp / "final_model.joblib"

            if model_dir and (Path(model_dir) / "final_preprocessing_pipeline.joblib").is_file():
                # Copy real pipeline so pipeline hash passes, then corrupt classifier
                pipe_file.write_bytes((Path(model_dir) / "final_preprocessing_pipeline.joblib").read_bytes())
                model_file.write_bytes(b"corrupted_classifier_bytes")
                with self.assertRaises(ValueError) as ctx:
                    verify_and_load_frozen_model(tmp)
                self.assertIn("Frozen classifier SHA-256 mismatch", str(ctx.exception))
            else:
                pipe_file.write_bytes(b"corrupted_pipeline")
                model_file.write_bytes(b"corrupted_classifier")
                with self.assertRaises(ValueError) as ctx:
                    verify_and_load_frozen_model(tmp)
                self.assertIn("SHA-256 mismatch", str(ctx.exception))

    def test_case_g_artifact_file_is_symlink(self):
        """Case G: Symlinks are strictly prohibited for model directories and artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            real_file = tmp / "real.joblib"
            real_file.write_bytes(b"some content")

            # Test 1: directory symlink
            sym_dir = tmp / "symlink_dir"
            sym_dir.symlink_to(tmp, target_is_directory=True)
            with self.assertRaises(RuntimeError) as ctx:
                verify_and_load_frozen_model(sym_dir)
            self.assertIn("Symlink substitution detected", str(ctx.exception))

            # Test 2: file symlink inside real directory
            sym_pipe = tmp / "final_preprocessing_pipeline.joblib"
            sym_pipe.symlink_to(real_file)
            real_model = tmp / "final_model.joblib"
            real_model.write_bytes(b"some content")
            with self.assertRaises(RuntimeError) as ctx:
                verify_and_load_frozen_model(tmp)
            self.assertIn("Symlink substitution detected at pipeline artifact", str(ctx.exception))

    def test_case_h_authentic_model_artifacts_mounted_correctly(self):
        """Case H: Verify authentic frozen artifact SHA-256 hashes and architecture if NEUROADHD_MODEL_DIR is set."""
        model_dir = os.environ.get("NEUROADHD_MODEL_DIR")
        if not model_dir or not (Path(model_dir) / "final_model.joblib").is_file():
            self.skipTest(
                "NEUROADHD_MODEL_DIR is not set or artifacts are unavailable. "
                "Frozen artifact checksum and architectural verification was not performed."
            )

        pipeline, model = verify_and_load_frozen_model(model_dir)

        # Preprocessing pipeline checks
        self.assertEqual(list(pipeline.named_steps.keys()), EXPECTED_STEP_NAMES)
        self.assertEqual(len(pipeline.named_steps["imputer"].statistics_), 19900)
        self.assertEqual(len(pipeline.named_steps["scaler"].mean_), 19900)
        self.assertEqual(pipeline.named_steps["select"].k, 100)
        actual_indices = list(pipeline.named_steps["select"].get_support(indices=True))
        self.assertEqual([int(x) for x in actual_indices], EXPECTED_SELECTED_FEATURE_INDICES)

        # Classifier checks
        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.max_depth, 3)
        self.assertEqual(model.class_weight, "balanced")
        self.assertEqual(model.random_state, 42)
        self.assertEqual(list(model.classes_), [0, 1])
        self.assertEqual(model.n_features_in_, 100)

    def test_require_model_inference_fails_closed_when_missing(self):
        """Explicitly requiring model inference without valid model dir must fail closed."""
        from scripts.run_poc_v2 import run_poc_v2_pipeline
        old_val = os.environ.pop("NEUROADHD_MODEL_DIR", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                with self.assertRaises(RuntimeError) as ctx:
                    run_poc_v2_pipeline(
                        bold_path=tmp / "bold.nii.gz",
                        bold_json_path=tmp / "bold.json",
                        t1w_path=tmp / "t1w.nii.gz",
                        atlas_path=tmp / "atlas.nii.gz",
                        template_path=tmp / "template.nii.gz",
                        output_dir=tmp / "out",
                        require_model_inference=True,
                    )
                self.assertIn("SAFETY VIOLATION", str(ctx.exception))
        finally:
            if old_val is not None:
                os.environ["NEUROADHD_MODEL_DIR"] = old_val


class TestFrozenArtifactIntegration(unittest.TestCase):
    """4. Actual frozen artifact integration tests (Case I: zero-refit inference pass)."""

    def test_case_i_actual_synthetic_inference_completes_successfully(self):
        """Case I: Verify end-to-end inference pass with real frozen artifacts when NEUROADHD_MODEL_DIR is set."""
        model_dir = os.environ.get("NEUROADHD_MODEL_DIR")
        if not model_dir or not (Path(model_dir) / "final_model.joblib").is_file():
            self.skipTest(
                "NEUROADHD_MODEL_DIR is not set or artifacts are unavailable. "
                "Real frozen artifact integration test was not performed."
            )

        pipeline, model = verify_and_load_frozen_model(model_dir)

        # 1. Generate synthetic 60-timepoint x 200-ROI signals using our connectivity generator
        np.random.seed(123)
        synthetic_signals = np.random.randn(60, 200)

        # 2. Compute 19,900 Fisher-z connectivity vector
        z_vector, names = compute_connectivity_vector(synthetic_signals)
        self.assertEqual(len(z_vector), 19900)
        self.assertEqual(len(names), 19900)
        self.assertEqual(names, get_feature_names())
        self.assertTrue(np.all(np.isfinite(z_vector)))

        # 3. Transform via loaded frozen pipeline (transform() only, NO fit())
        X_raw = z_vector.reshape(1, -1)
        self.assertEqual(X_raw.shape, (1, 19900))

        X_selected = pipeline.transform(X_raw)
        self.assertEqual(X_selected.shape, (1, 100))

        # 4. Predict via loaded frozen classifier
        probs = model.predict_proba(X_selected)[0]
        pred = model.predict(X_selected)[0]

        self.assertIn(pred, [0, 1])
        self.assertEqual(len(probs), 2)
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertTrue(0.0 <= probs[0] <= 1.0)
        self.assertTrue(0.0 <= probs[1] <= 1.0)

        # 5. Verify through run_frozen_inference helper
        res = run_frozen_inference(z_vector, pipeline, model)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["features_in"], 19900)
        self.assertEqual(res["features_selected"], 100)
        self.assertFalse(res["refit_performed"])
        self.assertEqual(res["predicted_class"], pred)


if __name__ == "__main__":
    unittest.main()
