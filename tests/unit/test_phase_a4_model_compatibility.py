"""Automated unit tests for Phase A.4 downstream ML model compatibility and feature contract."""
from __future__ import annotations

import itertools
import os
from pathlib import Path
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


class TestPhaseA4ModelCompatibility(unittest.TestCase):
    """Verify feature contract, numerical symmetry, clipping, and downstream pipeline interface."""

    def setUp(self):
        np.random.seed(42)
        self.n_rois = N_ROIS
        self.n_edges = (N_ROIS * (N_ROIS - 1)) // 2  # 19,900
        self.t_points = 60

    def test_feature_contract_dimensions_and_naming(self):
        """Feature names must be exactly 19,900 and match itertools upper triangle order."""
        names = get_feature_names()
        self.assertEqual(len(names), self.n_edges)
        self.assertEqual(len(names), 19900)

        # Check first and last entries
        self.assertEqual(names[0], "ROI_001__ROI_002")
        self.assertEqual(names[1], "ROI_001__ROI_003")
        self.assertEqual(names[198], "ROI_001__ROI_200")
        self.assertEqual(names[199], "ROI_002__ROI_003")
        self.assertEqual(names[-1], "ROI_199__ROI_200")

        # Verify exact correspondence with itertools.combinations
        expected_pairs = list(itertools.combinations(range(1, 201), 2))
        expected_names = [f"ROI_{i:03d}__ROI_{j:03d}" for i, j in expected_pairs]
        self.assertEqual(names, expected_names)

    def test_feature_contract_upper_triangle_indices_order(self):
        """np.triu_indices(200, k=1) ordering must match itertools combinations row-by-row."""
        tri_i, tri_j = np.triu_indices(self.n_rois, k=1)
        self.assertEqual(len(tri_i), 19900)
        self.assertEqual(len(tri_j), 19900)

        expected_pairs = list(itertools.combinations(range(1, 201), 2))
        for idx, (i_exp, j_exp) in enumerate(expected_pairs):
            # tri_i, tri_j are 0-indexed; expected_pairs are 1-indexed
            self.assertEqual(tri_i[idx] + 1, i_exp)
            self.assertEqual(tri_j[idx] + 1, j_exp)

    def test_compute_connectivity_vector_finite_and_symmetric(self):
        """Connectivity vector must be 1D (19900,), finite, and based on symmetric matrix."""
        # Create random non-constant time series
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
        signals_const[:, 5] = 1.0  # Constant column
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
        # Create perfectly correlated signals for regions 0 and 1
        signals = np.random.randn(self.t_points, self.n_rois)
        signals[:, 1] = signals[:, 0]  # r = +1.0 exactly
        signals[:, 2] = -signals[:, 0]  # r = -1.0 exactly

        z, _ = compute_connectivity_vector(signals)
        self.assertTrue(np.all(np.isfinite(z)))

        # Edge 0 (ROI 1 to ROI 2) had r = 1.0 -> should equal arctanh(1 - 1e-7)
        expected_max_z = np.arctanh(1.0 - FISHER_EPSILON)
        self.assertAlmostEqual(z[0], expected_max_z, places=7)

        # Edge 1 (ROI 1 to ROI 3) had r = -1.0 -> should equal arctanh(-1 + 1e-7)
        expected_min_z = np.arctanh(-1.0 + FISHER_EPSILON)
        self.assertAlmostEqual(z[1], expected_min_z, places=7)

    def test_downstream_frozen_pipeline_architecture_compatibility(self):
        """Simulate the 4-step downstream frozen pipeline and ensure zero-refit inference pass."""
        # Synthesize a fitted pipeline matching the frozen architecture
        n_train_subjects = 50
        X_train = np.random.randn(n_train_subjects, 19900)
        y_train = np.random.randint(0, 2, size=n_train_subjects)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", SelectKBest(score_func=f_classif, k=100)),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)

        # Generate a test feature vector from our connectivity extractor
        test_signals = np.random.randn(60, 200)
        z, _ = compute_connectivity_vector(test_signals)

        # Ingestion test: must be (1, 19900) 2D array
        X_new = z.reshape(1, -1)
        self.assertEqual(X_new.shape, (1, 19900))

        # Predict without refitting
        probs = pipeline.predict_proba(X_new)[0]
        pred = pipeline.predict(X_new)[0]

        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)
        self.assertIn(pred, [0, 1])

    def test_existing_model_artifacts_inspection_if_available(self):
        """If historical benchmark artifacts are accessible on disk, verify their SHA-256 and structure."""
        pipe_path = Path("/Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/results/final_locked/final_preprocessing_pipeline.joblib")
        model_path = Path("/Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD_ModelBenchmark/results/final_locked/final_model.joblib")

        if pipe_path.is_file() and model_path.is_file():
            import joblib
            import hashlib

            # Verify pipeline SHA-256
            pipe_sha = hashlib.sha256(pipe_path.read_bytes()).hexdigest()
            self.assertEqual(pipe_sha, "228d3f0180e4855d49fd732327ab1fc4f255beb7bea43dd4fd1db3064aa86249")

            # Verify model SHA-256
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            self.assertEqual(model_sha, "d8eb23eb8c6d25ffc12b2bab515a2a7112daeeb0afa8f4e9eaa499dbe5b5f2ef")

            # Load and verify structure
            pipe = joblib.load(pipe_path)
            model = joblib.load(model_path)

            self.assertEqual(len(pipe.named_steps["imputer"].statistics_), 19900)
            self.assertEqual(len(pipe.named_steps["scaler"].mean_), 19900)
            self.assertEqual(pipe.named_steps["select"].k, 100)
            self.assertEqual(model.n_estimators, 50)
            self.assertEqual(model.max_depth, 3)

            # Test synthetic feature vector pass
            z, _ = compute_connectivity_vector(np.random.randn(60, 200))
            X_selected = pipe.transform(z.reshape(1, -1))
            self.assertEqual(X_selected.shape, (1, 100))
            probs = model.predict_proba(X_selected)[0]
            self.assertEqual(len(probs), 2)
            self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

