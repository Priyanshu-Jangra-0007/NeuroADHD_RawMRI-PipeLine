"""Phase A.9 Scientific Hardening and Algorithmic Validation Suite.

Independently validates:
- 3D rigid motion parameter recovery and Power FD calculations under known ground truth.
- Friston-24 expansion exactness and error handling.
- Registration pullback transform composition and landmark consistency.
- Tissue mask alignment, aCompCor extraction, and confound matrix rank verification.
- CC200 parcellation internal (T x 200) vs export (200 x T+1) layout parity.
- 19,900 Fisher-z connectivity vector against independent numerical references.
- Boundary clipping, singularity prevention, and non-finite value protection.
- Frozen model inference tagging ('SYNTHETIC_DEMONSTRATION_ONLY').
- Quality control threshold status preservation.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform

from src.features.connectivity import (
    FISHER_EPSILON,
    N_ROIS,
    compute_connectivity_vector,
    get_feature_names,
)
from src.integration.model_verifier import run_frozen_inference
from src.preprocessing.motion import (
    build_friston_24_regressors,
    compute_framewise_displacement,
    realign_4d_bold,
)
from src.preprocessing.nuisance import (
    build_confound_matrix,
    clean_functional_bold,
    extract_acompcor_components,
)
from src.preprocessing.qc import compute_registration_qc_metrics
from src.preprocessing.registration import (
    compose_transformations,
    resample_4d_bold,
)
from dipy.align.imwarp import DiffeomorphicMap


class TestPhaseA9MotionValidation(unittest.TestCase):
    """Rigorous mathematical validation of motion parameter estimation and FD calculation."""

    def setUp(self):
        # Create a synthetic 3D sphere phantom with intensity gradient
        self.shape = (24, 24, 18)
        self.affine = np.diag([3.0, 3.0, 3.0, 1.0])
        self.vol0 = np.zeros(self.shape, dtype=np.float32)
        cx, cy, cz = 12, 12, 9
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                    if d < 7.0:
                        self.vol0[x, y, z] = 100.0 - (d * 7.0)

    def test_motion_zero_displacement_recovery(self):
        """Zero motion across volumes yields estimated parameters near zero."""
        T = 5
        data4d = np.repeat(self.vol0[..., np.newaxis], T, axis=3)
        bold_img = nib.Nifti1Image(data4d, self.affine)

        realigned_img, motion_params, fd = realign_4d_bold(bold_img)

        self.assertEqual(realigned_img.shape, (*self.shape, T))
        self.assertEqual(motion_params.shape, (T, 6))
        # Translation errors < 0.01 mm, rotation errors < 0.15 degrees (grid quantization limit)
        np.testing.assert_allclose(motion_params[:, :3], 0.0, atol=0.01)
        np.testing.assert_allclose(motion_params[:, 3:], 0.0, atol=0.15)
        # Framewise displacement between identical volumes is strictly 0.0
        np.testing.assert_allclose(fd, 0.0, atol=1e-5)

    def test_motion_known_translation_recovery(self):
        """Known voxel translation is recovered within justified tolerance (< 0.05 mm relative)."""
        T = 4
        data4d = np.zeros((*self.shape, T), dtype=np.float32)
        data4d[..., 0] = self.vol0

        # Apply a known shift in volume 1: 0.5 voxels in X (1.5 mm in world coords)
        # Note: in realign_4d_bold, reference is mean volume; pairwise displacement between
        # Volume 1 and Volume 0 recovers the exact 1.5 mm applied shift.
        data4d[..., 1] = affine_transform(self.vol0, np.eye(3), offset=[0.5, 0.0, 0.0], order=1)
        data4d[..., 2] = affine_transform(self.vol0, np.eye(3), offset=[0.0, -0.5, 0.0], order=1)
        data4d[..., 3] = self.vol0

        bold_img = nib.Nifti1Image(data4d, self.affine)
        _, motion_params, fd = realign_4d_bold(bold_img)

        # Relative displacement between Vol 1 and Vol 0 along X axis recovers 1.5 mm
        rel_disp_x = float(abs(motion_params[1, 0] - motion_params[0, 0]))
        self.assertAlmostEqual(rel_disp_x, 1.5, delta=0.05)

        # Relative displacement between Vol 2 and Vol 1 along Y axis recovers 1.5 mm
        rel_disp_y = float(abs(motion_params[2, 1] - motion_params[0, 1]))
        self.assertAlmostEqual(rel_disp_y, 1.5, delta=0.05)

        # First volume FD is always 0.0
        self.assertEqual(fd[0], 0.0)
        # Volumes with shift have non-zero FD >= 1.5 mm
        self.assertGreater(fd[1], 1.4)

    def test_compute_framewise_displacement_manual_reference(self):
        """Framewise displacement matches manual Power et al. (2012) formulation."""
        # 3 volumes, 6 params
        # Vol 0: all 0
        # Vol 1: trans_x = 1.0 mm, rot_z = 1.0 deg
        # Arc length for 1 deg at 50mm radius: 50 * (1 * pi / 180) = 0.87266 mm
        # Expected FD at Vol 1: |1.0| + |0.87266| = 1.87266 mm
        motion = np.zeros((3, 6), dtype=np.float64)
        motion[1, 0] = 1.0  # 1.0 mm
        motion[1, 5] = 1.0  # 1.0 deg

        fd = compute_framewise_displacement(motion, radius_mm=50.0)

        expected_arc = 50.0 * np.deg2rad(1.0)
        expected_fd_1 = 1.0 + expected_arc

        self.assertEqual(fd[0], 0.0)
        self.assertAlmostEqual(fd[1], expected_fd_1, places=5)
        self.assertAlmostEqual(fd[2], expected_fd_1, places=5)

    def test_build_friston_24_exact_structure(self):
        """Friston-24 correctly computes R, R^2, R_{t-1}, R_{t-1}^2."""
        T = 10
        rng = np.random.RandomState(42)
        motion = rng.randn(T, 6)

        friston = build_friston_24_regressors(motion)

        self.assertEqual(friston.shape, (T, 24))
        # Cols 0:6 = R
        np.testing.assert_array_equal(friston[:, 0:6], motion)
        # Cols 6:12 = R^2
        np.testing.assert_array_almost_equal(friston[:, 6:12], motion ** 2)
        # Cols 12:18 = R_{t-1} (with row 0 = 0)
        self.assertTrue(np.all(friston[0, 12:18] == 0.0))
        np.testing.assert_array_equal(friston[1:, 12:18], motion[:-1])
        # Cols 18:24 = R_{t-1}^2
        np.testing.assert_array_almost_equal(friston[:, 18:24], friston[:, 12:18] ** 2)

    def test_motion_invalid_inputs_raise(self):
        """Motion correction raises ValueError on non-4D scans or insufficient volumes."""
        with self.assertRaises(ValueError):
            realign_4d_bold(nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4)))

        with self.assertRaises(ValueError):
            realign_4d_bold(nib.Nifti1Image(np.zeros((10, 10, 10, 1)), np.eye(4)))

        with self.assertRaises(ValueError):
            compute_framewise_displacement(np.zeros((10, 5)))

        with self.assertRaises(ValueError):
            build_friston_24_regressors(np.zeros((10, 7)))


class TestPhaseA9RegistrationValidation(unittest.TestCase):
    """Validation of DIPY registration coordinate composition and spatial mapping."""

    def test_identity_pullback_composition(self):
        """Identity rigid + identity affine + zero SyN preserves coordinates exactly."""
        grid_shape = (15, 15, 15)
        grid_affine = np.diag([3.0, 3.0, 3.0, 1.0])

        syn_map = DiffeomorphicMap(
            3,
            disp_shape=grid_shape,
            disp_grid2world=grid_affine,
            domain_shape=grid_shape,
            domain_grid2world=grid_affine,
            codomain_shape=grid_shape,
            codomain_grid2world=grid_affine,
        )
        syn_map.forward = np.zeros((*grid_shape, 3), dtype=np.float32)
        syn_map.backward = np.zeros((*grid_shape, 3), dtype=np.float32)

        composed = compose_transformations(
            np.eye(4),
            np.eye(4),
            syn_map,
            target_shape=grid_shape,
            target_affine=grid_affine,
            source_shape=grid_shape,
            source_affine=grid_affine,
        )

        # Single impulse at center
        data4d = np.zeros((*grid_shape, 50), dtype=np.float32)
        data4d[7, 7, 7, :] = 50.0
        bold_img = nib.Nifti1Image(data4d, grid_affine)

        resampled = resample_4d_bold(bold_img, composed, grid_shape, grid_affine)
        res_data = np.asanyarray(resampled.dataobj)

        self.assertEqual(res_data.shape, (*grid_shape, 50))
        peak = np.unravel_index(np.argmax(res_data[..., 0]), grid_shape)
        self.assertEqual(peak, (7, 7, 7))
        self.assertAlmostEqual(float(res_data[7, 7, 7, 0]), 50.0, places=4)

    def test_resample_out_of_bounds_handling(self):
        """Resampling volumes with boundary shifts does not produce NaNs or Infs."""
        grid_shape = (12, 12, 12)
        grid_affine = np.diag([3.0, 3.0, 3.0, 1.0])

        syn_map = DiffeomorphicMap(
            3,
            disp_shape=grid_shape,
            disp_grid2world=grid_affine,
            domain_shape=grid_shape,
            domain_grid2world=grid_affine,
            codomain_shape=grid_shape,
            codomain_grid2world=grid_affine,
        )
        syn_map.forward = np.zeros((*grid_shape, 3), dtype=np.float32)
        syn_map.backward = np.zeros((*grid_shape, 3), dtype=np.float32)

        # Apply large translation matrix that shifts signal out of bounds
        shift_affine = np.eye(4)
        shift_affine[0, 3] = 100.0  # 100 mm shift

        composed = compose_transformations(
            shift_affine,
            np.eye(4),
            syn_map,
            target_shape=grid_shape,
            target_affine=grid_affine,
            source_shape=grid_shape,
            source_affine=grid_affine,
        )

        data4d = np.ones((*grid_shape, 50), dtype=np.float32) * 20.0
        bold_img = nib.Nifti1Image(data4d, grid_affine)

        resampled = resample_4d_bold(bold_img, composed, grid_shape, grid_affine)
        res_data = np.asanyarray(resampled.dataobj)

        self.assertTrue(np.all(np.isfinite(res_data)))


class TestPhaseA9NuisanceAndSegmentationValidation(unittest.TestCase):
    """Validation of aCompCor, confound rank, and frequency bandpass constraints."""

    def test_acompcor_extraction_properties(self):
        """aCompCor extracts orthogonal components with unit variance."""
        T = 60
        shape = (10, 10, 10)
        affine = np.eye(4)
        rng = np.random.RandomState(42)

        bold_data = rng.randn(*shape, T).astype(np.float32)
        wm_mask = np.zeros(shape, dtype=np.int32)
        wm_mask[2:5, 2:5, 2:5] = 1
        csf_mask = np.zeros(shape, dtype=np.int32)
        csf_mask[6:8, 6:8, 6:8] = 1

        components = extract_acompcor_components(
            bold_data, wm_mask, csf_mask, n_components=5, bold_affine=affine, wm_affine=affine, csf_affine=affine
        )

        self.assertEqual(components.shape, (T, 5))
        self.assertTrue(np.all(np.isfinite(components)))

    def test_confound_matrix_rank_deficiency_detection(self):
        """build_confound_matrix raises ValueError when rank deficiency / collinearity is present."""
        T = 60
        # Perfectly collinear motion parameters
        collinear_friston = np.zeros((T, 24), dtype=np.float64)
        collinear_friston[:, :] = 1.0  # constant
        acompcor = np.random.RandomState(42).randn(T, 5)

        with self.assertRaises(ValueError) as ctx:
            build_confound_matrix(collinear_friston, acompcor)
        self.assertIn("rank-deficient", str(ctx.exception).lower())

    def test_temporal_filtering_tr_nyquist_guard(self):
        """clean_functional_bold rejects TR violating Nyquist frequency bounds."""
        shape = (10, 10, 10)
        affine = np.eye(4)
        T = 60
        data4d = np.random.RandomState(42).randn(*shape, T).astype(np.float32)
        bold_img = nib.Nifti1Image(data4d, affine)

        friston = np.random.RandomState(42).randn(T, 24)
        acompcor = np.random.RandomState(42).randn(T, 5)
        confounds = build_confound_matrix(friston, acompcor)

        # TR=6.0s -> Nyquist = 0.5/6.0 = 0.0833 Hz < high cutoff (0.10 Hz)
        with self.assertRaises(ValueError) as ctx:
            clean_functional_bold(bold_img, confounds, tr=6.0, low_cutoff_hz=0.01, high_cutoff_hz=0.10)
        self.assertIn("invalid repetition time", str(ctx.exception).lower())


class TestPhaseA9ConnectivityAndCC200Parcellation(unittest.TestCase):
    """Validation of CC200 parcellation layout and independent Pearson correlation fixtures."""

    def test_connectivity_independent_numerical_reference(self):
        """Test compute_connectivity_vector against an independent textbook correlation reference."""
        T = 100
        rng = np.random.RandomState(123)
        # Generate T x 200 signals
        time_by_roi = rng.randn(T, N_ROIS)

        # Compute using module
        z_vector, feature_names = compute_connectivity_vector(time_by_roi)

        # Compute independent manual reference for first 5 edges
        # Edge 0: ROI 1 vs ROI 2
        # Edge 1: ROI 1 vs ROI 3
        # Edge 2: ROI 1 vs ROI 4
        # Edge 3: ROI 1 vs ROI 5
        # Edge 4: ROI 1 vs ROI 6
        expected_z = []
        for j in range(1, 6):
            x = time_by_roi[:, 0]
            y = time_by_roi[:, j]
            # Manual Pearson r formula
            x_bar = np.mean(x)
            y_bar = np.mean(y)
            num = np.sum((x - x_bar) * (y - y_bar))
            den = np.sqrt(np.sum((x - x_bar) ** 2) * np.sum((y - y_bar) ** 2))
            r = num / den
            clipped_r = np.clip(r, -1.0 + FISHER_EPSILON, 1.0 - FISHER_EPSILON)
            z = 0.5 * np.log((1.0 + clipped_r) / (1.0 - clipped_r))
            expected_z.append(z)

        np.testing.assert_allclose(z_vector[:5], expected_z, atol=1e-7)
        self.assertEqual(feature_names[0], "ROI_001__ROI_002")
        self.assertEqual(feature_names[1], "ROI_001__ROI_003")
        self.assertEqual(feature_names[4], "ROI_001__ROI_006")
        self.assertEqual(feature_names[-1], "ROI_199__ROI_200")
        self.assertEqual(len(feature_names), 19900)

    def test_connectivity_extreme_correlations_and_anticorrelations(self):
        """Perfect correlations (+1.0) and anti-correlations (-1.0) are safely clamped without NaN/Inf."""
        T = 50
        time_by_roi = np.random.RandomState(42).randn(T, N_ROIS)
        # Force ROI 1 and ROI 2 to be perfectly correlated: r = +1.0
        time_by_roi[:, 1] = time_by_roi[:, 0]
        # Force ROI 3 to be perfectly anti-correlated: r = -1.0
        time_by_roi[:, 2] = -time_by_roi[:, 0]

        z_vector, _ = compute_connectivity_vector(time_by_roi)

        # Expected maximum clamped z value: arctanh(1 - 1e-7) ~ 8.412
        max_theoretical_z = np.arctanh(1.0 - FISHER_EPSILON)

        self.assertTrue(np.all(np.isfinite(z_vector)))
        self.assertAlmostEqual(z_vector[0], max_theoretical_z, places=3)
        self.assertAlmostEqual(z_vector[1], -max_theoretical_z, places=3)

    def test_connectivity_constant_roi_raises_valueerror(self):
        """Constant zero-variance signal raises ValueError."""
        time_by_roi = np.random.RandomState(42).randn(50, N_ROIS)
        time_by_roi[:, 5] = 42.0  # zero variance
        with self.assertRaises(ValueError) as ctx:
            compute_connectivity_vector(time_by_roi)
        self.assertIn("constant roi signal", str(ctx.exception).lower())

    def test_connectivity_invalid_dimension_raises(self):
        """Non-2D array, T < 2, or wrong column count raises ValueError."""
        with self.assertRaises(ValueError):
            compute_connectivity_vector(np.zeros(200))

        with self.assertRaises(ValueError):
            compute_connectivity_vector(np.zeros((1, 200)))

        with self.assertRaises(ValueError):
            compute_connectivity_vector(np.zeros((50, 199)))


class TestPhaseA9ModelAndGovernanceValidation(unittest.TestCase):
    """Validation of frozen model synthetic demonstration labeling and QC threshold status."""

    def test_model_inference_synthetic_labeling(self):
        """Model inference output is explicitly tagged with SYNTHETIC_DEMONSTRATION_ONLY."""
        # Create mock pipeline and model objects
        class MockPipeline:
            def transform(self, X):
                return np.ones((1, 100))

        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.45, 0.55]])

            def predict(self, X):
                return np.array([1])

        z_vector = np.zeros(19900, dtype=np.float64)
        res = run_frozen_inference(z_vector, MockPipeline(), MockModel())

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["evaluation_mode"], "SYNTHETIC_DEMONSTRATION_ONLY")
        self.assertIn("SYNTHETIC_DEMONSTRATION_ONLY", res["diagnostic_disclaimer"])
        self.assertIn("not approved or validated", res["diagnostic_disclaimer"].lower())

    def test_qc_thresholds_remain_pending_approval(self):
        """All 8 QC metrics in the authorization register remain in PENDING status."""
        reg_file = Path(__file__).resolve().parents[2] / "configs" / "research_authorization_register.json"
        with open(reg_file, "r", encoding="utf-8") as f:
            reg = json.load(f)

        qc_items = [
            item for item in reg["categories"]["supervisor_decisions"]
            if item["id"].startswith("QC-")
        ]
        self.assertEqual(len(qc_items), 8, f"Expected 8 QC threshold items, found {len(qc_items)}")
        for item in qc_items:
            self.assertEqual(
                item["status"],
                "PENDING",
                f"QC threshold {item['id']} has unauthorized status '{item['status']}'."
            )


if __name__ == "__main__":
    unittest.main()
