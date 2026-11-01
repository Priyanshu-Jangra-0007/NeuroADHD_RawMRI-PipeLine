"""Unit tests for nuisance confound matrix generation and simultaneous temporal filtering."""
from __future__ import annotations

import unittest
import nibabel as nib
import numpy as np
import pandas as pd

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


class TestNuisanceRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.t_vol = 60
        # Realistic head motion trajectory (random walk drift + jitter)
        # 3 translations (mm), 3 rotations (degrees)
        steps = np.random.normal(0, 0.05, (self.t_vol, 6))
        self.motion_6 = np.cumsum(steps, axis=0)

    def test_compute_framewise_displacement(self):
        fd = compute_framewise_displacement(self.motion_6, radius_mm=50.0)

        self.assertEqual(len(fd), self.t_vol)
        # First volume must be strictly 0.0
        self.assertEqual(fd[0], 0.0)
        # All FD values must be non-negative
        self.assertTrue(np.all(fd >= 0.0))

        # Check explicit manual calculation for volume 1
        t_diff = np.abs(self.motion_6[1, :3] - self.motion_6[0, :3])
        r_diff_deg = np.abs(self.motion_6[1, 3:] - self.motion_6[0, 3:])
        r_diff_mm = 50.0 * np.deg2rad(r_diff_deg)
        expected_fd_1 = float(np.sum(t_diff) + np.sum(r_diff_mm))
        self.assertAlmostEqual(fd[1], expected_fd_1, places=6)

    def test_build_friston_24_regressors(self):
        friston = build_friston_24_regressors(self.motion_6)

        self.assertEqual(friston.shape, (self.t_vol, 24))
        # Columns 0..5: R
        np.testing.assert_allclose(friston[:, :6], self.motion_6)
        # Columns 6..11: R^2
        np.testing.assert_allclose(friston[:, 6:12], self.motion_6 ** 2)
        # Columns 12..17: R_{t-1} with t0=0
        expected_lag = np.vstack([np.zeros((1, 6)), self.motion_6[:-1, :]])
        np.testing.assert_allclose(friston[:, 12:18], expected_lag)
        # Columns 18..23: R_{t-1}^2
        np.testing.assert_allclose(friston[:, 18:24], expected_lag ** 2)

    def test_extract_acompcor_components(self):
        # 4D BOLD array (15, 15, 10, 60)
        bold_data = np.random.normal(100.0, 10.0, (15, 15, 10, self.t_vol)).astype(np.float32)

        # Create disjoint eroded WM and CSF masks
        wm_mask = np.zeros((15, 15, 10), dtype=bool)
        wm_mask[4:8, 4:8, 3:7] = True

        csf_mask = np.zeros((15, 15, 10), dtype=bool)
        csf_mask[1:3, 1:3, 1:4] = True

        compcor = extract_acompcor_components(bold_data, wm_mask, csf_mask, n_components=5)

        self.assertEqual(compcor.shape, (self.t_vol, 5))
        # All values must be finite
        self.assertTrue(np.all(np.isfinite(compcor)))
        # Components must have zero mean
        np.testing.assert_allclose(np.mean(compcor, axis=0), np.zeros(5), atol=1e-5)

    def test_extract_acompcor_insufficient_voxels_fails(self):
        bold_data = np.random.normal(100.0, 10.0, (10, 10, 5, self.t_vol)).astype(np.float32)
        tiny_mask = np.zeros((10, 10, 5), dtype=bool)
        tiny_mask[0, 0, 0] = True  # only 1 voxel < 5 components

        with self.assertRaises(ValueError) as ctx:
            extract_acompcor_components(bold_data, tiny_mask, np.zeros_like(tiny_mask), n_components=5)
        self.assertIn("Insufficient physiological voxels", str(ctx.exception))

    def test_build_confound_matrix_structure_and_rank(self):
        friston = build_friston_24_regressors(self.motion_6)
        compcor = np.random.normal(0, 1, (self.t_vol, 5))

        df = build_confound_matrix(friston, compcor)

        # Exactly 31 columns
        self.assertEqual(df.shape, (self.t_vol, 31))

        # Check column names
        expected_cols = (
            [f"motion_friston_{i:02d}" for i in range(1, 25)]
            + [f"aCompCor_{i:02d}" for i in range(1, 6)]
            + ["poly_linear", "poly_quadratic"]
        )
        self.assertEqual(list(df.columns), expected_cols)

        # Verify GSR is strictly excluded
        self.assertNotIn("global_signal", df.columns)
        self.assertNotIn("global_mean", df.columns)

        # Verify Mean tissue signals are strictly excluded
        self.assertNotIn("mean_wm", df.columns)
        self.assertNotIn("mean_csf", df.columns)
        self.assertNotIn("mean_gm", df.columns)

        # Verify full column rank (no rank deficiency)
        rank = np.linalg.matrix_rank(df.to_numpy())
        self.assertEqual(rank, 31)

    def test_clean_functional_bold(self):
        # Create 4D BOLD volume with known temporal signal
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
        bold_data = np.random.normal(100.0, 5.0, (12, 12, 8, self.t_vol)).astype(np.float32)
        bold_img = nib.Nifti1Image(bold_data, affine)

        friston = build_friston_24_regressors(self.motion_6)
        compcor = np.random.normal(0, 1, (self.t_vol, 5))
        df_confounds = build_confound_matrix(friston, compcor)

        cleaned_img = clean_functional_bold(
            bold_img,
            df_confounds,
            tr=2.0,
            low_cutoff_hz=0.01,
            high_cutoff_hz=0.10,
        )

        cleaned_data = np.asanyarray(cleaned_img.dataobj)

        self.assertEqual(cleaned_data.shape, (12, 12, 8, self.t_vol))
        self.assertTrue(np.all(np.isfinite(cleaned_data)))
        # Mean across time for residual signals should be close to 0 (average absolute mean < 1.0)
        mean_time = np.mean(cleaned_data, axis=3)
        self.assertLess(float(np.mean(np.abs(mean_time))), 1.0)

    def test_clean_functional_bold_rejects_invalid_tr(self):
        bold_img = nib.Nifti1Image(np.zeros((10, 10, 5, 55), dtype=np.float32), np.eye(4))
        df = pd.DataFrame(np.zeros((55, 31)))

        with self.assertRaises(ValueError) as ctx:
            clean_functional_bold(bold_img, df, tr=10.0)  # TR=10.0 > 5.0s
        self.assertIn("invalid repetition time", str(ctx.exception))

    def test_extract_acompcor_mismatched_wm_mask_shape_fails(self):
        """WM mask with wrong spatial shape must raise ValueError."""
        bold_data = np.random.normal(100.0, 10.0, (15, 15, 10, self.t_vol)).astype(np.float32)

        # WM mask shape doesn't match BOLD spatial (15, 15, 10)
        wrong_wm = np.ones((20, 20, 15), dtype=bool)
        csf_mask = np.zeros((15, 15, 10), dtype=bool)
        csf_mask[1:3, 1:3, 1:4] = True

        with self.assertRaises(ValueError) as ctx:
            extract_acompcor_components(bold_data, wrong_wm, csf_mask, n_components=5)
        self.assertIn("WM mask shape", str(ctx.exception))
        self.assertIn("does not match", str(ctx.exception))

    def test_extract_acompcor_mismatched_csf_mask_shape_fails(self):
        """CSF mask with wrong spatial shape must raise ValueError."""
        bold_data = np.random.normal(100.0, 10.0, (15, 15, 10, self.t_vol)).astype(np.float32)

        wm_mask = np.zeros((15, 15, 10), dtype=bool)
        wm_mask[4:8, 4:8, 3:7] = True
        wrong_csf = np.ones((12, 12, 8), dtype=bool)

        with self.assertRaises(ValueError) as ctx:
            extract_acompcor_components(bold_data, wm_mask, wrong_csf, n_components=5)
        self.assertIn("CSF mask shape", str(ctx.exception))
        self.assertIn("does not match", str(ctx.exception))

    def test_extract_acompcor_4d_mask_fails(self):
        """4D mask (not 3D) must raise ValueError."""
        bold_data = np.random.normal(100.0, 10.0, (15, 15, 10, self.t_vol)).astype(np.float32)

        wm_4d = np.ones((15, 15, 10, 3), dtype=bool)
        csf_mask = np.zeros((15, 15, 10), dtype=bool)

        with self.assertRaises(ValueError) as ctx:
            extract_acompcor_components(bold_data, wm_4d, csf_mask, n_components=5)
        self.assertIn("3D WM mask", str(ctx.exception))

    def test_extract_acompcor_nonfinite_bold_fails(self):
        """BOLD with NaN values must raise ValueError before CompCor extraction."""
        bold_data = np.random.normal(100.0, 10.0, (15, 15, 10, self.t_vol)).astype(np.float32)
        bold_data[3, 3, 3, 5] = np.nan  # inject NaN

        wm_mask = np.zeros((15, 15, 10), dtype=bool)
        wm_mask[4:8, 4:8, 3:7] = True
        csf_mask = np.zeros((15, 15, 10), dtype=bool)
        csf_mask[1:3, 1:3, 1:4] = True

        with self.assertRaises(ValueError) as ctx:
            extract_acompcor_components(bold_data, wm_mask, csf_mask, n_components=5)
        self.assertIn("non-finite", str(ctx.exception))

    def test_clean_functional_bold_nyquist_violation_fails(self):
        """High cutoff at or above Nyquist frequency must raise ValueError."""
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
        bold_img = nib.Nifti1Image(
            np.random.normal(100.0, 5.0, (10, 10, 5, self.t_vol)).astype(np.float32), affine
        )
        df = pd.DataFrame(np.random.normal(0, 1, (self.t_vol, 31)))

        # TR=2.0s -> Nyquist=0.25 Hz; high_cutoff=0.30 Hz > 0.25 Hz -> FAIL
        with self.assertRaises(ValueError) as ctx:
            clean_functional_bold(bold_img, df, tr=2.0, low_cutoff_hz=0.01, high_cutoff_hz=0.30)
        self.assertIn("Nyquist", str(ctx.exception))

    def test_clean_functional_bold_bandpass_order_fails(self):
        """Low cutoff >= high cutoff must raise ValueError."""
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
        bold_img = nib.Nifti1Image(
            np.random.normal(100.0, 5.0, (10, 10, 5, self.t_vol)).astype(np.float32), affine
        )
        df = pd.DataFrame(np.random.normal(0, 1, (self.t_vol, 31)))

        with self.assertRaises(ValueError) as ctx:
            clean_functional_bold(bold_img, df, tr=2.0, low_cutoff_hz=0.10, high_cutoff_hz=0.05)
        self.assertIn("strictly less than", str(ctx.exception))

    def test_clean_functional_bold_nonfinite_bold_fails(self):
        """BOLD image with Inf values must raise ValueError before cleaning."""
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
        data = np.random.normal(100.0, 5.0, (10, 10, 5, self.t_vol)).astype(np.float32)
        data[2, 2, 2, 10] = np.inf  # inject Inf
        bold_img = nib.Nifti1Image(data, affine)

        df = pd.DataFrame(np.random.normal(0, 1, (self.t_vol, 31)))

        with self.assertRaises(ValueError) as ctx:
            clean_functional_bold(bold_img, df, tr=2.0, low_cutoff_hz=0.01, high_cutoff_hz=0.10)
        self.assertIn("non-finite", str(ctx.exception))

    def test_build_confound_matrix_rank_deficiency_reported(self):
        """Rank-deficient confound matrix must raise ValueError with condition number."""
        friston = build_friston_24_regressors(self.motion_6)
        # Duplicate column into aCompCor to create perfect collinearity
        degenerate_acompcor = np.zeros((self.t_vol, 5))
        degenerate_acompcor[:, 0] = friston[:, 0]
        degenerate_acompcor[:, 1] = friston[:, 1]
        degenerate_acompcor[:, 2] = friston[:, 2]
        degenerate_acompcor[:, 3] = friston[:, 3]
        degenerate_acompcor[:, 4] = friston[:, 4]

        with self.assertRaises(ValueError) as ctx:
            build_confound_matrix(friston, degenerate_acompcor)
        self.assertIn("rank-deficient", str(ctx.exception))
        self.assertIn("Condition number", str(ctx.exception))

    def test_extract_acompcor_mismatched_mask_affine_fails(self):
        """WM/CSF mask with mismatched NIfTI affine must raise ValueError."""
        bold_affine = np.diag([3.0, 3.0, 3.0, 1.0])
        mismatched_affine = np.diag([2.0, 2.0, 2.0, 1.0])  # different resolution / space

        bold_img = nib.Nifti1Image(
            np.random.normal(100.0, 10.0, (15, 15, 10, self.t_vol)).astype(np.float32),
            bold_affine,
        )
        wm_mask = np.zeros((15, 15, 10), dtype=bool)
        wm_mask[4:8, 4:8, 3:7] = True
        csf_mask = np.zeros((15, 15, 10), dtype=bool)
        csf_mask[1:3, 1:3, 1:4] = True

        wm_img_mismatched = nib.Nifti1Image(wm_mask.astype(np.int32), mismatched_affine)
        csf_img_matched = nib.Nifti1Image(csf_mask.astype(np.int32), bold_affine)

        # WM affine mismatch must fail
        with self.assertRaises(ValueError) as ctx:
            extract_acompcor_components(bold_img, wm_img_mismatched, csf_img_matched, n_components=5)
        self.assertIn("WM mask affine does not match", str(ctx.exception))

        # CSF affine mismatch must fail
        wm_img_matched = nib.Nifti1Image(wm_mask.astype(np.int32), bold_affine)
        csf_img_mismatched = nib.Nifti1Image(csf_mask.astype(np.int32), mismatched_affine)
        with self.assertRaises(ValueError) as ctx:
            extract_acompcor_components(bold_img, wm_img_matched, csf_img_mismatched, n_components=5)
        self.assertIn("CSF mask affine does not match", str(ctx.exception))

    def test_motion_parameter_translation_direction_and_units(self):
        """Realign must accurately extract known translation direction and mm units."""
        # Static volume with sphere in center
        shape = (20, 20, 16)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])  # 2mm voxels
        vol0 = np.zeros(shape, dtype=np.float32)
        vol0[7:13, 7:13, 5:11] = 100.0

        # Volume 1: translated by +1 voxel (+2.0 mm) along X
        vol1 = np.zeros(shape, dtype=np.float32)
        vol1[8:14, 7:13, 5:11] = 100.0

        # Volume 2: translated by -1 voxel (-2.0 mm) along Y
        vol2 = np.zeros(shape, dtype=np.float32)
        vol2[7:13, 6:12, 5:11] = 100.0

        data = np.stack([vol0, vol1, vol2], axis=-1)
        bold_img = nib.Nifti1Image(data, affine)

        _, params, fd = realign_4d_bold(bold_img)

        # Vol 0 should have ~0 translation relative to mean
        # Vol 1 vs Vol 2 differential translation: Vol 1 has +X, Vol 2 has -Y
        # Verify relative translation between Vol 1 and Vol 0 along X is positive
        diff_vol1_vol0 = params[1] - params[0]
        self.assertAlmostEqual(diff_vol1_vol0[0], 2.0, delta=0.5)

        diff_vol2_vol0 = params[2] - params[0]
        self.assertAlmostEqual(diff_vol2_vol0[1], -2.0, delta=0.5)

    def test_framewise_displacement_analytical_components(self):
        """Verify Power et al. (2012) FD calculation against exact analytical ground truth."""
        # 4 volumes
        # Vol 0: origin
        # Vol 1: +1.5 mm X translation -> delta_T = 1.5 mm
        # Vol 2: +2.0 deg X rotation -> delta_R = 50 * (pi/180) * 2.0 = 1.74533 mm
        # Vol 3: +1.0 mm Y translation + 1.0 deg Z rotation -> delta_T = 1.0, delta_R = 50*(pi/180)*1.0 = 0.87266 mm
        motion = np.zeros((4, 6))
        motion[0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        motion[1] = [1.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        motion[2] = [1.5, 0.0, 0.0, 2.0, 0.0, 0.0]
        motion[3] = [1.5, 1.0, 0.0, 2.0, 0.0, 1.0]

        fd = compute_framewise_displacement(motion, radius_mm=50.0)

        expected_fd_0 = 0.0
        expected_fd_1 = 1.5
        expected_fd_2 = 50.0 * np.deg2rad(2.0)
        expected_fd_3 = 1.0 + 50.0 * np.deg2rad(1.0)

        self.assertAlmostEqual(fd[0], expected_fd_0, places=6)
        self.assertAlmostEqual(fd[1], expected_fd_1, places=6)
        self.assertAlmostEqual(fd[2], expected_fd_2, places=6)
        self.assertAlmostEqual(fd[3], expected_fd_3, places=6)


if __name__ == "__main__":
    unittest.main()
