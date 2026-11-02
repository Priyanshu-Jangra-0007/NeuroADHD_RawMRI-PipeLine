"""Unit tests for QC visualization functions and spatial geometry validation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from src.preprocessing.qc import (
    compute_registration_qc_metrics,
    generate_before_after_qc_figure,
    generate_coregistration_qc_overlay,
    generate_landmark_qc_overlay,
    generate_normalization_qc_overlay,
    generate_parcellation_qc_overlay,
    generate_registration_qc_overlay,
    generate_tissue_mask_qc_overlay,
    validate_qc_grid_alignment,
)


class TestQCVisualization(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp_dir.name)

        self.shape = (20, 20, 16)
        self.affine = np.diag([3.0, 3.0, 3.0, 1.0])

        bg = np.zeros(self.shape, dtype=np.float32)
        bg[4:16, 4:16, 3:13] = 100.0
        self.bg_img = nib.Nifti1Image(bg, self.affine)

        fg = np.zeros(self.shape, dtype=np.float32)
        fg[6:14, 6:14, 5:11] = 80.0
        self.fg_img = nib.Nifti1Image(fg, self.affine)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_validate_qc_grid_alignment_rejects_shape_mismatch(self):
        mismatched_img = nib.Nifti1Image(np.zeros((15, 15, 10), dtype=np.float32), self.affine)
        with self.assertRaises(ValueError) as ctx:
            validate_qc_grid_alignment(self.bg_img, mismatched_img, "bg", "fg")
        self.assertIn("shape", str(ctx.exception))

    def test_validate_qc_grid_alignment_rejects_affine_mismatch(self):
        diff_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        mismatched_img = nib.Nifti1Image(np.zeros(self.shape, dtype=np.float32), diff_affine)
        with self.assertRaises(ValueError) as ctx:
            validate_qc_grid_alignment(self.bg_img, mismatched_img, "bg", "fg")
        self.assertIn("affine mismatch", str(ctx.exception))

    def test_generate_registration_qc_overlay_creates_valid_file(self):
        out_path = self.work_dir / "reg_qc.png"
        res = generate_registration_qc_overlay(self.bg_img, self.fg_img, out_path, title="Test Reg")
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 5000)
        self.assertEqual(res, out_path)

    def test_generate_coregistration_qc_overlay(self):
        out_path = self.work_dir / "coreg_qc.png"
        res = generate_coregistration_qc_overlay(self.bg_img, self.fg_img, out_path)
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 5000)

    def test_generate_normalization_qc_overlay(self):
        out_path = self.work_dir / "norm_qc.png"
        res = generate_normalization_qc_overlay(self.bg_img, self.fg_img, out_path)
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 5000)

    def test_generate_tissue_mask_qc_overlay(self):
        wm_data = np.zeros(self.shape, dtype=np.int32)
        wm_data[7:13, 7:13, 6:10] = 1
        csf_data = np.zeros(self.shape, dtype=np.int32)
        csf_data[5:7, 5:7, 5:7] = 1

        wm_img = nib.Nifti1Image(wm_data, self.affine)
        csf_img = nib.Nifti1Image(csf_data, self.affine)

        out_path = self.work_dir / "tissue_qc.png"
        res = generate_tissue_mask_qc_overlay(self.bg_img, wm_img, csf_img, out_path)
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 5000)

    def test_generate_parcellation_qc_overlay(self):
        atlas_data = np.zeros(self.shape, dtype=np.int32)
        atlas_data[5:10, 5:10, 5:10] = 1
        atlas_data[10:15, 10:15, 10:15] = 2
        atlas_img = nib.Nifti1Image(atlas_data, self.affine)

        out_path = self.work_dir / "parc_qc.png"
        res = generate_parcellation_qc_overlay(self.bg_img, atlas_img, out_path)
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 5000)

    def test_compute_registration_qc_metrics_passing_and_failing(self):
        # High correlation / overlap
        identical_metrics = compute_registration_qc_metrics(self.bg_img, self.bg_img)
        self.assertAlmostEqual(identical_metrics["pearson_r"], 1.0, places=4)
        self.assertAlmostEqual(identical_metrics["rmse"], 0.0, places=4)
        self.assertAlmostEqual(identical_metrics["dice"], 1.0, places=4)
        self.assertTrue(identical_metrics["qc_pass"])

        # Low correlation / failing metrics
        inverted_data = np.zeros(self.shape, dtype=np.float32)
        inverted_data[0:4, 0:4, 0:4] = 100.0  # Non-overlapping
        inverted_img = nib.Nifti1Image(inverted_data, self.affine)

        failing_metrics = compute_registration_qc_metrics(self.bg_img, inverted_img)
        self.assertFalse(failing_metrics["qc_pass"])
        self.assertLess(failing_metrics["dice"], 0.1)

        # With explicit mask
        mask_data = np.zeros(self.shape, dtype=np.int32)
        mask_data[4:16, 4:16, 3:13] = 1
        mask_img = nib.Nifti1Image(mask_data, self.affine)
        masked_metrics = compute_registration_qc_metrics(self.bg_img, self.fg_img, mask=mask_img)
        self.assertIn("pearson_r", masked_metrics)
        self.assertGreater(masked_metrics["n_voxels_evaluated"], 0)

    def test_generate_before_after_qc_figure(self):
        before_data = np.zeros(self.shape, dtype=np.float32)
        before_data[2:10, 2:10, 2:10] = 50.0
        before_img = nib.Nifti1Image(before_data, self.affine)

        out_path = self.work_dir / "before_after_qc.png"
        res = generate_before_after_qc_figure(self.bg_img, before_img, self.fg_img, out_path)
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 5000)
        self.assertEqual(res, out_path)

    def test_generate_landmark_qc_overlay_and_tre(self):
        # Target landmarks in mm
        t_lms = np.array([
            [12.0, 15.0, 18.0],
            [24.0, 30.0, 36.0],
            [40.0, 40.0, 30.0],
        ])
        # Warped landmarks with small perturbation (mean TRE ~ 1.0 mm)
        w_lms = t_lms + np.array([
            [0.5, 0.5, 0.5],
            [-0.5, 0.0, 0.5],
            [0.2, -0.4, 0.1],
        ])

        out_path = self.work_dir / "landmark_qc.png"
        res_path, metrics = generate_landmark_qc_overlay(
            background_img=self.bg_img,
            target_landmarks_mm=t_lms,
            warped_landmarks_mm=w_lms,
            output_png_path=out_path,
            max_allowed_tre_mm=3.0,
        )
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 5000)
        self.assertEqual(res_path, out_path)
        self.assertTrue(metrics["qc_pass"])
        self.assertLess(metrics["mean_tre_mm"], 2.0)
        self.assertLess(metrics["max_tre_mm"], 3.0)

        # Test failing TRE
        w_lms_bad = t_lms + 10.0
        out_bad = self.work_dir / "landmark_qc_bad.png"
        _, bad_metrics = generate_landmark_qc_overlay(
            background_img=self.bg_img,
            target_landmarks_mm=t_lms,
            warped_landmarks_mm=w_lms_bad,
            output_png_path=out_bad,
            max_allowed_tre_mm=3.0,
        )
        self.assertFalse(bad_metrics["qc_pass"])
        self.assertGreater(bad_metrics["max_tre_mm"], 10.0)

        # Test shape validation
        with self.assertRaises(ValueError):
            generate_landmark_qc_overlay(self.bg_img, t_lms[:2], w_lms, out_path)



if __name__ == "__main__":
    unittest.main()

