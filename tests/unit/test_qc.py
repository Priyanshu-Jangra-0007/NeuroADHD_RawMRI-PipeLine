"""Unit tests for QC visualization functions and spatial geometry validation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from src.preprocessing.qc import (
    generate_coregistration_qc_overlay,
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


if __name__ == "__main__":
    unittest.main()

