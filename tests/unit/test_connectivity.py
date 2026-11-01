"""Unit tests for CC200 parcellation timeseries extraction and 19,900 Fisher-z connectivity features."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
import nibabel as nib
import numpy as np
import pandas as pd

from src.features.connectivity import (
    FISHER_EPSILON,
    N_ROIS,
    compute_connectivity_vector,
    get_feature_names,
)
from src.parcellation.extractor import (
    CANONICAL_CC200_AFFINE,
    CANONICAL_CC200_SHA256,
    CANONICAL_CC200_SHAPE,
    extract_cc200_timeseries,
    verify_canonical_cc200_atlas,
)


class TestConnectivityAndParcellation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.t_vol = 70

        # Generate synthetic (T, 200) timeseries
        self.timeseries = np.random.normal(0, 1.0, (self.t_vol, N_ROIS))

    def test_feature_names_count_and_lexicographic_order(self):
        names = get_feature_names()
        self.assertEqual(len(names), 19900)

        # First edge
        self.assertEqual(names[0], "ROI_001__ROI_002")
        # Second edge
        self.assertEqual(names[1], "ROI_001__ROI_003")
        # Edge between 1 and 200
        self.assertEqual(names[198], "ROI_001__ROI_200")
        # Edge between 2 and 3
        self.assertEqual(names[199], "ROI_002__ROI_003")
        # Last edge
        self.assertEqual(names[-1], "ROI_199__ROI_200")

    def test_compute_connectivity_vector_shape_and_finite(self):
        z_vector, edge_names = compute_connectivity_vector(self.timeseries)

        self.assertEqual(len(z_vector), 19900)
        self.assertEqual(len(edge_names), 19900)
        self.assertTrue(np.all(np.isfinite(z_vector)))
        self.assertEqual(z_vector.dtype, np.float64)

    def test_boundary_clipping_prevents_infinity(self):
        # Create identical signals for two ROIs (correlation = 1.0)
        corr_timeseries = self.timeseries.copy()
        corr_timeseries[:, 1] = corr_timeseries[:, 0]  # ROI 1 and ROI 2 perfectly correlated
        corr_timeseries[:, 2] = -corr_timeseries[:, 0]  # ROI 1 and ROI 3 perfectly anti-correlated

        z_vector, _ = compute_connectivity_vector(corr_timeseries)

        # arctanh(1.0 - 1e-7) ~ 8.41
        max_possible_z = np.arctanh(1.0 - FISHER_EPSILON)
        min_possible_z = np.arctanh(-1.0 + FISHER_EPSILON)

        self.assertAlmostEqual(z_vector[0], max_possible_z, places=5)
        self.assertAlmostEqual(z_vector[1], min_possible_z, places=5)
        self.assertFalse(np.isinf(z_vector).any())

    def test_constant_signal_raises_valueerror(self):
        bad_timeseries = self.timeseries.copy()
        bad_timeseries[:, 50] = 5.0  # Constant signal in ROI 51

        with self.assertRaises(ValueError) as ctx:
            compute_connectivity_vector(bad_timeseries)
        self.assertIn("Constant ROI signal detected", str(ctx.exception))

    def test_extract_cc200_timeseries_structure(self):
        # Create synthetic atlas with 200 unique integer labels
        atlas_data = np.zeros((20, 20, 20), dtype=np.int32)
        # Create 200 small parcels
        idx = 1
        for x in range(1, 19, 3):
            for y in range(1, 19, 3):
                for z in range(1, 19, 3):
                    if idx <= 200:
                        atlas_data[x : x + 2, y : y + 2, z : z + 2] = idx
                        idx += 1

        atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))

        # Synthetic cleaned 4D BOLD image
        bold_data = np.random.normal(100.0, 5.0, (20, 20, 20, 55)).astype(np.float32)
        bold_img = nib.Nifti1Image(bold_data, np.eye(4))

        df, signals = extract_cc200_timeseries(bold_img, atlas_img)

        # Format matches historical C-PAC roi_CC200.csv layout:
        # 200 rows, column 0 is 'node/volume' with integer labels 1..200
        # remaining columns are 0, 1, ..., T-1
        self.assertEqual(df.shape, (200, 56))  # 1 label column + 55 time columns
        self.assertEqual(df.columns[0], "node/volume")
        np.testing.assert_array_equal(df["node/volume"].to_numpy(), np.arange(1, 201))

        self.assertEqual(signals.shape, (55, 200))

    def test_extract_cc200_timeseries_shape_mismatch_raises_valueerror(self):
        """extract_cc200_timeseries must raise ValueError on spatial dimension mismatch."""
        atlas_data = np.zeros((20, 20, 20), dtype=np.int32)
        atlas_data[5:15, 5:15, 5:15] = np.random.randint(1, 201, (10, 10, 10))
        # Ensure all 200 labels are present for validation
        atlas_data.flat[:200] = np.arange(1, 201)
        atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))

        # BOLD with mismatched shape (24, 20, 20) instead of (20, 20, 20)
        bold_data = np.random.normal(100.0, 5.0, (24, 20, 20, 55)).astype(np.float32)
        bold_img = nib.Nifti1Image(bold_data, np.eye(4))

        with self.assertRaises(ValueError) as ctx:
            extract_cc200_timeseries(bold_img, atlas_img)
        self.assertIn("Spatial dimension mismatch", str(ctx.exception))

    def test_extract_cc200_timeseries_affine_mismatch_raises_valueerror(self):
        """extract_cc200_timeseries must raise ValueError on affine grid mismatch."""
        atlas_data = np.zeros((20, 20, 20), dtype=np.int32)
        atlas_data.flat[:200] = np.arange(1, 201)
        atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))

        # BOLD with matching shape (20, 20, 20) but mismatched affine
        bold_data = np.random.normal(100.0, 5.0, (20, 20, 20, 55)).astype(np.float32)
        mismatched_affine = np.diag([3.0, 3.0, 3.0, 1.0])
        bold_img = nib.Nifti1Image(bold_data, mismatched_affine)

        with self.assertRaises(ValueError) as ctx:
            extract_cc200_timeseries(bold_img, atlas_img)
        self.assertIn("Affine mismatch", str(ctx.exception))

    def test_actual_cc200_atlas_extraction_and_feature_contract(self):
        """Audit real project CC200 atlas file and verify downstream feature extraction contract."""
        atlas_path = Path("data/atlases/cc200_roi_atlas.nii.gz")
        self.assertTrue(atlas_path.exists(), f"CC200 atlas file missing at {atlas_path}")

        # Compute SHA256 checksum
        hasher = hashlib.sha256()
        with open(atlas_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        atlas_sha256 = hasher.hexdigest()

        expected_sha256 = "9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650"
        self.assertEqual(
            atlas_sha256,
            expected_sha256,
            f"CC200 atlas SHA256 mismatch: {atlas_sha256} != {expected_sha256}",
        )

        # Load real atlas
        atlas_img = nib.load(str(atlas_path))
        self.assertEqual(atlas_img.shape, (63, 75, 61))
        zooms = tuple(np.round(atlas_img.header.get_zooms()[:3], 1))
        self.assertEqual(zooms, (3.0, 3.0, 3.0))

        atlas_data = np.asanyarray(atlas_img.dataobj, dtype=np.int32)
        unique_labels = np.unique(atlas_data[atlas_data > 0])
        self.assertEqual(len(unique_labels), 200)
        np.testing.assert_array_equal(unique_labels, np.arange(1, 201))

        # Generate synthetic 4D BOLD volume on exact CC200 grid
        t_vol = 60
        t_grid = np.arange(t_vol, dtype=np.float32)
        bold_data = np.zeros((*atlas_img.shape, t_vol), dtype=np.float32)

        for roi_id in range(1, 201):
            mask = atlas_data == roi_id
            roi_sig = np.sin(2 * np.pi * roi_id * t_grid / 100.0) + np.random.normal(0, 0.05, t_vol)
            bold_data[mask, :] = roi_sig.astype(np.float32)

        bold_img = nib.Nifti1Image(bold_data, atlas_img.affine, atlas_img.header)

        # Extract time series
        df, signals = extract_cc200_timeseries(bold_img, atlas_img)

        # Verify layout matching historical C-PAC roi_CC200.csv
        self.assertEqual(df.shape, (200, t_vol + 1))
        self.assertEqual(df.columns[0], "node/volume")
        np.testing.assert_array_equal(df["node/volume"].to_numpy(), np.arange(1, 201))
        self.assertEqual(signals.shape, (t_vol, 200))

        # Compute 19,900 Fisher-z connectivity vector
        z_vector, edge_names = compute_connectivity_vector(signals)
        self.assertEqual(len(z_vector), 19900)
        self.assertEqual(len(edge_names), 19900)
        self.assertTrue(np.all(np.isfinite(z_vector)))
        self.assertEqual(z_vector.dtype, np.float64)

        # Verify exact lexicographical edge ordering
        expected_names = get_feature_names()
        self.assertEqual(edge_names, expected_names)


    def test_verify_canonical_cc200_atlas_success(self):
        """Canonical CC200 atlas file must pass complete verification."""
        atlas_path = Path("data/atlases/cc200_roi_atlas.nii.gz")
        res = verify_canonical_cc200_atlas(atlas_path)
        self.assertTrue(res["verified"])
        self.assertEqual(res["sha256"], CANONICAL_CC200_SHA256)
        self.assertEqual(res["shape"], CANONICAL_CC200_SHAPE)
        self.assertEqual(res["n_rois"], 200)

    def test_verify_canonical_cc200_atlas_missing_file_raises_filenotfound(self):
        """Missing file must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            verify_canonical_cc200_atlas(Path("data/atlases/non_existent_atlas.nii.gz"))

    def test_verify_canonical_cc200_atlas_corrupt_checksum_raises_valueerror(self):
        """Modified or unauthorized file must raise ValueError on SHA-256 mismatch."""
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
            # Write arbitrary content
            Path(tf.name).write_bytes(b"dummy corrupt nifti header")
            with self.assertRaises(ValueError) as ctx:
                verify_canonical_cc200_atlas(tf.name)
            self.assertIn("SHA-256 mismatch", str(ctx.exception))

    def test_verify_canonical_cc200_atlas_arbitrary_filename_success(self):
        """Canonical CC200 atlas must be verified by cryptographic hash and geometry regardless of filename."""
        real_atlas = Path("data/atlases/cc200_roi_atlas.nii.gz")
        with tempfile.TemporaryDirectory() as td:
            arbitrary_path = Path(td) / "unrelated_arbitrary_name.nii.gz"
            arbitrary_path.write_bytes(real_atlas.read_bytes())
            res = verify_canonical_cc200_atlas(arbitrary_path)
            self.assertTrue(res["verified"])
            self.assertEqual(res["sha256"], CANONICAL_CC200_SHA256)
            self.assertEqual(res["shape"], CANONICAL_CC200_SHAPE)
            self.assertEqual(res["n_rois"], 200)

    def test_verify_canonical_cc200_atlas_wrong_atlas_with_canonical_filename_rejected(self):
        """Atlas named 'cc200_roi_atlas.nii.gz' with non-canonical content must be rejected."""
        with tempfile.TemporaryDirectory() as td:
            canonical_named_bad_file = Path(td) / "cc200_roi_atlas.nii.gz"
            canonical_named_bad_file.write_bytes(b"non-canonical fake atlas data")
            with self.assertRaises(ValueError) as ctx:
                verify_canonical_cc200_atlas(canonical_named_bad_file)
            self.assertIn("SHA-256 mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

