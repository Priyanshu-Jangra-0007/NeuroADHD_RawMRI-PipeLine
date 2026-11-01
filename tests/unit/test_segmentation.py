"""Unit tests for anatomical brain extraction, bias correction, and HMRF tissue segmentation."""
from __future__ import annotations

import unittest
import nibabel as nib
import numpy as np

from src.preprocessing.segmentation import (
    correct_bias_field,
    create_eroded_tissue_masks,
    extract_anatomical_brain,
    segment_t1w_tissue,
    validate_tissue_masks,
)


class TestSegmentation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.affine = np.diag([2.0, 2.0, 2.0, 1.0])

        # Create synthetic 3D head volume
        shape = (40, 40, 40)
        x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]
        center = (20, 20, 20)

        # Distances from center
        r_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2

        self.head = np.zeros(shape, dtype=np.float32)
        # Brain sphere: radius <= 14
        self.brain_mask = r_sq <= 14 ** 2
        # Skull shell: 15 <= radius <= 17
        skull_shell = (r_sq >= 15 ** 2) & (r_sq <= 17 ** 2)

        # Tissue structures inside brain
        # CSF: outer rim (11 <= r <= 14)
        self.csf_region = self.brain_mask & (r_sq >= 11 ** 2)
        # GM: middle layer (7 <= r < 11)
        self.gm_region = self.brain_mask & (r_sq >= 7 ** 2) & (~self.csf_region)
        # WM: core (r < 7)
        self.wm_region = self.brain_mask & (~self.csf_region) & (~self.gm_region)

        # Populate intensities with realistic continuous distribution (SNR ~ 15-30)
        self.head[self.csf_region] = 30.0 + np.random.normal(0, 3, np.sum(self.csf_region))
        self.head[self.gm_region] = 80.0 + np.random.normal(0, 6, np.sum(self.gm_region))
        self.head[self.wm_region] = 140.0 + np.random.normal(0, 8, np.sum(self.wm_region))
        self.head[skull_shell] = 120.0 + np.random.normal(0, 5, np.sum(skull_shell))

        self.t1w_img = nib.Nifti1Image(self.head, self.affine)

    def test_extract_anatomical_brain(self):
        brain_img, mask_img = extract_anatomical_brain(self.t1w_img, median_radius=2, numpass=2)

        brain_data = np.asanyarray(brain_img.dataobj)
        mask_data = np.asanyarray(mask_img.dataobj)

        self.assertEqual(brain_data.shape, (40, 40, 40))
        self.assertEqual(mask_data.shape, (40, 40, 40))
        # Non-brain voxels must be strictly 0
        self.assertTrue(np.all(brain_data[mask_data == 0] == 0.0))
        # Brain voxels must have non-zero elements
        self.assertGreater(np.count_nonzero(mask_data), 1000)

    def test_correct_bias_field_reduces_variance(self):
        # Create homogeneous parenchyma with a smooth multiplicative bias field (30% variation)
        homo_brain = np.zeros((40, 40, 40), dtype=np.float32)
        homo_brain[self.brain_mask] = 100.0 + np.random.normal(0, 2, np.sum(self.brain_mask))

        x_coords = np.linspace(-1, 1, 40)[:, None, None]
        bias_true = 1.0 + 0.3 * x_coords
        corrupted = (homo_brain * bias_true).astype(np.float32)

        brain_img = nib.Nifti1Image(corrupted, self.affine)
        mask_img = nib.Nifti1Image(self.brain_mask.astype(np.int32), self.affine)

        cv_before = float(np.std(corrupted[self.brain_mask]) / np.mean(corrupted[self.brain_mask]))

        corrected_img, bias_img = correct_bias_field(brain_img, mask_img, order=2)

        corrected_data = np.asanyarray(corrected_img.dataobj)
        bias_data = np.asanyarray(bias_img.dataobj)
        cv_after = float(np.std(corrected_data[self.brain_mask]) / np.mean(corrected_data[self.brain_mask]))

        # Bias correction must reduce variance across homogenous parenchyma
        self.assertLess(cv_after, cv_before)
        # Bias field must be strictly positive
        self.assertTrue(np.all(bias_data[self.brain_mask] > 0))

    def test_correct_bias_field_rejects_empty_mask(self):
        empty_mask = nib.Nifti1Image(np.zeros((40, 40, 40), dtype=np.int32), self.affine)
        with self.assertRaises(ValueError) as ctx:
            correct_bias_field(self.t1w_img, empty_mask)
        self.assertIn("empty brain mask", str(ctx.exception))

    def test_segment_t1w_tissue_probabilities(self):
        # Prepare clean brain-extracted image
        brain_data = (self.head * self.brain_mask).astype(np.float32)
        brain_img = nib.Nifti1Image(brain_data, self.affine)
        mask_img = nib.Nifti1Image(self.brain_mask.astype(np.int32), self.affine)

        res = segment_t1w_tissue(brain_img, mask_img, n_classes=3, beta=0.1, max_iter=20)

        self.assertIn("prob_csf", res)
        self.assertIn("prob_gm", res)
        self.assertIn("prob_wm", res)
        self.assertIn("segmentation", res)

        prob_csf = np.asanyarray(res["prob_csf"].dataobj)
        prob_gm = np.asanyarray(res["prob_gm"].dataobj)
        prob_wm = np.asanyarray(res["prob_wm"].dataobj)
        seg = np.asanyarray(res["segmentation"].dataobj)

        # Check bounds [0.0, 1.0] and no NaNs
        self.assertFalse(np.isnan(prob_csf).any())
        self.assertFalse(np.isnan(prob_gm).any())
        self.assertFalse(np.isnan(prob_wm).any())
        self.assertTrue(np.all((prob_csf >= 0.0) & (prob_csf <= 1.0)))
        self.assertTrue(np.all((prob_gm >= 0.0) & (prob_gm <= 1.0)))
        self.assertTrue(np.all((prob_wm >= 0.0) & (prob_wm <= 1.0)))

        # Non-brain voxels must have exactly 0 probability
        self.assertTrue(np.all(prob_csf[~self.brain_mask] == 0.0))
        self.assertTrue(np.all(prob_gm[~self.brain_mask] == 0.0))
        self.assertTrue(np.all(prob_wm[~self.brain_mask] == 0.0))
        self.assertTrue(np.all(seg[~self.brain_mask] == 0))

        # Inside brain mask, probabilities sum close to 1.0
        prob_sum = (prob_csf + prob_gm + prob_wm)[self.brain_mask]
        self.assertTrue(np.all(prob_sum > 0.90))

        # Check discrete segmentation labels: 0=bg, 1=CSF, 2=GM, 3=WM
        unique_labels = set(np.unique(seg))
        self.assertTrue(unique_labels.issubset({0, 1, 2, 3}))

        # Verify tissue order: CSF mean < GM mean < WM mean
        mean_csf = float(np.mean(brain_data[seg == 1])) if np.any(seg == 1) else 0.0
        mean_gm = float(np.mean(brain_data[seg == 2])) if np.any(seg == 2) else 0.0
        mean_wm = float(np.mean(brain_data[seg == 3])) if np.any(seg == 3) else 0.0
        self.assertLess(mean_csf, mean_gm)
        self.assertLess(mean_gm, mean_wm)

    def test_create_eroded_tissue_masks_and_validation(self):
        brain_data = (self.head * self.brain_mask).astype(np.float32)
        brain_img = nib.Nifti1Image(brain_data, self.affine)
        mask_img = nib.Nifti1Image(self.brain_mask.astype(np.int32), self.affine)

        prob_maps = segment_t1w_tissue(brain_img, mask_img, n_classes=3, beta=0.1, max_iter=20)

        # Generate eroded masks with mandatory reference T1w image
        masks = create_eroded_tissue_masks(
            prob_maps, reference_t1w_img=brain_img, wm_threshold=0.80, csf_threshold=0.80, erosion_voxels=1
        )

        self.assertIn("mask_wm_eroded", masks)
        self.assertIn("mask_csf_eroded", masks)
        self.assertIn("mask_wm_raw", masks)
        self.assertIn("mask_csf_raw", masks)

        wm_raw = np.asanyarray(masks["mask_wm_raw"].dataobj)
        wm_eroded = np.asanyarray(masks["mask_wm_eroded"].dataobj)
        csf_eroded = np.asanyarray(masks["mask_csf_eroded"].dataobj)

        # 1-voxel erosion must shrink mask or maintain containment
        self.assertLessEqual(np.count_nonzero(wm_eroded), np.count_nonzero(wm_raw))
        # Eroded WM and eroded CSF masks must be mutually exclusive
        overlap = np.count_nonzero((wm_eroded > 0) & (csf_eroded > 0))
        self.assertEqual(overlap, 0)

        # Validate masks pass quality checks
        val_res = validate_tissue_masks(masks, min_wm_voxels=5, min_csf_voxels=5)
        self.assertEqual(val_res["status"], "VALIDATED")
        self.assertEqual(val_res["overlap_voxel_count"], 0)

    def test_segmentation_dice_overlap_against_ground_truth(self):
        """Evaluate synthetic segmentation accuracy using Dice coefficient against known ground truth.

        Note: Synthetic segmentation Dice scores establish algorithmic implementation correctness
        and numerical convergence on controlled mathematical phantoms; they do NOT establish
        biological or anatomical segmentation accuracy on real human in-vivo MRI scans.
        """
        brain_data = (self.head * self.brain_mask).astype(np.float32)
        brain_img = nib.Nifti1Image(brain_data, self.affine)
        mask_img = nib.Nifti1Image(self.brain_mask.astype(np.int32), self.affine)

        res = segment_t1w_tissue(brain_img, mask_img, n_classes=3, beta=0.1, max_iter=20)
        seg = np.asanyarray(res["segmentation"].dataobj)

        def dice(pred: np.ndarray, gt: np.ndarray) -> float:
            inter = np.count_nonzero(pred & gt)
            total = np.count_nonzero(pred) + np.count_nonzero(gt)
            return float(2.0 * inter / total) if total > 0 else 1.0

        # Discrete labels: 1=CSF, 2=GM, 3=WM
        dice_csf = dice(seg == 1, self.csf_region)
        dice_gm = dice(seg == 2, self.gm_region)
        dice_wm = dice(seg == 3, self.wm_region)

        # On known spherical ground-truth geometry, segmentation must achieve high fidelity (> 0.90)
        self.assertGreater(dice_csf, 0.90, f"CSF Dice {dice_csf:.3f} below 0.90")
        self.assertGreater(dice_gm, 0.90, f"GM Dice {dice_gm:.3f} below 0.90")
        self.assertGreater(dice_wm, 0.90, f"WM Dice {dice_wm:.3f} below 0.90")

    def test_one_voxel_erosion_behavior_and_mutual_exclusivity(self):
        """Test expected 1-voxel 3D erosion behavior: shrinkage, boundary removal, and mutual exclusivity."""
        from scipy.ndimage import generate_binary_structure, binary_erosion

        # Create synthetic probability maps with well-separated solid tissue cores
        prob_wm = np.zeros((30, 30, 30), dtype=np.float32)
        prob_wm[10:20, 10:20, 10:20] = 0.99  # Solid 10x10x10 cube

        prob_csf = np.zeros((30, 30, 30), dtype=np.float32)
        prob_csf[2:8, 2:8, 2:8] = 0.99       # Solid 6x6x6 cube

        ref_img = nib.Nifti1Image(np.zeros((30, 30, 30), dtype=np.float32), self.affine)
        prob_maps = {
            "prob_wm": nib.Nifti1Image(prob_wm, self.affine),
            "prob_csf": nib.Nifti1Image(prob_csf, self.affine),
        }

        masks = create_eroded_tissue_masks(
            prob_maps, reference_t1w_img=ref_img, wm_threshold=0.96, csf_threshold=0.96, erosion_voxels=1
        )

        raw_wm = np.asanyarray(masks["mask_wm_raw"].dataobj) > 0
        eroded_wm = np.asanyarray(masks["mask_wm_eroded"].dataobj) > 0
        raw_csf = np.asanyarray(masks["mask_csf_raw"].dataobj) > 0
        eroded_csf = np.asanyarray(masks["mask_csf_eroded"].dataobj) > 0

        # 1. Volume shrinkage
        self.assertLess(np.count_nonzero(eroded_wm), np.count_nonzero(raw_wm))
        self.assertGreater(np.count_nonzero(eroded_wm), 0)
        self.assertLess(np.count_nonzero(eroded_csf), np.count_nonzero(raw_csf))
        self.assertGreater(np.count_nonzero(eroded_csf), 0)

        # 2. Strict containment: all eroded voxels must lie strictly within the raw mask
        self.assertTrue(np.all(raw_wm[eroded_wm]))
        self.assertTrue(np.all(raw_csf[eroded_csf]))

        # 3. Exterior boundary removal: 1-voxel 6-connectivity erosion removes exterior surface
        struct_6 = generate_binary_structure(3, 1)
        expected_eroded_wm = binary_erosion(raw_wm, structure=struct_6, iterations=1)
        expected_eroded_csf = binary_erosion(raw_csf, structure=struct_6, iterations=1)
        np.testing.assert_array_equal(eroded_wm, expected_eroded_wm)
        np.testing.assert_array_equal(eroded_csf, expected_eroded_csf)

        # 4. Mutual exclusivity
        overlap = np.count_nonzero(eroded_wm & eroded_csf)
        self.assertEqual(overlap, 0, "Eroded WM and CSF masks must be strictly mutually exclusive")

    def test_correct_bias_field_full_resolution_anatomical_memory(self):
        """Full anatomical volume (176 x 256 x 256) must correct rapidly within 2.5 GB ceiling.

        Validates reliable peak RSS (high-water mark) memory usage using an isolated process
        to avoid pollution from sequential test executions.
        """
        import subprocess
        import sys

        code = """
import numpy as np
import nibabel as nib
from src.preprocessing.segmentation import correct_bias_field
from src.preprocessing.registration import check_process_memory_gb

shape = (176, 256, 256)
cz, cy, cx = 88, 128, 128
zz, yy, xx = np.ogrid[:176, :256, :256]
dist_sq = ((zz - cz) / 60.0) ** 2 + ((yy - cy) / 90.0) ** 2 + ((xx - cx) / 90.0) ** 2
brain_mask = dist_sq <= 1.0

grad_3d = np.broadcast_to((1.0 + 0.3 * (xx / 256.0)).astype(np.float32), shape)
brain_data = np.zeros(shape, dtype=np.float32)
brain_data[brain_mask] = (100.0 * grad_3d)[brain_mask]

brain_img = nib.Nifti1Image(brain_data, np.eye(4))
mask_img = nib.Nifti1Image(brain_mask.astype(np.int32), np.eye(4))

corr_img, bias_img = correct_bias_field(brain_img, mask_img, order=2)
peak_rss_gb = check_process_memory_gb()

print(f"SHAPE:{corr_img.shape}")
print(f"PEAK_RSS_GB:{peak_rss_gb:.4f}")
"""
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        lines = proc.stdout.strip().split("\n")
        shape_line = [line for line in lines if line.startswith("SHAPE:")][0]
        peak_line = [line for line in lines if line.startswith("PEAK_RSS_GB:")][0]

        self.assertEqual(shape_line, "SHAPE:(176, 256, 256)")
        peak_mem = float(peak_line.split(":")[1])
        # Peak memory must remain well beneath the 2.5 GB ceiling (typically ~0.67 GB)
        self.assertLess(peak_mem, 2.5, f"Peak RSS {peak_mem:.2f} GB exceeded 2.5 GB ceiling")
        self.assertGreater(peak_mem, 0.1, "Sanity check: Peak RSS should be non-trivial")

    def test_validate_tissue_masks_fails_on_empty(self):
        empty_mask = nib.Nifti1Image(np.zeros((30, 30, 30), dtype=np.int32), self.affine)
        masks = {
            "mask_wm_eroded": empty_mask,
            "mask_csf_eroded": empty_mask,
        }
        with self.assertRaises(ValueError) as ctx:
            validate_tissue_masks(masks, min_wm_voxels=50, min_csf_voxels=20)
        self.assertIn("insufficient voxels", str(ctx.exception))

    def test_validate_tissue_masks_fails_on_shape_mismatch(self):
        wm_mask = nib.Nifti1Image(np.ones((30, 30, 30), dtype=np.int32), self.affine)
        csf_mask = nib.Nifti1Image(np.ones((25, 25, 25), dtype=np.int32), self.affine)
        masks = {"mask_wm_eroded": wm_mask, "mask_csf_eroded": csf_mask}
        with self.assertRaises(ValueError) as ctx:
            validate_tissue_masks(masks, min_wm_voxels=50, min_csf_voxels=20)
        self.assertIn("Shape mismatch", str(ctx.exception))

    def test_validate_tissue_masks_fails_on_affine_mismatch(self):
        wm_mask = nib.Nifti1Image(np.ones((30, 30, 30), dtype=np.int32), np.diag([2.0, 2.0, 2.0, 1.0]))
        csf_mask = nib.Nifti1Image(np.ones((30, 30, 30), dtype=np.int32), np.diag([3.0, 3.0, 3.0, 1.0]))
        masks = {"mask_wm_eroded": wm_mask, "mask_csf_eroded": csf_mask}
        with self.assertRaises(ValueError) as ctx:
            validate_tissue_masks(masks, min_wm_voxels=50, min_csf_voxels=20)
        self.assertIn("affine mismatch", str(ctx.exception))

    def test_validate_tissue_masks_fails_on_overlap(self):
        data = np.zeros((30, 30, 30), dtype=np.int32)
        data[10:15, 10:15, 10:15] = 1
        wm_mask = nib.Nifti1Image(data, self.affine)
        csf_mask = nib.Nifti1Image(data, self.affine)
        masks = {"mask_wm_eroded": wm_mask, "mask_csf_eroded": csf_mask}
        with self.assertRaises(ValueError) as ctx:
            validate_tissue_masks(masks, min_wm_voxels=50, min_csf_voxels=20)
        self.assertIn("overlap", str(ctx.exception))

    def test_validate_tissue_masks_success_production_thresholds(self):
        wm_data = np.zeros((30, 30, 30), dtype=np.int32)
        wm_data[5:15, 5:15, 5:15] = 1  # 1000 voxels > 50
        csf_data = np.zeros((30, 30, 30), dtype=np.int32)
        csf_data[20:25, 20:25, 20:25] = 1  # 125 voxels > 20
        masks = {
            "mask_wm_eroded": nib.Nifti1Image(wm_data, self.affine),
            "mask_csf_eroded": nib.Nifti1Image(csf_data, self.affine),
        }
        res = validate_tissue_masks(masks, min_wm_voxels=50, min_csf_voxels=20)
        self.assertEqual(res["status"], "VALIDATED")
        self.assertEqual(res["wm_voxel_count"], 1000)
        self.assertEqual(res["csf_voxel_count"], 125)
        self.assertEqual(res["overlap_voxel_count"], 0)

    def test_create_eroded_tissue_masks_rejects_missing_reference_t1w(self):
        """create_eroded_tissue_masks must reject missing or None reference_t1w_img."""
        prob_wm = nib.Nifti1Image(np.zeros((10, 10, 10)), self.affine)
        prob_csf = nib.Nifti1Image(np.zeros((10, 10, 10)), self.affine)
        with self.assertRaises(ValueError) as ctx:
            create_eroded_tissue_masks({"prob_wm": prob_wm, "prob_csf": prob_csf}, reference_t1w_img=None)
        self.assertIn("reference_t1w_img (nib.Nifti1Image) is mandatory", str(ctx.exception))

    def test_create_eroded_tissue_masks_rejects_missing_keys(self):
        """create_eroded_tissue_masks must reject dictionaries missing prob_wm or prob_csf."""
        ref_t1w = nib.Nifti1Image(np.zeros((10, 10, 10)), self.affine)
        with self.assertRaises(ValueError) as ctx:
            create_eroded_tissue_masks(
                {"prob_wm": nib.Nifti1Image(np.zeros((10, 10, 10)), self.affine)},
                reference_t1w_img=ref_t1w,
            )
        self.assertIn("must contain 'prob_wm' and 'prob_csf'", str(ctx.exception))

    def test_create_eroded_tissue_masks_rejects_shape_mismatch_between_wm_and_csf(self):
        """create_eroded_tissue_masks must reject shape mismatch between prob_wm and prob_csf."""
        prob_wm = nib.Nifti1Image(np.zeros((10, 10, 10)), self.affine)
        prob_csf = nib.Nifti1Image(np.zeros((10, 10, 12)), self.affine)
        ref_t1w = nib.Nifti1Image(np.zeros((10, 10, 10)), self.affine)
        with self.assertRaises(ValueError) as ctx:
            create_eroded_tissue_masks({"prob_wm": prob_wm, "prob_csf": prob_csf}, reference_t1w_img=ref_t1w)
        self.assertIn("shape mismatch", str(ctx.exception))

    def test_create_eroded_tissue_masks_rejects_affine_mismatch_between_wm_and_csf(self):
        """create_eroded_tissue_masks must reject affine mismatch between prob_wm and prob_csf."""
        prob_wm = nib.Nifti1Image(np.zeros((10, 10, 10)), self.affine)
        prob_csf = nib.Nifti1Image(np.zeros((10, 10, 10)), np.diag([3.0, 3.0, 3.0, 1.0]))
        ref_t1w = nib.Nifti1Image(np.zeros((10, 10, 10)), self.affine)
        with self.assertRaises(ValueError) as ctx:
            create_eroded_tissue_masks({"prob_wm": prob_wm, "prob_csf": prob_csf}, reference_t1w_img=ref_t1w)
        self.assertIn("affine mismatch", str(ctx.exception))

    def test_create_eroded_tissue_masks_rejects_reference_t1w_shape_mismatch(self):
        """create_eroded_tissue_masks must reject shape mismatch against reference_t1w_img."""
        prob_wm = nib.Nifti1Image(np.zeros((10, 10, 10)), self.affine)
        prob_csf = nib.Nifti1Image(np.zeros((10, 10, 10)), self.affine)
        ref_t1w = nib.Nifti1Image(np.zeros((10, 10, 12)), self.affine)
        with self.assertRaises(ValueError) as ctx:
            create_eroded_tissue_masks(
                {"prob_wm": prob_wm, "prob_csf": prob_csf}, reference_t1w_img=ref_t1w
            )
        self.assertIn("geometry mismatch", str(ctx.exception))

    def test_create_eroded_tissue_masks_rejects_reference_t1w_affine_mismatch(self):
        """create_eroded_tissue_masks must reject affine mismatch against reference_t1w_img."""
        prob_wm = nib.Nifti1Image(np.zeros((10, 10, 10)), self.affine)
        prob_csf = nib.Nifti1Image(np.zeros((10, 10, 10)), self.affine)
        ref_t1w = nib.Nifti1Image(np.zeros((10, 10, 10)), np.diag([3.0, 3.0, 3.0, 1.0]))
        with self.assertRaises(ValueError) as ctx:
            create_eroded_tissue_masks(
                {"prob_wm": prob_wm, "prob_csf": prob_csf}, reference_t1w_img=ref_t1w
            )
        self.assertIn("affine mismatch", str(ctx.exception))

    def test_create_eroded_tissue_masks_accepts_valid_reference_t1w(self):
        """create_eroded_tissue_masks succeeds when reference_t1w_img matches probability geometry."""
        data_wm = np.zeros((20, 20, 20), dtype=np.float32)
        data_wm[8:12, 8:12, 8:12] = 0.99
        data_csf = np.zeros((20, 20, 20), dtype=np.float32)
        data_csf[2:5, 2:5, 2:5] = 0.99
        prob_wm = nib.Nifti1Image(data_wm, self.affine)
        prob_csf = nib.Nifti1Image(data_csf, self.affine)
        ref_t1w = nib.Nifti1Image(np.zeros((20, 20, 20), dtype=np.float32), self.affine)

        masks = create_eroded_tissue_masks(
            {"prob_wm": prob_wm, "prob_csf": prob_csf},
            reference_t1w_img=ref_t1w,
            wm_threshold=0.96,
            csf_threshold=0.96,
            erosion_voxels=1,
        )
        self.assertIn("mask_wm_eroded", masks)
        self.assertIn("mask_csf_eroded", masks)
        self.assertEqual(masks["mask_wm_eroded"].shape, (20, 20, 20))
        np.testing.assert_allclose(masks["mask_wm_eroded"].affine, self.affine)

    def test_production_orchestration_passes_reference_t1w_to_erosion(self):
        """Verify that the production call sequence delivers the corrected T1w reference image to erosion."""
        # Step 3a: Brain extraction & bias correction
        brain_data = (self.head * self.brain_mask).astype(np.float32)
        brain_img = nib.Nifti1Image(brain_data, self.affine)
        mask_img = nib.Nifti1Image(self.brain_mask.astype(np.int32), self.affine)
        t1w_corrected_img, _ = correct_bias_field(brain_img, mask_img, order=2)

        # Step 3b: Tissue segmentation
        prob_maps = segment_t1w_tissue(t1w_corrected_img, mask_img, n_classes=3, beta=0.1, max_iter=20)

        # Step 3c: Production call with reference_t1w_img=t1w_corrected_img
        masks = create_eroded_tissue_masks(
            prob_maps,
            reference_t1w_img=t1w_corrected_img,
            wm_threshold=0.80,
            csf_threshold=0.80,
            erosion_voxels=1,
        )
        self.assertIn("mask_wm_eroded", masks)
        self.assertIn("mask_csf_eroded", masks)
        self.assertEqual(masks["mask_wm_eroded"].shape, t1w_corrected_img.shape)
        np.testing.assert_allclose(masks["mask_wm_eroded"].affine, t1w_corrected_img.affine)

        # Verifies that passing a corrupted/mismatched reference volume is rejected
        mismatched_t1w = nib.Nifti1Image(np.zeros((30, 30, 30)), self.affine)
        with self.assertRaises(ValueError) as ctx:
            create_eroded_tissue_masks(
                prob_maps,
                reference_t1w_img=mismatched_t1w,
                wm_threshold=0.80,
                csf_threshold=0.80,
                erosion_voxels=1,
            )
        self.assertIn("geometry mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
