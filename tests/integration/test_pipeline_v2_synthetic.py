"""End-to-end integration test of Corrected PoC V2 preprocessing pipeline on synthetic data.

Validates the full execution chain from raw BIDS inputs to CC200 ROI time series
and 19,900 Fisher-z connectivity features under the 2.5 GB peak-memory ceiling.
Strictly uses synthetic data: NO real participant MRI data is accessed or processed.
Enforces authentic historical 0.96 tissue thresholding and 1-voxel erosion without fallback.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from dipy.align.imaffine import AffineMap

from src.features.connectivity import compute_connectivity_vector
from src.parcellation.extractor import extract_cc200_timeseries
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
from src.preprocessing.qc import (
    generate_coregistration_qc_overlay,
    generate_normalization_qc_overlay,
    generate_parcellation_qc_overlay,
    generate_tissue_mask_qc_overlay,
)
from src.preprocessing.registration import (
    check_process_memory_gb,
    compose_transformations,
    compute_spatial_dice,
    generate_functional_reference,
    register_epi_to_t1w,
    register_t1w_to_mni,
    resample_4d_bold,
    resample_mask_to_target,
)
from src.preprocessing.segmentation import (
    correct_bias_field,
    create_eroded_tissue_masks,
    extract_anatomical_brain,
    segment_t1w_tissue,
    validate_tissue_masks,
)


class TestPipelineV2SyntheticIntegration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp_dir.name)

        self.t_vol = 50
        self.epi_shape = (30, 30, 20)
        self.t1_shape = (40, 40, 40)
        self.target_shape = (30, 30, 20)

        self.epi_affine = np.diag([3.0, 3.0, 3.0, 1.0])
        self.t1_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        self.target_affine = np.diag([3.0, 3.0, 3.0, 1.0])

        # 1. Realistic synthetic anatomical T1w head volume with internal ventricles and thick WM lobes
        x, y, z = np.ogrid[: self.t1_shape[0], : self.t1_shape[1], : self.t1_shape[2]]
        c = (20, 20, 20)
        r_sq = (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2

        self.t1_brain_mask = r_sq <= 17 ** 2
        skull = (r_sq >= 18 ** 2) & (r_sq <= 19 ** 2)

        # Internal lateral/central ventricles (CSF, low intensity ~25)
        ventricles = ((x - 20) ** 2 / 6 ** 2 + (y - 20) ** 2 / 6 ** 2 + (z - 20) ** 2 / 6 ** 2) <= 1.0

        # Subcortical and deep White Matter (WM, high intensity ~150)
        wm = self.t1_brain_mask & (~ventricles) & (r_sq <= 13 ** 2)

        # Cortical Gray Matter ribbon (GM, intermediate intensity ~85)
        gm = self.t1_brain_mask & (~ventricles) & (~wm)
        csf = ventricles

        t1_data = np.zeros(self.t1_shape, dtype=np.float32)
        t1_data[csf] = 25.0 + np.random.normal(0, 2, np.sum(csf))
        t1_data[gm] = 85.0 + np.random.normal(0, 3, np.sum(gm))
        t1_data[wm] = 150.0 + np.random.normal(0, 3, np.sum(wm))
        t1_data[skull] = 110.0 + np.random.normal(0, 3, np.sum(skull))
        self.t1w_img = nib.Nifti1Image(t1_data, self.t1_affine)

        # 2. Synthetic 4D BOLD series with physiological signal & head motion
        bold_data = np.zeros((*self.epi_shape, self.t_vol), dtype=np.float32)
        ex, ey, ez = np.ogrid[: self.epi_shape[0], : self.epi_shape[1], : self.epi_shape[2]]
        ec = (15, 15, 10)
        er_sq = (ex - ec[0]) ** 2 + (ey - ec[1]) ** 2 + (ez - ec[2]) ** 2
        epi_brain = er_sq <= 9 ** 2

        for t in range(self.t_vol):
            vol = np.zeros(self.epi_shape, dtype=np.float32)
            vol[epi_brain] = 100.0 + np.sin(0.1 * t) * 5.0 + np.random.normal(0, 2, np.sum(epi_brain))
            bold_data[..., t] = vol
        self.bold_img = nib.Nifti1Image(bold_data, self.epi_affine)

        # 3. Synthetic Target MNI Template
        mni_data = np.zeros(self.target_shape, dtype=np.float32)
        mni_data[er_sq <= 9 ** 2] = 100.0
        self.mni_img = nib.Nifti1Image(mni_data, self.target_affine)

        # 4. Synthetic CC200 Atlas (200 discrete parcels)
        atlas_data = np.zeros(self.target_shape, dtype=np.int32)
        brain_indices = np.argwhere(er_sq <= 9 ** 2)
        n_brain = len(brain_indices)
        chunk_size = max(1, n_brain // 200)
        for i in range(200):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < 199 else n_brain
            for coord in brain_indices[start:end]:
                atlas_data[tuple(coord)] = i + 1
        self.atlas_img = nib.Nifti1Image(atlas_data, self.target_affine)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_full_poc_v2_synthetic_execution_and_memory(self):
        """Execute complete PoC V2 preprocessing pipeline on synthetic data and verify memory limit."""
        mem_start = check_process_memory_gb()
        self.assertLess(mem_start, 2.5, "Initial memory exceeds 2.5 GB ceiling")

        # Step 1: Realignment & Motion Estimation
        realigned_bold, motion_params, fd = realign_4d_bold(self.bold_img)
        self.assertEqual(realigned_bold.shape, (*self.epi_shape, self.t_vol))
        self.assertEqual(motion_params.shape, (self.t_vol, 6))
        self.assertEqual(len(fd), self.t_vol)

        # Compute Friston-24 regressors
        friston_24 = build_friston_24_regressors(motion_params)
        self.assertEqual(friston_24.shape, (self.t_vol, 24))

        # Step 2: Functional Reference Image & Intracranial Mask
        mean_epi, epi_brain_mask = generate_functional_reference(realigned_bold)
        self.assertEqual(mean_epi.shape, self.epi_shape)

        # Step 3: Anatomical Brain Extraction (Skull Stripping with filled cavities)
        t1_brain, t1_brain_mask = extract_anatomical_brain(self.t1w_img, median_radius=2, numpass=2)
        self.assertEqual(t1_brain.shape, self.t1_shape)

        # Step 4: Bias-Field Correction
        t1_corrected, bias_field = correct_bias_field(t1_brain, t1_brain_mask, order=2)
        self.assertEqual(t1_corrected.shape, self.t1_shape)

        # Step 5: Anatomical Tissue Segmentation (GMM)
        prob_maps = segment_t1w_tissue(t1_corrected, t1_brain_mask, n_classes=3, method="gmm")
        self.assertIn("prob_csf", prob_maps)
        self.assertIn("prob_wm", prob_maps)

        # Step 6: Historical 0.96 Thresholding & Morphological Erosion
        # Enforce production parameters without threshold dilution or artificial injection
        tissue_masks = create_eroded_tissue_masks(
            prob_maps, wm_threshold=0.96, csf_threshold=0.96, erosion_voxels=1
        )
        val_res = validate_tissue_masks(tissue_masks, min_wm_voxels=50, min_csf_voxels=20)
        self.assertEqual(val_res["status"], "VALIDATED")
        self.assertGreaterEqual(val_res["wm_voxel_count"], 50)
        self.assertGreaterEqual(val_res["csf_voxel_count"], 20)
        self.assertEqual(val_res["overlap_voxel_count"], 0)

        # Step 7: Coregistration (EPI to T1w)
        m_epi_to_t1w, _ = register_epi_to_t1w(mean_epi, t1_brain, level_iters=[10])

        # Step 8: Nonlinear Normalization (T1w to MNI)
        m_t1w_to_mni, syn_map = register_t1w_to_mni(
            t1_brain, self.mni_img, affine_level_iters=[10], syn_level_iters=[5]
        )

        # Step 9: Composed Single-Step Reslicing Transformation
        composed_map = compose_transformations(
            m_epi_to_t1w,
            m_t1w_to_mni,
            syn_map,
            target_shape=self.target_shape,
            target_affine=self.target_affine,
            source_shape=self.epi_shape,
            source_affine=self.epi_affine,
        )

        # Resample native 4D BOLD directly into target grid in a single step
        resampled_4d = resample_4d_bold(
            realigned_bold,
            composed_map,
            target_shape=self.target_shape,
            target_affine=self.target_affine,
            interpolation="linear",
        )
        self.assertEqual(resampled_4d.shape, (*self.target_shape, self.t_vol))

        # Also resample eroded tissue masks into target space using nearest-neighbor
        # NOTE: Tissue masks originate in T1w space, so we use the T1w-to-MNI mapping (syn_map),
        # NOT the composed EPI-to-MNI mapping (which includes EPI-to-T1w coregistration).
        wm_eroded_mni = resample_mask_to_target(
            tissue_masks["mask_wm_eroded"], syn_map, self.target_shape, self.target_affine
        )
        csf_eroded_mni = resample_mask_to_target(
            tissue_masks["mask_csf_eroded"], syn_map, self.target_shape, self.target_affine
        )

        # Step 10: Extract 5 aCompCor components directly from registered tissue masks
        # Zero artificial fallback injection
        compcor_5 = extract_acompcor_components(
            resampled_4d, wm_eroded_mni, csf_eroded_mni, n_components=5
        )
        self.assertEqual(compcor_5.shape, (self.t_vol, 5))

        # Step 11: Build 31-column Confound Matrix
        df_confounds = build_confound_matrix(friston_24, compcor_5)
        self.assertEqual(df_confounds.shape, (self.t_vol, 31))

        # Step 12: Simultaneous Filtering & Nuisance Regression
        cleaned_bold = clean_functional_bold(
            resampled_4d, df_confounds, tr=2.0, low_cutoff_hz=0.01, high_cutoff_hz=0.10
        )
        self.assertEqual(cleaned_bold.shape, (*self.target_shape, self.t_vol))

        # Step 13: Extract CC200 Timeseries
        roi_df, signals = extract_cc200_timeseries(cleaned_bold, self.atlas_img)
        self.assertEqual(roi_df.shape, (200, self.t_vol + 1))
        self.assertEqual(signals.shape, (self.t_vol, 200))

        # Step 14: Compute 19,900 Fisher-z Connectivity Features
        z_vector, edge_names = compute_connectivity_vector(signals)
        self.assertEqual(len(z_vector), 19900)
        self.assertEqual(len(edge_names), 19900)
        self.assertTrue(np.all(np.isfinite(z_vector)))

        # Step 15: Generate Visual QC Figures using strictly grid-aligned inputs
        # 1. EPI-to-T1w Coregistration QC: resample mean EPI into T1w space using coregistration affine
        aff_epi_to_t1w = AffineMap(
            m_epi_to_t1w,
            domain_grid_shape=self.t1_shape,
            domain_grid2world=self.t1_affine,
            codomain_grid_shape=self.epi_shape,
            codomain_grid2world=self.epi_affine,
        )
        epi_resampled_t1w = nib.Nifti1Image(
            aff_epi_to_t1w.transform(np.asanyarray(mean_epi.dataobj), interpolation="linear"),
            self.t1_affine,
        )
        qc_coreg_path = self.work_dir / "qc_registration_epi_to_t1w.png"
        generate_coregistration_qc_overlay(t1_brain, epi_resampled_t1w, qc_coreg_path)
        self.assertTrue(qc_coreg_path.is_file())
        self.assertGreater(qc_coreg_path.stat().st_size, 5000)

        # 2. T1w-to-MNI Normalization QC: transform T1w into MNI grid using syn_map
        t1w_resampled_mni = nib.Nifti1Image(
            syn_map.transform(
                np.asanyarray(t1_corrected.dataobj),
                interpolation="linear",
                out_shape=self.target_shape,
                out_grid2world=self.target_affine,
            ),
            self.target_affine,
        )
        qc_norm_path = self.work_dir / "qc_registration_t1w_to_mni.png"
        generate_normalization_qc_overlay(self.mni_img, t1w_resampled_mni, qc_norm_path)
        self.assertTrue(qc_norm_path.is_file())
        self.assertGreater(qc_norm_path.stat().st_size, 5000)

        # 3. Tissue Mask QC: eroded WM and CSF on anatomical T1w
        qc_tissue_path = self.work_dir / "qc_tissue_masks.png"
        generate_tissue_mask_qc_overlay(
            t1_brain,
            tissue_masks["mask_wm_eroded"],
            tissue_masks["mask_csf_eroded"],
            qc_tissue_path,
        )
        self.assertTrue(qc_tissue_path.is_file())
        self.assertGreater(qc_tissue_path.stat().st_size, 5000)

        # 4. CC200 Parcellation QC: parcels on mean cleaned BOLD
        qc_parcellation_path = self.work_dir / "qc_parcellation_overlay.png"
        mean_cleaned = nib.Nifti1Image(
            np.mean(np.asanyarray(cleaned_bold.dataobj), axis=3), self.target_affine
        )
        generate_parcellation_qc_overlay(
            mean_cleaned, self.atlas_img, qc_parcellation_path, title="QC: CC200 Parcellation Overlay"
        )
        self.assertTrue(qc_parcellation_path.is_file())
        self.assertGreater(qc_parcellation_path.stat().st_size, 5000)

        # Assert Peak RAM < 2.5 GB
        mem_peak = check_process_memory_gb()
        self.assertLess(
            mem_peak,
            2.5,
            f"End-to-end synthetic pipeline exceeded 2.5 GB peak memory: {mem_peak:.2f} GB",
        )
        print(
            f"\n[SYNTHETIC END-TO-END PIPELINE SUCCESS] Start RAM: {mem_start:.2f} GB, "
            f"Peak RAM: {mem_peak:.2f} GB (Ceiling: 2.50 GB)"
        )


if __name__ == "__main__":
    unittest.main()
