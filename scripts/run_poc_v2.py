#!/usr/bin/env python3
"""PoC V2 single-subject preprocessing pipeline runner — STUB (NOT for execution yet).

DO NOT RUN until code review is approved by the project owner.

Implements the corrected V2 preprocessing chain:
1. BIDS validation and sidecar parsing (TR extraction)
2. 3D rigid motion realignment (DIPY) -> motion parameters + FD
3. Functional reference generation (temporal mean)
4. EPI-to-T1w rigid coregistration (DIPY 6-DOF)
5. T1w-to-MNI nonlinear normalization (DIPY SyN)
6. Composed single-step resampling into CC200 grid
7. Anatomical brain extraction (DIPY median_otsu)
8. Bias-field correction (polynomial, log-space)
9. Tissue segmentation (GMM: CSF, GM, WM probability maps)
10. Eroded WM and CSF masks (threshold=0.96, 1-voxel erosion)
11. Tissue mask registration to CC200 grid
12. aCompCor extraction from registered masks (5 components)
13. Friston-24 motion regressor expansion
14. Confound matrix assembly (31 columns)
15. Simultaneous nuisance regression + 0.01-0.10 Hz bandpass filtering
16. CC200 ROI time series extraction (200 parcels)
17. 19,900 Fisher-z pairwise connectivity features
18. QC overlay generation (registration, parcellation)
19. Provenance JSON report

Key corrections vs. V1 (run_single_scan_poc.py):
- V1 called normalize_to_mni() directly without EPI-to-T1w coregistration -> WRONG
- V1 used brain-mask CompCor instead of anatomical WM/CSF masks -> WRONG
- V1 did not perform tissue segmentation -> WRONG
- V2 uses separate T1w processing path with skull-strip -> bias correct -> segment -> erode -> register

Memory ceiling: 2.5 GB peak RSS (Apple Silicon M2, 8 GB physical RAM).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_poc_v2_stub():
    """Illustrate the V2 pipeline call graph. Not for execution."""

    raise NotImplementedError(
        "PoC V2 pipeline execution requires explicit approval from the project owner. "
        "This stub documents the corrected module call chain only. "
        "Review docs/POC_V2_REMEDIATION_REPORT.md before proceeding."
    )

    # ----------------------------------------------------------------
    # The corrected V2 pipeline call chain (for documentation):
    # ----------------------------------------------------------------
    # Step 0: Load inputs, parse TR from BIDS sidecar
    # bold_img = nib.load(RAW_BOLD)
    # t1w_img = nib.load(RAW_T1)
    # atlas_img = nib.load(ATLAS_NII)
    # tr = parse_tr_from_sidecar(RAW_JSON)

    # Step 1: Motion realignment
    # realigned_img, motion_params, fd = realign_4d_bold(bold_img)

    # Step 2: Functional reference
    # mean_epi_img, epi_brain_mask = generate_functional_reference(realigned_img)

    # Step 3: T1w brain extraction + bias correction + segmentation
    # t1w_brain_img, t1w_mask_img = extract_anatomical_brain(t1w_img)
    # t1w_corrected_img, bias_img = correct_bias_field(t1w_brain_img, t1w_mask_img)
    # prob_maps = segment_t1w_tissue(t1w_corrected_img, t1w_mask_img)
    # eroded_masks = create_eroded_tissue_masks(prob_maps, reference_t1w_img=t1w_corrected_img, wm_threshold=0.96, csf_threshold=0.96)
    # validate_tissue_masks(eroded_masks)

    # Step 4: EPI -> T1w rigid coregistration
    # rigid_epi_to_t1w, rigid_affine_map = register_epi_to_t1w(mean_epi_img, t1w_brain_img)

    # Step 5: T1w -> MNI SyN registration
    # affine_t1w_to_mni, syn_mapping = register_t1w_to_mni(t1w_corrected_img, mni_template_img)

    # Step 6: Compose and resample BOLD -> CC200 grid (single interpolation step)
    # composed_map = compose_transformations(
    #     rigid_epi_to_t1w=rigid_epi_to_t1w,
    #     affine_t1w_to_mni=affine_t1w_to_mni,
    #     syn_mapping=syn_mapping,
    #     target_shape=atlas_img.shape,        # CC200: (63, 75, 61)
    #     target_affine=atlas_img.affine,      # CC200 affine
    #     source_shape=realigned_img.shape[:3], # native EPI
    #     source_affine=realigned_img.affine,
    # )
    # norm_bold_img = resample_4d_bold(realigned_img, composed_map,
    #                                   target_shape=atlas_img.shape[:3],
    #                                   target_affine=atlas_img.affine)

    # Step 7: Register tissue masks to CC200 standard space
    # NOTE: Tissue masks originate in T1w space, so we use the T1w-to-MNI mapping (syn_mapping),
    # NOT the composed EPI-to-MNI mapping (which includes EPI-to-T1w coregistration).
    # wm_eroded_mni = resample_mask_to_target(eroded_masks['mask_wm_eroded'], syn_mapping,
    #                                         atlas_img.shape[:3], atlas_img.affine)
    # csf_eroded_mni = resample_mask_to_target(eroded_masks['mask_csf_eroded'], syn_mapping,
    #                                          atlas_img.shape[:3], atlas_img.affine)

    # Step 8: aCompCor extraction using verified standard-space masks
    # Verifies both NIfTI spatial shape and grid affine match against norm_bold_img
    # acompcor_5 = extract_acompcor_components(norm_bold_img, wm_eroded_mni, csf_eroded_mni, n_components=5)

    # Step 9: Confound matrix
    # friston_24 = build_friston_24_regressors(motion_params)
    # confounds_df = build_confound_matrix(friston_24, acompcor_5)

    # Step 10: Nuisance regression + bandpass
    # cleaned_bold = clean_functional_bold(norm_bold_img, confounds_df, tr=tr)

    # Step 11: CC200 parcellation
    # roi_df, signals = extract_cc200_timeseries(cleaned_bold, atlas_img)

    # Step 12: Connectivity features
    # z_vector, edge_names = compute_connectivity_vector(signals)

    # Step 13: QC overlays
    # generate_registration_qc_overlay(mni_template_img, norm_bold_mean_img, ...)
    # generate_parcellation_qc_overlay(norm_bold_mean_img, atlas_img, ...)


if __name__ == "__main__":
    run_poc_v2_stub()

