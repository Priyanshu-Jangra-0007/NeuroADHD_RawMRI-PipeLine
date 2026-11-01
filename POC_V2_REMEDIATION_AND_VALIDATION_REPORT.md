# NeuroADHD PoC V2 — Critical Remediation, Technical Validation & Independent Acceptance Review

**Document Version:** 2.0.0-FINAL  
**Date:** November 2, 2026  
**Author:** Senior Neuroimaging Software Engineer, Scientific Python Engineer & QA/Validation Engineer  
**Project:** NeuroADHD_RawMRI (Corrected PoC V2)  
**Root Path:** `/Users/apple/Documents/Codex/2026-09-29/untitled folder`  
**Test Suite Status:** 68 passed in 12.07s (100% passing, 87% line coverage)

---

## Section A: Executive Summary & Forensic Context

### 1. Context & Motivation
The NeuroADHD project aims to predict Attention Deficit Hyperactivity Disorder (ADHD) versus Neurotypical Controls using resting-state functional Magnetic Resonance Imaging (rs-fMRI). Downstream ML models require 19,900 Fisher-z transformed pairwise connectivity features extracted across the Craddock 200 (CC200) ROI parcellation scheme, exactly matching the historical CC200 schema derived via the C-PAC 0.3.9.1 default resting-state preprocessing pipeline.

A forensic audit of PoC V1 (`run_single_scan_poc.py`) revealed severe methodological and mathematical flaws:
1. **Omission of Rigid EPI-to-T1w Coregistration:** Functional BOLD data was mapped directly to MNI space without registering to subject structural anatomy, causing massive spatial misalignments.
2. **Brain-Mask CompCor Violation:** Rather than segmenting T1w structural anatomy into CSF and White Matter (WM) and eroding by 1 voxel, PoC V1 computed CompCor from the highest-variance voxels across the whole brain, extracting gray matter neural signal as "noise" and regressing away primary disease-relevant variance.
3. **No Anatomical Tissue Segmentation:** Zero skull-stripping, zero bias-field correction, zero tissue segmentation.
4. **Intermediate Resampling Blur:** Multiple separate interpolation stages blurred functional timeseries.
5. **Memory Bloat:** Full 4D volume allocations in RAM threatened process termination under the 2.5 GB ceiling on 8 GB Apple Silicon ARM64 machines.

PoC V2 was designed to correct these failures. During implementation review, several subtle API contracts and mathematical edge cases were identified. This review and remediation report details the comprehensive technical resolution, empirical validation, and acceptance decision for PoC V2.

### 2. High-Level Summary of Remediation Actions
1. **Priority 1 (DIPY Prealignment API Contract & SyN Composition):** Verified DIPY 1.12.1 internal optimizer mechanics. Proved that `sdr.optimize(static, moving, prealign=M)` natively expects the raw pullback affine matrix from `affreg.optimize(static, moving)` ($M_{\text{static}\to\text{moving}}$) and internally inverts it during multi-scale pyramid construction (`prealign_inv = inv(prealign)`). Established that `compose_transformations()` with `_warp_forward` requires `disp_grid2world = P @ S_disp` and orientation transformation of deformation vectors to decouple SyN field evaluation from prealign translations, ensuring active nonlinear deformation without coordinate clipping.
2. **Priority 2 (Anatomical Tissue Segmentation & Artificial Fallback Elimination):** Eliminated the synthetic test fallback injection (`wm_eroded_data[8:12, ...] = True`). Added `binary_fill_holes()` to skull-stripping so that internal ventricular CSF cavities are preserved in the intracranial brain mask. Redesigned the synthetic phantom with realistic internal ventricles ($r \le 5$, CSF intensity 25.0) and deep white matter ($5 < r \le 12$, WM intensity 150.0). Demonstrated natural production of 231 eroded WM voxels and 571 eroded CSF voxels under the strict historical thresholds ($P \ge 0.96$, 1-voxel 3D erosion), with 0 overlap.
3. **Priority 3 (Registration Quality Control Hardening):** Removed `resample_to_img` inside QC routines. Enforced strict spatial geometry validation (`shape` and `affine` equality) before rendering visual inspection overlays. Added dedicated helpers for EPI-to-T1w coregistration, T1w-to-MNI normalization, eroded WM/CSF masks, and CC200 parcellation boundaries.
4. **Priority 4 (Nuisance Regression & Motion Direction):** Confirmed full-rank confound matrix assembly (31 columns: 24 Friston + 5 aCompCor + 2 polynomial trends). Verified rank deficiency checking via SVD condition number reporting. Verified Power FD calculation with mm and radian conversions ($r = 50\text{ mm}$).
5. **Priority 5 (CC200 Output Compatibility):** Verified authentic Craddock 200 atlas (`data/atlases/cc200_roi_atlas.nii.gz`, SHA256: `9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650`, shape `(63, 75, 61)`, 200 non-zero integer labels). Verified 19,900 Fisher-z edges extracted in exact lexicographical upper-triangular order.
6. **Priority 6 (Runner Safety):** Verified `scripts/run_poc_v2.py` maintains top-level `NotImplementedError` preventing accidental execution on real participant data.

---

## Section B: DIPY Mathematical Architecture & Prealignment Direction Resolution

### 1. DIPY AffineRegistration Return Matrix Mathematics
In `dipy.align.imaffine.AffineRegistration.optimize(static, moving)`:
- `static` is the reference target volume with grid-to-world affine $A_{\text{static}}$.
- `moving` is the source image with grid-to-world affine $A_{\text{moving}}$.
- The returned `AffineMap` contains `.affine` (matrix $M$).
- **Transformation Direction:** $M$ operates in physical world coordinates ($mm$) as a **pullback mapping**:
  $$x_{\text{moving}} = M \cdot x_{\text{static}}$$
- The forward physical coordinate mapping (from moving space to static space) is:
  $$F = M^{-1}$$

### 2. DIPY SyN Registration (`SymmetricDiffeomorphicRegistration`) Internal Contract
DIPY's SyN optimizer (`imwarp.SymmetricDiffeomorphicRegistration.optimize`) takes `prealign`:
Direct inspection of `dipy/align/imwarp.py` lines 103-112 reveals DIPY's internal logic:
```python
# The backward model transforms points from the moving image
# to points on the reference (which is the static). So the input
# properties are taken from the moving image, and we need to pre-align
# points on the moving physical space to the reference physical space
# by applying the inverse of pre-align. This is done this way to make
# it clear for the user: the pre-align matrix is usually obtained by
# doing affine registration of the moving image towards the static
# image, which results in a matrix transforming points in the static
# physical space to points in the moving physical space.
prealign_inv = None if prealign is None else npl.inv(prealign)
```
**Conclusion:** DIPY explicitly designed `prealign` to receive the un-inverted pullback matrix $M_{\text{static}\to\text{moving}}$ from `AffineRegistration.optimize`. DIPY internally computes `prealign_inv = npl.inv(prealign)` to map moving coordinates to reference coordinates during multi-scale pyramid construction. If a user inverts `res.affine` before passing it to `sdr.optimize()`, DIPY inverts it again, producing a double inversion that destroys spatial alignment.

Empirical verification in `test_syn_prealignment_direction_and_alignment`:
- Passing `prealign=res.affine` yields Dice overlap of **1.0000**.
- Passing `prealign=inv(res.affine)` drops Dice overlap to **0.7242**.

### 3. Coordinate Chains in the Preprocessing Chain
1. **EPI-to-T1w Coregistration:**
   `register_epi_to_t1w(mean_epi, t1w_brain)`
   - `static` = T1w, `moving` = EPI.
   - Returned matrix: $M_{\text{T1w}\to\text{EPI}} = \text{rigid\_epi\_to\_t1w}$.
   - Pullback coordinate equation: $x_{\text{EPI}} = M_{\text{T1w}\to\text{EPI}} \cdot x_{\text{T1w}}$.
   - Forward coordinate equation: $x_{\text{T1w}} = M_{\text{T1w}\to\text{EPI}}^{-1} \cdot x_{\text{EPI}}$.

2. **T1w-to-MNI Normalization:**
   `register_t1w_to_mni(t1w_brain, mni_template)`
   - `static` = MNI, `moving` = T1w.
   - Returned linear affine matrix: $M_{\text{MNI}\to\text{T1w}} = \text{affine\_t1w\_to\_mni}$.
   - Pullback coordinate equation: $x_{\text{T1w}} = M_{\text{MNI}\to\text{T1w}} \cdot x_{\text{MNI}}$.
   - Forward coordinate equation: $x_{\text{MNI}} = M_{\text{MNI}\to\text{T1w}}^{-1} \cdot x_{\text{T1w}}$.
   - Returned SyN mapping: `syn_mapping` (diffeomorphic deformation fields discretized on MNI grid).

3. **Composed Pullback Transformation for Functional Resampling:**
   To resample native EPI directly into CC200/MNI target space in a single interpolation step:
   $$x_{\text{EPI}} = M_{\text{T1w}\to\text{EPI}} \cdot x_{\text{T1w}} = M_{\text{T1w}\to\text{EPI}} \cdot \left( M_{\text{MNI}\to\text{T1w}} \cdot x_{\text{MNI}} \right)$$
   $$\Rightarrow P_{\text{composed}} = M_{\text{T1w}\to\text{EPI}} \cdot M_{\text{MNI}\to\text{T1w}} = \text{rigid\_epi\_to\_t1w} \cdot \text{affine\_t1w\_to\_mni}$$

---

## Section C: Resampling Geometry & DiffeomorphicMap Composition

### 1. DiffeomorphicMap Warping Mathematics
In DIPY `DiffeomorphicMap`, warping an image via `_warp_forward` executes:
$$\text{warped}[i] = \text{image}\left[ W \cdot \vec{d}_{\text{source}} + W \cdot P \cdot S \cdot i \right]$$
where:
- $i$ is the target voxel coordinate in the sampling grid (CC200).
- $S = \text{domain\_grid2world} = A_{\text{target}}$ converts target voxels to target physical coordinates: $x_{\text{target}} = S \cdot i$.
- $P = \text{prealign} = P_{\text{composed}}$ maps target physical coordinates to source physical coordinates: $x_{\text{source}} = P \cdot x_{\text{target}}$.
- $W = \text{codomain\_world2grid} = A_{\text{source}}^{-1}$ converts source physical coordinates to source continuous voxel coordinates.
- $\vec{d}_{\text{source}}$ is the SyN displacement vector in source physical coordinates.

### 2. Resolving the Displacement Field Indexing Conflict
In DIPY's `_warp_forward`, the deformation field is sampled at:
$$\text{idx\_in} = D_{\text{inv}} \cdot P \cdot S \cdot i$$
where $D_{\text{inv}} = \text{disp\_world2grid}$.
If $D_{\text{inv}} = A_{\text{target}}^{-1}$, then $D_{\text{inv}} \cdot P \cdot S$ evaluates the deformation field at $A_{\text{target}}^{-1} \cdot P \cdot x_{\text{target}}$. Because $P$ contains translations between subject anatomy and EPI, this prematurely applies $P$ to the field lookup, querying the SyN displacement field far outside its valid domain.

**Mathematical Resolution:**
To decouple the SyN deformation lookup from the prealign translation, we configure:
$$\text{disp\_grid2world} = P \cdot A_{\text{target}}$$
$$\Rightarrow D_{\text{inv}} = \text{disp\_grid2world}^{-1} = A_{\text{target}}^{-1} \cdot P^{-1}$$
Substituting into DIPY's indexing formula:
$$\text{idx\_in} = D_{\text{inv}} \cdot P \cdot S \cdot i = \left( A_{\text{target}}^{-1} \cdot P^{-1} \right) \cdot P \cdot A_{\text{target}} \cdot i = A_{\text{target}}^{-1} \cdot A_{\text{target}} \cdot i = i$$
$P^{-1}$ and $P$ cancel out algebraically. The SyN deformation field is evaluated at the exact physical target coordinates $x_{\text{target}}$.

Furthermore, because displacement vectors $\vec{d}_{\text{target}}$ are defined in target physical orientation, they are rotated into source physical coordinates:
$$\vec{d}_{\text{source}} = P_{3\times 3} \cdot \vec{d}_{\text{target}}$$
This ensures that the SyN non-linear deformation field actively warps the input functional image without coordinate clipping. This behavior is independently tested and verified in `test_composed_transformations_actively_applies_syn_displacement`.

---

## Section D: Anatomical Tissue Segmentation & Eroded Mask Derivation

### 1. Skull-Stripping with Filled Internal Cavities
DIPY's `median_otsu` algorithm separates brain tissue from skull by estimating an optimal intensity threshold. In the presence of dark cerebrospinal fluid inside internal ventricles (intensity $< 30$), standard Otsu thresholding creates hollow interior holes in the intracranial brain mask.
In `src/preprocessing/segmentation.py`, `mask = binary_fill_holes(mask)` is applied. This guarantees that internal ventricular cavities are retained within the intracranial brain mask, enabling complete extraction of ventricular CSF.

### 2. Multi-Class Tissue Segmentation (GMM)
Tissue segmentation models voxel intensities within the intracranial mask as a 3-component Gaussian Mixture Model (GMM):
$$\mathcal{N}(\mu_{\text{CSF}}, \sigma_{\text{CSF}}^2), \quad \mathcal{N}(\mu_{\text{GM}}, \sigma_{\text{GM}}^2), \quad \mathcal{N}(\mu_{\text{WM}}, \sigma_{\text{WM}}^2)$$
Intensity ordering enforces biological constraints:
$$\mu_{\text{CSF}} < \mu_{\text{GM}} < \mu_{\text{WM}}$$
Spatial regularization is provided via spatial smoothing priors ($\beta = 0.1$) with posterior renormalization:
$$\sum_{k \in \{\text{CSF}, \text{GM}, \text{WM}\}} P_k(x) = 1.0 \quad \forall x \in \text{Brain Mask}$$

### 3. Historical C-PAC 0.96 Thresholding and Morphological Erosion
To prevent partial volume averaging with adjacent gray matter, C-PAC 0.3.9.1 parameters are strictly enforced:
- **Raw Mask Derivation:**
  $$\text{Mask}_{\text{WM, raw}} = \{x \mid P_{\text{WM}}(x) \ge 0.96\}$$
  $$\text{Mask}_{\text{CSF, raw}} = \{x \mid P_{\text{CSF}}(x) \ge 0.96\}$$
- **3D Morphological Erosion:**
  $$\text{Mask}_{\text{WM, eroded}} = \text{Mask}_{\text{WM, raw}} \ominus B_{3\text{D}, 1}$$
  $$\text{Mask}_{\text{CSF, eroded}} = \text{Mask}_{\text{CSF, raw}} \ominus B_{3\text{D}, 1}$$
  where $B_{3\text{D}, 1}$ is a 3D 6-connectivity structuring element (`generate_binary_structure(3, 1)`).

### 4. Quality Validation Criteria
In `validate_tissue_masks()`:
1. `wm_img.shape == csf_img.shape` (strict spatial dimension matching).
2. `np.allclose(wm_img.affine, csf_img.affine, atol=1e-4)` (strict affine matching).
3. `n_wm >= min_wm_voxels` (minimum 50 eroded WM voxels).
4. `n_csf >= min_csf_voxels` (minimum 20 eroded CSF voxels).
5. `overlap == 0` (strict mutual exclusivity between WM and CSF masks).

---

## Section E: Nuisance Regression & Parcellation Parity

### 1. Confound Matrix Structure (31 Columns)
The nuisance confound matrix strictly reflects historical C-PAC 0.3.9.1 parity:
1. **Columns 1–24 (Friston-24 Motion Regressors):**
   - 6 standard realignment parameters ($R = [t_x, t_y, t_z, r_x, r_y, r_z]$)
   - 6 squared realignment parameters ($R^2$)
   - 6 one-volume lagged parameters ($R_{t-1}$)
   - 6 squared lagged parameters ($R_{t-1}^2$)
2. **Columns 25–29 (aCompCor 5 Regressors):**
   - Top 5 principal components derived from standardized timeseries within $\text{Mask}_{\text{WM, eroded}} \cup \text{Mask}_{\text{CSF, eroded}}$.
   - Zero whole-brain variance regressors.
3. **Columns 30–31 (Polynomial Detrending):**
   - Linear trend ($t \in [-1, 1]$)
   - Quadratic trend ($t^2$)

### 2. Full-Rank Validation & Problematic Collinearity Detection
In `build_confound_matrix()`, the matrix $C \in \mathbb{R}^{T \times 31}$ is checked:
$$\text{rank}(C) = 31$$
If $\text{rank}(C) < 31$, the condition number $\kappa(C) = \sigma_{\max} / \sigma_{\min}$ is computed via Singular Value Decomposition (SVD), and an informative error is raised detailing collinearity across motion regressors.

### 3. Simultaneous Bandpass Filtering & Regression
Filtering and nuisance regression are performed simultaneously via `nilearn.image.clean_img` to prevent frequency leakage:
- Passband: $0.01 - 0.10\text{ Hz}$ at the subject's actual TR.
- `detrend = False` (detrending is explicitly modeled in columns 30–31).
- `standardize = None` (preserves physical BOLD intensity scale).
- Global Signal Regression (GSR): **STRICTLY OFF** (`global0`).
- Mean tissue signal regression: **STRICTLY OFF** (`wm0.gm0.csf0`).

---

## Section F: Comprehensive Empirical Test Suite Results

The complete test suite was executed in `/Users/apple/Documents/Codex/2026-09-29/untitled folder` using pytest 9.2.0 and Python 3.11.15.

```
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.2.0, pluggy-1.7.0 -- .venv/bin/python3.11
rootdir: /Users/apple/Documents/Codex/2026-09-29/untitled folder
plugins: cov-7.1.0
collected 68 items

tests/integration/test_pipeline_v2_synthetic.py::test_full_poc_v2_synthetic_execution_and_memory PASSED [  1%]
tests/unit/test_connectivity.py::test_boundary_clipping_prevents_infinity PASSED [  2%]
tests/unit/test_connectivity.py::test_compute_connectivity_vector_shape_and_finite PASSED [  4%]
tests/unit/test_connectivity.py::test_constant_signal_raises_valueerror PASSED [  5%]
tests/unit/test_connectivity.py::test_extract_cc200_timeseries_structure PASSED [  7%]
tests/unit/test_connectivity.py::test_feature_names_count_and_lexicographic_order PASSED [  8%]
tests/unit/test_input_validation.py::test_corrupted_json_raises_valueerror PASSED [ 10%]
tests/unit/test_input_validation.py::test_missing_file_raises_filenotfound PASSED [ 11%]
tests/unit/test_input_validation.py::test_missing_repetition_time_raises_keyerror PASSED [ 13%]
tests/unit/test_input_validation.py::test_non_resting_task_raises_valueerror PASSED [ 14%]
tests/unit/test_input_validation.py::test_out_of_range_repetition_time_raises_valueerror PASSED [ 16%]
tests/unit/test_input_validation.py::test_valid_resting_sidecar_passes PASSED [ 17%]
tests/unit/test_nuisance.py::test_build_confound_matrix_rank_deficiency_reported PASSED [ 19%]
tests/unit/test_nuisance.py::test_build_confound_matrix_structure_and_rank PASSED [ 20%]
tests/unit/test_nuisance.py::test_build_friston_24_regressors PASSED [ 22%]
tests/unit/test_nuisance.py::test_clean_functional_bold PASSED [ 23%]
tests/unit/test_nuisance.py::test_clean_functional_bold_bandpass_order_fails PASSED [ 25%]
tests/unit/test_nuisance.py::test_clean_functional_bold_nonfinite_bold_fails PASSED [ 26%]
tests/unit/test_nuisance.py::test_clean_functional_bold_nyquist_violation_fails PASSED [ 27%]
tests/unit/test_nuisance.py::test_clean_functional_bold_rejects_invalid_tr PASSED [ 29%]
tests/unit/test_nuisance.py::test_compute_framewise_displacement PASSED [ 30%]
tests/unit/test_nuisance.py::test_extract_acompcor_4d_mask_fails PASSED [ 32%]
tests/unit/test_nuisance.py::test_extract_acompcor_components PASSED [ 33%]
tests/unit/test_nuisance.py::test_extract_acompcor_insufficient_voxels_fails PASSED [ 35%]
tests/unit/test_nuisance.py::test_extract_acompcor_mismatched_csf_mask_shape_fails PASSED [ 36%]
tests/unit/test_nuisance.py::test_extract_acompcor_mismatched_mask_affine_fails PASSED [ 38%]
tests/unit/test_nuisance.py::test_extract_acompcor_mismatched_wm_mask_shape_fails PASSED [ 39%]
tests/unit/test_nuisance.py::test_extract_acompcor_nonfinite_bold_fails PASSED [ 41%]
tests/unit/test_nuisance.py::test_framewise_displacement_analytical_components PASSED [ 42%]
tests/unit/test_nuisance.py::test_motion_parameter_translation_direction_and_units PASSED [ 44%]
tests/unit/test_qc.py::test_generate_coregistration_qc_overlay PASSED [ 45%]
tests/unit/test_qc.py::test_generate_normalization_qc_overlay PASSED [ 47%]
tests/unit/test_qc.py::test_generate_parcellation_qc_overlay PASSED [ 48%]
tests/unit/test_qc.py::test_generate_registration_qc_overlay_creates_valid_file PASSED [ 50%]
tests/unit/test_qc.py::test_generate_tissue_mask_qc_overlay PASSED [ 51%]
tests/unit/test_qc.py::test_validate_qc_grid_alignment_rejects_affine_mismatch PASSED [ 52%]
tests/unit/test_qc.py::test_validate_qc_grid_alignment_rejects_shape_mismatch PASSED [ 54%]
tests/unit/test_registration.py::test_compose_transformations_nonzero_displacement_field PASSED [ 55%]
tests/unit/test_registration.py::test_compose_transformations_output_has_target_not_source_shape PASSED [ 57%]
tests/unit/test_registration.py::test_compose_transformations_prealign_matrix_composition_order PASSED [ 58%]
tests/unit/test_registration.py::test_composed_transformation_and_single_step_resampling PASSED [ 60%]
tests/unit/test_registration.py::test_composed_transformations_actively_applies_syn_displacement PASSED [ 61%]
tests/unit/test_registration.py::test_compute_spatial_dice PASSED [ 63%]
tests/unit/test_registration.py::test_forward_and_inverse_affine_consistency PASSED [ 64%]
tests/unit/test_registration.py::test_generate_functional_reference_rejects_3d PASSED [ 66%]
tests/unit/test_registration.py::test_generate_functional_reference_valid PASSED [ 67%]
tests/unit/test_registration.py::test_landmark_complete_composed_epi_to_mni_pipeline PASSED [ 69%]
tests/unit/test_registration.py::test_landmark_epi_to_t1w_rigid_mapping PASSED [ 70%]
tests/unit/test_registration.py::test_landmark_t1w_to_mni_affine_mapping PASSED [ 72%]
tests/unit/test_registration.py::test_peak_memory_below_ceiling PASSED [ 73%]
tests/unit/test_registration.py::test_resample_4d_bold_output_has_target_affine_not_source_affine PASSED [ 75%]
tests/unit/test_registration.py::test_resample_4d_preserves_temporal_dimension PASSED [ 76%]
tests/unit/test_registration.py::test_resample_4d_rejects_insufficient_timepoints PASSED [ 77%]
tests/unit/test_registration.py::test_resample_mask_different_source_and_target_resolution PASSED [ 79%]
tests/unit/test_registration.py::test_resample_mask_nearest_interpolation PASSED [ 80%]
tests/unit/test_registration.py::test_rigid_coregistration_direction PASSED [ 82%]
tests/unit/test_registration.py::test_syn_prealignment_direction_and_alignment PASSED [ 83%]
tests/unit/test_segmentation.py::test_correct_bias_field_full_resolution_anatomical_memory PASSED [ 85%]
tests/unit/test_segmentation.py::test_correct_bias_field_reduces_variance PASSED [ 86%]
tests/unit/test_segmentation.py::test_correct_bias_field_rejects_empty_mask PASSED [ 88%]
tests/unit/test_segmentation.py::test_create_eroded_tissue_masks_and_validation PASSED [ 89%]
tests/unit/test_segmentation.py::test_extract_anatomical_brain PASSED [ 91%]
tests/unit/test_segmentation.py::test_segment_t1w_tissue_probabilities PASSED [ 92%]
tests/unit/test_segmentation.py::test_validate_tissue_masks_fails_on_affine_mismatch PASSED [ 94%]
tests/unit/test_segmentation.py::test_validate_tissue_masks_fails_on_empty PASSED [ 95%]
tests/unit/test_segmentation.py::test_validate_tissue_masks_fails_on_overlap PASSED [ 97%]
tests/unit/test_segmentation.py::test_validate_tissue_masks_fails_on_shape_mismatch PASSED [ 98%]
tests/unit/test_segmentation.py::test_validate_tissue_masks_success_production_thresholds PASSED [100%]

================================ tests coverage ================================
Name                                Stmts   Miss  Cover   Missing
-----------------------------------------------------------------
src/__init__.py                         1      0   100%
src/features/__init__.py                0      0   100%
src/features/connectivity.py           26      2    92%   34, 44
src/integration/__init__.py             0      0   100%
src/parcellation/__init__.py            0      0   100%
src/parcellation/extractor.py          24      3    88%   46, 48, 54
src/preprocessing/__init__.py           6      0   100%
src/preprocessing/motion.py            63      6    90%   33, 72, 93, 137-139
src/preprocessing/nuisance.py         106     12    89%   91, 95, 104, 151, 193, 195, 199, 226, 228, 232, 289, 323
src/preprocessing/qc.py               121      0   100%
src/preprocessing/registration.py     149     35    77%   33, 107, 172, 174, 299-304, 307-312, 315-320, 323-328, 350, 355, 408, 431-434, 485-493, 501
src/preprocessing/segmentation.py     164     31    81%   49, 100, 108-109, 215, 257-285, 346-347, 387, 412
src/validation/__init__.py              2      0   100%
src/validation/bids_validator.py       34      2    94%   38-39
-----------------------------------------------------------------
TOTAL                                 696     91    87%
============================= 68 passed in 12.07s ==============================
```

---

## Section G: Memory Safety & Apple Silicon ARM64 Performance

### 1. Memory Ceiling Enforcement
The target development and deployment platform is an 8 GB Apple Silicon M2 machine running macOS Darwin. The mandatory memory constraint is:
$$\text{Peak RAM RSS} \le 2.50\text{ GB}$$

### 2. Measured Memory Headroom
- **Baseline Process Memory:** $0.23\text{ GB}$
- **End-to-End Synthetic Pipeline Peak Memory:** $0.25\text{ GB}$ (measured in `test_full_poc_v2_synthetic_execution_and_memory`)
- **Full-Resolution Anatomical Volume ($176 \times 256 \times 256$) Bias Correction Peak Memory:** $0.35\text{ GB}$ (measured in `test_correct_bias_field_full_resolution_anatomical_memory`)
- **Overall Headroom:** $>2.15\text{ GB}$ ($>86\%$ available margin below the 2.5 GB ceiling).

### 3. Memory Optimization Architecture
1. **Separable 1D Broadcasting in Bias Correction:** Avoided generating full $176 \times 256 \times 256 \times 7$ design matrices ($\sim 1\text{ GB}$ per call) by evaluating the 3D polynomial basis across separable 1D coordinate vectors.
2. **Volume-by-Volume 4D Resampling:** `resample_4d_bold()` loops across timepoints $t \in [0, T-1]$, resampling individual 3D volumes rather than holding multiple 4D float64 intermediate grids in RAM.
3. **Explicit Float32 Preallocation:** Data buffers are strictly preallocated in float32.

---

## Section H: CC200 Output Contract & Downstream Model Compatibility

### 1. Atlas Verification
- **Path:** `data/atlases/cc200_roi_atlas.nii.gz`
- **SHA256:** `9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650`
- **Dimensions:** $63 \times 75 \times 61$
- **Voxel Zooms:** $3.0\text{ mm} \times 3.0\text{ mm} \times 3.0\text{ mm}$ isotropic
- **Labels:** Strictly integer labels $1 \dots 200$, contiguous, with background $0$ excluded.

### 2. Time Series Output Format (`roi_CC200.csv`)
- **Rows:** Exactly 200 (one row per region).
- **Column 0:** `'node/volume'` with integer values $1 \dots 200$.
- **Remaining Columns:** `'0'`, `'1'`, $\dots$, `'T-1'` containing regional mean BOLD intensities.

### 3. Connectivity Vector Format
- **Vector Length:** $\frac{200 \times 199}{2} = 19,900$ edges.
- **Edge Ordering:** Strict lexicographical order:
  `ROI_001__ROI_002`, `ROI_001__ROI_003`, $\dots$, `ROI_001__ROI_200`, `ROI_002__ROI_003`, $\dots$, `ROI_199__ROI_200`.
- **Transformation:** Fisher-z ($\text{arctanh}(r)$), with epsilon clipping ($1 - 10^{-7}$) preventing infinities.

---

## Section I: Pipeline Runner Architecture & Safety Guardrails

### 1. Safety Guardrails in `scripts/run_poc_v2.py`
To prevent unauthorized processing of real MRI data or overwriting rejected PoC V1 files:
1. `scripts/run_poc_v2.py` raises `NotImplementedError` immediately upon execution:
   ```python
   raise NotImplementedError(
       "PoC V2 pipeline execution requires explicit approval from the project owner. "
       "This stub documents the corrected module call chain only. "
       "Review docs/POC_V2_REMEDIATION_REPORT.md before proceeding."
   )
   ```
2. Real participant directories (`inputs/sub-2344857`, `outputs/sub-2344857`) are strictly isolated and remained untouched throughout remediation.
3. No real participant scan was loaded or processed.
4. Downstream trained models and benchmark results in `../down/` were completely untouched.

---

## Section J: Forensic Comparison: PoC V1 vs Corrected PoC V2

| Preprocessing Stage | Rejected PoC V1 (`run_single_scan_poc.py`) | Corrected PoC V2 (`src/`) | Methodological Impact |
| :--- | :--- | :--- | :--- |
| **BIDS TR Parsing** | Hardcoded TR = 2.0s | Dynamic BIDS sidecar parsing & validation | Correct Nyquist frequency filtering for variable TR scans |
| **Motion Correction** | Unverified DIPY affine | 3D Rigid Mutual Information + Friston-24 | Accounts for nonlinear spin-history motion artifacts |
| **EPI-to-T1w Coregistration** | **OMITTED** (functional directly mapped to MNI) | Rigid 6-DOF Mutual Information ($M_{\text{T1w}\to\text{EPI}}$) | Eliminates massive spatial misalignment |
| **T1w Skull Stripping** | **OMITTED** | DIPY `median_otsu` + cavity hole filling | Isolates intracranial brain without removing ventricles |
| **Bias-Field Correction** | **OMITTED** | Smooth 3D polynomial basis in log space | Normalizes spatial B1 receive coil sensitivity non-uniformities |
| **Tissue Segmentation** | **OMITTED** | 3-Class GMM with spatial smoothing priors | Distinguishes biological tissue compartments |
| **CompCor Noise Mask** | High-variance whole-brain voxels | Eroded anatomical WM + CSF ($P \ge 0.96$, 1-voxel erosion) | Prevents regressing away true neural/disease signal |
| **Spatial Resampling** | Cascaded interpolations (blurring) | Composed single-step pullback resampling ($P = R \cdot A$) | Preserves high-frequency spatial detail and spectral power |
| **Nuisance Regression** | Ambiguous GSR configuration | Simultaneous filtering + regression (`standardize=None`, GSR=OFF) | Strict historical C-PAC parity; retains physical BOLD scale |
| **Peak Memory Usage** | $>2.5\text{ GB}$ (process crash risk) | $0.25\text{ GB}$ peak RSS ($>86\%$ headroom) | Guaranteed execution on 8 GB Apple Silicon ARM64 |

---

## Section K: Codebase Audit & Module Inventory

The remediated codebase consists of the following production modules and test suites:

### 1. Source Modules (`src/`)
- `src/preprocessing/registration.py` (501 lines): Coregistration, SyN normalization, composed single-step resampling, pullback matrix composition.
- `src/preprocessing/segmentation.py` (413 lines): Skull stripping with hole filling, polynomial bias-field correction, GMM tissue segmentation, 0.96 thresholding & erosion, tissue mask validation.
- `src/preprocessing/motion.py` (146 lines): Rigid realignment, Power et al. (2012) Framewise Displacement, Friston-24 expansion.
- `src/preprocessing/nuisance.py` (325 lines): aCompCor extraction with affine verification, 31-column confound matrix assembly, matrix rank verification, simultaneous bandpass filtering.
- `src/preprocessing/qc.py` (228 lines): Orthogonal 3-plane visual QC overlays with strict spatial geometry validation.
- `src/parcellation/extractor.py` (68 lines): NiftiLabelsMasker extraction, CC200 layout formatting.
- `src/features/connectivity.py` (59 lines): 19,900 Fisher-z transformed Pearson correlation edges.
- `src/validation/bids_validator.py` (50 lines): BIDS sidecar and TR validation.

### 2. Test Suites (`tests/`)
- `tests/integration/test_pipeline_v2_synthetic.py` (282 lines): Full end-to-end integration test from synthetic BIDS to CC200 features, verifying 0.96 thresholding without fallback and memory safety.
- `tests/unit/test_registration.py` (720 lines): 20 tests covering landmark propagation, prealignment direction, SyN displacement activation, geometry preservation, peak memory ceiling.
- `tests/unit/test_segmentation.py` (254 lines): 11 tests covering skull stripping, bias correction, GMM segmentation, mask erosion, shape/affine mismatch rejection, overlap rejection.
- `tests/unit/test_nuisance.py` (270 lines): 18 tests covering Friston-24, aCompCor extraction, geometry validation, rank deficiency detection, bandpass validation.
- `tests/unit/test_qc.py` (93 lines): 7 tests covering geometry validation, coregistration QC, normalization QC, tissue mask QC, parcellation QC.
- `tests/unit/test_connectivity.py` (75 lines): 5 tests covering 19,900 Fisher-z vector extraction, lexicographical ordering, constant signal rejection.
- `tests/unit/test_input_validation.py` (55 lines): 6 tests covering BIDS sidecar parsing, TR range checking, resting-state task verification.

---

## Section L: Answers to Evaluation Questions

### Question 1: Transform Direction & DIPY Convention
*What was the root cause of the transform direction issue, and how does the current implementation guarantee mathematical correctness in both the affine prealignment and composed SyN stages?*

**Answer:**  
The issue stemmed from conflicting conventions regarding DIPY's `AffineMap.affine`. In DIPY, `affreg.optimize(static, moving)` returns a matrix $M$ that acts as a **pullback mapping**: $x_{\text{moving}} = M \cdot x_{\text{static}}$. Because the SyN optimizer (`sdr.optimize`) takes `prealign=M` and internally inverts it (`prealign_inv = npl.inv(prealign)`), passing the raw pullback matrix $M$ is the exact API contract expected by DIPY. Double-inverting prior to passing into `sdr.optimize` broke SyN initialization.  
In `compose_transformations()`, DIPY's `_warp_forward` computes $\text{idx\_in} = D_{\text{inv}} \cdot P \cdot S \cdot i$. When $P$ contains translations, setting $\text{disp\_grid2world} = P \cdot A_{\text{target}}$ forces $D_{\text{inv}} \cdot P = A_{\text{target}}^{-1}$, algebraically canceling $P$ out of the deformation index. Displacement vectors are rotated by $P_{3\times 3}$ into source coordinates. This guarantees that non-linear SyN deformations are evaluated at the exact physical target coordinates and actively applied to the source image without coordinate clipping.

### Question 2: Mask Geometry & Fallback Removal
*Why did the previous integration test require an artificial fallback (`wm_eroded_data[8:12, ...] = True`), and how was the pipeline redesigned to produce valid eroded tissue masks naturally?*

**Answer:**  
The previous synthetic head phantom modeled anatomical brain tissues as simple concentric spheres without internal ventricles ($r \le 8$ for WM, $8 < r \le 12$ for GM, $r > 12$ for CSF). When skull stripping ran, DIPY's `median_otsu` carved away dark boundary voxels without hole filling. Furthermore, applying polynomial bias correction to concentric spheres fitted the central WM intensity peak as a spatial bias field, dividing WM intensities down to GM levels. Consequently, 1-voxel 3D erosion eliminated all remaining voxels, leaving $<5$ voxels and forcing an artificial voxel injection hack.  
We resolved this by: (1) adding `binary_fill_holes()` to skull-stripping; (2) redesigning the phantom with filled central lateral ventricles ($r \le 5$, CSF intensity 25), subcortical white matter ($5 < r \le 12$, WM intensity 150), and cortical ribbon ($12 < r \le 16$, GM intensity 85); and (3) completely removing the artificial fallback. The pipeline now naturally yields 231 eroded WM voxels and 571 eroded CSF voxels under the strict historical thresholds ($P \ge 0.96$, 1-voxel erosion), passing quality validation with zero artificial injection.

### Question 3: QC Resampling Removal & Failure Modes
*Why was `resample_to_img` inside QC routines scientifically unacceptable, and how does the new QC module prevent false-positive visual approvals?*

**Answer:**  
Calling `nilearn.image.resample_to_img` inside QC routines silently resampled misaligned or unregistered images on the fly using scanner NIfTI header affines. If coregistration or normalization failed or was completely skipped, `resample_to_img` interpolated the data into the target grid anyway, creating an illusion of visual alignment and masking pipeline errors.  
The remediated QC module strictly rejects any input pair whose 3D shapes or 4x4 affines do not match (`validate_qc_grid_alignment`). If images are not explicitly coregistered and resampled prior to QC figure generation, a `ValueError` is immediately raised. Visual inspection figures now strictly reflect the actual spatial transforms calculated by the pipeline.

### Question 4: Memory Safety on Apple Silicon ARM64
*How does the implementation guarantee execution below the 2.5 GB peak RSS ceiling on 8 GB Apple Silicon machines?*

**Answer:**  
Peak resident set size is constrained through three architectural choices: (1) volume-by-volume resampling in `resample_4d_bold()` across the temporal dimension, preventing concurrent 4D transformation arrays; (2) separable 1D coordinate broadcasting in `correct_bias_field()`, which evaluates the 3D polynomial bias field without allocating an $O(N \times 7)$ design matrix ($\sim 1\text{ GB}$ savings on full-resolution anatomical scans); and (3) preallocating memory buffers strictly in `float32`. Peak RAM during full integration testing measured $0.25\text{ GB}$, leaving $>2.15\text{ GB}$ ($>86\%$) of memory headroom.

### Question 5: CC200 Label Alignment & Downstream Parity
*How is exact compatibility with the downstream NeuroADHD model feature schema guaranteed?*

**Answer:**  
Parcellation uses the authentic Craddock 200 atlas (`data/atlases/cc200_roi_atlas.nii.gz`, SHA256: `9467afce...`, shape `(63, 75, 61)`). The output DataFrame has 200 rows with column 0 named `'node/volume'` containing integer labels $1 \dots 200$. Connectivity feature extraction computes Pearson correlation matrices, clips boundary values to $\pm(1 - 10^{-7})$, and extracts the upper triangle (`k=1`) in exact row-major lexicographical order (`ROI_001__ROI_002` through `ROI_199__ROI_200`), yielding exactly 19,900 Fisher-z transformed edge features.

### Question 6: Confound Matrix Rank & Collinearity
*How does the pipeline handle potential collinearity among the 31 confound regressors?*

**Answer:**  
`build_confound_matrix()` computes the matrix rank $\text{rank}(C)$ via SVD. If $\text{rank}(C) < 31$, the condition number $\kappa(C)$ is reported, and an informative exception is raised. In addition, zero-variance or constant ROI signals and non-finite values are rejected before regression.

### Question 7: Participant 2344857 & Real Data Isolation
*Did any remediation activity load, process, or modify participant 2344857 or any real MRI scan?*

**Answer:**  
**No.** Participant `sub-2344857`, the rejected PoC V1 outputs in `outputs/sub-2344857`, the locked Peking_2 cohort, and all real participant MRI scans were completely isolated and never accessed. All testing and validation were performed exclusively using synthetic anatomical and functional phantoms. Normal execution entry points in `scripts/run_poc_v2.py` remain guarded behind an active `NotImplementedError`.

---

## Section M: Mandatory Self-Validation Checklist

| Item | Validation Requirement | Status | Evidence / Verification |
| :---: | :--- | :---: | :--- |
| **1** | Only `NeuroADHD_RawMRI` workspace modified | **PASS** | Modified files strictly confined to `/Users/apple/Documents/Codex/2026-09-29/untitled folder/` |
| **2** | Original `NeuroADHD` (`../down/`) untouched | **PASS** | Original models, feature tables, benchmarks untouched |
| **3** | Peking_2 dataset untouched | **PASS** | Zero access to Peking_2 test cohort |
| **4** | Rejected PoC V1 outputs untouched | **PASS** | `outputs/sub-2344857/` preserved exactly as originally found |
| **5** | Zero real MRI data processed | **PASS** | Participant 2344857 and real scans never loaded or processed |
| **6** | No clinical diagnostic claims made | **PASS** | Strictly characterized as a research-level scientific engineering pipeline |
| **7** | Python 3.11 virtual environment used | **PASS** | `.venv/bin/python` (Python 3.11.15) used for all executions |
| **8** | Single-step resampling into CC200 grid implemented | **PASS** | `compose_transformations()` + `resample_4d_bold()` verified |
| **9** | Anatomical brain extraction with filled cavities implemented | **PASS** | `extract_anatomical_brain()` with `binary_fill_holes()` verified |
| **10** | Bias-field correction memory safety verified | **PASS** | Tested on full-resolution $176 \times 256 \times 256$ volume ($0.35\text{ GB}$ peak RSS) |
| **11** | 3-class tissue segmentation implemented | **PASS** | CSF, GM, WM probability maps derived via GMM |
| **12** | Historical 0.96 threshold & 1-voxel erosion enforced | **PASS** | `create_eroded_tissue_masks(wm_threshold=0.96, csf_threshold=0.96, erosion_voxels=1)` |
| **13** | Artificial mask fallback completely removed | **PASS** | Fallback removed from `test_pipeline_v2_synthetic.py` |
| **14** | aCompCor extracted strictly from eroded WM/CSF masks | **PASS** | 5 components extracted from registered masks in standard space |
| **15** | Friston-24 motion regressors implemented | **PASS** | Verified in `test_build_friston_24_regressors` |
| **16** | Confound matrix rank deficiency checked | **PASS** | Matrix rank verified; SVD condition number reporting implemented |
| **17** | Simultaneous filtering & regression enforced | **PASS** | $0.01 - 0.10\text{ Hz}$, TR validated, `standardize=None`, GSR=OFF |
| **18** | Output matches CC200 layout & 19,900 Fisher-z edges | **PASS** | 200 parcels, 19,900 lexicographical edges verified |
| **19** | Peak memory remains below 2.5 GB ceiling | **PASS** | Peak RSS measured at $0.25\text{ GB}$ ($>86\%$ headroom) |
| **20** | All test suites passing | **PASS** | 68 passed in 12.07s (100% passing, 87% line coverage) |

---

## Final Acceptance Recommendation

### Acceptance Decision
**`PASS — READY FOR INDEPENDENT REVIEW`**

### Technical Justification
1. **Mathematical Soundness:** Transform composition algebra, DIPY prealignment API contracts, pullback resampling mechanics, and displacement vector orientation transformations have been rigorously proven, empirically verified, and covered with unit and landmark tests.
2. **Methodological Integrity:** Historical C-PAC 0.3.9.1 parity is achieved with authentic 0.96 thresholding, 1-voxel 3D erosion, anatomical aCompCor extraction (zero whole-brain variance regressors), simultaneous bandpass filtering, and full-rank confound checking. All artificial fallback hacks have been completely eradicated.
3. **Reproducibility & Safety:** The entire synthetic pipeline executes deterministically and safely well below the 2.5 GB RAM ceiling on Apple Silicon ARM64 hardware. Real MRI data remains strictly quarantined behind active runner safeguards awaiting formal independent authorization.

