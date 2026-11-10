# PoC V2 Remediation Report

> [!WARNING]
> **Superseded by [`POC_V2_FINAL_REVIEW_REMEDIATION.md`](<your-project-folder>/docs/POC_V2_FINAL_REVIEW_REMEDIATION.md)**
> Sections discussing affine matrix-composition order and transformation geometry in this historical document are superseded by [`POC_V2_FINAL_REVIEW_REMEDIATION.md`](<your-project-folder>/docs/POC_V2_FINAL_REVIEW_REMEDIATION.md), which reflects the canonical implemented codebase in [`src/preprocessing/registration.py`](<your-project-folder>/src/preprocessing/registration.py) (`compose_transformations`). This document is preserved for historical audit provenance.

**Date**: 2026-11-01  
**Project**: NeuroADHD_RawMRI  
**Phase**: PoC V2 — Code Review Remediation  
**Host Hardware**: Apple Silicon M2, 8 GB physical RAM, macOS Darwin ARM64  
**Status**: REMEDIATION COMPLETE — 47/47 TESTS PASSING — AWAITING EXPLICIT USER APPROVAL FOR REAL-SCAN EXECUTION

---

## 1. Executive Summary

Following code review of the initial PoC V2 implementation report, all five critical blocking issues and remediation priorities have been systematically addressed, verified, and hardened:

1. **DiffeomorphicMap Sampling Geometry & Prealign Direction Fixed**: Reconciled the inverted `domain` / `codomain` semantics and corrected the non-commutative prealign composition order ($\text{prealign} = M_{\text{T1w}\to\text{MNI}} \cdot M_{\text{EPI}\to\text{T1w}}$) in `compose_transformations()`.
2. **Geometry-Sensitive Registration Unit Tests Added**: Created 5 new unit tests using non-identical spatial dimensions, anisotropic affines, distinct voxel resolutions (2.0 mm vs. 3.0 mm), non-zero translations, and non-trivial SyN displacement fields.
3. **Nuisance Regression Hardened**: Added strict 3D mask dimensionality and BOLD spatial grid shape validation in `extract_acompcor_components()`, Nyquist frequency upper-bound enforcement and passband ordering in `clean_functional_bold()`, finite-data verification, and SVD condition-number reporting for rank-deficient confound matrices. Added 8 new unit tests covering all failure modes.
4. **Implementation Report Reconciled**: Clarified that matrix rank checking was present in `build_confound_matrix()` and has now been upgraded with SVD diagnostics. Documented the pytest rootdir canonical resolution (`untitled folder` physical path vs. `NeuroADHD_RawMRI` symlink).
5. **Code Review Package & V2 Stub Delivered**: Audited `motion.py`, `segmentation.py`, `qc.py`, and `registration.py`. Formally documented critical architectural deficiencies in the rejected V1 runner (`scripts/run_single_scan_poc.py`). Created the modular V2 pipeline stub (`scripts/run_poc_v2.py`) with a strict runtime safety guard (`NotImplementedError`) preventing unauthorized execution.

All 47 unit and integration tests are passing with 86% overall codebase coverage. Peak process RSS remained at **0.29 GB**, well within the 2.5 GB hardware ceiling.

---

## 2. Files Changed

| File Path | Nature of Change | Priority | Before Behavior | Remediated Behavior |
|:---|:---|:---:|:---|:---|
| [`src/preprocessing/registration.py`](<your-project-folder>/src/preprocessing/registration.py) | Bug fix & interface refinement | **P1** | `domain` was native EPI and `codomain` was CC200; `prealign` was $M_{\text{EPI}\to\text{T1w}} \cdot M_{\text{T1w}\to\text{MNI}}$ (inverted) | Explicit `target_shape/affine` (domain=CC200) and `source_shape/affine` (codomain=EPI); `prealign` corrected to $M_{\text{T1w}\to\text{MNI}} \cdot M_{\text{EPI}\to\text{T1w}}$ |
| [`tests/unit/test_registration.py`](<your-project-folder>/tests/unit/test_registration.py) | Unit test expansion | **P2** | 9 tests; tested only equal-sized grids and near-identity transforms | 14 tests (+5 new); tests distinct grid shapes, LAS orientation affine, prealign non-commutativity, label preservation across resolutions, and non-zero deformation fields |
| [`src/preprocessing/nuisance.py`](<your-project-folder>/src/preprocessing/nuisance.py) | Hardening & input validation | **P3** | No mask shape check against BOLD grid; no non-finite BOLD check; no Nyquist check; basic rank check without condition number | Strict 3D mask checks; shape equality with `bold.shape[:3]`; non-finite BOLD rejected; Nyquist limit enforced ($f_{\text{high}} < 0.5/\text{TR}$); SVD condition number diagnostics |
| [`tests/unit/test_nuisance.py`](<your-project-folder>/tests/unit/test_nuisance.py) | Unit test expansion | **P3** | 7 tests; tested valid inputs and basic length mismatch | 15 tests (+8 new); tests WM/CSF spatial mismatch, 4D mask rejection, non-finite input rejection, Nyquist violations, inverted bandpass, and rank deficiency with SVD reporting |
| [`tests/integration/test_pipeline_v2_synthetic.py`](<your-project-folder>/tests/integration/test_pipeline_v2_synthetic.py) | Test update | **P1** | Passed legacy `domain_shape=epi_shape`, `codomain_shape=target_shape` | Uses explicit `target_shape=target_shape`, `source_shape=epi_shape` matching corrected DIPY semantics |
| [`docs/POC_V2_IMPLEMENTATION_REPORT.md`](<your-project-folder>/docs/POC_V2_IMPLEMENTATION_REPORT.md) | Documentation addendum | **P4** | Described initial V2 state; lacked remediation findings | Added Section 9 documenting remediation findings, DIPY semantics, SVD rank checks, and pytest rootdir |
| [`scripts/run_poc_v2.py`](<your-project-folder>/scripts/run_poc_v2.py) | New pipeline stub | **P5** | Did not exist (only rejected V1 runner existed) | Modular stub documenting the 19-step V2 execution graph with safety guard (`raise NotImplementedError`) |

---

## 3. Priority 1: DiffeomorphicMap Sampling Geometry Fix

### 3.1 Problem and Root Cause Analysis

In DIPY's `imwarp.DiffeomorphicMap`, coordinate evaluation occurs via pullback sampling:
- **`domain_shape` / `domain_grid2world`**: The **target output grid** (CC200 standard space) where the output image is defined. When calling `.transform(image)`, DIPY iterates over voxel coordinates in this domain grid.
- **`codomain_shape` / `codomain_grid2world`**: The **source/moving image space** (native EPI) from which voxel intensities are interpolated.
- **`prealign`**: A 4x4 affine matrix in physical world coordinates mapping points from the moving space (codomain / EPI) into the static space (domain / MNI).

In the previous implementation of `compose_transformations()`:
1. `domain_shape` was incorrectly assigned to native EPI, and `codomain_shape` to CC200. This inverted the definition of sampling space.
2. The `prealign` composition matrix order was:
   $$\text{prealign}_{\text{prior}} = M_{\text{EPI}\to\text{T1w}} \cdot M_{\text{T1w}\to\text{MNI}} \quad \text{(INCORRECT)}$$
   Because matrix transformation operates from right to left on coordinate vectors ($\vec{x}_{\text{out}} = A B \vec{x}_{\text{in}}$), applying $M_{\text{T1w}\to\text{MNI}}$ first to native EPI coordinates is physically meaningless.

### 3.2 Corrected Implementation

In [`src/preprocessing/registration.py`](<your-project-folder>/src/preprocessing/registration.py):
```python
def compose_transformations(
    rigid_epi_to_t1w: np.ndarray,
    affine_t1w_to_mni: np.ndarray,
    syn_mapping: DiffeomorphicMap,
    target_shape: Optional[Tuple[int, int, int]] = None,      # CC200 target grid (domain)
    target_affine: Optional[np.ndarray] = None,               # CC200 affine (domain_grid2world)
    source_shape: Optional[Tuple[int, int, int]] = None,      # native EPI shape (codomain)
    source_affine: Optional[np.ndarray] = None,               # native EPI affine (codomain_grid2world)
    *,
    domain_shape: Optional[Tuple[int, int, int]] = None,
    domain_affine: Optional[np.ndarray] = None,
    codomain_shape: Optional[Tuple[int, int, int]] = None,
    codomain_affine: Optional[np.ndarray] = None,
) -> DiffeomorphicMap:
    ...
    # Corrected physical-space composition: EPI -> T1w -> MNI
    prealign_composed = affine_t1w_to_mni @ rigid_epi_to_t1w

    composed_map = DiffeomorphicMap(
        3,
        disp_shape=syn_mapping.disp_shape,
        disp_grid2world=syn_mapping.disp_grid2world,
        domain_shape=target_shape,         # Target CC200 grid
        domain_grid2world=target_affine,   # Target CC200 affine
        codomain_shape=source_shape,       # Native EPI grid
        codomain_grid2world=source_affine, # Native EPI affine
        prealign=prealign_composed,        # EPI world -> MNI world
    )
    composed_map.forward = syn_mapping.forward
    composed_map.backward = syn_mapping.backward
    return composed_map
```

When `resample_4d_bold(bold_img, composed_map, target_shape, target_affine)` calls `composed_map.transform(vol_3d)`, DIPY now correctly evaluates native EPI voxels directly onto the target CC200 grid in a single interpolation step.

---

## 4. Priority 2: Geometry-Sensitive Tests

To ensure registration correctness under non-trivial spatial transformations, five new test cases were added to [`tests/unit/test_registration.py`](<your-project-folder>/tests/unit/test_registration.py):

1. **`test_compose_transformations_output_has_target_not_source_shape`**:
   - Source EPI grid: $(20, 18, 15)$ @ $3.5\text{ mm}$ isotropic.
   - Target CC200 grid: $(24, 22, 18)$ with LAS orientation affine ($[-3.0, 3.0, 3.0]$ voxel zooms and non-zero origins).
   - Verifies `composed_map.domain_shape == target_shape` and `composed_map.codomain_shape == source_shape`.
2. **`test_resample_4d_bold_output_has_target_affine_not_source_affine`**:
   - Resamples a 4D BOLD series ($T=52$) from native $(18, 16, 14)$ space to target $(20, 18, 15)$ space.
   - Verifies that output image affine strictly matches `target_affine` (not source affine) and preserves all 52 timepoints.
3. **`test_compose_transformations_prealign_matrix_composition_order`**:
   - Constructs non-commutative affine matrices (translation + mild anisotropy scaling).
   - Directly asserts `prealign == affine_t1w_to_mni @ rigid_epi_to_t1w` and verifies it diverges from the erroneous reverse order.
4. **`test_resample_mask_different_source_and_target_resolution`**:
   - Evaluates nearest-neighbor interpolation on discrete tissue labels ($0, 1, 2$) across a resolution transition from $2.0\text{ mm}$ (T1w) to $3.0\text{ mm}$ (CC200).
   - Verifies exact preservation of integer label values $\{0, 1, 2\}$ without interpolation artifact fractions.
5. **`test_compose_transformations_nonzero_displacement_field`**:
   - Constructs a synthetic SyN mapping with a constant $+2.0\text{ mm}$ forward and $-2.0\text{ mm}$ backward deformation field.
   - Verifies that the composed mapping preserves non-zero displacement vectors intact alongside grid geometry.

---

## 5. Priority 3: Nuisance Regression Hardening

### 5.1 Hardened Input Validations Added

In [`src/preprocessing/nuisance.py`](<your-project-folder>/src/preprocessing/nuisance.py):

#### `extract_acompcor_components()`
- **3D Mask Dimensionality Check**: Rejects any non-3D mask array with `ValueError("Expected 3D WM/CSF mask, got ND array")`.
- **Mask-BOLD Spatial Grid Alignment**: Asserts `eroded_wm_mask.shape == bold_data.shape[:3]` and `eroded_csf_mask.shape == bold_data.shape[:3]`. This guarantees tissue masks have been registered into the standard CC200 space before aCompCor extraction, preventing silent dimension broadcasting bugs.
- **Pre-Cleaning Finite Value Validation**: Rejects BOLD series containing `NaN` or `Inf` prior to PCA extraction with `ValueError("BOLD data contains N non-finite values")`.

#### `clean_functional_bold()`
- **Nyquist Frequency Upper Boundary**: Calculates Nyquist limit $f_{\text{Nyquist}} = 0.5 / \text{TR}$. If $f_{\text{high}} \ge f_{\text{Nyquist}}$, raises `ValueError("High cutoff frequency meets or exceeds Nyquist frequency")`.
- **Passband Boundary Ordering**: Validates $0 < f_{\text{low}} < f_{\text{high}}$ (e.g. rejects $0.10\text{ Hz} \to 0.05\text{ Hz}$).
- **Standard-Space Finite Value Check**: Rejects any standard-space BOLD series with non-finite voxel intensities prior to Nilearn filtering.
- **Historical Parity Enforced**: Preserves `standardize=None`, `detrend=False`, GSR strictly OFF, and mean tissue signal regression strictly OFF.

#### `build_confound_matrix()`
- **SVD Condition Number & Rank Reporting**: Evaluates matrix rank via `np.linalg.matrix_rank()`. If $\text{rank} < 31$, computes Singular Value Decomposition ($\sigma_1 / \sigma_{31}$) and raises:
  ```
  ValueError: Confound matrix is rank-deficient: rank 27 < 31 columns. Condition number: 1.4e+16.
  Check motion parameters for perfect collinearity...
  ```

### 5.2 Unit Tests Added (`tests/unit/test_nuisance.py`)

Eight new unit tests verify each error condition:
1. `test_extract_acompcor_mismatched_wm_mask_shape_fails`: Catches spatial dimension mismatch between WM mask and BOLD.
2. `test_extract_acompcor_mismatched_csf_mask_shape_fails`: Catches spatial dimension mismatch between CSF mask and BOLD.
3. `test_extract_acompcor_4d_mask_fails`: Rejects 4D mask arrays.
4. `test_extract_acompcor_nonfinite_bold_fails`: Injects `NaN` into BOLD series and verifies clean rejection.
5. `test_clean_functional_bold_nyquist_violation_fails`: Tests $\text{TR}=2.0\text{ s}$ ($f_{\text{Nyquist}}=0.25\text{ Hz}$) with $f_{\text{high}}=0.30\text{ Hz}$; verifies exception.
6. `test_clean_functional_bold_bandpass_order_fails`: Tests $f_{\text{low}}=0.10 \ge f_{\text{high}}=0.05$; verifies exception.
7. `test_clean_functional_bold_nonfinite_bold_fails`: Injects `Inf` into standard-space BOLD; verifies rejection before cleaning.
8. `test_build_confound_matrix_rank_deficiency_reported`: Injects collinear motion regressors; verifies rank deficiency exception and SVD condition number reporting.

---

## 6. Priority 4: Implementation Report Reconciliation

Two specific items raised in the code review were reconciled:

1. **Confound Matrix Full Rank Verification**:
   - The initial implementation report accurately stated that full column rank was checked (`np.linalg.matrix_rank(df.to_numpy()) == 31`). However, the original error message did not diagnostic details.
   - Remediated implementation now uses SVD to compute singular values and outputs the exact condition number alongside the rank deficiency count.
2. **Pytest Root Directory Name**:
   - The pytest output displays `rootdir: <your-project-folder>`.
   - This occurs because the physical directory on disk was originally created with that folder name. A filesystem symbolic link at `<your-project-folder>` points directly to `untitled folder`.
   - When pytest initializes, Python's `os.path.realpath` resolves the canonical physical path. Both paths access identical source files; all tests and changes execute directly on the active NeuroADHD_RawMRI codebase.

---

## 7. Priority 5: Code Review Package & V2 Pipeline Runner

### 7.1 Cross-Module Integration Audit

| Module | Verification Findings | Status |
|:---|:---|:---:|
| [`src/preprocessing/motion.py`](<your-project-folder>/src/preprocessing/motion.py) | `realign_4d_bold()` returns 3-tuple `(realigned_img, motion_params, fd)`. Motion parameters adhere to mm for translations $[0:3]$ and degrees for rotations $[3:6]$. FD uses Power et al. (2012) formulation with standard $50\text{ mm}$ cortical radius. Friston-24 expansion produces exact $(T, 24)$ shape with zero-initial-lag conditions. | **VERIFIED** |
| [`src/preprocessing/segmentation.py`](<your-project-folder>/src/preprocessing/segmentation.py) | Skull-stripping via `median_otsu` isolates brain parenchyma. Smooth polynomial bias field models multiplicative log-intensity non-uniformity without dark-CSF edge distortion. GMM tissue classification orders classes strictly by ascending intensity: CSF (dim) < GM (mid) < WM (bright). Eroded masks apply historical C-PAC thresholds ($P \ge 0.96$) and 1-voxel 3D 6-connectivity morphological erosion. | **VERIFIED** |
| [`src/preprocessing/qc.py`](<your-project-folder>/src/preprocessing/qc.py) | `generate_registration_qc_overlay()` automatically resamples contour foreground into background space via `nilearn.image.resample_to_img` when shapes/affines differ, preventing out-of-bounds index errors. `generate_parcellation_qc_overlay()` uses Laplacian 3D filtering to render crisp parcel boundaries over mean functional slices. | **VERIFIED** |
| [`src/preprocessing/registration.py`](<your-project-folder>/src/preprocessing/registration.py) | Remediated domain/codomain geometry and prealign direction. Resamples 4D series volume-by-volume with float32 preallocation, keeping peak memory below $0.3\text{ GB}$. Nearest-neighbor mask resampling preserves integer labels. Spatial Dice utility validated. | **VERIFIED** |

### 7.2 Forensic Audit of Rejected V1 Runner (`scripts/run_single_scan_poc.py`)

A comparative review of the rejected V1 script confirmed critical architectural flaws:
1. **Direct Normalization without Coregistration**: V1 called `normalize_to_mni(realigned_img, atlas_img)` directly on BOLD, skipping EPI-to-T1w coregistration entirely.
2. **Brain-Mask CompCor vs. Anatomical aCompCor**: V1 computed CompCor from a simple whole-brain intensity threshold mask (`mean_norm > 0.15 * max`), extracting signals from Gray Matter and violating historical C-PAC aCompCor methodology.
3. **No Anatomical Tissue Segmentation**: Structural T1w scans were never skull-stripped, bias-corrected, segmented, or eroded.
4. **Resampling Blur**: V1 applied separate spatial steps rather than composed single-step pullback resampling.

### 7.3 Remediated V2 Pipeline Runner Stub (`scripts/run_poc_v2.py`)

The new runner stub [`scripts/run_poc_v2.py`](<your-project-folder>/scripts/run_poc_v2.py) defines the complete 19-step call graph matching the approved V2 specification:

```
Raw BIDS Functional (BOLD) & Structural (T1w) Scans
  ├── Functional Path:
  │     └── 3D Rigid Motion Realignment (DIPY) -> Motion Params + FD.1D
  │           └── Generate Functional Reference (Temporal Mean BOLD)
  │                 └── Coregister EPI to Skull-Stripped T1w (6 DOF Rigid)
  ├── Structural Path:
  │     └── Anatomical Brain Extraction (median_otsu)
  │           └── 3D Polynomial Bias-Field Correction (log space)
  │                 └── GMM Tissue Classification -> CSF, GM, WM Probabilities
  │                       └── Threshold (P >= 0.96) & 1-Voxel 3D Erosion -> WM/CSF Masks
  │                             └── Nonlinear SyN Normalization (T1w to MNI152)
  ├── Composed Pullback Resampling:
  │     └── Compose: [EPI->T1w] ∘ [T1w->MNI Affine] ∘ [SyN Warp]
  │           ├── Single-step 4D BOLD resampling into exact CC200 grid
  │           └── Nearest-neighbor resampling of eroded WM/CSF masks into CC200 grid
  ├── Nuisance Confound Construction:
  │     ├── 24 Friston Motion Regressors
  │     ├── 5 aCompCor Regressors (from verified standard-space WM/CSF masks)
  │     └── Linear & Quadratic Polynomial Trends
  │           └── Full Rank & SVD Condition Diagnostics (31 columns)
  ├── Residual Cleaning & Temporal Filtering:
  │     └── Simultaneous Nuisance Regression + 0.01-0.10 Hz Bandpass (GSR OFF, standardize=None)
  ├── Parcellation & Feature Extraction:
  │     └── CC200 Regional Signal Extraction (200 ROIs)
  │           └── 19,900 Fisher-z Pairwise Connectivity Features
  └── Quality Control:
        ├── Registration QC Overlays (EPI on T1w, T1w on MNI, BOLD on CC200)
        ├── CC200 Parcellation Alignment Overlays
        └── Structured Provenance Report (JSON)
```

> [!CAUTION]
> **SAFETY GUARD**: `run_poc_v2.py` includes an active top-level `raise NotImplementedError` ensuring it **cannot be executed** until explicit user authorization is provided.

---

## 8. Test Results and Code Coverage

### 8.1 Full Test Suite Execution

```
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.2.0, pluggy-1.7.0 -- <your-project-folder>/.venv/bin/python3.11
rootdir: <your-project-folder>
plugins: cov-7.1.0
collected 47 items

tests/integration/test_pipeline_v2_synthetic.py::TestPipelineV2SyntheticIntegration::test_full_poc_v2_synthetic_execution_and_memory PASSED [  2%]
tests/unit/test_connectivity.py::TestConnectivityAndParcellation::test_boundary_clipping_prevents_infinity PASSED [  4%]
tests/unit/test_connectivity.py::TestConnectivityAndParcellation::test_compute_connectivity_vector_shape_and_finite PASSED [  6%]
tests/unit/test_connectivity.py::TestConnectivityAndParcellation::test_constant_signal_raises_valueerror PASSED [  8%]
tests/unit/test_connectivity.py::TestConnectivityAndParcellation::test_extract_cc200_timeseries_structure PASSED [ 10%]
tests/unit/test_connectivity.py::TestConnectivityAndParcellation::test_feature_names_count_and_lexicographic_order PASSED [ 12%]
tests/unit/test_input_validation.py::TestBIDSInputValidation::test_corrupted_json_raises_valueerror PASSED [ 14%]
tests/unit/test_input_validation.py::TestBIDSInputValidation::test_missing_file_raises_filenotfound PASSED [ 17%]
tests/unit/test_input_validation.py::TestBIDSInputValidation::test_missing_repetition_time_raises_keyerror PASSED [ 19%]
tests/unit/test_input_validation.py::TestBIDSInputValidation::test_non_resting_task_raises_valueerror PASSED [ 21%]
tests/unit/test_input_validation.py::TestBIDSInputValidation::test_out_of_range_repetition_time_raises_valueerror PASSED [ 23%]
tests/unit/test_input_validation.py::TestBIDSInputValidation::test_valid_resting_sidecar_passes PASSED [ 25%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_build_confound_matrix_rank_deficiency_reported PASSED [ 27%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_build_confound_matrix_structure_and_rank PASSED [ 29%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_build_friston_24_regressors PASSED [ 31%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_clean_functional_bold PASSED [ 34%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_clean_functional_bold_bandpass_order_fails PASSED [ 36%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_clean_functional_bold_nonfinite_bold_fails PASSED [ 38%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_clean_functional_bold_nyquist_violation_fails PASSED [ 40%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_clean_functional_bold_rejects_invalid_tr PASSED [ 42%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_compute_framewise_displacement PASSED [ 44%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_extract_acompcor_4d_mask_fails PASSED [ 46%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_extract_acompcor_components PASSED [ 48%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_extract_acompcor_insufficient_voxels_fails PASSED [ 51%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_extract_acompcor_mismatched_csf_mask_shape_fails PASSED [ 53%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_extract_acompcor_mismatched_wm_mask_shape_fails PASSED [ 55%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_extract_acompcor_nonfinite_bold_fails PASSED [ 57%]
tests/unit/test_registration.py::TestRegistration::test_compose_transformations_nonzero_displacement_field PASSED [ 59%]
tests/unit/test_registration.py::TestRegistration::test_compose_transformations_output_has_target_not_source_shape PASSED [ 61%]
tests/unit/test_registration.py::TestRegistration::test_compose_transformations_prealign_matrix_composition_order PASSED [ 63%]
tests/unit/test_registration.py::TestRegistration::test_composed_transformation_and_single_step_resampling PASSED [ 65%]
tests/unit/test_registration.py::TestRegistration::test_compute_spatial_dice PASSED [ 68%]
tests/unit/test_registration.py::TestRegistration::test_generate_functional_reference_rejects_3d PASSED [ 70%]
tests/unit/test_registration.py::TestRegistration::test_generate_functional_reference_valid PASSED [ 72%]
tests/unit/test_registration.py::TestRegistration::test_peak_memory_below_ceiling PASSED [ 74%]
tests/unit/test_registration.py::TestRegistration::test_resample_4d_bold_output_has_target_affine_not_source_affine PASSED [ 76%]
tests/unit/test_registration.py::TestRegistration::test_resample_4d_preserves_temporal_dimension PASSED [ 78%]
tests/unit/test_registration.py::TestRegistration::test_resample_4d_rejects_insufficient_timepoints PASSED [ 80%]
tests/unit/test_registration.py::TestRegistration::test_resample_mask_different_source_and_target_resolution PASSED [ 82%]
tests/unit/test_registration.py::TestRegistration::test_resample_mask_nearest_interpolation PASSED [ 85%]
tests/unit/test_registration.py::TestRegistration::test_rigid_coregistration_direction PASSED [ 87%]
tests/unit/test_segmentation.py::TestSegmentation::test_correct_bias_field_reduces_variance PASSED [ 89%]
tests/unit/test_segmentation.py::TestSegmentation::test_correct_bias_field_rejects_empty_mask PASSED [ 91%]
tests/unit/test_segmentation.py::TestSegmentation::test_create_eroded_tissue_masks_and_validation PASSED [ 93%]
tests/unit/test_segmentation.py::TestSegmentation::test_extract_anatomical_brain PASSED [ 95%]
tests/unit/test_segmentation.py::TestSegmentation::test_segment_t1w_tissue_probabilities PASSED [ 97%]
tests/unit/test_segmentation.py::TestSegmentation::test_validate_tissue_masks_fails_on_empty PASSED [100%]

============================== 47 passed in 6.17s ==============================
```

### 8.2 Codebase Test Coverage Breakdown

```
Name                                Stmts   Miss  Cover
-------------------------------------------------------
src/__init__.py                         1      0   100%
src/features/__init__.py                0      0   100%
src/features/connectivity.py           26      2    92%
src/integration/__init__.py             0      0   100%
src/parcellation/__init__.py            0      0   100%
src/parcellation/extractor.py          24      3    88%
src/preprocessing/__init__.py           6      0   100%
src/preprocessing/motion.py            65      6    91%
src/preprocessing/nuisance.py          88     12    86%
src/preprocessing/qc.py                83      0   100%
src/preprocessing/registration.py     132     29    78%
src/preprocessing/segmentation.py     147     31    79%
src/validation/__init__.py              2      0   100%
src/validation/bids_validator.py       34      2    94%
-------------------------------------------------------
TOTAL                                 608     85    86%
```

---

## 9. Hardware Resource & Memory Measurements

Measured on macOS Darwin ARM64 (Apple Silicon M2, 8 GB RAM):

| Execution Phase | Measured Peak RSS | Permitted Ceiling | Safety Headroom | Verdict |
|:---|:---:|:---:|:---:|:---:|
| **Full Module Imports** | `0.2246 GB` | $2.50\text{ GB}$ | $+2.28\text{ GB}$ | **PASS** |
| **Unit Test Suite (47 tests)** | `0.2612 GB` | $2.50\text{ GB}$ | $+2.24\text{ GB}$ | **PASS** |
| **Synthetic End-to-End Pipeline** | `0.2897 GB` | $2.50\text{ GB}$ | $+2.21\text{ GB}$ | **PASS** |

The volume-by-volume resampling strategy in `resample_4d_bold()` and explicit garbage collection between major registration phases guarantee that physical memory consumption remains well below the strict 2.5 GB ceiling.

---

## 10. Open Historical Differences & Unresolved Blockers

| Topic | Resolution Status | Technical Characterization |
|:---|:---:|:---|
| **DiffeomorphicMap Sampling Direction** | **RESOLVED** | Fixed inverted domain/codomain parameters; corrected prealign multiplication order. |
| **Mask Shape & Finite BOLD Verification** | **RESOLVED** | Added 3D mask assertions, spatial shape equality checks with BOLD, and NaN/Inf rejection. |
| **Nyquist & Bandpass Ordering** | **RESOLVED** | Enforced $f_{\text{high}} < 0.5/\text{TR}$ and $0 < f_{\text{low}} < f_{\text{high}}$. |
| **Rank Deficiency Diagnostics** | **RESOLVED** | SVD condition number calculated and formatted on singular confound matrices. |
| **FSL FAST vs. DIPY/Sklearn GMM** | **DOCUMENTED DEVIATION** | Historical C-PAC 0.3.9.1 used FSL FAST (C++ binary). PoC V2 uses Python-native GMM / HMRF to achieve 100% Apple Silicon ARM64 compatibility without heavy containerized dependencies. |
| **ANTs SyN vs. DIPY SyN** | **DOCUMENTED DEVIATION** | Historical C-PAC used ANTs `antsRegistrationSyN.sh`. PoC V2 uses DIPY Symmetric Diffeomorphic Registration (CCMetric) within the 2.5 GB ceiling. |
| **Real-Scan Pipeline Execution** | **PAUSED — AWAITING APPROVAL** | `scripts/run_poc_v2.py` has an active safety guard. Subject 2344857 will not be processed until the user reviews and approves this remediation package. |

---

## 11. Strict Constraints Compliance Attestation

1. **Subject 2344857 Processing**: Strictly prohibited and **NOT** executed.
2. **Cohort Scans**: No raw functional or structural subject scans were accessed or processed.
3. **Peking_2 Test Partition**: Remained completely untouched and unreferenced.
4. **Original NeuroADHD Project**: The standalone project in `../down/` was not modified in any way.
5. **Rejected V1 Outputs**: All historical V1 outputs in `outputs/sub-2344857/` remain intact and unmodified.
6. **Reference Atlas**: `data/atlases/cc200_roi_atlas.nii.gz` was preserved without alteration.
7. **Downstream ML Inference**: No model inference, training, or re-tuning was initiated.
8. **Historical Preprocessing Parameters**:
   - Bandpass filter: $0.01 - 0.10\text{ Hz}$ at subject TR.
   - Global Signal Regression (GSR): **STRICTLY OFF** (`global0`).
   - Mean tissue signal regression: **STRICTLY OFF** (`wm0.gm0.csf0`).
   - Nilearn amplitude standardization: `standardize=None` strictly preserved.

