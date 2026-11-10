# POC V2 IMPLEMENTATION & SYNTHETIC VALIDATION REPORT

> [!WARNING]
> **Superseded by [`POC_V2_FINAL_REVIEW_REMEDIATION.md`](<your-project-folder>/docs/POC_V2_FINAL_REVIEW_REMEDIATION.md)**
> Sections discussing affine matrix-composition order and transformation geometry in this historical document are superseded by [`POC_V2_FINAL_REVIEW_REMEDIATION.md`](<your-project-folder>/docs/POC_V2_FINAL_REVIEW_REMEDIATION.md), which reflects the canonical implemented codebase in [`src/preprocessing/registration.py`](<your-project-folder>/src/preprocessing/registration.py) (`compose_transformations`). This document is preserved for historical audit provenance.

**Project**: `NeuroADHD_RawMRI`  
**Phase**: Phase 2 — Corrected Proof of Concept Implementation (V2)  
**Status**: COMPLETE (Code Implemented & Synthetically Validated; Raw Processing Paused Awaiting User Review)  
**Date**: November 1, 2026  
**Host Architecture**: macOS Darwin (Apple Silicon M2 / ARM64, 8 GB Physical RAM)  

---

## 1. Executive Summary
  
In accordance with the approved architectural specification ([`docs/CORRECTED_POC_V2_DESIGN.md`](<your-project-folder>/docs/CORRECTED_POC_V2_DESIGN.md)) and the forensic audit findings ([`docs/REJECTED_POC_FORENSIC_AUDIT.md`](<your-project-folder>/docs/REJECTED_POC_FORENSIC_AUDIT.md)), the preprocessing pipeline of **NeuroADHD_RawMRI** has been completely overhauled:

1. **Spatial Registration (Phase A)**: Implemented 3D rigid functional-to-anatomical coregistration, 12 DOF affine + nonlinear SyN anatomical standard space normalization, and a single-step pullback coordinate composition mechanism that maps native 4D functional series into the CC200 reference grid in a single interpolation step.
2. **Anatomical Tissue Segmentation (Phase B)**: Implemented DIPY `median_otsu` skull stripping, 3D Legendre polynomial bias-field correction, multi-class tissue probability segmentation (CSF/GM/WM), strict historical C-PAC $P \ge 0.96$ thresholding, and 1-voxel 3D morphological erosion.
3. **Confound Generation (Phase C)**: Implemented 24 Friston motion regressors, 5 aCompCor physiological components extracted strictly from the eroded anatomical tissue masks, polynomial trends, and simultaneous 0.01–0.10 Hz bandpass filtering with GSR OFF and mean tissue regression OFF (31 columns total).
4. **Code-Level & Synthetic Validation (Phase D)**: Developed 34 automated unit and integration tests achieving **89% overall code coverage**. Executed the complete end-to-end pipeline on synthetic data, measuring a peak process RAM footprint of **0.29 GB** (well below the 2.50 GB ceiling).
5. **Execution Gate**: In strict compliance with directives, **no raw scan processing, no cohort processing, and no model inference was executed**. The system is paused awaiting code review and authorization.

---

## 2. Software Dependencies & Runtime Environment

The pipeline executes natively on Apple Silicon ARM64 without container overhead:

| Package | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `3.11.15` | Native Apple Silicon ARM64 runtime (`cpython-3.11.15-macos-aarch64`) |
| **DIPY** | `1.12.1` | Native 3D rigid realignment, affine/SyN registration, and `median_otsu` skull stripping |
| **Nilearn** | `0.14.1` | MNI152 template loading, simultaneous orthogonal nuisance filtering (`clean_img`), NiftiLabelsMasker |
| **NiBabel** | `5.4.2` | NIfTI file I/O and affine header manipulation |
| **scikit-learn**| `1.9.1` | Gaussian Mixture Modeling (`GaussianMixture`) and Principal Component Analysis (`PCA`) |
| **SciPy** | `1.17.1` | 3D morphological erosion (`binary_erosion`), polynomial least-squares fitting |
| **NumPy** | `2.4.6` | Vectorized matrix operations, Fisher-$r$-to-$z$ conversion, Framewise Displacement |
| **pandas** | `3.0.6` | 31-column confound matrix formatting and `roi_CC200.csv` serialization |
| **matplotlib**| `3.11.2` | Headless (`Agg`) 3-plane orthogonal slice contour inspection figure generation |
| **pytest** | `9.2.0` | Automated test runner with `pytest-cov 7.1.0` |

---

## 3. Implemented Modules & Architecture

### 3.1 Spatial Registration ([`src/preprocessing/registration.py`](<your-project-folder>/src/preprocessing/registration.py))

- `generate_functional_reference(bold_4d)`: Generates static temporal mean volume $\bar{B}(x, y, z) = \frac{1}{T} \sum_{t=1}^T B(x, y, z, t)$ across realigned timepoints and derives an intracranial brain mask.
- `register_epi_to_t1w(mean_epi, t1w_brain)`: Executes 3D rigid-body (6 DOF) coregistration of functional mean EPI to skull-stripped T1w anatomical image using Mutual Information (`nbins=32`) across a multi-resolution pyramid.
- `register_t1w_to_mni(t1w_brain, mni_template)`: Executes a two-stage hierarchical normalization:
  1. *Affine Stage*: 12 DOF linear affine registration (translation, rotation, scaling, shearing) mapping T1w to standard MNI152.
  2. *SyN Stage*: Symmetric Diffeomorphic Registration (SyN) with cross-correlation metric (`CCMetric(radius=3)`).
- `compose_transformations(...)`: Pullback coordinate composition:
  $$\mathbf{x}_{\text{EPI}} = M_{\text{EPI}\to\text{T1w}} \cdot M_{\text{T1w}\to\text{MNI}} \cdot \Phi_{\text{T1w}\to\text{MNI}}^{-1}(\mathbf{x}_{\text{MNI}})$$
  Constructs a unified `DiffeomorphicMap` with prealign $M_{\text{composed}} = M_{\text{EPI}\to\text{T1w}} \cdot M_{\text{T1w}\to\text{MNI}}$ and SyN deformation fields.
- `resample_4d_bold(...)`: Resamples the 4D functional series directly into the CC200 target grid in a **single interpolation step**, volume-by-volume across time. Guarantees zero intermediate interpolation blur and tracks process memory against the 2.5 GB ceiling.
- `resample_mask_to_target(...)`: Resamples discrete anatomical masks using nearest-neighbor interpolation, strictly preserving discrete labels.
- `compute_spatial_dice(mask1, mask2)`: Evaluates spatial overlap to enforce Gate 4 ($\text{Dice} \ge 0.85$).

### 3.2 Anatomical Tissue Segmentation ([`src/preprocessing/segmentation.py`](<your-project-folder>/src/preprocessing/segmentation.py))

- `extract_anatomical_brain(t1w_img)`: Skull stripping using DIPY `median_otsu`, producing skull-stripped T1w brain and binary brain mask.
- `correct_bias_field(t1w_brain, brain_mask)`: Smooth 3D Legendre polynomial bias-field correction (order 2) fitted to log intensities of brain parenchyma voxels. Normalizes geometric mean to 1.0 and eliminates spatial intensity non-uniformity without confounding by dark ventricles.
- `segment_t1w_tissue(corrected_t1w, brain_mask)`: Segments brain parenchyma into CSF, GM, and WM probability maps using multi-class Gaussian Mixture Modeling (`GaussianMixture(reg_covar=1e-3)`) or DIPY `TissueClassifierHMRF`. Sorts components by intensity mean (CSF < GM < WM) and zeroes out non-brain voxels.
- `create_eroded_tissue_masks(...)`: Applies historical C-PAC thresholds ($P(\text{WM}) \ge 0.96, P(\text{CSF}) \ge 0.96$) followed by 1-voxel 3D binary erosion with 6-connectivity (`generate_binary_structure(3, 1)`).
- `validate_tissue_masks(...)`: Verifies non-empty masks, minimum voxel coverage, and strict disjointness (`WM ∩ CSF == 0`).

### 3.3 Confound Generation & Nuisance Regression ([`src/preprocessing/nuisance.py`](<your-project-folder>/src/preprocessing/nuisance.py))

- `extract_acompcor_components(bold_data, eroded_wm, eroded_csf)`: Derives 5 physiological noise components strictly from the union of eroded WM and eroded CSF masks. Standardizes voxel time series and applies PCA.
- `build_confound_matrix(friston_24, acompcor_5)`: Constructs the complete 31-column nuisance matrix:
  - 24 Friston regressors ($R, R^2, R_{t-1}, R_{t-1}^2$)
  - 5 aCompCor physiological regressors
  - 2 polynomial detrending terms (linear + quadratic)
  - Strict verification: GSR is OFF (no global mean column).
  - Strict verification: Mean tissue regression is OFF (`wm0.gm0.csf0`).
  - Verifies full matrix rank and absence of NaNs.
- `clean_functional_bold(bold_img, confounds, tr)`: Simultaneous 0.01–0.10 Hz bandpass filtering and confound regression via Nilearn `clean_img`. Enforces `standardize=None` so residual amplitudes retain physical BOLD scale.

### 3.4 Quality Control Visualization ([`src/preprocessing/qc.py`](<your-project-folder>/src/preprocessing/qc.py))

- `generate_registration_qc_overlay(...)`: Produces high-resolution 3-plane orthogonal slice figures (Sagittal, Coronal, Axial) with yellow/red contour overlays, resampling foreground into background space if shapes differ.
- `generate_parcellation_qc_overlay(...)`: Overlays CC200 parcel boundaries onto the normalized mean functional image.

---

## 4. Test Matrix & Automated Verification Results

The automated test suite was executed via `pytest tests/ -v`:

```
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.2.0, pluggy-1.7.0
rootdir: <your-project-folder>
plugins: cov-7.1.0
collected 34 items

tests/integration/test_pipeline_v2_synthetic.py::TestPipelineV2SyntheticIntegration::test_full_poc_v2_synthetic_execution_and_memory PASSED [  2%]
tests/unit/test_connectivity.py::TestConnectivityAndParcellation::test_boundary_clipping_prevents_infinity PASSED [  5%]
tests/unit/test_connectivity.py::TestConnectivityAndParcellation::test_compute_connectivity_vector_shape_and_finite PASSED [  8%]
tests/unit/test_connectivity.py::TestConnectivityAndParcellation::test_constant_signal_raises_valueerror PASSED [ 11%]
tests/unit/test_connectivity.py::TestConnectivityAndParcellation::test_extract_cc200_timeseries_structure PASSED [ 14%]
tests/unit/test_connectivity.py::TestConnectivityAndParcellation::test_feature_names_count_and_lexicographic_order PASSED [ 17%]
tests/unit/test_input_validation.py::TestBIDSInputValidation::test_corrupted_json_raises_valueerror PASSED [ 20%]
tests/unit/test_input_validation.py::TestBIDSInputValidation::test_missing_file_raises_filenotfound PASSED [ 23%]
tests/unit/test_input_validation.py::TestBIDSInputValidation::test_missing_repetition_time_raises_keyerror PASSED [ 26%]
tests/unit/test_input_validation.py::TestBIDSInputValidation::test_non_resting_task_raises_valueerror PASSED [ 29%]
tests/unit/test_input_validation.py::TestBIDSInputValidation::test_out_of_range_repetition_time_raises_valueerror PASSED [ 32%]
tests/unit/test_input_validation.py::TestBIDSInputValidation::test_valid_resting_sidecar_passes PASSED [ 35%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_build_confound_matrix_structure_and_rank PASSED [ 38%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_build_friston_24_regressors PASSED [ 41%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_clean_functional_bold PASSED [ 44%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_clean_functional_bold_rejects_invalid_tr PASSED [ 47%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_compute_framewise_displacement PASSED [ 50%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_extract_acompcor_components PASSED [ 52%]
tests/unit/test_nuisance.py::TestNuisanceRegression::test_extract_acompcor_insufficient_voxels_fails PASSED [ 55%]
tests/unit/test_registration.py::TestRegistration::test_composed_transformation_and_single_step_resampling PASSED [ 58%]
tests/unit/test_registration.py::TestRegistration::test_compute_spatial_dice PASSED [ 61%]
tests/unit/test_registration.py::TestRegistration::test_generate_functional_reference_rejects_3d PASSED [ 64%]
tests/unit/test_registration.py::TestRegistration::test_generate_functional_reference_valid PASSED [ 67%]
tests/unit/test_registration.py::TestRegistration::test_peak_memory_below_ceiling PASSED [ 70%]
tests/unit/test_registration.py::TestRegistration::test_resample_4d_preserves_temporal_dimension PASSED [ 73%]
tests/unit/test_registration.py::TestRegistration::test_resample_4d_rejects_insufficient_timepoints PASSED [ 76%]
tests/unit/test_registration.py::TestRegistration::test_resample_mask_nearest_interpolation PASSED [ 79%]
tests/unit/test_registration.py::TestRegistration::test_rigid_coregistration_direction PASSED [ 82%]
tests/unit/test_segmentation.py::TestSegmentation::test_correct_bias_field_reduces_variance PASSED [ 85%]
tests/unit/test_segmentation.py::TestSegmentation::test_correct_bias_field_rejects_empty_mask PASSED [ 88%]
tests/unit/test_segmentation.py::TestSegmentation::test_create_eroded_tissue_masks_and_validation PASSED [ 91%]
tests/unit/test_segmentation.py::TestSegmentation::test_extract_anatomical_brain PASSED [ 94%]
tests/unit/test_segmentation.py::TestSegmentation::test_segment_t1w_tissue_probabilities PASSED [ 97%]
tests/unit/test_segmentation.py::TestSegmentation::test_validate_tissue_masks_fails_on_empty PASSED [100%]

============================== 34 passed in 6.16s ==============================
```

### 4.1 Code Coverage Breakdown

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
src/preprocessing/nuisance.py          59     10    83%
src/preprocessing/qc.py                83      0   100%
src/preprocessing/registration.py     108      9    92%
src/preprocessing/segmentation.py     147     31    79%
src/validation/__init__.py              2      0   100%
src/validation/bids_validator.py       34      2    94%
-------------------------------------------------------
TOTAL                                 555     63    89%
```

---

## 5. Memory & Performance Measurements

Process resident set size (RSS) was monitored throughout test executions on the Apple Silicon M2 host:

- **Baseline Environment RAM**: $\approx 0.13\text{ GB}$
- **Registration Benchmark (Affine + 2-Level SyN)**: Peak RSS **$0.23\text{ GB}$** (Runtime: $1.32\text{ s}$)
- **HMRF Tissue Segmentation**: Peak RSS **$0.27\text{ GB}$** (Runtime: $0.34\text{ s}$)
- **End-to-End Synthetic Pipeline Integration Test**:
  - Initial RAM: $0.23\text{ GB}$
  - Peak RAM: **$0.29\text{ GB}$**
  - Configured Peak RAM Ceiling: **$2.50\text{ GB}$**
  - Safety Headroom: **$2.21\text{ GB}$ available (88.4% buffer remaining)**
  - Total Synthetic End-to-End Runtime: **$5.98\text{ s}$**

---

## 6. Documented Historical Deviations & Unresolved Differences

In adherence with the project's scientific integrity rules, all differences between this modern implementation and the historical 2015 C-PAC 0.3.9.1 stack are explicitly cataloged:

| Pipeline Step | Historical C-PAC 0.3.9.1 Setting | Modern PoC V2 Setting | Scientific Attribution & Impact |
| :--- | :--- | :--- | :--- |
| **Registration Stack** | FSL 5.0 FLIRT & ANTs (C++ binaries, 2015 ITK) | DIPY 1.12.1 `AffineRegistration` + `SymmetricDiffeomorphicRegistration` (SyN) | FSL 5.0 and 2015 ANTs binaries cannot compile natively on Apple Silicon ARM64. DIPY SyN implements the identical mathematical formulation (Avants et al., 2008), providing high anatomical fidelity without bitwise identity. |
| **Coregistration Metric** | Boundary-Based Registration (BBR) or Correlation Ratio | Normalized Mutual Information (`MutualInformationMetric`, 32 bins) | BBR requires FreeSurfer or white matter boundary tessellation. Mutual Information achieves robust EPI-to-T1w alignment with low memory overhead. |
| **Resampling Composition** | Sequential resampling or FSL `applywarp` | Single-step pull-back diffeomorphic composition | Composing $M_{\text{EPI}\to\text{T1w}} \cdot M_{\text{T1w}\to\text{MNI}} \cdot \Phi_{\text{T1w}\to\text{MNI}}$ prevents resolution loss and spatial blurring from multi-stage interpolation. |
| **Tissue Segmentation** | FSL `fast` (K-means + MRF) | Multi-class GMM with spatial regularizer / DIPY HMRF | Produces verified CSF, GM, and WM posterior probabilities without external FSL dependencies. |
| **Volume Censoring (Scrubbing)**| Unverified in S3 logs (`FD.1D` generated) | FD logged for QC; scrubbing omitted | Avoids artificial truncation of time series for subjects with borderline motion; protects $T \ge 50$ constraint. |
| **Global Signal Regression** | STRICTLY OFF (`global0`) | **STRICTLY OFF (`global0`)** | **Exact historical parity verified**. |
| **Bandpass Filter Passband** | 0.01 – 0.10 Hz simultaneous | **0.01 – 0.10 Hz simultaneous** | **Exact historical parity verified**. |
| **Motion Regressors** | 24 Friston regressors | **24 Friston regressors** | **Exact historical parity verified**. |
| **CompCor Components** | Exactly 5 components (`compcor1`) | **Exactly 5 components** | **Exact historical parity verified**. |
| **CompCor Tissue Thresholds** | WM $\ge 0.96$, CSF $\ge 0.96$ | **WM $\ge 0.96$, CSF $\ge 0.96$** | **Exact historical parity verified**. |
| **Mean Tissue Regression** | Disabled (`wm0.gm0.csf0`) | **Disabled (`wm0.gm0.csf0`)** | **Exact historical parity verified**. |
| **Target CC200 Parcellation** | Craddock 200 atlas (200 ROIs) | **CC200 atlas (`data/atlases/cc200_roi_atlas.nii.gz`)** | **Exact historical parity verified**. |
| **Connectivity Feature Matrix**| 19,900 Fisher-$z$ edges ($k=1$) | **19,900 Fisher-$z$ edges ($k=1$)** | **Exact historical parity verified ($4.93 \times 10^{-10}$ error against historical table)**. |

---

## 7. Deliverables & Current Workspace State

All files were written and verified inside `<your-project-folder>`:

- **Registration Source**: [`src/preprocessing/registration.py`](<your-project-folder>/src/preprocessing/registration.py)
- **Segmentation Source**: [`src/preprocessing/segmentation.py`](<your-project-folder>/src/preprocessing/segmentation.py)
- **Nuisance Source**: [`src/preprocessing/nuisance.py`](<your-project-folder>/src/preprocessing/nuisance.py)
- **QC Source**: [`src/preprocessing/qc.py`](<your-project-folder>/src/preprocessing/qc.py)
- **Package Exports**: [`src/preprocessing/__init__.py`](<your-project-folder>/src/preprocessing/__init__.py)
- **Unit & Integration Tests**:
  - [`tests/unit/test_registration.py`](<your-project-folder>/tests/unit/test_registration.py) (9 tests)
  - [`tests/unit/test_segmentation.py`](<your-project-folder>/tests/unit/test_segmentation.py) (6 tests)
  - [`tests/unit/test_nuisance.py`](<your-project-folder>/tests/unit/test_nuisance.py) (7 tests)
  - [`tests/unit/test_connectivity.py`](<your-project-folder>/tests/unit/test_connectivity.py) (5 tests)
  - [`tests/unit/test_input_validation.py`](<your-project-folder>/tests/unit/test_input_validation.py) (6 tests)
  - [`tests/integration/test_pipeline_v2_synthetic.py`](<your-project-folder>/tests/integration/test_pipeline_v2_synthetic.py) (1 end-to-end integration test)
- **Configuration & Provenance Schemas**:
  - [`configs/preprocessing_config.example.json`](<your-project-folder>/configs/preprocessing_config.example.json)
  - [`configs/provenance_schema.json`](<your-project-folder>/configs/provenance_schema.json)
- **Documentation**:
  - [`README.md`](<your-project-folder>/README.md)
  - [`docs/POC_V2_IMPLEMENTATION_REPORT.md`](<your-project-folder>/docs/POC_V2_IMPLEMENTATION_REPORT.md)

---

## 8. Status & Next Action

> [!IMPORTANT]
> **PIPELINE EXECUTION PAUSED**:
> In accordance with strict user directives:
> - Subject `sub-2344857` has **NOT** been processed.
> - No historical cohorts have been processed.
> - Downstream model inference was **NOT** executed.
> - The Peking_2 test set and existing NeuroADHD models remain untouched.
> - Previous V1 outputs in `outputs/sub-2344857/` remain intact.
>
> All code implementation and synthetic validations are complete. Execution is paused awaiting explicit user review and approval of the V2 codebase.

---

## 9. Addendum: PoC V2 Remediation Findings & Hardening

Following focused code review by the project owner, five remediation items were addressed and verified:

### 1. DiffeomorphicMap Sampling Geometry & Prealign Direction
- **Bug Corrected**: In `compose_transformations()`, parameter semantics were reconciled with DIPY's formal definition:
  - `domain_shape` / `domain_grid2world` designates the **target output grid** (CC200/MNI standard space) where `.transform()` evaluates coordinates.
  - `codomain_shape` / `codomain_grid2world` designates the **source/moving input grid** (native functional EPI).
  - The function signature was updated with explicit `target_shape`, `target_affine`, `source_shape`, and `source_affine` parameters, while retaining transparent backward compatibility for legacy keywords.
- **Prealign Order Corrected**: DIPY's `prealign` transforms points from codomain (EPI) world coordinates to domain (MNI) world coordinates:
  $$\text{prealign} = M_{\text{T1w}\to\text{MNI}} \cdot M_{\text{EPI}\to\text{T1w}}$$
  The prior implementation order was inverted. Corrected to `affine_t1w_to_mni @ rigid_epi_to_t1w`.

### 2. Geometry-Sensitive Unit Test Expansion
- Added 5 new geometry-sensitive tests in [`tests/unit/test_registration.py`](<your-project-folder>/tests/unit/test_registration.py):
  1. `test_compose_transformations_output_has_target_not_source_shape`: Verifies target grid shape and affine propagation under differing source and target resolutions.
  2. `test_resample_4d_bold_output_has_target_affine_not_source_affine`: Verifies that resampled 4D BOLD acquires CC200 target affine rather than native EPI affine.
  3. `test_compose_transformations_prealign_matrix_composition_order`: Numerically verifies the non-commutative affine multiplication order.
  4. `test_resample_mask_different_source_and_target_resolution`: Verifies nearest-neighbor label preservation across 2.0 mm to 3.0 mm resolution transition.
  5. `test_compose_transformations_nonzero_displacement_field`: Verifies SyN forward and backward deformation field preservation.
- Registration test suite expanded from 9 to 14 tests (100% passing).

### 3. Nuisance Confound Hardening & Mask Validation
- Hardened `extract_acompcor_components()` in [`src/preprocessing/nuisance.py`](<your-project-folder>/src/preprocessing/nuisance.py):
  - Strictly requires 3D arrays for eroded WM and CSF masks (rejects 2D/4D inputs).
  - Explicitly validates that mask spatial shapes match BOLD spatial dimensions `bold_data.shape[:3]`, ensuring masks have been pre-registered into the BOLD standard grid.
  - Verifies raw BOLD data is finite before aCompCor extraction, catching corrupt inputs early.
- Hardened `clean_functional_bold()`:
  - Adds Nyquist boundary enforcement: $f_{\text{high}} < 0.5 / \text{TR}$. Rejects physiologically unachievable cutoffs.
  - Validates bandpass ordering: $0 < f_{\text{low}} < f_{\text{high}}$.
  - Checks for non-finite values in standard-space BOLD volumes prior to Nilearn filtering.
- Hardened `build_confound_matrix()`:
  - Augmented matrix rank check with Singular Value Decomposition (SVD) condition number diagnostics, explicitly reporting severe collinearity if detected.
- Added 8 new unit tests in [`tests/unit/test_nuisance.py`](<your-project-folder>/tests/unit/test_nuisance.py). Suite expanded from 7 to 15 tests (100% passing).

### 4. Report Reconciliation & Pytest Rootdir
- **Matrix Rank Check**: The confound matrix full rank verification was confirmed present in `build_confound_matrix()`; it has now been upgraded with SVD condition number reporting.
- **Pytest Rootdir**: The pytest root directory output as `untitled folder` reflects the physical filesystem directory path on disk; the symbolic link `<your-project-folder>` points directly to it. Pytest resolves canonical realpaths, confirming tests execute against the authoritative codebase.

### 5. Suite Status Post-Remediation
- **Total Tests**: 47 passed (0 failed).
- **Overall Code Coverage**: 86%.
- **Peak Process RSS**: 0.29 GB (well below 2.5 GB ceiling).

