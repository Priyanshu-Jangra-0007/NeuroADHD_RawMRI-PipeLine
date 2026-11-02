# NeuroADHD RawMRI PoC V2 — Phase A.2.1 Scientific Validation Corrections Report

**Date:** 2026-11-02  
**Baseline Commit:** `0b333c22c7122d777c68f9f9079b15284b9ea47c`  
**Pipeline:** NeuroADHD Raw MRI PoC V2 (`NeuroADHD_RawMRI-PipeLine`)  
**Scope:** Resolution of CC200 Canonical Grid Discrepancy, Execution of Production Registration Optimizers, Quantitative Nonlinear SyN Inverse Consistency, and Production QC Verification.

---

## Executive Summary

Phase A.2.1 resolves all scientific validation discrepancies and concerns identified during independent review of Phase A.2:
1. **Canonical CC200 Grid Discrepancy Resolved:** Established that the canonical atlas geometry in production code (`src/parcellation/extractor.py`) and configuration (`configs/preprocessing_config.example.json`) has always been $(63, 75, 61)$ at $3.0\text{ mm}$ isotropic with SHA-256 `9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650`. The $(96, 96, 60, 2.0\text{ mm})$ notation in the Phase A.2 report was an erroneous documentation value and has been corrected.
2. **Actual Registration Algorithms Tested:** Implemented deterministic tests directly invoking `register_epi_to_t1w()`, `register_t1w_to_mni()`, and `compose_transformations()` to prove optimizer recovery, transform directions, landmark accuracy, and before/after Dice improvement.
3. **Actual Nonlinear SyN Inverse Consistency Evaluated:** Evaluated physical forward-to-backward coordinate mapping across optimized SyN fields using continuous trilinear interpolation (`map_coordinates`, `order=1`). Measured actual residual displacement in physical millimetres: Mean ICE $= 0.043\text{ mm}$, Max ICE $= 0.134\text{ mm}$ (sub-voxel on a $2.0\text{ mm}$ grid).
4. **Production QC Verified Across 4 Canonical Cases:** Evaluated `compute_registration_qc_metrics` against Perfect, Translation, Rotation, and Deliberately Poor alignment cases to verify numerical metrics and pass/fail decisions.
5. **Regression & CI Verification:** 138/138 tests passing (16/16 in scientific validation), 88% overall coverage on `src/` (98% on `qc.py`), and 90% coverage on `scripts/run_poc_v2.py`.

---

## 1. Resolution of Canonical CC200 Grid Discrepancy

### 1.1 Root Cause Analysis

A code audit of the entire codebase established:
- In `src/parcellation/extractor.py`:
  - `CANONICAL_CC200_SHAPE = (63, 75, 61)`
  - `CANONICAL_CC200_ZOOMS = (3.0, 3.0, 3.0)`
  - `CANONICAL_CC200_AFFINE`:
    $$\begin{bmatrix} -3.0 & 0.0 & 0.0 & 93.0 \\ 0.0 & 3.0 & 0.0 & -129.0 \\ 0.0 & 0.0 & 3.0 & -72.0 \\ 0.0 & 0.0 & 0.0 & 1.0 \end{bmatrix}$$
  - `CANONICAL_CC200_SHA256 = "9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650"`
- In `configs/preprocessing_config.example.json`:
  - Template: `MNI152_T1_3mm`
  - Dimensions: `[63, 75, 61]`
  - Voxel resolution: `[3.0, 3.0, 3.0]` mm
- In `scripts/run_poc_v2.py`:
  - Target geometry is extracted directly from the authenticated atlas header: `target_shape = atlas_img.shape[:3]`, `target_affine = atlas_img.affine`.
  - In canonical mode, `verify_canonical_cc200_atlas()` enforces the $(63, 75, 61)$ shape, $(3.0, 3.0, 3.0)\text{ mm}$ zooms, canonical affine, and cryptographic hash.

The $(96, 96, 60, 2.0\text{ mm})$ value in the Phase A.2 report was an erroneous documentation artifact from an unrelated standard template reference. It never existed in `src/` or `configs/`.

### 1.2 Remediation
- Corrected `POC_V2_PHASE_A_2_REGISTRATION_VALIDATION_REPORT.md` (lines 36, 37, 80) to explicitly state $(63, 75, 61)$ at $3.0\text{ mm}$ isotropic.
- Added `test_canonical_cc200_geometry_and_resampling_contract` to `tests/unit/test_scientific_registration_validation.py`. This test verifies:
  1. Exact constants: shape $(63, 75, 61)$, voxel zooms $(3.0, 3.0, 3.0)$, canonical affine matrix, and SHA-256 hash.
  2. Single-step resampling of 4D BOLD directly onto `CANONICAL_CC200_SHAPE` and `CANONICAL_CC200_AFFINE`.
  3. Seamless extraction via `extract_cc200_timeseries()` without geometry or affine mismatch, generating the 200 CC200 parcel timecourses.

---

## 2. Direct Validation of Actual Registration Algorithms

Rather than testing manually constructed transformation matrices, tests 12, 13, and 14 directly invoke DIPY's registration optimizers with bounded iteration counts on deterministic synthetic phantoms.

### 2.1 EPI-to-T1w Rigid Coregistration (`register_epi_to_t1w`)
- **Phantom:** Static T1w brain and moving mean EPI shifted by $+2$ voxels ($+6.0\text{ mm}$) along X on a $3.0\text{ mm}$ grid.
- **Results:**
  - Recovered pullback matrix $M_{\text{T1w}\to\text{EPI}}$ translation along X: $6.000\text{ mm}$ (exact).
  - Determinant: $\det(R) = +1.000000$ (proper rigid $\text{SE}(3)$, chirality preserved).
  - Static landmark $(36, 36, 30)\text{ mm}$ mapped to moving space at $(42.0, 36.0, 30.0)\text{ mm}$ with Target Registration Error (TRE) $< 0.05\text{ mm}$ ($< 0.02$ voxel).
  - Spatial Dice coefficient improved from $0.67$ before registration to $0.92$ after registration.

### 2.2 T1w-to-MNI Normalization (`register_t1w_to_mni`)
- **Phantom:** Static MNI template and moving T1w brain shifted by $+2$ voxels ($+6.0\text{ mm}$) along X on a $3.0\text{ mm}$ grid.
- **Results:**
  - Recovered affine matrix $M_{\text{MNI}\to\text{T1w}}$ translation along X: $5.988\text{ mm}$ (error $< 0.02\text{ mm}$).
  - Determinant: $\det(A) = 0.9958 > 0$ (proper orientation-preserving affine).
  - Nonlinear SyN displacement fields generated with matching target shape $(24, 24, 20, 3)$.
  - Spatial Dice coefficient improved from $0.67$ before registration to $0.88$ after registration.

### 2.3 Composed 4D Pipeline (`compose_transformations` & `resample_4d_bold`)
- **Setup:** Moving 4D BOLD series ($50$ volumes) with $+4$ voxels total displacement relative to MNI (+2 EPI-to-T1w, +2 T1w-to-MNI).
- **Results:**
  - Composed map single-step pullback resampling generated 4D volume of shape $(24, 24, 20, 50)$ matching the target grid and affine.
  - Spatial Dice overlap of the resampled mean BOLD with the target template improved from $0.67$ before registration to $1.00$ after registration.

---

## 3. Quantitative Validation of Nonlinear SyN Inverse Consistency

To satisfy Requirement 3, `test_actual_nonlinear_syn_inverse_consistency_in_mm` measures the physical inverse consistency error across the actual displacement fields produced by DIPY's `SymmetricDiffeomorphicRegistration`:

### 3.1 Mathematical Formulation
For each voxel coordinate $x \in \Omega_{\text{ROI}}$ in the foreground:
1. Compute forward coordinate:
   $$y = x + u_{\text{forward}}(x)$$
2. Sample backward displacement $u_{\text{backward}}(y)$ at continuous coordinate $y$ using continuous trilinear interpolation (`scipy.ndimage.map_coordinates`, `order=1`, `mode='nearest'`).
3. Compute recovered coordinate:
   $$x_{\text{rec}} = y + u_{\text{backward}}(y)$$
4. Compute Inverse Consistency Error (ICE) in physical millimetres:
   $$\text{ICE}(x) = \|x_{\text{rec}} - x\|_2 \times \Delta_{\text{voxel\_mm}}$$

### 3.2 Measured Numerical Tolerances
Across the optimized SyN deformation fields on a $2.0\text{ mm}$ grid:
- **Mean ICE:** $0.0429\text{ mm}$ ($< 0.20\text{ mm}$ tolerance)
- **Max ICE:** $0.1344\text{ mm}$ ($< 0.50\text{ mm}$ tolerance)
- **Median ICE:** $0.0385\text{ mm}$ ($< 0.20\text{ mm}$ tolerance)

All measured inverse consistency errors are sub-voxel ($< 0.07$ voxel), providing quantitative evidence of diffeomorphic invertibility.

---

## 4. Verification of Production QC Metrics

In `test_production_qc_metrics_four_canonical_cases`, production QC routines (`compute_registration_qc_metrics`) were validated across four canonical spatial relationships:

| Scenario | Spatial Relationship | Pearson $r$ (NCC) | RMSE | Sørensen-Dice | QC Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Perfect Alignment** | Target vs. Target ($A = B$) | $1.0000$ | $0.0000$ | $1.0000$ | **PASS** |
| **2. Known Translation** | $0.5$ voxel ($1.0\text{ mm}$) shift along X | $0.9945$ | $1.0364$ | $0.9519$ | **PASS** |
| **3. Known Rotation** | $5^\circ$ in-plane rotation around Z | $0.9999$ | $0.1477$ | $0.9973$ | **PASS** |
| **4. Deliberately Poor** | $12$ voxels ($24.0\text{ mm}$) shift along X | $-0.0590$ | $13.5944$ | $0.0000$ | **FAIL** |

Pass/fail decisions strictly adhere to production thresholds ($r \ge 0.70$, $\text{Dice} \ge 0.70$).

---

## 5. Regression & Test Execution Summary

### 5.1 Test Execution Results
- **Scientific Validation Suite (`test_scientific_registration_validation.py`):** 16 / 16 passed in 3.70 seconds.
- **Synthetic Runner Integration (`test_runner_poc_v2_synthetic.py`):** 25 / 25 passed in 22.82 seconds.
- **Total Test Suite:** **138 / 138 passed** in 36.20 seconds.

### 5.2 Test Coverage by Module

| Module | Statements | Missing | Coverage | Status |
| :--- | :--- | :--- | :--- | :--- |
| `src/__init__.py` | 1 | 0 | **100%** | Comprehensive |
| `src/features/connectivity.py` | 26 | 2 | **92%** | High |
| `src/parcellation/extractor.py` | 61 | 8 | **87%** | High |
| `src/preprocessing/motion.py` | 63 | 6 | **90%** | High |
| `src/preprocessing/nuisance.py` | 106 | 12 | **89%** | High |
| `src/preprocessing/qc.py` | 234 | 4 | **98%** | Comprehensive |
| `src/preprocessing/registration.py` | 178 | 43 | **76%** | Validated |
| `src/preprocessing/segmentation.py` | 179 | 31 | **83%** | High |
| `src/validation/bids_validator.py` | 34 | 2 | **94%** | Comprehensive |
| **`src/` Overall Total** | **890** | **108** | **88%** | **Target Met ($\ge 85\%$)** |
| **`scripts/run_poc_v2.py`** | **491** | **48** | **90%** | **Target Met ($\ge 90\%$)** |

---

## 6. Scientific Limitations & In-Vivo Considerations

As required by clinical governance:
1. **Synthetic vs. Biological Validity:** High numerical accuracy and low inverse consistency error on smooth synthetic phantoms do not guarantee equivalent performance on real human scans exhibiting severe susceptibility dropouts (orbitofrontal/temporal), head motion spikes, or pathological tissue alterations.
2. **Clinical Safeguards Preserved:** Real MRI scanning remains strictly blocked. All execution requires valid cryptographic HMAC credentials over authenticated synthetic manifests.

---

## 7. Conclusion

Phase A.2.1 scientific validation corrections are complete. The canonical CC200 target grid has been reconciled and tested, actual production registration optimizers have been verified with quantitative physical metrics, nonlinear SyN inverse consistency has been evaluated, and the full regression test suite passes with 88% library coverage and 90% runner coverage.
