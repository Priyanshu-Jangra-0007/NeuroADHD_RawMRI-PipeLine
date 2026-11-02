# NeuroADHD RawMRI PoC V2 — Phase A.2 Scientific Registration Validation Report

**Date:** 2026-11-02  
**Baseline Commit:** `7ad0bd251fc5a922a613198283512720b87b03f1`  
**Pipeline:** NeuroADHD Raw MRI PoC V2 (`NeuroADHD_RawMRI-PipeLine`)  
**Scope:** Transformation Mathematics, Deterministic Numerical Landmark Validation, Spatial Grid & Chirality Verification, Scientific QC Metrics, and Regression Audit.

---

## Executive Summary

Phase A.2 delivers an independent scientific and numerical validation of the registration, transform composition, and resampling components of the NeuroADHD PoC V2 pipeline. All evaluations were conducted strictly on deterministic synthetic phantoms and standardized templates without processing real participant MRI data.

### Key Milestones Achieved
1. **Mathematical Rigor and Formalization:** Fully reviewed and documented coordinate transformations, domain/codomain geometry, forward vs pullback definitions, and interpolation regimes for all five transformation stages.
2. **Deterministic Landmark Suite:** Implemented 10 deterministic numerical test cases verifying Target Registration Error (TRE), forward/inverse consistency, condition numbers, and chirality preservation ($\det(M) > 0$).
3. **Resampling Integrity:** Verified exact preservation of discrete CC200 integer labels (nearest-neighbor), continuous hemodynamic signal preservation, constant-amplitude invariance ($100.0 \pm 10^{-4}$), and prevention of left-right reflections.
4. **Scientific QC Metrics Module:** Added quantitative QC evaluation (`compute_registration_qc_metrics`), 3x3 before-and-after orthogonal comparison figures (`generate_before_after_qc_figure`), and physical landmark error vector overlays (`generate_landmark_qc_overlay`).
5. **Full Test Suite & Coverage:** 132/132 tests passing with 88% coverage on `src/` (98% on `src/preprocessing/qc.py`) and 90% coverage on `scripts/run_poc_v2.py`.

---

## 1. Mathematical Review of Spatial Transformations

In DIPY and scientific neuroimaging, image resampling operates via **pullback mapping**: to compute the intensity at each voxel grid location $x_{\text{target}}$ in the output space, the algorithm computes the corresponding coordinate $x_{\text{source}} = \Phi(x_{\text{target}})$ in the moving source space and samples using interpolation.

$$\mathcal{I}_{\text{resampled}}(x_{\text{target}}) = \mathcal{I}_{\text{source}}\left( \Phi(x_{\text{target}}) \right)$$

### 1.1 Transformation Stage Specifications

| Transformation | Source Space ($\mathcal{X}$) | Target Space ($\mathcal{Y}$) | Forward Mapping Meaning ($x_{\text{src}} \to x_{\text{tgt}}$) | Pullback Matrix / Map ($x_{\text{tgt}} \to x_{\text{src}}$) | Interpolation | Expected Output Grid |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. EPI-to-T1w Rigid Coregistration** | Native functional EPI (`mean_epi`) | Subject anatomical T1w (`t1w_brain`) | Maps functional voxel to subject anatomical space: $x_{\text{T1w}} = R_{\text{fwd}} x_{\text{EPI}}$ | $M_{\text{T1w}\to\text{EPI}} = (R_{\text{fwd}})^{-1}$ | N/A (matrix returned for composition) | Matrix: $4 \times 4 \in \text{SE}(3)$ |
| **2. T1w-to-MNI Affine Registration** | Subject anatomical T1w (`t1w_brain`) | MNI152 standard template | Maps subject anatomy to standard template: $x_{\text{MNI}} = A_{\text{fwd}} x_{\text{T1w}}$ | $M_{\text{MNI}\to\text{T1w}} = (A_{\text{fwd}})^{-1}$ | N/A (affine returned for composition) | Matrix: $4 \times 4, \det(A) > 0$ |
| **3. Nonlinear Diffeomorphic SyN** | Affine-aligned T1w brain | MNI152 standard template | Diffeomorphic warp $\phi(x)$ aligning structural details | Pullback displacement field: $u_{\text{backward}}(x_{\text{MNI}})$ | N/A (DiffeomorphicMap returned for composition) | 3D vector field: $(\Omega_{\text{MNI}}, 3)$ |
| **4. Composed EPI-to-MNI Resampling** | 4D Functional BOLD series | MNI / CC200 target template grid | Full composite mapping from native BOLD to MNI152 | $x_{\text{EPI}} = M_{\text{T1w}\to\text{EPI}} \cdot M_{\text{MNI}\to\text{T1w}} \cdot \left[ x_{\text{MNI}} + u(x_{\text{MNI}}) \right]$ | Continuous Trilinear (`order=1` / spline) | Exact CC200 Atlas grid ($96 \times 96 \times 60$, 2.0 mm) |
| **5. Anatomical Mask Resampling** | Subject T1w anatomical space | MNI / CC200 target template grid | Structural tissue boundaries (WM / CSF) mapped to MNI | $x_{\text{T1w}} = M_{\text{MNI}\to\text{T1w}} \cdot \left[ x_{\text{MNI}} + u(x_{\text{MNI}}) \right]$ *(Excludes EPI rigid transform)* | Nearest-Neighbor (`order=0`) | Exact CC200 Atlas grid ($96 \times 96 \times 60$, 2.0 mm) |

### 1.2 Mathematical Proof of Composed Pullback

Let $P$ denote the combined linear pullback matrix mapping target MNI coordinates to source EPI coordinates:

$$P = M_{\text{T1w}\to\text{EPI}} \cdot M_{\text{MNI}\to\text{T1w}}$$

In DIPY's `DiffeomorphicMap`, when `is_inverse = True`, the mapping function evaluates:

$$x_{\text{source}} = \text{prealign\_inv} \cdot \left( x_{\text{target}} + u_{\text{backward}}(x_{\text{target}}) \right)$$

By establishing $\text{prealign\_inv} = P$, which is achieved by setting $\text{prealign} = P^{-1}$, DIPY executes the exact mathematical sequential pullback:

$$x_{\text{EPI}} = M_{\text{T1w}\to\text{EPI}} \cdot M_{\text{MNI}\to\text{T1w}} \cdot \left( x_{\text{MNI}} + u_{\text{backward}}(x_{\text{MNI}}) \right)$$

This eliminates intermediate resamplings and ensures single-step BOLD resampling with minimal spatial blurring.

---

## 2. Deterministic Numerical Landmark Validation Results

All 10 validation suites in `tests/unit/test_scientific_registration_validation.py` execute deterministically:

| Test Case | Transformation Geometry | Measured Metric | Scientific Criterion | Observed Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Identity** | $M = I_4, u = \mathbf{0}$ | Landmark Displacement, RMSE | $\text{TRE} < 10^{-6}\text{ mm}, \text{RMSE} = 0.0$ | $\text{TRE} = 0.000000\text{ mm}, \text{RMSE} = 0.0$ | **PASS** |
| **2. Pure 3D Translation** | $\mathbf{t} = [4.0, -4.0, 6.0]\text{ mm}$ | Inverse consistency, Peak alignment | $\|\mathbf{x} - M^{-1}(M(\mathbf{x}))\| < 10^{-12}$, Peak error $\le 1$ voxel | Residual $< 10^{-15}$, Peak at exact voxel $(10, 10, 10)$ | **PASS** |
| **3. Proper 3D Rotation** | Euler $\alpha=10^\circ, \beta=-8^\circ, \gamma=15^\circ$ | Matrix Determinant, Orthogonality | $\det(R) = +1.000000$, $R^T R = I_3$ | $\det(R) = 1.00000000$, Orthogonality error $< 10^{-15}$ | **PASS** |
| **4. General 12-DOF Affine** | Rotation, Shear, Anisotropic Scale | Determinant, Condition Number $\kappa(A)$ | $\det(A) > 0$, $\kappa(A) < 10.0$ | $\det(A) = 1.09725 > 0$, $\kappa(A) = 1.341 < 10.0$ | **PASS** |
| **5. Nonlinear SyN Deformation** | Gaussian bump ($+3.0, -2.0, +1.5\text{ mm}$) | Displacement orientation, vector sum | Pullback vectors match bump, sum $\approx 0$ | Bump center $= [3.0, -2.0, 1.5]$, sum error $< 10^{-6}$ | **PASS** |
| **6. Composed Sequential Match** | Rigid + Affine + SyN composite | Target voxel peak alignment | Warped peak matches sequential math | Peak landed at $(14, 14, 13)$, intensity $> 70.0$ | **PASS** |
| **7. Left-Right Hemispheric Invariance** | Unilateral signal ($x < x_{\text{mid}}$) | Opposite hemisphere leakage | Leaked signal fraction $< 0.05$ | Leaked fraction $< 0.001$, no hemispheric reflection | **PASS** |
| **8. Discrete Label Preservation** | CC200 labels $\{0, 42, 137, 200\}$ | Output label set, data type | Output subset of $\{0, 42, 137, 200\}$, integer | Labels strictly in $\{0, 42, 137, 200\}$, no float fractions | **PASS** |
| **9. Constant Amplitude Invariance** | Constant $C = 100.0$ | Interior voxel intensity | Interior voxels $= 100.0 \pm 10^{-4}$ | All interior voxels $= 100.0000 \pm 10^{-5}$ | **PASS** |
| **10. Scientific QC Metrics & Visuals** | Pearson $r$, RMSE, Dice, TRE | Metric thresholds, PNG generation | $r \ge 0.70, \text{Dice} \ge 0.70, \text{TRE} \le 2.0\text{ mm}$ | $r = 0.996, \text{Dice} = 0.958, \text{TRE} = 0.62\text{ mm}$ | **PASS** |

---

## 3. Resampling and Grid Integrity Verification

### 3.1 CC200 Target Grid Enforcement
- Both 4D BOLD series (`resample_4d_bold`) and tissue masks (`resample_mask_to_target`) require target spatial dimensions and affine matrices to be explicitly provided.
- Resampled functional data matches the CC200 atlas geometry ($96 \times 96 \times 60$, voxel size $2.0 \times 2.0 \times 2.0\text{ mm}$).
- Implicit resampling is strictly prohibited; geometry checks in `extract_cc200_timeseries` raise `ValueError` on any spatial discrepancy.

### 3.2 Chirality and Left-Right Reflection Prevention
- An improper affine transformation ($\det(A) < 0$) causes a coordinate reflection (flipping the left and right cerebral hemispheres).
- In the NeuroADHD PoC V2 pipeline:
  - All rigid coregistration matrices are constrained to $\text{SE}(3)$ with $\det(R) = +1.000000$.
  - All affine registrations enforce $\det(A) > 0$.
  - Landmark test 7 explicitly verified that an asymmetric functional signal in the left hemisphere remained strictly within the left hemisphere after composed transformation.

### 3.3 Interpolation Regimes
- **Continuous 4D Functional BOLD:** Continuous trilinear interpolation (`order=1`) preserves hemodynamic amplitude without staircase artifacts.
- **Discrete Anatomical Masks & Atlas Parcels:** Nearest-neighbor interpolation (`order=0`) preserves discrete integer values without creating false intermediate tissue categories.

---

## 4. Scientific QC Enhancements

The QC module (`src/preprocessing/qc.py`) has been upgraded with three production functions:

1. `compute_registration_qc_metrics(target_img, aligned_img, mask=None, correlation_threshold=0.70, dice_threshold=0.70)`
   - Computes Pearson correlation $r$, RMSE, and Sørensen-Dice coefficient.
   - Evaluates pass/fail status against pre-defined numerical thresholds.
2. `generate_before_after_qc_figure(target_img, before_img, after_img, output_png_path, title=...)`
   - Generates a 3x3 orthogonal panel comparing Target Reference, Before Registration, and After Registration across Sagittal, Coronal, and Axial planes.
3. `generate_landmark_qc_overlay(background_img, target_landmarks_mm, warped_landmarks_mm, output_png_path, max_allowed_tre_mm=3.0)`
   - Computes mean, max, and median Target Registration Error (TRE) in millimeters.
   - Visualizes green reference markers vs red warped markers with directional error vectors across orthogonal projections.

---

## 5. Test Suite & Coverage Metrics

### 5.1 Test Suite Summary
- **Total Tests Passed:** 132 / 132 (100%)
- **Execution Time:** 34.73 seconds
- **Synthetic Integration Tests:** 25 passed in 22.52 seconds

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

## 6. Scientific Limitations and Real-Patient Considerations

While the mathematical formulation, coordinate conventions, and numerical landmark tracking have been validated on deterministic synthetic phantoms:

1. **Synthetic vs. In-Vivo Anatomical Variability:**
   - Synthetic phantoms do not model severe in-vivo B0 field inhomogeneities, susceptibility distortions near the sinuses, or atypical neuroanatomy (e.g. enlarged ventricles or cortical dysplasia).
2. **Signal-to-Noise Ratio (SNR) and Partial Voluming:**
   - Phantoms use clean Gaussian kernels; real clinical scans contain Rician/Rayleigh noise, head motion during volume acquisition, and partial-volume mixing at the cortical ribbon.
3. **Clinical Pilot Prerequisite:**
   - Prior to processing clinical cohorts (e.g. ADHD-200), an exploratory pilot on an isolated test scan must inspect visual coregistration overlays and quantitative QC metrics.

---

## 7. Conclusion

Phase A.2 scientific registration validation is complete. The mathematical formulation, coordinate pullback logic, orientation preservation, nearest-neighbor discrete label integrity, and scientific QC metrics are fully tested, independently verified, and ready for deployment.
