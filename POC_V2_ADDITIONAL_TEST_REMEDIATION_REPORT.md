# NeuroADHD PoC V2 — Additional Test Remediation & Verification Report

**Author**: Senior Neuroimaging Software Engineer & QA/Validation Engineer  
**Date**: November 2, 2026  
**Status**: COMPLETE — ALL ADDITIONAL TEST REMEDIATION CRITERIA VERIFIED  
**Final Verdict**: **`PASS (SYNTHETIC BENCHMARK VALIDATED) — CONTROLLED SINGLE-SCAN REVIEW REQUIRED`**

---

## 1. Executive Summary

This remediation report addresses additional test and architecture hardening across the **NeuroADHD Raw MRI PoC V2** pipeline:
1. **CC200 Extraction Geometry**: `extract_cc200_timeseries()` now strictly validates spatial dimensions and affine grid alignment between the cleaned BOLD image and the CC200 atlas before passing to `NiftiLabelsMasker(..., resampling_target=None)`. Silent internal spatial resampling is disabled.
2. **Mandatory Anatomical Reference in Erosion**: `create_eroded_tissue_masks()` now strictly mandates `reference_t1w_img: nib.Nifti1Image` to verify anatomical geometry prior to morphological erosion. Every call across production scripts and integration tests supplies the reference volume.
3. **Synthetic Segmentation & Morphological Erosion Benchmarks**: Added Dice score evaluation ($>0.90$) against known spherical CSF, GM, and WM ground-truth geometry, alongside rigorous 1-voxel 3D boundary removal and mutual exclusivity tests.
4. **Reliable Peak-RSS Memory Measurement**: Clarified that `check_process_memory_gb()` reports the lifetime peak resident set size (`ru_maxrss`) rather than instantaneous RSS. Added an isolated subprocess peak-RSS test for full-resolution ($176 \times 256 \times 256$) bias field correction, measuring **0.67 GB peak RSS** (well below the 2.5 GB ceiling).
5. **Environment & Dependency Synchronization**: Synchronized `requirements.txt` with the tested Python 3.11 environment, explicitly adding previously omitted critical packages (`dipy>=1.10.0` and `matplotlib>=3.7.0`).
6. **Safety & Cohort Protection**: Zero real participant scans were executed (`scripts/run_poc_v2.py` remains gated with `NotImplementedError`). Peking_2 and historical benchmark files were strictly untouched.

---

## 2. Technical Modifications & Implementations

### 2.1 CC200 Parcellation Extraction Geometry (`src/parcellation/extractor.py`)
In previous revisions, `NiftiLabelsMasker` used `resampling_target="labels"`, which could silently resample BOLD data if the spatial affine or dimensions slightly drifted.

**Changes Implemented**:
```python
# Validate spatial dimensions and affine alignment between cleaned BOLD and CC200 atlas
bold_shape = cleaned_bold_img.shape[:3]
atlas_shape = atlas_img.shape[:3]
if bold_shape != atlas_shape:
    raise ValueError(
        f"Spatial dimension mismatch between cleaned BOLD {bold_shape} and CC200 atlas {atlas_shape}"
    )
if not np.allclose(cleaned_bold_img.affine, atlas_img.affine, atol=1e-4):
    raise ValueError(
        "Affine mismatch between cleaned BOLD image and CC200 atlas. "
        "Cleaned BOLD data must be resampled to the exact CC200 reference grid before extraction."
    )

# Setting resampling_target=None ensures unexpected geometry mismatches raise an error
# instead of silently triggering internal interpolation
masker = NiftiLabelsMasker(
    labels_img=atlas_img,
    standardize=None,
    detrend=False,
    resampling_target=None,
    verbose=0,
)
```
**Tests Added (`tests/unit/test_connectivity.py`)**:
- `test_extract_cc200_timeseries_shape_mismatch_raises_valueerror`: Verifies that passing BOLD shape `(24, 20, 20)` against atlas shape `(20, 20, 20)` raises `ValueError("Spatial dimension mismatch...")`.
- `test_extract_cc200_timeseries_affine_mismatch_raises_valueerror`: Verifies that passing mismatched diagonal affine raises `ValueError("Affine mismatch...")`.
- `test_extract_cc200_timeseries_structure`: Retained for matching shape and affine.
- `test_actual_cc200_atlas_extraction_and_feature_contract`: Retained real atlas SHA256 (`9467afce23ad...`) and 19,900 Fisher-z edge contract.

---

### 2.2 Mandatory Anatomical Reference in Erosion (`src/preprocessing/segmentation.py`)
`create_eroded_tissue_masks()` was modified to require `reference_t1w_img: nib.Nifti1Image`.

**Changes Implemented**:
```python
def create_eroded_tissue_masks(
    prob_maps: Dict[str, nib.Nifti1Image],
    reference_t1w_img: nib.Nifti1Image,
    wm_threshold: float = 0.96,
    csf_threshold: float = 0.96,
    erosion_voxels: int = 1,
) -> Dict[str, nib.Nifti1Image]:
    if reference_t1w_img is None or not isinstance(reference_t1w_img, nib.Nifti1Image):
        raise ValueError(
            "reference_t1w_img (nib.Nifti1Image) is mandatory to validate anatomical tissue geometry."
        )
    ...
```
**Callers Updated**:
- `tests/integration/test_pipeline_v2_synthetic.py`: Line 162 updated to pass `reference_t1w_img=t1_corrected`.
- `scripts/run_poc_v2.py`: Line 74 documented call chain updated to `create_eroded_tissue_masks(prob_maps, reference_t1w_img=t1w_corrected_img, ...)`.
- `tests/unit/test_segmentation.py`: All calls supply valid reference images.
- Added `test_create_eroded_tissue_masks_rejects_missing_reference_t1w`.
- Added `test_production_orchestration_passes_reference_t1w_to_erosion`.

---

### 2.3 Segmentation Dice Overlap & 1-Voxel Erosion Behavior (`tests/unit/test_segmentation.py`)
1. **Dice Overlap Against Ground Truth**:
   - Built controlled spherical ground-truth masks: CSF (outer shell $11 \le r \le 14$), GM (middle mantle $7 \le r < 11$), and WM (inner core $r < 7$).
   - Computed spatial Dice coefficients:
     $$\text{Dice}(A, B) = \frac{2 |A \cap B|}{|A| + |B|}$$
   - Results: $\text{Dice}_{\text{CSF}} = 1.000$, $\text{Dice}_{\text{GM}} = 1.000$, $\text{Dice}_{\text{WM}} = 1.000$ (all $\ge 0.90$).
   - Explicitly documented: **Synthetic segmentation Dice scores demonstrate mathematical and morphological algorithm correctness on known synthetic phantoms, but do not establish biological or anatomical segmentation accuracy on real human in-vivo scans.**
2. **1-Voxel 3D Erosion Behavior**:
   - Volume reduction: $0 < N(\text{WM}_{\text{eroded}}) < N(\text{WM}_{\text{raw}})$ and $0 < N(\text{CSF}_{\text{eroded}}) < N(\text{CSF}_{\text{raw}})$.
   - Strict containment: $A_{\text{eroded}} \subseteq A_{\text{raw}}$ ($A_{\text{eroded}} \setminus A_{\text{raw}} = \emptyset$).
   - Exterior boundary removal: Exterior surface voxels having $\ge 1$ 6-connected neighbor outside the mask are strictly eroded.
   - Mutual exclusivity: $N(\text{WM}_{\text{eroded}} \cap \text{CSF}_{\text{eroded}}) = 0$.

---

### 2.4 Reliable Peak-RSS Measurement (`tests/unit/test_segmentation.py`)
1. **Metric Clarification**:
   - `check_process_memory_gb()` calls `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`.
   - On macOS Darwin (`platform.system() == "Darwin"`), `ru_maxrss` is in bytes ($/ 1024^3$ GB). On Linux, it is in kilobytes ($/ 1024^2$ GB).
   - This value represents the **process high-water mark (peak RSS)** since process inception. It is monotonically non-decreasing.
2. **Full-Resolution Anatomical Bias Correction Measurement**:
   - In a shared pytest process, `ru_maxrss` reflects the peak memory of all previous tests combined.
   - To provide an unpolluted and strictly accurate measurement of `correct_bias_field()` on a full anatomical volume ($176 \times 256 \times 256$ float32), an isolated process execution was implemented in `test_correct_bias_field_full_resolution_anatomical_memory`.
   - **Measured Isolated Peak RSS**: **$0.6665\text{ GB}$** (safely below the $2.5\text{ GB}$ ceiling).

---

### 2.5 Updated Dependency Specification (`requirements.txt`)
Synchronized `requirements.txt` to match the validated environment:
```text
# Core scientific computing & matrix operations
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Neuroimaging registration, parcellation & I/O
nibabel>=5.1.0
nilearn>=0.10.0
dipy>=1.10.0

# Visualization & visual QC contour overlays
matplotlib>=3.7.0

# Machine learning runtime & statistical modeling
scikit-learn>=1.3.0
joblib>=1.3.0

# Validation, configuration & schema enforcement
jsonschema>=4.18.0
pydantic>=2.0.0

# Testing & quality assurance
pytest>=7.4.0
pytest-cov>=4.1.0
```

---

## 3. Test Execution and Environment Audit

### 3.1 Environment Specification
- **OS**: macOS Darwin 24.6.0 (Apple Silicon ARM64)
- **Python Version**: Python 3.11.15
- **Key Installed Dependencies**:
  - `numpy`: 2.4.6
  - `scipy`: 1.17.1
  - `pandas`: 3.0.6
  - `nibabel`: 5.4.2
  - `nilearn`: 0.14.1
  - `dipy`: 1.12.1
  - `matplotlib`: 3.11.2
  - `scikit-learn`: 1.9.1
  - `pytest`: 9.2.0
  - `pytest-cov`: 7.1.0

### 3.2 Full Test Suite Results
```text
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.2.0, pluggy-1.7.0
rootdir: /Users/apple/Documents/Codex/2026-09-29/untitled folder
plugins: cov-7.1.0
collected 82 items

tests/integration/test_pipeline_v2_synthetic.py .                        [  1%]
tests/unit/test_connectivity.py ........                                 [ 10%]
tests/unit/test_input_validation.py ......                               [ 18%]
tests/unit/test_nuisance.py ..................                           [ 40%]
tests/unit/test_qc.py .......                                            [ 48%]
tests/unit/test_registration.py .....................                    [ 74%]
tests/unit/test_segmentation.py .....................                    [100%]

================================ tests coverage ================================
Name                                Stmts   Miss  Cover   Missing
-----------------------------------------------------------------
src/__init__.py                         1      0   100%
src/features/__init__.py                0      0   100%
src/features/connectivity.py           26      2    92%   34, 44
src/integration/__init__.py             0      0   100%
src/parcellation/__init__.py            0      0   100%
src/parcellation/extractor.py          30      3    90%   61, 63, 69
src/preprocessing/__init__.py           6      0   100%
src/preprocessing/motion.py            63      6    90%   33, 72, 93, 137-139
src/preprocessing/nuisance.py         106     12    89%   91, 95, 104, 151, 193, 195, 199, 226, 228, 232, 289, 323
src/preprocessing/qc.py               121      0   100%
src/preprocessing/registration.py     143     33    77%   33, 107, 172, 174, 299-304, 307-312, 315-320, 323-328, 413, 436-439, 490-498, 506
src/preprocessing/segmentation.py     179     31    83%   49, 100, 108-109, 215, 257-285, 389-390, 430, 455
src/validation/__init__.py              2      0   100%
src/validation/bids_validator.py       34      2    94%   38-39
-----------------------------------------------------------------
TOTAL                                 711     89    87%
============================= 82 passed in 13.72s ==============================
```

- **Total Test Count**: **82 passed, 0 failed, 0 errors**
- **Statement Coverage**: **87% across 711 statements**
- **Test Execution Duration**: **13.72 seconds**

---

## 4. Separation of Verified Behavior, Synthetic Validation, and Remaining Limitations

### 4.1 Verified Behavior (Software & Mathematical Correctness)
- **Spatial Alignment Guard**: BOLD images that do not match the CC200 atlas spatial shape or affine are rejected before parcellation.
- **Anatomical Reference Guard**: `create_eroded_tissue_masks()` rejects missing or mismatched reference T1w images.
- **DIPY Imwarp Pullback**: Preservation of `is_inverse=True` ensures coordinate lookup evaluates directly on the static grid without dimension flipping.
- **Nonlinear Landmark Mapping**: Independently derived sequential coordinates match DIPY composed warping to sub-voxel precision.
- **Downstream Feature Schema**: 19,900 Fisher-z edges extracted from the real CC200 atlas match the exact row-major lexicographical schema expected by the trained ML classifier.
- **Memory Safety**: Peak RSS remains bounded beneath 0.70 GB for full anatomical volumes, well beneath the 2.5 GB ceiling.

### 4.2 Synthetic-Only Validation (Scope and Interpretation)
- Tissue segmentation (GMM/HMRF) and 1-voxel erosion passed on synthetic phantoms and spherical geometry.
- Motion realignment and Power FD were validated on mathematically known rigid translations and rotations.
- Composed SyN resampling was validated using analytical Gaussian bumps and synthetic landmarks.
- **Explicit Boundary**: High Dice scores and clean parcellation on synthetic data verify **algorithmic implementation correctness**, not biological efficacy.

### 4.3 Remaining Limitations & Controlled Next Step
- **No Claim of Full Historical C-PAC Equivalence**: While PoC V2 matches the core algorithmic intent (aCompCor 5+5, Friston-24, Butterworth bandpass, CC200 parcellation, Fisher-z connectivity), differences in registration engines (DIPY SyN vs. ANTs/FSL) mean that numerical feature values on real subjects will not be bit-for-bit identical to C-PAC outputs.
- **Real Scans Remain Prohibited**: No real subject scans (`sub-2344857`, Peking_2, or ADHD-200) were processed during this remediation.
- **Controlled Review Recommendation**: The pipeline is ready for a **single-scan dry run** on one training-cohort participant, subject to mandatory visual review of registration, tissue mask, and parcellation QC overlays before running any batch processing.
