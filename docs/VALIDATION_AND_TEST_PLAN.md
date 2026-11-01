# VALIDATION AND TEST PLAN

## 1. Overview & Testing Philosophy

To prevent silent failures, data leakage, and invalid predictions, every layer of the `NeuroADHD_RawMRI` pipeline is accompanied by automated unit tests, integration tests, and manual quality control (QC) checkpoints.

In accordance with strict project rules:
- **No test is claimed to have passed unless it was actually executed.**
- Tests planned for future phases are explicitly marked **`PLANNED / NOT YET RUN`**.
- Tests verifying the current Phase 1 specification and configuration are marked **`EXECUTED`** once verified.

---

## 2. Test Catalog & Verification Matrix

### 2.1 Input Validation Tests (`tests/unit/test_input_validation.py`)

| Test ID | Test Description | Test Type | Expected Outcome | Phase 1 Status |
|---|---|---|---|---|
| `VAL-01` | Non-existent file path | Unit | Raises `FileNotFoundError` | **`PLANNED`** |
| `VAL-02` | Empty file (0 bytes) | Unit | Raises `ValueError: Uploaded file is empty` | **`PLANNED`** |
| `VAL-03` | Corrupted NIfTI image | Unit | Raises `nibabel.filebasedimages.ImageFileError` | **`PLANNED`** |
| `VAL-04` | 3D Structural image passed as functional | Unit | Raises `ValueError: Expected 4D BOLD, found 3D` | **`PLANNED`** |
| `VAL-05` | Scan has fewer than 50 timepoints | Unit | Raises `ValueError: Insufficient timepoints (T < 50)` | **`PLANNED`** |
| `VAL-06` | Missing BOLD JSON sidecar | Unit | Raises `FileNotFoundError: Missing sidecar` | **`PLANNED`** |
| `VAL-07` | Sidecar missing `RepetitionTime` | Unit | Raises `KeyError: RepetitionTime required` | **`PLANNED`** |
| `VAL-08` | Sidecar specifies non-resting task | Unit | Raises `ValueError: Unsupported task (must be 'rest')` | **`PLANNED`** |
| `VAL-09` | Non-finite values in functional image | Unit | Aborts with error identifying non-finite voxels | **`PLANNED`** |

### 2.2 Preprocessing & Registration Tests (`tests/integration/test_preprocessing.py`)

| Test ID | Test Description | Test Type | Expected Outcome | Phase 1 Status |
|---|---|---|---|---|
| `PRP-01` | Coregistration fails to converge | Integration | Aborts pipeline; logs failure; cleans temp files | **`PLANNED`** |
| `PRP-02` | MNI template space mismatch | Unit | Detects affine discrepancy; resamples correctly | **`PLANNED`** |
| `PRP-03` | CC200 atlas mask file missing | Integration | Aborts before preprocessing with path error | **`PLANNED`** |
| `PRP-04` | Signal filtering retains 0.01–0.10 Hz band | Unit | Spectral power verified in passband | **`PLANNED`** |
| `PRP-05` | Nuisance regression preserves zero mean | Unit | Cleaned BOLD voxels have mean $\approx 0$ | **`PLANNED`** |
| `PRP-06` | Extreme head motion detected | Integration | Generates motion warning in QC manifest | **`PLANNED`** |

### 2.3 CC200 Parcellation & Feature Tests (`tests/unit/test_cc200_features.py`)

| Test ID | Test Description | Test Type | Expected Outcome | Phase 1 Status |
|---|---|---|---|---|
| `FET-01` | Parcellation produces exactly 200 ROIs | Unit | Shape is strictly `(200, T)` | **`PLANNED`** |
| `FET-02` | ROI labels are strictly 1..200 in order | Unit | Row index array equals `np.arange(1, 201)` | **`PLANNED`** |
| `FET-03` | Zero-variance ROI signal detected | Unit | Aborts with `ValueError: Constant ROI signals` | **`PLANNED`** |
| `FET-04` | Correlation matrix symmetrization | Unit | Confirms `corr == corr.T` and `diag == 1.0` | **`PLANNED`** |
| `FET-05` | Upper-triangle edge count is 19,900 | Unit | Length of edge vector is exactly 19,900 | **`PLANNED`** |
| `FET-06` | Boundary clipping prevents infinities | Unit | Values at $\pm 1.0$ are clipped to $\pm (1 - 10^{-7})$ | **`PLANNED`** |
| `FET-07` | Feature naming matches lexicographic order | Unit | Names match `ROI_001__ROI_002` .. `ROI_199__ROI_200` | **`PLANNED`** |

### 2.4 End-to-End Inference Tests (`tests/integration/test_model_inference.py`)

| Test ID | Test Description | Test Type | Expected Outcome | Phase 1 Status |
|---|---|---|---|---|
| `INF-01` | Preprocessing pipeline artifact missing | Integration | Raises `FileNotFoundError` | **`PLANNED`** |
| `INF-02` | Model checkpoint missing | Integration | Raises `FileNotFoundError` | **`PLANNED`** |
| `INF-03` | Transforms 19,900 features to 100 features | Integration | Output shape is strictly `(1, 100)` | **`PLANNED`** |
| `INF-04` | Frozen pipeline does not call fit() | Unit | Checks learned attributes are unmodified | **`PLANNED`** |
| `INF-05` | Output contains required disclaimers | Unit | Validates disclaimer string in response | **`PLANNED`** |

---

## 3. Execution Commands for Automated Suite

Once Phase 2 is authorized, the complete automated test suite will be run from the project root via:

```bash
# Run unit tests only
pytest tests/unit -v

# Run full test suite with coverage
pytest --cov=src tests/ -v
```
