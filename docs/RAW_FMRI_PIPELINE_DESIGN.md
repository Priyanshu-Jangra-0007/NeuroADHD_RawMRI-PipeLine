# RAW fMRI PIPELINE DESIGN: ARCHITECTURE & WORKFLOW

## 1. System Overview

The `NeuroADHD_RawMRI` preprocessing pipeline bridges the gap between raw DICOM/BIDS neuroimaging data and the existing frozen `NeuroADHD` classification model. It transforms raw 4D resting-state fMRI volumes into the verified `roi_CC200.csv` format using parameter settings calibrated to the historical ADHD-200 CC200 derivatives.

```
RAW BIDS DIRECTORY (sub-<ID>/func/*_bold.nii.gz + sidecar)
                          ↓
[Step 1: Input Validation & Header Integrity Check]
                          ↓
[Step 2: Motion Correction (3dvolreg / MCFLIRT) & FD Generation]
                          ↓
[Step 3: Coregistration & Standard Space Normalization (MNI152)]
                          ↓
[Step 4: Nuisance Regression & Temporal Bandpass Filtering]
  - Linear & Quadratic Detrending
  - 24 Motion Regressors (Friston 24)
  - aCompCor (5 components, CSF/WM masks)
  - Bandpass Filter (0.01 - 0.10 Hz)
  - Global Signal Regression: OFF (global0)
                          ↓
[Step 5: CC200 Atlas Parcellation & Mean Signal Extraction]
  - Map CC200 ROI mask onto normalized BOLD space
  - Extract spatial mean time series across voxels per ROI
                          ↓
[Step 6: Signal Quality & Variance Validation]
  - Verify exactly 200 ROIs
  - Confirm std(ROI) > 0 for all regions (no constant signals)
                          ↓
OUTPUT: roi_CC200.csv (200 rows x T columns)
```

---

## 2. Detailed Processing Steps

### Step 1: Input Validation & Header Check
- Validates BIDS naming conventions.
- Checks BOLD 4D dimensionality and ensures minimum temporal length ($T \ge 50$).
- Parses $T_R$ from the sidecar JSON.
- Generates initial QC report and provenance checksums.

### Step 2: Functional Motion Correction
- Computes rigid-body motion parameters (3 translations in mm, 3 rotations in degrees) using the mean volume as reference.
- Generates 6 realignment parameters and calculates Framewise Displacement (FD).
- Constructs the 24-parameter Friston motion matrix (6 standard parameters, 6 temporal derivatives, and their quadratic terms: $6 + 6 + 6 + 6 = 24$).

### Step 3: Spatial Registration & MNI Normalization
- Rigid/Affine coregistration of the mean functional image to the subject's anatomical T1w (if present) or directly to standard MNI152 space.
- Non-linear spatial normalization to MNI152 standard space (target resolution $3 \times 3 \times 3 \text{ mm}^3$ or matching historical grid).
- Generates brain mask to restrict subsequent operations to intracranial voxels.

### Step 4: Nuisance Regression & Temporal Filtering
In strict accordance with the historical findings in `docs/HISTORICAL_PREPROCESSING_AUDIT.md`:
- **Nuisance Matrix ($M$) Construction**:
  - Linear and quadratic polynomial trends (`linear1.quadratic1`).
  - 24 Friston motion regressors (`motion1`).
  - 5 anatomical CompCor (aCompCor) components extracted from high-probability eroded CSF and WM masks (`_compcor_ncomponents_5_selector_pc10`, CSF threshold 0.96, WM threshold 0.96).
  - Global signal is **NOT** included (`global0`).
  - Mean WM/CSF signals are **NOT** added separately (`wm0.csf0.gm0`), as variance is accounted for by CompCor.
- **Simultaneous Filtering & Confound Regression**:
  - Regression and bandpass filtering (0.01 – 0.10 Hz) must be performed simultaneously or sequentially in an orthogonal manner to prevent frequency leakage and reintroduction of noise artifacts.

### Step 5: CC200 Parcellation & ROI Extraction
- Loads the Craddock 200 (CC200) functional atlas mask in standard MNI space.
- Verifies that the atlas contains exactly 200 discrete integer labels $\{1, 2, \dots, 200\}$.
- For each ROI $k \in \{1, \dots, 200\}$:
  $$s_k(t) = \frac{1}{|V_k|} \sum_{v \in V_k} B(v, t)$$
  where $V_k$ is the set of brain voxels belonging to region $k$, and $B(v, t)$ is the cleaned BOLD signal at voxel $v$ and timepoint $t$.

### Step 6: ROI Validation & Output Formatting
- Verifies that every region $k \in \{1, \dots, 200\}$ contains at least one voxel covered by the acquisition.
- Computes temporal standard deviation $\sigma_k = \operatorname{std}(s_k, \text{ddof}=0)$.
  - If $\sigma_k == 0.0$ for any $k$, abort with a fatal QC error: `"Zero-variance ROI detected; Pearson correlation undefined"`.
- Formats output table:
  - Header: `node/volume,0,1,2,...,T-1`
  - Rows: 200 rows with 1-based index in column 0.
- Writes to `roi_CC200.csv`.

---

## 3. Toolchain Evaluation & Rationale

For Phase 2 implementation, three execution modalities were evaluated:

1. **Native Python 3 Stack (Nilearn + Nibabel + Scipy)**:
   - **Pros**: Zero Docker requirement, runs natively on macOS Apple Silicon (ARM64), lightweight memory footprint, fully reproducible in pure Python.
   - **Cons**: Requires manual implementation of spatial registration (or reliance on lightweight ANTs/NiftyReg binaries).
   - **Suitability**: Highly suitable for motion correction, nuisance regression, filtering, and CC200 extraction.

2. **Containerized Standard Pipeline (fMRIPrep / C-PAC Modern Docker)**:
   - **Pros**: Gold-standard anatomical registration and surface processing.
   - **Cons**: Docker daemon currently inactive on host; requires heavy image downloads (>15 GB); heavy memory footprint (>12 GB RAM) which exceeds host physical RAM (8 GB).

3. **Hybrid Workflow (Recommended for Phase 2 PoC)**:
   - Utilize a lightweight, native Python workflow (`nilearn.signal.clean`, `nilearn.maskers.NiftiLabelsMasker`) with standardized MNI normalization templates.
   - Validates memory safety on 8 GB RAM without depending on background virtualization daemons.
