# DESIGN SPECIFICATION: CORRECTED SINGLE-SUBJECT PIPELINE (VERSION 2)
**Project**: `NeuroADHD_RawMRI`  
**Target Subject**: `sub-2344857` (Kennedy Krieger Institute — KKI, Training Cohort)  
**Document Status**: DESIGN SPECIFICATION ONLY — NO CODE EXECUTED / NO SCRIPTS MODIFIED  
**Date**: October 21, 2026  

---

## 1. Executive Context & Objectives

The forensic audit documented in [`docs/REJECTED_POC_FORENSIC_AUDIT.md`](<your-project-folder>/docs/REJECTED_POC_FORENSIC_AUDIT.md) definitively established why the initial Phase 2 proof-of-concept prototype failed to achieve feature concordance with historical C-PAC 0.3.9.1 derivatives ($r = 0.2512$):
1. **Omitted Spatial Registration**: Functional volumes in native scanner coordinates were resampled directly into the standard space grid using `nilearn.image.resample_to_img()` without computing or applying rigid EPI-to-T1w coregistration or nonlinear T1w-to-MNI normalization. CC200 parcels were applied to mismatched anatomical tissue and non-brain regions.
2. **Unsegmented aCompCor Masking**: Five CompCor components were derived from the top 10% variance voxels across an unsegmented whole-brain intensity mask, inappropriately regressing out cortical gray matter fluctuations.

### Objective of Design v2
This document specifies a complete, mathematically sound, parameter-calibrated resting-state fMRI preprocessing architecture for single-subject processing. It provides:
- A verified, composable spatial-registration transformation chain.
- An anatomically grounded tissue segmentation and nuisance-regression workflow.
- Strict multi-stage validation gates and acceptance criteria.
- Complete preservation of upstream project assets, model weights, and the Peking_2 locked test set.

---

## 2. Spatial-Registration Architecture & Transformation Chain

```
[Raw 4D BOLD]                                      [Raw 3D T1w]
      ↓                                                  ↓
[3D Rigid Motion Realignment]                      [Anatomical Brain Extraction]
  - Target: static mean EPI                          - Skull-stripped T1w brain
  - Trilinear / sinc interpolation                       ↓
  - Output: motion_params (6)                      [Tissue Segmentation]
  - Output: FD.1D                                    - WM mask (p >= 0.96, eroded)
  - Output: realigned mean EPI                       - CSF mask (p >= 0.96, eroded)
      │                                                  │
      └──────────────────┬───────────────────────────────┘
                         ↓
      [Stage 1: Functional-to-Anatomical Coregistration]
        - Moving: Realigned Mean EPI
        - Static: Skull-stripped T1w
        - Transform: Rigid 3D (6 DOF) -> M_EPI->T1w
        - Metric: Normalized Mutual Information (NMI)
                         ↓
      [Stage 2: Anatomical-to-MNI152 Normalization]
        - Moving: Skull-stripped T1w
        - Static: Authentic MNI152 2mm/3mm Brain Template
        - Step 2a: Affine 3D (12 DOF) -> M_T1w->MNI
        - Step 2b: Symmetric Diffeomorphic (SyN) Nonlinear Warp -> Φ_T1w->MNI
                         ↓
      [Stage 3: Composed Single-Step Reslicing]
        - Composite mapping: Φ_T1w->MNI ∘ M_T1w->MNI ∘ M_EPI->T1w
        - Applied to 4D realigned BOLD series into MNI CC200 grid
        - Interpolation: Continuous 5th-order B-spline / trilinear
                         ↓
      [Stage 4: Spatial Correspondence Verification]
        - Automated contour overlay QC inspection
        - Voxel overlap threshold check against MNI brain mask
                         ↓
      [CC200 Atlas Parcellation] (Only after Stage 4 passes)
```

### 2.1 Transformation Chain Details

#### Step 1: Functional Motion Correction
- **Input**: Raw 4D functional series ($64 \times 64 \times 47 \times 152$, $T_R = 2.5\text{ s}$).
- **Reference Volume**: Static temporal mean across all timepoints ($\bar{B}(x, y, z) = \frac{1}{T} \sum_{t=1}^T B(x, y, z, t)$).
- **Transformation**: 3D rigid-body transformation (6 degrees of freedom: 3 translations in mm, 3 rotations in degrees).
- **Optimization Metric**: Mutual Information (32 bins).
- **Outputs**:
  - Realigned 4D BOLD series.
  - Motion parameters text file ($152 \times 6$).
  - Framewise Displacement vector ($152 \times 1$) computed according to Power et al. (2012) with $r = 50\text{ mm}$.
  - Static temporal mean volume of the realigned functional series ($\text{EPI}_{\text{ref}}$).

#### Step 2: EPI-to-T1w Coregistration
- **Moving Image**: Subject mean functional volume ($\text{EPI}_{\text{ref}}$), skull-stripped using an intensity-based intracranial mask.
- **Fixed Image**: Subject anatomical volume ($\text{T1w}_{\text{brain}}$), skull-stripped.
- **Transformation Model**: Rigid-body affine transformation ($M_{\text{EPI}\to\text{T1w}}$, $4 \times 4$ matrix, 6 DOF).
- **Similarity Metric**: Normalized Mutual Information (NMI) or Boundary-Based Registration (BBR) cost function.
- **Direction**: Forward mapping from native functional space to native anatomical T1w space.

#### Step 3: Subject T1w to MNI152 Normalization
- **Moving Image**: Subject skull-stripped anatomical volume ($\text{T1w}_{\text{brain}}$).
- **Fixed Reference**: Verified MNI152 standard brain template (MNI152NLin6Asym or MNI152_T1_2mm_brain matching C-PAC).
- **Multi-Resolution Hierarchical Alignment**:
  1. *Linear Affine Alignment*: 12 degrees of freedom (translation, rotation, scaling, shearing) yielding affine matrix $M_{\text{T1w}\to\text{MNI}}$.
  2. *Nonlinear Diffeomorphic Alignment*: Symmetric Diffeomorphic Registration (SyN) with cross-correlation (CC) metric, computing forward displacement field $\mathbf{u}(x, y, z)$ and inverse field $\mathbf{u}^{-1}(x, y, z)$, denoted $\Phi_{\text{T1w}\to\text{MNI}}$.

#### Step 4: Mathematical Composition and Single-Step Reslicing
To avoid the spatial blurring and resolution degradation caused by sequential resampling (resampling at motion correction, then coregistration, then normalization), the spatial mappings must be composed mathematically:

$$\mathbf{x}_{\text{MNI}} = \Phi_{\text{T1w}\to\text{MNI}} \left( M_{\text{T1w}\to\text{MNI}} \cdot M_{\text{EPI}\to\text{T1w}} \cdot \mathbf{x}_{\text{EPI}} \right)$$

- **Pull-Back Coordinate Mapping**:
  For each voxel coordinate $\mathbf{x}_{\text{MNI}}$ in the target CC200 atlas grid ($63 \times 75 \times 61$, $3.0\text{ mm}$ isotropic):
  $$\mathbf{x}_{\text{EPI}} = M_{\text{EPI}\to\text{T1w}}^{-1} \cdot M_{\text{T1w}\to\text{MNI}}^{-1} \cdot \Phi_{\text{T1w}\to\text{MNI}}^{-1}(\mathbf{x}_{\text{MNI}})$$
- **Interpolation Scheme**: Continuous windowed sinc or 5th-order B-spline interpolation for the functional series; nearest-neighbor interpolation for discrete parcel masks.

### 2.2 Toolchain Evaluation for Apple Silicon (8 GB RAM)

| Backend Candidate | Native Apple Silicon | RAM Footprint (8 GB Peak) | Historical C-PAC Compatibility | Implementation Complexity | Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DIPY Native Python** (`dipy.align`) | Yes (Pure ARM64 / NumPy / Cython) | **Low** (1.5 – 2.5 GB peak) | Approximate (SyN algorithm is mathematically comparable, but not identical to ANTs C++) | Minimal (already installed in local `.venv`) | **Recommended for PoC v2** under strict RAM safety |
| **ANTsPy** (`antspyx`) | Yes (Prebuilt ARM64 wheels available) | **Moderate to High** (3.5 – 5.5 GB peak during 4-level SyN) | **High** (identical underlying C++ ANTs codebase used in C-PAC / fMRIPrep) | Moderate (requires package install) | **Strong Alternative** if DIPY concordance is insufficient |
| **FSL 5.0 / AFNI 2011 (Docker)** | No (Requires x86_64 emulation via Rosetta 2) | **Prohibitive** (> 12 GB RAM required for container overhead) | Exact (original 2015 historical stack) | High (Docker daemon inactive, disk heavy) | **Rejected** (incompatible with 8 GB physical host) |

> [!IMPORTANT]
> **Toolchain Caveat**: Neither DIPY SyN nor ANTsPy reproduces historical C-PAC 0.3.9.1 outputs to bitwise identity, because exact 2015 random seeds, ITK versions, and library compilations cannot be recovered. The objective is **anatomical and statistical compatibility**, not bitwise emulation.

---

## 3. Nuisance-Regression Architecture & Physiological Modeling

### 3.1 Anatomical White Matter and CSF Mask Derivation
Historical C-PAC 0.3.9.1 derivatives for KKI used anatomical CompCor (`compcor1`, 5 components, `_csf_threshold_0.96/_wm_threshold_0.96`). Extracting components from whole-brain high-variance voxels was a primary failure mode of PoC v1.

#### Anatomical Mask Protocol:
1. **Tissue Probability Segmentation**:
   - Segment the subject's skull-stripped T1w scan into Gray Matter (GM), White Matter (WM), and Cerebrospinal Fluid (CSF) probability maps.
   - Using native Apple Silicon Python: Fit a multi-class Gaussian Mixture Model (GMM) or DIPY Expectation-Maximization tissue classifier (`dipy.segment.tissue.TissueClassifierHMRF`) with spatial neighborhood priors.
2. **Thresholding**:
   - Strict thresholding adhering to historical C-PAC parameters:
     - $P(\text{WM}) \ge 0.96$
     - $P(\text{CSF}) \ge 0.96$
3. **Morphological Erosion**:
   - Apply a 1-voxel binary erosion (3D structuring element, radius 1) to both the WM and CSF masks to eliminate partial volume averaging with adjacent cortical and subcortical gray matter.
4. **Spatial Transformation to Standard Space**:
   - Resample eroded anatomical masks into the CC200 MNI grid using the inverse transformation or direct forward warp with nearest-neighbor interpolation.

### 3.2 Confound Regressor Construction (31 Columns)

The complete nuisance matrix $\mathbf{X}_{\text{nuisance}} \in \mathbb{R}^{T \times 31}$ must consist of:
1. **24 Friston Motion Regressors** (`motion_friston_01` to `motion_friston_24`):
   - 6 standard realignment parameters ($R = [t_x, t_y, t_z, r_x, r_y, r_z]$).
   - 6 squared realignment parameters ($R^2$).
   - 6 backward temporal derivatives ($R_{t-1}$, with $t_0$ set to 0).
   - 6 squared backward derivatives ($R_{t-1}^2$).
2. **5 Anatomical CompCor Components** (`aCompCor_01` to `aCompCor_05`):
   - Union of eroded WM and eroded CSF masks.
   - Voxel signals within this physiological noise region are standardized (zero mean, unit variance).
   - Principal Component Analysis (PCA) extracts the top 5 eigenvectors explaining physiological variance.
3. **Polynomial Trends** (`poly_linear`, `poly_quadratic`):
   - Linear trend ($t$, centered on zero: $\operatorname{linspace}(-1, 1, T)$).
   - Quadratic trend ($t^2$).
4. **Excluded Parameters (Strict Historical Parity)**:
   - Global Signal Regression: **STRICTLY OFF** (`global0`).
   - Mean Tissue Signals: **OFF** (`wm0.gm0.csf0`) because variance is accounted for by aCompCor.

### 3.3 Simultaneous Regression & Filtering
To prevent frequency leakage and re-injection of nuisance artifacts into filtered passbands:
- Nuisance regression and temporal bandpass filtering ($0.01 \le f \le 0.10\text{ Hz}$) must be executed **simultaneously** using orthogonal projection or `nilearn.image.clean_img()`.
- Standardize parameter in `clean_img`: Must remain `standardize=False` so that residual BOLD values retain their physical amplitude scale prior to ROI averaging.

### 3.4 Summary: Verified Evidence vs. Unresolved Settings

| Pipeline Component | Historical Setting | Status | Action in Design v2 |
| :--- | :--- | :--- | :--- |
| **GSR** | Disabled (`global0`) | **`VERIFIED`** | Regressors omit global mean signal |
| **Bandpass Filtering** | $0.01 - 0.10\text{ Hz}$ | **`VERIFIED`** | Simultaneous bandpass filter in `clean_img` |
| **Friston-24 Regressors** | 24 motion terms | **`VERIFIED`** | $R, R^2, R_{t-1}, R_{t-1}^2$ included in confounds |
| **CompCor Component Count** | Exactly 5 components | **`VERIFIED`** | Top 5 PCA components from physiological mask |
| **Tissue Thresholds** | $\text{WM} \ge 0.96, \text{CSF} \ge 0.96$ | **`VERIFIED`** | Segmented T1w probability masks thresholded at 0.96 |
| **Tissue Erosion** | 1-voxel erosion | **`PROVISIONAL`** | Enforced to prevent gray matter partial voluming |
| **Scrubbing / Censoring** | Threshold unspecified in logs | **`UNRESOLVED`** | No volumes censored; FD recorded for QC only |
| **Slice Timing Correction** | Unverified for KKI site | **`UNRESOLVED`** | Omitted in v2 to prevent interpolation artifacts |
| **Exact Registration Cost** | BBR vs NMI in C-PAC 0.3.9.1 | **`UNRESOLVED`** | Use Mutual Information with rigid coregistration |

---

## 4. Multi-Stage Validation Gates & Acceptance Criteria

Processing must proceed through explicit sequential gates. If any gate fails, the pipeline must immediately abort and output a structured diagnostic report without overwriting prior outputs.

```
[Gate 1: Pre-Execution Resource Check] ──(Pass)──>
[Gate 2: Raw Input & BIDS Sidecar Check] ──(Pass)──>
[Gate 3: Atlas Integrity & Grid Validation] ──(Pass)──>
[Gate 4: Spatial Alignment & Coregistration QC] ──(Pass)──>
[Gate 5: CC200 ROI Coverage & Signal QC] ──(Pass)──>
[Gate 6: Edge Concordance & Feature Validation] ──(Pass)──>
[Gate 7: Provenance & Output Manifest Generation]
```

### Gate 1: Pre-Execution Environment & Resource Budget
- **Memory Check**: Verify $\ge 3.5\text{ GB}$ of uncompressed RAM headroom prior to execution.
- **Disk Check**: Verify $\ge 10\text{ GB}$ free disk space in `NeuroADHD_RawMRI`.
- **Sandbox Isolation**: Confirm that all outputs target `NeuroADHD_RawMRI/outputs/sub-2344857_v2/` and that no upstream project paths are targeted.

### Gate 2: Input Integrity & Metadata Validation
- **Raw BOLD**: Check existence, size ($60,405,324\text{ B}$), and SHA-256 (`350615addd323635ef7f5691c8267df216743d661c8a0f19042e0e86f0f70ae8`).
- **Raw T1w**: Check existence, size ($5,834,214\text{ B}$), and SHA-256 (`4e3de7bd16469b1588b02e97ca5ef124809629aa9d2112f7eae498e9bce18838`).
- **Sidecar JSON**: Verify $T_R = 2.5\text{ s}$, slice count $= 47$, temporal volumes $T = 152 \ge 50$.

### Gate 3: Atlas Provenance & Grid Check
- **Atlas File**: `data/atlases/cc200_roi_atlas.nii.gz` (SHA-256: `9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650`).
- **Dimensions**: Exactly $(63, 75, 61)$ with voxel zoom $(3.0, 3.0, 3.0)\text{ mm}$.
- **Labels**: Exactly 201 unique values (integers $0 \dots 200$, where 0 is background and $1 \dots 200$ are anatomical regions).

### Gate 4: Spatial Alignment & Coregistration QC
Before parcellation is permitted to run, spatial alignment must be verified both quantitatively and visually:
1. **Quantitative Voxel Overlap**:
   - Compute the spatial Dice coefficient between the normalized subject brain mask and the standard MNI152 brain mask:
     $$\operatorname{Dice}(A, B) = \frac{2 |A \cap B|}{|A| + |B|} \ge 0.85$$
   - Failure to achieve $\operatorname{Dice} \ge 0.85$ triggers an automatic pipeline stop.
2. **Visual QC Image Generation**:
   - Save high-resolution PNG inspection figures:
     - `qc_registration_epi_to_t1w.png`: Contours of coregistered EPI overlaid on anatomical T1w.
     - `qc_registration_t1w_to_mni.png`: Contours of normalized T1w overlaid on MNI152 template.
     - `qc_parcellation_overlay.png`: CC200 parcel boundaries overlaid on normalized functional mean image.

### Gate 5: ROI Coverage & Time-Series Quality
- **Coverage**: All 200 parcels must contain at least 5 functional voxels ($|V_k| \ge 5\ \forall k \in \{1 \dots 200\}$).
- **Temporal Standard Deviation**:
  $$\sigma_k = \operatorname{std}(s_k, \text{ddof}=0) > 0.0 \quad \forall k \in \{1 \dots 200\}$$
  If any region has $\sigma_k == 0.0$, the pipeline halts immediately.
- **Matrix Dimension**: Extracted time series must be exactly $(200, 152)$.

### Gate 6: Connectivity Feature Vector Alignment & Concordance Metrics
1. **Authoritative String-Name Matching**:
   All 19,900 pairwise edges are mapped using exact lexicographical string names (`ROI_001__ROI_002` through `ROI_199__ROI_200`), avoiding column positional errors.
2. **Comparative Metrics Evaluated**:
   - Pearson correlation coefficient ($r$).
   - Mean Absolute Error (MAE).
   - Root Mean Squared Error (RMSE).
   - Linear regression slope ($m$) and intercept ($b$).
   - Coefficient of determination ($R^2$).
3. **Status of Threshold ($r \ge 0.85$)**:
   The previously suggested threshold of $r \ge 0.85$ is recognized as a **provisional engineering indicator of pipeline health**, not mathematical proof of equivalence. A lower concordance (e.g. $r \approx 0.65\text{--}0.80$) with visually verified registration and correct anatomical masks may represent the limit of cross-toolchain reproducibility between modern DIPY/ANTs and 2015 C-PAC.

### Gate 7: Provenance & Output Manifest
- Generate `outputs/sub-2344857_v2/provenance.json` detailing:
  - Exact UTC timestamp and Git commit hash (if available).
  - Software versions (`dipy`, `nilearn`, `nibabel`, `numpy`, `scipy`).
  - SHA-256 hashes of all input files, intermediate transformation matrices, and output artifacts.
  - Complete quantitative metrics from Gates 1 through 6.

---

## 5. Resource Estimates & Execution Plan

### 5.1 Memory & Storage Budgets (Apple M2, 8 GB RAM)
- **Memory Footprint**:
  - Raw 4D functional array ($64 \times 64 \times 47 \times 152$ float32): $\approx 117\text{ MB}$.
  - Normalized 4D functional array ($63 \times 75 \times 61 \times 152$ float32): $\approx 175\text{ MB}$.
  - Multi-resolution SyN displacement field: $\approx 850\text{ MB}$.
  - Peak RAM estimate: **$2.2\text{ GB}$** (well within the safe 5.5 GB headroom limit).
- **Disk Storage**:
  - Intermediate transformation files and QC PNGs: $\approx 45\text{ MB}$.
  - Final artifacts (`roi_CC200.csv`, `motion_parameters.txt`, `FD.1D`, `confounds.tsv`, `validation_report.json`): $\approx 500\text{ KB}$.
  - Total disk requirement: $< 100\text{ MB}$.

### 5.2 Next Steps & Execution Gate
This document constitutes a **complete design specification**. In accordance with instructions:
- **No processing has been started.**
- **No existing scripts or files have been modified.**
- **The previous outputs in `outputs/sub-2344857/` remain intact.**
- **The Peking_2 test set and upstream NeuroADHD model artifacts remain strictly isolated.**

**Next Action Required**: Await explicit user authorization before writing implementation code, downloading templates, or initiating execution.
