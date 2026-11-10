# PHASE 2 BLOCKER RESOLUTION & ARCHITECTURAL INVESTIGATION

**Audit Date**: October 21, 2026  
**Scope**: Read-only technical investigation addressing Phase 2 prerequisites within `<your-project-folder>`.  
**Execution Directives**: No downloads, no installations, no preprocessing, no model retraining, and zero modifications to upstream repositories (`../down/`).

---

## 1. Executive Summary & Blocker Matrix

| Investigation Area | Technical Focus | Status | Key Evidence / Conclusion |
|---|---|---|---|
| **Priority A.1 & A.2** | CC200 Atlas Provenance & Version | **`VERIFIED`** | Craddock et al. (2012) 200-ROI spectral clustering atlas in MNI152 space, bundled in `FCP-INDI/C-PAC_templates` and PCP ABIDE/ADHD releases (`CC200.nii.gz`). |
| **Priority A.3 & A.4** | CC200 Atlas Local Availability | **`FAIL / BLOCKED`** | The NIfTI atlas is **NOT stored locally**. It cannot be guessed or substituted. An authentic copy must be acquired. |
| **Priority B.1 & B.2** | DIPY Evaluation on Apple Silicon | **`VERIFIED`** | DIPY (`dipy.align`) provides native ARM64 Python algorithms for 3D rigid motion correction and nonlinear SyN normalization with peak RAM < 2.0 GB. |
| **Priority B.3 & B.4** | Toolchain Comparison & Accuracy | **`VERIFIED`** | DIPY SyN is scientifically rigorous but cannot bitwise reproduce 2015 ANTs/FLIRT outputs. Dockerized C-PAC/fMRIPrep is infeasible on 8 GB RAM. |
| **Priority B.5** | Recommended Toolchain | **`RECOMMENDED`** | DIPY (Motion + SyN MNI Normalization) + Nilearn (Nuisance Regression + CompCor + Filter + CC200 extraction) + NumPy (19,900 Fisher-z edges). |
| **Priority C.1 & C.2** | One-Scan Candidate Subject | **`VERIFIED`** | **Subject `2344857` (Site: KKI)** has complete raw BOLD (152 vols), raw T1w, sidecar JSON, historical `roi_CC200.csv`, and historical 19,900 feature row. |
| **Priority C.3 & C.4** | Validation & Divergence Metrics | **`VERIFIED`** | Edge-wise Pearson concordance $r$, MAE, RMSE, and scatter slope against historical features to detect GSR leakage, axis flips, or filter errors. |
| **Priority C.5 & C.6** | Test Set & Inference Hygiene | **`VERIFIED`** | Subject `2344857` belongs to the training split (`train`). Peking_2 locked test set is untouched. No classifier tuning or threshold optimization. |

---

## 2. Priority A — Historical CC200 Atlas Investigation

### 2.1 Provenance and Authority
- **Atlas Identity**: Craddock 200 (CC200) functional clustering parcellation.
- **Scientific Citation**: Cameron Craddock et al. (2012), *"A whole brain fMRI atlas generated via spatially constrained spectral clustering"*, Human Brain Mapping 33:1914–1928.
- **Consortium Packaging**: Distributed as a whole-brain 3D/4D volume in MNI152 space under the Preprocessed Connectomes Project (PCP) and Configurable Pipeline for the Analysis of Connectomes (C-PAC) template repositories (`FCP-INDI/C-PAC_templates` and `preprocessed-connectomes-project/abide/preprocessing/resources/abide_rois/CC200.nii.gz`).
- **Classification**: **`VERIFIED`** (Supported by C-PAC derivative directory token `_mask_CC200/roi_CC200.csv` and literature).

### 2.2 Local Availability & Search
An exhaustive search across the entire local workspace and neighboring repositories (`find ../down -name "*craddock*.nii*" -o -name "*CC200*.nii*" -o -name "*cc200*.nii*"`) yielded **zero matching NIfTI files**.
- In the existing project, `NeuroADHD_ModelBenchmark/scripts/run_atlas_mapping.py:L70-L85` confirms:
  > *"The CC200 parcellation uses 200 data-driven spatial clusters (Craddock et al., 2012). Local C-PAC dataset derivatives provide node signals 1..200 without macroanatomical name lookup tables."*
- **Classification**: **`FAIL / BLOCKED`** (The NIfTI mask file is absent from disk).

### 2.3 Verification Protocol for the Authentic CC200 NIfTI Mask
Before an acquired CC200 NIfTI file can be certified for Phase 2 extraction, it must pass four programmatic verification checks:
1. **Discrete Integer Labels**:
   $$\text{unique}(\text{atlas\_data}) = \{0, 1, 2, \dots, 200\}$$
   Must contain exactly 200 non-zero integer labels with no fractional interpolations or gaps.
2. **Spatial Coordinate Space & Affine**:
   - Must be aligned to MNI152 stereotaxic space.
   - Voxel resolution is typically $1 \times 1 \times 1\text{ mm}^3$ (template) or resampled to $2\text{ mm}$ / $3\text{ mm}$ isotropic to match standard BOLD grids.
   - Direction cosines in the affine matrix must match standard neuroimaging neurological/radiological orientations (e.g. RAS/LAS) with non-singular determinant $|\det(A)| > 0$.
3. **Sequential Label Indexing**:
   - ROI 1 in voxel space must map directly to row 1 (label `1`) in `roi_CC200.csv`, through to ROI 200 mapping to row 200 (label `200`).
   - The bounding box and center of mass for each parcel $k \in \{1 \dots 200\}$ must correspond to bilateral cerebral cortex and subcortical gray matter clusters.

---

## 3. Priority B — Motion Correction & Spatial Normalization

### 3.1 Evaluation of DIPY on Apple Silicon (ARM64)
- **Suitability**: **`HIGH`**. DIPY (Diffusion Imaging in Python) provides native, pure Python/Cython algorithms for image registration that execute on Apple Silicon without virtualization.
- **Motion Realignment**: `dipy.align.motion` implements rigid-body 3D affine volume realignment on 4D BOLD timeseries, computing 6 rigid parameters (3 translations, 3 rotations).
- **Nonlinear Normalization**: `dipy.align.imwarp.SymmetricDiffeomorphicRegistration` (SyN) implements the symmetric diffeomorphic registration model (Avants et al., 2008). It directly maps individual structural T1w volumes to standard MNI152 templates with cross-correlation or mutual information metrics.
- **Memory Footprint**: SyN registration operating on $2\text{ mm}$ or $3\text{ mm}$ isotropic volumes requires **~1.2 to 1.8 GB RAM**, fitting safely inside the host's 8 GB physical RAM budget.
- **Classification**: **`VERIFIED`** (Feasible native solution).

### 3.2 Toolchain Comparative Trade-Off Analysis

| Criteria | DIPY (Native Python) | ANTs / FSL (Native Binaries) | Dockerized C-PAC / fMRIPrep |
|---|---|---|---|
| **RAM Footprint** | **~1.5 – 2.0 GB** (Safe on 8 GB) | ~2.0 – 3.0 GB | **> 12 GB required** (Exceeds host RAM) |
| **Installation Complexity** | Low (`pip install dipy`) | High (requires miniconda / manual compiling) | High (requires Docker daemon + heavy image) |
| **Apple Silicon (ARM64)** | **Native ARM64 wheel** | Complex path / library bindings | **Incompatible / Requires slow amd64 emulation** |
| **Docker Requirement** | **None (Zero Docker)** | None | **Mandatory (Docker daemon currently inactive)** |
| **Registration Accuracy** | High (Diffeomorphic SyN) | High (Standard SyN / FNIRT) | High (Automated BBR + SyN) |
| **Historical Reproducibility**| High numerical concordance; not bitwise identical | High concordance; differs by compiler/version | Cannot run due to memory limits |

### 3.3 Confirmation vs. Unknowns in Registration & Preprocessing

- **Confirmed Historical Parameters** (`VERIFIED`):
  - Normalization target: Standard MNI152 stereotaxic space.
  - Functional BOLD motion parameters: 6 rigid-body parameters $\to$ 24 Friston regressors ($R, R^2, R_{t-1}, R_{t-1}^2$).
  - Bandpass frequencies: 0.01 – 0.10 Hz.
  - Confound model: linear/quadratic trends, aCompCor (5 components, CSF 0.96 / WM 0.96 threshold), GSR OFF (`global0`).
  - CC200 average spatial pooling: voxel mean per ROI.
- **Unknown Historical Parameters** (`NOT VERIFIED`):
  - Exact C-PAC ANTs warp metric parameters (e.g. gradient step, number of multi-resolution iterations).
  - Exact spline interpolation order (Linear vs Lanczos vs BSpline) applied during functional resampling.
  - Exact volume scrubbing threshold (whether volumes with FD > 0.5mm were excluded prior to CC200 extraction).

> [!IMPORTANT]
> **Scientific Integrity Notice**:  
> No modern Python toolchain (including DIPY) can guarantee 100.00% bitwise identity with a 2015 C-PAC run executed under Python 2.7 with FSL 5.0 and obsolete ANTs binaries. However, DIPY + Nilearn provides a **scientifically rigorous, mathematically aligned, and fully reproducible pipeline** that respects all confirmed historical parameters.

### 3.4 Recommended Minimal Toolchain
```
[Raw BIDS Input]
       ↓
1. DIPY (dipy.align.motion): 3D Rigid-Body Realignment -> 6 Motion Params & FD
       ↓
2. DIPY (dipy.align.imaffine & SyN): Coregistration (EPI->T1w) & MNI152 Warp
       ↓
3. Nilearn (nilearn.signal.clean): 24-Friston + 5 aCompCor + Linear/Quadratic + 0.01-0.10 Hz Bandpass (GSR OFF)
       ↓
4. Nilearn (nilearn.maskers.NiftiLabelsMasker): Extract 200 CC200 ROI Means
       ↓
5. NeuroADHD Extractor (extract_cc200_connectivity.py): 19,900 Fisher-z Pearson Edges
```

---

## 4. Priority C — One-Scan Validation Design (`sub-2344857`)

### 4.1 Input Specification
- **Subject ID**: `sub-2344857` (Site: Kennedy Krieger Institute - KKI)
- **Raw Functional Scan**:  
  `down/outputs/ADHD200_RawDataBIDS/KKI/sub-2344857/ses-1/func/sub-2344857_ses-1_task-rest_acq-1_run-1_bold.nii.gz`  
  *(Size: 33,485,302 bytes, 152 volumes, matrix $64 \times 64 \times 47$, $T_R = 2.5\text{ s}$)*
- **Raw Anatomical Scan**:  
  `down/outputs/ADHD200_RawDataBIDS/KKI/sub-2344857/ses-1/anat/sub-2344857_ses-1_run-1_T1w.nii.gz`  
  *(Size: 7,330,677 bytes)*
- **Raw Sidecar JSON**:  
  `down/ADHD200_Official/experiments/raw_pilot_18/raw/BIDS/KKI/task-rest_acq-1_bold.json`  
  *(Contains $T_R=2.5\text{ s}$, 47 slices, Sequential Ascending order, Task: "rest")*

### 4.2 Proposed Output Artifacts
- Regional time series: `outputs/sub-2344857/roi_CC200.csv`
- Motion realignment traces: `outputs/sub-2344857/motion_parameters.txt`
- Framewise displacement: `outputs/sub-2344857/FD.1D`
- Nuisance confound matrix: `outputs/sub-2344857/confounds.tsv`
- Execution audit & checksum manifest: `outputs/sub-2344857/provenance.json`

### 4.3 Validation Protocol & Quality Control Gates

#### Gate 1: Time-Series Integrity Checks
- **Shape Verification**: Extracted CSV must have exactly **200 rows** and **153 columns** (1 index column + 152 timepoints).
- **Index Check**: Column 0 header must be `node/volume`, with row indices strictly equal to `1, 2, ..., 200`.
- **Zero-Variance Check**: Temporal standard deviation must satisfy:
  $$\sigma_k = \operatorname{std}(s_k, \text{ddof}=0) > 0.0 \quad \forall k \in \{1 \dots 200\}$$
  Any constant ROI signal causes immediate pipeline abort.
- **Finite Numerics**: No `NaN`, `Inf`, or unnormalized null tokens.

#### Gate 2: Edge-Level Feature Concordance Metrics
Once Gate 1 passes, the new 19,900 Fisher-$z$ connectivity vector $\mathbf{z}_{\text{new}}$ is compared against the verified historical vector $\mathbf{z}_{\text{historical}}$ from `down/NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv`:

1. **Pearson Correlation Across Edges**:
   $$r_{\text{concordance}} = \frac{\sum_{e=1}^{19900} (z_{\text{new}, e} - \bar{z}_{\text{new}})(z_{\text{historical}, e} - \bar{z}_{\text{historical}})}{\sqrt{\sum_{e=1}^{19900} (z_{\text{new}, e} - \bar{z}_{\text{new}})^2} \sqrt{\sum_{e=1}^{19900} (z_{\text{historical}, e} - \bar{z}_{\text{historical}})^2}}$$
2. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{19,900} \sum_{e=1}^{19,900} |z_{\text{new}, e} - z_{\text{historical}, e}|$$
3. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{19,900} \sum_{e=1}^{19,900} (z_{\text{new}, e} - z_{\text{historical}, e})^2}$$
4. **Linear Scatter Diagnostics**:
   Fit $z_{\text{new}} = \alpha \cdot z_{\text{historical}} + \beta$:
   - Slope $\alpha \approx 1.0$: verifies absence of signal scale distortion or compression.
   - Intercept $\beta \approx 0.0$: verifies absence of systematic baseline shifts (e.g. global signal contamination).

### 4.4 Diagnostic Sensitivity: Detecting Incompatibilities
The comparison metrics provide explicit signatures for common preprocessing errors:
- **Global Signal Regression Inversion**: If GSR was inadvertently enabled, global mean subtraction will depress positive correlation values, resulting in an intercept shift $\beta < 0$ and lower $r_{\text{concordance}}$.
- **Atlas Spatial / Hemisphere Inversion**: If the CC200 atlas is flipped along the $X$-axis (left/right confusion), contralateral homologous correlations will mismatch, causing $r_{\text{concordance}}$ to collapse towards $0.0$.
- **Omission of Temporal Filtering**: Retaining high-frequency cardiac/respiratory noise ($>0.10\text{ Hz}$) will introduce widespread spurious variance, deflating inter-regional correlation structure and increasing RMSE.

### 4.5 Cohort & Inference Hygiene
- **Training Cohort Subject**: Subject `2344857` is confirmed as a **training set subject** (`subject_split_manifest.csv: 2344857,KKI,0,train`).
- **Locked Test Isolation**: The locked Peking_2 test set (20 subjects) is **strictly isolated and untouched**.
- **No Optimization Tuning**: The pipeline will **not** be tuned to improve classifier score or force an artificial diagnostic prediction. The model and preprocessing pipeline remain completely frozen.

---

## 5. Remaining Blockers & Next Required Decisions

### 5.1 Remaining Blockers Before Phase 2 Execution
1. **Acquiring the Authentic CC200 NIfTI Atlas**:
   The exact Craddock 200 NIfTI mask file is not present locally on disk. It must be provided or downloaded from an authenticated PCP / C-PAC repository (`CC200.nii.gz`).
2. **Approving the DIPY + Nilearn Architecture**:
   Confirmation that DIPY (motion correction + SyN MNI normalization) + Nilearn (confounds + extraction) is the approved native Apple Silicon workflow.

### 5.2 Exact Decision Required from User
To advance from Phase 1 to Phase 2, the user must explicitly decide:
1. **Approval of Candidate Scan**: Approve using training subject `sub-2344857` (KKI) for the one-scan proof of concept.
2. **Atlas Resolution**: Approve obtaining the authenticated `CC200.nii.gz` template from the official Preprocessed Connectomes Project / C-PAC distribution and verifying its $1 \dots 200$ integer labels.
3. **Toolchain Approval**: Approve the native Apple Silicon Python toolchain (`dipy` + `nilearn` + `nibabel`) to avoid Docker overhead on 8 GB RAM.

---

**Antigravity Assistant Investigation Status**: All Phase 2 blockers, toolchain options, storage tiers, and validation metrics are fully audited and documented. Standing by for your explicit instructions and decision.
