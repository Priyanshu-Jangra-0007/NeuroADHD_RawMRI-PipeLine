# FORENSIC AUDIT OF THE REJECTED SINGLE-SUBJECT PROTOTYPE (PHASE 2)
**Project**: `NeuroADHD_RawMRI`  
**Target Subject**: `sub-2344857` (Kennedy Krieger Institute — KKI, Training Cohort)  
**Audit Scope**: Read-Only Inspection of Executed Code, Logs, Manifests, and Outputs  
**Date of Audit**: October 21, 2026  
**Status**: COMPLETE — REJECTED IMPLEMENTATION EXPLAINED  

---

## Executive Summary

A single-subject proof of concept (PoC) pipeline was executed on subject `sub-2344857` using `scripts/run_single_scan_poc.py`. The resulting derivative generated a Pearson concordance of **$r = 0.2512$** (linear regression slope: $0.2499$, MAE: $0.2683$) when compared against the authoritative 19,900 Fisher-$z$ connectivity features from the historical ADHD-200 C-PAC 0.3.9.1 pipeline. This fell far short of the project's provisional concordance threshold of $r \ge 0.85$, and the prototype was rejected.

This read-only forensic audit investigated the exact code executed, verified file identities and checksums, evaluated the feature comparison logic, and ranked the causes of failure by evidence strength.

### Key Audit Conclusions
1. **Root Cause Confirmed (Rank 1)**: Spatial registration to standard MNI152 space was **completely omitted**. Although anatomical coregistration functions were written and the T1w scan was loaded, the script never invoked coregistration. Instead, native-space functional data were merely resampled into the MNI template grid geometry using `nilearn.image.resample_to_img()` without any spatial alignment. The CC200 parcels were applied to brain tissue in native scanner coordinate space.
2. **Comparison Logic Validated**: The 19,900 Fisher-$z$ edges were compared by explicit string feature name rather than positional column alignment. Computing connectivity from the historical `roi_CC200.csv` using the Python feature extraction code reproduced the historical feature table row for `2344857` with a maximum absolute discrepancy of **$4.93 \times 10^{-10}$** (machine precision). The low concordance was caused by physical spatial misregistration, not software indexing or edge-ordering bugs.
3. **Raw File Discrepancy Resolved**: The raw BOLD scan on disk (`60,405,324` bytes, SHA-256: `350615addd323635ef7f5691c8267df216743d661c8a0f19042e0e86f0f70ae8`) perfectly matches the pre-existing backup manifest recorded on September 29, 2026. The previously cited size of `33,485,302` bytes was a typographical error in an earlier markdown report (`PHASE2_BLOCKER_RESOLUTION_INVESTIGATION.md`). The raw BOLD scan was never modified or overwritten.
4. **Variance Compression Did Not Cause Failure**: While post-regression residual signals had lower numerical variance than raw signals, Pearson correlation and Fisher-$z$ transformation are mathematically scale-invariant ($r = \frac{\operatorname{Cov}(X, Y)}{\sigma_X \sigma_Y}$). Variance compression cannot account for $r = 0.2512$.

---

## 1. Inspection of the Executed Implementation

A line-by-line inspection of `scripts/run_single_scan_poc.py`, helper modules under `src/`, and `outputs/sub-2344857/validation_report.json` was conducted to establish verified code evidence versus previous report claims.

### Summary Comparison: Previous Claims vs. Verified Code Evidence

| Feature / Step | Claimed in Previous Reports | Actual Executed Code Evidence | Evidence Path & Line Reference |
| :--- | :--- | :--- | :--- |
| **`clean_img` Standardization** | Stated to use standard settings or unstandardized residuals | `clean_img` explicitly passed `standardize=False` | [`src/preprocessing/nuisance.py:L122`](<your-project-folder>/src/preprocessing/nuisance.py#L122) |
| **Signal Standardization Timing** | Claimed standardized | Neither before nor after ROI averaging; `NiftiLabelsMasker` also set `standardize=False` | [`src/parcellation/extractor.py:L35`](<your-project-folder>/src/parcellation/extractor.py#L35) |
| **Motion Realignment** | DIPY 3D rigid realignment with mean reference | **Verified executed**: Each volume was transformed via DIPY `AffineRegistration` using `RigidTransform3D`; reference volume was the temporal mean (`np.mean(data, axis=3)`) | [`src/preprocessing/motion.py:L99-L120`](<your-project-folder>/src/preprocessing/motion.py#L99-L120) |
| **EPI-to-T1w Coregistration** | Stated to be part of normalization workflow | **Never executed**: `register_epi_to_t1w` was imported and `t1_img` was loaded, but the function was never called | [`scripts/run_single_scan_poc.py:L31,L99,L133`](<your-project-folder>/scripts/run_single_scan_poc.py#L31-L133) |
| **T1w-to-MNI Nonlinear Warp** | Described as SyN or affine to MNI152 | **Never executed**: No nonlinear warp (SyN/FNIRT/ANTs) was implemented or called | [`src/preprocessing/registration.py:L49-L80`](<your-project-folder>/src/preprocessing/registration.py#L49-L80) |
| **Claimed MNI Normalization** | "Normalized to MNI152 standard space" | Consisted solely of `nilearn.image.resample_to_img(bold_4d, target_atlas_img, interpolation="continuous")` | [`src/preprocessing/registration.py:L72-L78`](<your-project-folder>/src/preprocessing/registration.py#L72-L78) |
| **CompCor Masking** | Stated to match C-PAC aCompCor (WM/CSF) | Heuristic intensity threshold mask (`mean > 0.15 * max`), top 10% highest variance voxels across whole brain | [`scripts/run_single_scan_poc.py:L147`](<your-project-folder>/scripts/run_single_scan_poc.py#L147), [`src/preprocessing/nuisance.py:L47-L50`](<your-project-folder>/src/preprocessing/nuisance.py#L47-L50) |
| **Confound Matrix** | 24 motion + 5 CompCor + trends | 31 columns: 24 Friston regressors, 5 aCompCor components, 2 polynomial trends (linear, quadratic). GSR OFF. | [`src/preprocessing/nuisance.py:L68-L83`](<your-project-folder>/src/preprocessing/nuisance.py#L68-L83) |
| **Bandpass Filtering** | 0.01 - 0.10 Hz simultaneous filtering | Simultaneously filtered in `clean_img(low_pass=0.10, high_pass=0.01, t_r=2.5)` | [`src/preprocessing/nuisance.py:L123-L125`](<your-project-folder>/src/preprocessing/nuisance.py#L123-L125) |
| **ROI Extraction Dimensions** | 200 parcels, 152 timepoints | Shape: $(200, 153)$ (column 0: `node/volume`, columns 1..152: timepoints $0 \dots 151$) | [`src/parcellation/extractor.py:L62-L67`](<your-project-folder>/src/parcellation/extractor.py#L62-L67) |
| **Feature Extraction** | 19,900 Fisher-z edges | Authoritative upper-triangle extraction ($k=1$), clipped to $[-1+\epsilon, 1-\epsilon]$, transformed via $\operatorname{arctanh}(r)$ | [`src/features/connectivity.py:L47-L56`](<your-project-folder>/src/features/connectivity.py#L47-L56) |

---

## 2. Audit of the Comparison Logic

### 2.1 Static Inspection of Edge Matching
The historical connectivity matrix contains $\binom{200}{2} = 19,900$ unique pairwise edges. A potential failure mode in functional connectivity pipelines is edge misalignment caused by row-major vs. column-major flattening, varying 0-based vs. 1-based indexing, or column positional mismatch.

Static code inspection reveals:
1. **Authoritative Edge Names**: [`get_feature_names()`](<your-project-folder>/src/features/connectivity.py#L13-L16) generates edge names using `itertools.combinations(range(1, 201), 2)`. This yields:
   ```python
   ["ROI_001__ROI_002", "ROI_001__ROI_003", ..., "ROI_199__ROI_200"]
   ```
2. **Matrix Extraction**: [`compute_connectivity_vector()`](<your-project-folder>/src/features/connectivity.py#L50-L52) extracts edges via:
   ```python
   tri_i, tri_j = np.triu_indices(200, k=1)
   edge_corr = corr[tri_i, tri_j]
   ```
   In NumPy, `np.triu_indices(N, k=1)` traverses upper-triangle coordinates in strict row-major order: $(0, 1), (0, 2), \dots, (0, 199), (1, 2), \dots, (198, 199)$. This maps 1-to-1 to the pairs generated by `itertools.combinations(range(1, 201), 2)`.
3. **Explicit Key-Based Matching**: In [`scripts/run_single_scan_poc.py:L198`](<your-project-folder>/scripts/run_single_scan_poc.py#L198), edges are matched to the historical feature table using named column selection:
   ```python
   hist_table_edges = sub_row[edge_names].to_numpy(dtype=np.float64).flatten()
   ```
   The comparison does not rely on arbitrary column index positions; it indexes columns by exact string identity.

### 2.2 Numerical Proof of Mathematical Equivalence
In [`scripts/run_single_scan_poc.py:L186-L202`](<your-project-folder>/scripts/run_single_scan_poc.py#L186-L202), the pipeline re-extracted 19,900 Fisher-$z$ edges from the historical `roi_CC200.csv` using the Python feature extraction function and compared them directly against `sub_row[edge_names]` from `down/NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv`:
```
Verification: Historical CSV re-extraction matches feature table (Max Abs Diff: 4.93e-10)
```
- **Max Absolute Discrepancy**: $4.93 \times 10^{-10}$ (within double-precision float rounding error).
- **Correlation**: $r = 1.000000$.
- **Conclusion**: The comparison logic, edge names, ROI ordering, upper-triangle indexing, and Fisher-$z$ conversion are **100% mathematically correct and identical** to the historical pipeline. The failure of the prototype is not due to comparison logic.

---

## 3. Reconciliation of Raw BOLD File Identity

### 3.1 The Reported Discrepancy
- **Earlier report claim** (`PHASE2_BLOCKER_RESOLUTION_INVESTIGATION.md:L113`): `33,485,302` bytes.
- **Physical disk observation and latest report**: `60,405,324` bytes.

### 3.2 Audit of Manifests and On-Disk File Metadata
The absolute paths, filesystem metadata, and historical manifests were checked without altering any file:

1. **Target File Path**:
   `<external-reference-folder>/outputs/ADHD200_RawDataBIDS/KKI/sub-2344857/ses-1/func/sub-2344857_ses-1_task-rest_acq-1_run-1_bold.nii.gz`
2. **Current Filesystem Stat**:
   - Size: `60,405,324` bytes.
   - Last Modified Timestamp: September 29, 2026, 23:06.
   - SHA-256 Hash: `350615addd323635ef7f5691c8267df216743d661c8a0f19042e0e86f0f70ae8`.
3. **Historical Manifest Check**:
   Searching pre-existing project backup manifests created on September 29, 2026 reveals:
   - `<external-reference-folder>/BACKUP_MANIFEST.csv:1194`:
     ```csv
     outputs/ADHD200_RawDataBIDS/KKI/sub-2344857/ses-1/func/sub-2344857_ses-1_task-rest_acq-1_run-1_bold.nii.gz,60405324,350615addd323635ef7f5691c8267df216743d661c8a0f19042e0e86f0f70ae8,raw_data
     ```
   - `<external-reference-folder>/NeuroADHD_NewPC_Transfer_manifest.csv:267`:
     ```csv
     ADHD200_RawDataBIDS_subset/BIDS/KKI/sub-2344857/ses-1/func/sub-2344857_ses-1_task-rest_acq-1_run-1_bold.nii.gz,<external-reference-folder>/outputs/ADHD200_RawDataBIDS/KKI/sub-2344857/ses-1/func/sub-2344857_ses-1_task-rest_acq-1_run-1_bold.nii.gz,60405324,350615addd323635ef7f5691c8267df216743d661c8a0f19042e0e86f0f70ae8,official_raw_bids_local_subset,...
     ```
   - Pre-existing Anatomical Scan Manifest:
     Size: `5,834,214` bytes, SHA-256: `4e3de7bd16469b1588b02e97ca5ef124809629aa9d2112f7eae498e9bce18838`.

### 3.3 Provenance Verdict
The raw BOLD file currently on disk is byte-for-byte identical to the file cataloged when the project repository was initialized on September 29, 2026. The number `33,485,302` does not exist in any manifest, log, or filesystem record; it was an accidental typographical error in the markdown narrative of `PHASE2_BLOCKER_RESOLUTION_INVESTIGATION.md`. The integrity of the raw input data is completely uncompromised.

---

## 4. Assessment of Rejected Outputs & Root Cause Ranking

The rejected prototype recorded the following concordance metrics against the historical derivative:
- Pearson correlation: $r = 0.2512$
- Linear regression slope: $0.2499$ (ideal: $1.0000$)
- Linear regression intercept: $0.0072$ (ideal: $0.0000$)
- Mean Absolute Error (MAE): $0.2683$
- Root Mean Squared Error (RMSE): $0.3376$

### 4.1 Failure Causes Ranked by Evidence Strength

#### Rank 1 (Confirmed Root Cause): Complete Omission of Spatial Registration to MNI Template
- **Evidence Strength**: **DEFINITIVE (Code Inspection & Behavioral Proof)**.
- **Evidence**:
  1. [`scripts/run_single_scan_poc.py:L31`](<your-project-folder>/scripts/run_single_scan_poc.py#L31) imported `register_epi_to_t1w` and loaded `t1_img` (L99), but never called `register_epi_to_t1w`. The variable `t1_img` was never passed to any registration or transformation routine.
  2. [`src/preprocessing/registration.py:L49-L80`](<your-project-folder>/src/preprocessing/registration.py#L49-L80) defined `normalize_to_mni` as a simple call to `nilearn.image.resample_to_img()`.
  3. No affine transformation, no rigid EPI-to-T1w coregistration, and no nonlinear warp (SyN/FNIRT/ANTs) were ever computed or applied.
- **Scientific Impact**:
  Resampling an image into an atlas bounding box with `resample_to_img` only interpolates voxels based on the existing NIfTI header affine. In raw BIDS data, the functional affine reflects scanner magnet bore coordinates. Standard space CC200 parcels were therefore placed over arbitrary brain structures, ventricles, or outside the skull entirely. This anatomical mismatch directly explains why the correlation between regional signals and historical signals collapsed to $r = 0.2512$.

#### Rank 2 (Strongly Supported Contributor): aCompCor Masking and Physiological Confound Extraction
- **Evidence Strength**: **HIGH (Code Inspection)**.
- **Evidence**:
  In [`scripts/run_single_scan_poc.py:L147`](<your-project-folder>/scripts/run_single_scan_poc.py#L147), `brain_mask` was created using an intensity heuristic (`mean_norm > 0.15 * max`). In [`src/preprocessing/nuisance.py:L48-L53`](<your-project-folder>/src/preprocessing/nuisance.py#L48-L53), CompCor extracted the top 10% highest variance voxels across this entire brain mask.
- **Scientific Impact**:
  Official C-PAC 0.3.9.1 aCompCor extracts components strictly from **eroded anatomical White Matter (WM) and Cerebrospinal Fluid (CSF)** masks derived from tissue segmentation of the anatomical T1w volume. Extracting principal components from the top 10% variance of the whole brain captures genuine cortical gray matter fluctuations, inappropriately regressing out neural signal.

#### Rank 3 (Hypothesis / Minor Contributor): Motion Correction Toolchain Differences
- **Evidence Strength**: **MODERATE**.
- **Evidence**:
  The prototype used DIPY's `AffineRegistration(metric=MutualInformationMetric, transform=RigidTransform3D)` with temporal mean reference. C-PAC 0.3.9.1 historically used AFNI `3dvolreg` (or FSL `mcflirt`) with Fourier or sinc interpolation.
- **Scientific Impact**:
  Different motion correction implementations yield slightly different motion parameter trajectories and interpolation smoothing, which could slightly perturb the Friston-24 regressors and time series, but cannot account for an $r = 0.25$ collapse on its own.

#### Eliminated Factors
1. **ROI Indexing & Comparison Logic**: **ELIMINATED**. Verified with $4.93 \times 10^{-10}$ error against the historical feature table.
2. **Variance Compression**: **ELIMINATED as primary cause**. While post-regression residuals had standard deviations of $\sim 0.03\text{--}0.25$ compared to historical raw signals ($\sim 20\text{--}80$), Pearson correlation $r = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sqrt{\sum (x - \bar{x})^2 \sum (y - \bar{y})^2}}$ is mathematically invariant to positive scalar scaling. Variance scaling differences alone cannot explain the low concordance.
3. **Input Data Corruption**: **ELIMINATED**. Raw BOLD and T1w files have unmodified timestamps from September 29, 2026 and exact SHA-256 matches to the original backup manifest.

---

## 5. Unresolved Questions

1. **Exact C-PAC Coregistration Cost Function for KKI**:
   Did the original C-PAC 0.3.9.1 configuration for KKI use Boundary-Based Registration (BBR via FSL FLIRT) or Mutual Information (`corratio` / `mutualinfo`) for EPI-to-T1w coregistration?
2. **Nonlinear Normalization Resolution and Parameters**:
   Did C-PAC use ANTs (`antsRegistrationSyN.sh`) or FSL `fnirt` with 2mm or 3mm MNI152 non-linear templates?
3. **Anatomical Tissue Segmentation for aCompCor**:
   Did historical preprocessing use FSL `fast` with 50% probability thresholding and 1-voxel erosion for WM and CSF masks, or an AFNI-based segmentation?

---

## 6. Minimum Additional Checks Needed Before Another Scan Can Be Processed

Before any new scan or rerun is initiated, the following prerequisites must be formally verified in an approved plan:

1. **Implement and Validate Anatomical Processing**:
   - Implement T1w brain extraction (skull stripping) using an Apple Silicon-compatible tool (e.g. DIPY or HD-BET).
   - Implement tissue segmentation to produce verified anatomical White Matter (WM) and Cerebrospinal Fluid (CSF) probability masks.
2. **Implement and Validate Rigid EPI-to-T1w Coregistration**:
   - Align the mean EPI volume to the skull-stripped T1w anatomical scan.
   - Verify alignment using mutual information or boundary-based registration.
3. **Implement and Validate Nonlinear T1w-to-MNI Normalization**:
   - Compute a verified nonlinear spatial transformation (e.g., DIPY Symmetric Diffeomorphic Registration — SyN, or ANTs) mapping subject T1w to the standard MNI152 template.
4. **Composite Spatial Transformation**:
   - Concatenate the rigid EPI-to-T1w affine and the nonlinear T1w-to-MNI warp into a single resampling step to avoid multiple interpolation blurs.
5. **Spatial QC Check**:
   - Generate an automated slice overlay or report verifying that the CC200 atlas parcels overlay the normalized functional cortex accurately before computing ROI time series.
6. **Strict Execution Authorization**:
   - Await explicit user authorization before creating scripts or running any processing.

---
*Report compiled autonomously via read-only inspection. All existing files, outputs, and model artifacts remain unaltered.*
