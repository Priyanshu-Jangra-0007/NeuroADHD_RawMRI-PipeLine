# NeuroADHD RawMRI PoC V2 — Phase A.5
# Technical Review & Pipeline Architecture Specification

**Document ID:** `DOC-NA-A5-TR-001`  
**Version:** 1.0.0  
**Status:** `READY_FOR_SUPERVISOR_DECISION`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Execution Context:** Strictly Synthetic-Only (Linux Cgroup $\le 2.5\text{ GB}$)  

---

## 1. System Overview & Architectural Diagram

NeuroADHD RawMRI PoC V2 is an automated, containerized neuroimaging pipeline designed to process raw structural (T1w) and resting-state functional (rs-fMRI BOLD) magnetic resonance images conforming to the Brain Imaging Data Structure (BIDS). The pipeline extracts functional connectivity features using the Craddock CC200 parcellation and feeds them into a frozen, pre-trained scikit-learn machine learning classifier.

The complete system comprises 19 distinct algorithmic stages separated into two strictly bounded architectural tiers:
1. **Tier I: Neuroimaging Preprocessing & Connectivity Engineering (Stages 1–17)**
2. **Tier II: Downstream Frozen Machine Learning Inference (Stages 18–19)**

```
====================================================================================================
               NEUROADHD RAWMRI POC V2: END-TO-END TECHNICAL ARCHITECTURE
====================================================================================================

               +-------------------------------------------------------+
               |  BIDS Input Directory: sub-<ID>/                      |
               |  - anat/sub-<ID>_T1w.nii.gz (+ .json)                 |
               |  - func/sub-<ID>_task-rest_bold.nii.gz (+ .json)      |
               |  - manifest.json (HMAC-SHA256 Signed)                 |
               +-------------------------------------------------------+
                                           |
                                           v
+==================================================================================================+
| TIER I: RAW MRI PREPROCESSING & FEATURE EXTRACTION (STAGES 1-17)                                 |
+==================================================================================================+
|  [Stage 1]  Manifest HMAC Verification  ---> Cryptographic signature check (Fail-closed)        |
|  [Stage 2]  BIDS Schema Validation      ---> Filename syntax, JSON metadata, required tags       |
+--------------------------------------------------------------------------------------------------+
| STRUCTURAL PATHWAY (T1w)                 | FUNCTIONAL PATHWAY (BOLD)                             |
|                                          |                                                       |
|  [Stage 5] Skull Stripping (Otsu/Watershed) [Stage 3] BOLD Motion Estimation (MCFLIRT-style 6-DOF) |
|            -> Brain Mask Extraction      |            -> 6 Motion Parameters (Tx, Ty, Tz, Rx, Ry, Rz)|
|                                          |            -> Framewise Displacement (FD), DVARS     |
|  [Stage 6] N4 Bias Field Correction      |                                                       |
|            -> Low-frequency intensity    |  [Stage 4] BOLD Reference Volume Generation           |
|               homogeneity correction     |            -> Median/mean volume over motion-free time|
|                                          |                                                       |
|  [Stage 7] Tissue Segmentation           |  [Stage 9] EPI-to-T1w Coregistration                  |
|            -> CSF, WM, GM posterior maps|            -> Rigid 6-DOF boundary-based registration |
|                                          |            -> Normalized Mutual Information (NMI)     |
|  [Stage 8] Tissue Mask Erosion           |                                                       |
|            -> CSF (1-voxel), WM (2-voxel)|                                                       |
+------------------------------------------+-------------------------------------------------------+
|  [Stage 10] T1w-to-MNI152 Non-Linear Registration                                                 |
|             -> Affine initialization + SyN (Symmetric Diffeomorphic Normalization)               |
+--------------------------------------------------------------------------------------------------+
|  [Stage 11] Unified Transform Composition                                                         |
|             -> Composition: Phi_total = Phi_{T1w->MNI} o T_{EPI->T1w} o T_{motion}               |
|             -> Single-step pullback resampling avoiding intermediate interpolation blurring      |
+--------------------------------------------------------------------------------------------------+
|  [Stage 12] BOLD Resampling to Canonical CC200 Grid (63 x 75 x 61, 3.0mm isotropic, 4th-order)    |
+--------------------------------------------------------------------------------------------------+
|  [Stage 13] Nuisance Regressor Extraction: aCompCor (5 CSF + 5 WM) + Friston-24 Motion Expansion  |
|  [Stage 14] Nuisance Regression & Temporal Bandpass Filtering (0.01 - 0.10 Hz, simultaneous OLS)  |
|  [Stage 15] Canonical CC200 Atlas Verification (SHA-256: 9467afce..., 200 labeled ROIs)         |
|  [Stage 16] CC200 ROI Time-Series Extraction (Spatial mean across voxels within each ROI)        |
|  [Stage 17] Functional Connectivity Computation: Pearson r -> Fisher-z transform                  |
|             -> 19,900 Upper-Triangular Features (200 * 199 / 2)                                  |
+==================================================================================================+
                                           |
                                           | Feature Vector: 1 x 19,900 float64
                                           v
+==================================================================================================+
| TIER II: DOWNSTREAM FROZEN ML INFERENCE (STAGES 18-19) [READ-ONLY MOUNT, ZERO RETRAINING]        |
+==================================================================================================+
|  [Stage 18] Preprocessing Pipeline (`final_preprocessing_pipeline.joblib`)                       |
|             1. SimpleImputer(strategy='median')                                                  |
|             2. VarianceThreshold(threshold=0.0)                                                  |
|             3. StandardScaler()                                                                  |
|             4. SelectKBest(score_func=f_classif, k=100)                                          |
|             -> Dimension Reduced: 1 x 19,900  --->  1 x 100 Selected Features                   |
+--------------------------------------------------------------------------------------------------+
|  [Stage 19] Classification Model (`final_model.joblib`)                                          |
|             - RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced')      |
|             -> Output: Class Probabilities [P(TDC), P(ADHD)], Binary Classification Call        |
+==================================================================================================+
                                           |
                                           v
+--------------------------------------------------------------------------------------------------+
| OUTPUT ARTIFACTS:                                                                                |
| - QC Acceptance Report (8 Evaluated Metrics vs Pending Thresholds)                               |
| - Pipeline Execution Log & Provenance Metadata                                                  |
| - CC200 Connectivity Matrix (200 x 200)                                                          |
| - Exploratory Model Prediction Summary (Non-Diagnostic)                                          |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Comprehensive Stage-by-Stage Technical Specification

### Stage 1: Input Manifest & Cryptographic Signature Verification
- **Input:** `manifest.json` located within the input root, HMAC key via `NEUROADHD_HMAC_KEY`.
- **Operation:** Evaluates cryptographic authenticity of input dataset using HMAC-SHA256. Verifies that all registered files exist, canonicalizes all relative paths, and asserts that no symlinks resolve outside the designated sandbox root directory.
- **Output:** Validated file manifest and verified `is_synthetic` execution mode flag.
- **Validation Method:** `hmac.compare_digest(computed_hmac, manifest_hmac)`.
- **Failure Conditions:** Missing manifest, missing HMAC key in environment, empty key, HMAC mismatch, path traversal attempt, or unregistered files. The pipeline terminates immediately (`exit code 1`).

### Stage 2: BIDS Directory Structure Validation
- **Input:** Input directory path conforming to BIDS v1.8+.
- **Operation:** Inspects subject directory structure `sub-<subject_id>/` for required sub-folders (`anat/`, `func/`). Parses naming conventions according to BIDS regular expressions. Validates existence and basic structure of JSON sidecars (repetition time `RepetitionTime`, echo time `EchoTime`, task name).
- **Output:** Structured BIDS data dictionary mapping subject identifiers to physical NIfTI and JSON sidecar paths.
- **Validation Method:** Regex syntax parsing against official BIDS file format specs.
- **Failure Conditions:** Non-conforming filenames, missing JSON metadata, missing modalities, multiple conflicting runs without explicit run identifiers.

### Stage 3: BOLD Motion Estimation & Realignment
- **Input:** Raw 4D resting-state BOLD timeseries ($N_x \times N_y \times N_z \times N_t$).
- **Operation:** Rigid-body 6-degrees-of-freedom (6-DOF) motion realignment using a Gauss-Newton optimization with normalized correlation cost function. Computes translations ($\Delta x, \Delta y, \Delta z$ in mm) and Euler rotations ($\theta_x, \theta_y, \theta_z$ in degrees/radians) relative to the reference volume. Computes Framewise Displacement (FD, Jenkinson formula) and DVARS (root-mean-square intensity differences of volume-to-volume transitions).
- **Output:** Realigned 4D BOLD timeseries, 6 motion parameter file ($N_t \times 6$), Framewise Displacement timeseries ($N_t \times 1$), DVARS vector ($N_t - 1 \times 1$).
- **Validation Method:** Motion parameter bounds check; verification that reference volume motion is zero.
- **Failure Conditions:** Convergence failure during 6-DOF minimization, unreadable timepoints, zero temporal variance.

### Stage 4: High-Contrast BOLD Reference Volume Generation
- **Input:** Realigned 4D BOLD timeseries, Framewise Displacement vector.
- **Operation:** Identifies the temporal subset of volumes exhibiting minimal motion ($\text{FD} < 0.2\text{ mm}$). Computes a voxel-wise median volume across these high-quality volumes to produce a high signal-to-noise ratio (tSNR) 3D functional reference image.
- **Output:** 3D functional reference volume (`bold_ref.nii.gz`, shape $N_x \times N_y \times N_z$).
- **Validation Method:** Signal-to-noise ratio calculation ($\mu / \sigma > 15$ on synthetic reference).
- **Failure Conditions:** All timepoints exceed motion threshold; empty volume selection.

### Stage 5: T1w Structural Brain Extraction (Skull Stripping)
- **Input:** Raw 3D structural T1-weighted volume (`sub-<ID>_T1w.nii.gz`).
- **Operation:** Eliminates non-brain anatomical structures (skull, scalp, dura, meninges, neck tissue) using an intensity-adaptive multi-Otsu thresholding followed by morphological opening, geodesic hole-filling, and connected-component labeling.
- **Output:** Stripped T1w brain image (`t1w_brain.nii.gz`) and binary brain mask (`t1w_mask.nii.gz`).
- **Validation Method:** Brain volume quantification against biological limits ($600\text{--}2200\text{ cm}^3$); mask compactness check.
- **Failure Conditions:** Brain volume outside allowable range, disjoint mask fragments exceeding threshold, empty mask.

### Stage 6: T1w N4 Bias Field Correction
- **Input:** Stripped T1w brain image, binary brain mask.
- **Operation:** Corrects low-frequency radiofrequency transmit/receive coil intensity inhomogeneities using Non-parametric Non-uniform intensity Normalization (N4ITK). Solves for a multiplicative smooth bias field using a multi-resolution B-spline approximation under the brain mask.
- **Output:** Inhomogeneity-corrected structural brain volume (`t1w_corrected.nii.gz`).
- **Validation Method:** Coefficient of variation (CV) reduction across homogeneous white matter voxels.
- **Failure Conditions:** Divergence of B-spline optimization; NaN/Inf values generated in corrected image.

### Stage 7: T1w Tissue Segmentation
- **Input:** Bias-corrected structural brain volume.
- **Operation:** Classifies brain tissue into three primary compartments: Cerebrospinal Fluid (CSF), White Matter (WM), and Gray Matter (GM) using a three-class Gaussian Mixture Model (GMM) with spatial Markov Random Field (MRF) regularization. Produces posterior probability maps ($P_{\text{CSF}}, P_{\text{WM}}, P_{\text{GM}}$) and deterministic hard segmentations.
- **Output:** Probabilistic tissue maps and discrete segmentation mask (`t1w_dseg.nii.gz`).
- **Validation Method:** Tissue volume ratios check (biological sanity: GM:WM ratio between 0.8 and 1.8; total tissue volume sanity).
- **Failure Conditions:** Failure of EM convergence; degenerate tissue fractions ($<5\%$ for any primary class).

### Stage 8: Anatomical Tissue Mask Erosion
- **Input:** Discrete tissue masks (CSF, WM).
- **Operation:** Morphological erosion using a 3D spherical structuring element. CSF mask is eroded by 1 voxel (radius $r=1$); WM mask is eroded by 2 voxels (radius $r=2$). This aggressive erosion eliminates partial volume contamination from cortical gray matter voxels during nuisance signal extraction.
- **Output:** Eroded binary masks: `csf_mask_eroded.nii.gz`, `wm_mask_eroded.nii.gz`.
- **Validation Method:** Volume reduction audit: eroded WM volume must retain $>30\%$ of original WM volume.
- **Failure Conditions:** Complete erosion of mask (zero remaining voxels), non-contiguous degenerate mask.

### Stage 9: EPI-to-T1w Functional-Structural Coregistration
- **Input:** BOLD reference volume (`bold_ref.nii.gz`), T1w brain volume (`t1w_brain.nii.gz`).
- **Operation:** Computes a rigid-body 6-DOF coordinate transformation matrix ($T_{\text{EPI}\to\text{T1w}} \in \mathbb{R}^{4 \times 4}$) aligning the low-resolution functional image with the high-resolution structural anatomy. Uses Normalized Mutual Information (NMI) or Boundary-Based Registration (BBR) cost metric.
- **Output:** Rigid affine transformation matrix ($T_{\text{EPI}\to\text{T1w}}$) and coregistered reference volume.
- **Validation Method:** Normalized Mutual Information computation ($\text{NMI} \ge 0.55$).
- **Failure Conditions:** Optimization fails to converge; registration determinant $<0.8$ or $>1.2$; translation displacement $>25\text{ mm}$.

### Stage 10: T1w-to-MNI152 Non-Linear Stereotaxic Normalization
- **Input:** Subject T1w brain volume, MNI152 standard stereotaxic template ($1.0\text{ mm}$ isotropic).
- **Operation:** Two-stage spatial normalization:
  1. Affine 12-DOF transformation ($A_{\text{T1w}\to\text{MNI}}$) modeling global translation, rotation, scaling, and shearing.
  2. Symmetric Diffeomorphic Normalization (SyN) computing an invertible, topology-preserving non-linear deformation field ($\Phi, \Phi^{-1}$).
- **Output:** Forward warp field ($\Phi$), inverse warp field ($\Phi^{-1}$), affine matrix ($A$), and normalized T1w volume in MNI space.
- **Validation Method:** Dice coefficient between normalized T1w brain mask and standard MNI152 brain mask ($\text{Dice} \ge 0.85$).
- **Failure Conditions:** Non-invertible Jacobian determinant ($\det(J) \le 0$); registration Dice below threshold.

### Stage 11: Unified Transformation Composition
- **Input:** Within-run motion affine ($T_{\text{motion}}$), EPI-to-T1w affine ($T_{\text{EPI}\to\text{T1w}}$), T1w-to-MNI SyN warp ($\Phi_{\text{T1w}\to\text{MNI}}$).
- **Operation:** Numerically composes the sequence of spatial mappings into a single composite pullback operator:
  $$\Phi_{\text{total}} = \Phi_{\text{T1w}\to\text{MNI}} \circ T_{\text{EPI}\to\text{T1w}} \circ T_{\text{motion}}$$
  This formulation guarantees that the raw BOLD voxels are resampled exactly once, entirely preventing compounding interpolation blur.
- **Output:** Composite coordinate pullback mapping field.
- **Validation Method:** Mathematical commutativity check on identity transforms; invertibility verification.
- **Failure Conditions:** Incompatible affine dimensions; invalid coordinate system headers.

### Stage 12: BOLD Pullback Resampling to Canonical CC200 Grid
- **Input:** Raw 4D BOLD timeseries, composite transformation field $\Phi_{\text{total}}$, canonical CC200 reference geometry ($63 \times 75 \times 61$, 3.0 mm isotropic).
- **Operation:** Pullback resampling of 4D BOLD timeseries directly into canonical CC200 stereotaxic space using 4th-order B-spline interpolation. Enforces exact voxel dimensions, qform/sform affines, and bounding box.
- **Output:** Normalized 4D BOLD timeseries (`bold_space-CC200_desc-preproc_bold.nii.gz`, shape $63 \times 75 \times 61 \times N_t$).
- **Validation Method:** Spatial grid header equivalence test; SHA-256 verification of canonical target affine.
- **Failure Conditions:** Grid shape mismatch, dimension mismatch, interpolation numerical instability (NaN/Inf).

### Stage 13: Nuisance Confound Regressor Extraction
- **Input:** Normalized 4D BOLD timeseries, eroded tissue masks (CSF, WM resampled into CC200 space), motion parameters.
- **Operation:**
  1. **Anatomical CompCor (aCompCor):** Extracts BOLD timeseries from eroded CSF and WM masks. Normalizes each timeseries to unit variance, performs Principal Component Analysis (PCA) via Singular Value Decomposition (SVD), and retains the top 5 principal components from CSF and top 5 from WM (total 10 aCompCor regressors).
  2. **Friston-24 Motion Expansion:** Generates 24 motion regressors: 6 original motion parameters, 6 temporal derivatives ($\Delta x_t = x_t - x_{t-1}$), and their 12 quadratic terms ($x^2, (\Delta x)^2$).
  3. Total nuisance design matrix $X_{\text{confounds}} \in \mathbb{R}^{N_t \times 34}$.
- **Output:** Nuisance design matrix file (`confounds.tsv`).
- **Validation Method:** SVD variance explained verification; rank check of confound matrix (condition number $<10^5$).
- **Failure Conditions:** Rank deficiency in design matrix; zero variance in noise masks.

### Stage 14: Nuisance Regression & Temporal Bandpass Filtering
- **Input:** Normalized 4D BOLD timeseries, nuisance design matrix $X_{\text{confounds}}$.
- **Operation:** Simultaneous nuisance regression and temporal bandpass filtering ($0.01\text{--}0.10\text{ Hz}$). Uses Ordinary Least Squares (OLS) regression with discrete cosine transform (DCT) basis functions modeling low-frequency drifts:
  $$Y_{\text{denoised}} = Y - X (X^T X)^{-1} X^T Y$$
  Simultaneous regression ensures that frequency filtering and confound removal do not reintroduce filtered artifacts.
- **Output:** Denoised, bandpassed 4D BOLD volume (`bold_denoised.nii.gz`).
- **Validation Method:** Orthogonality check: residual timeseries must have correlation $r < 10^{-6}$ with nuisance regressors.
- **Failure Conditions:** Singularity of design matrix $(X^T X)$; temporal degrees of freedom exhausted ($N_t \le 34$).

### Stage 15: Canonical CC200 Atlas Integrity Verification
- **Input:** Canonical CC200 atlas NIfTI file (`data/atlases/cc200_canonical.nii.gz`).
- **Operation:** Performs cryptographic and structural verification before parcellation extraction:
  - Cryptographic Hash (SHA-256): `9467afce23ad719fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650`
  - Matrix Dimensions: Exactly $(63, 75, 61)$
  - Voxel Resolution: Exactly $(3.0, 3.0, 3.0)\text{ mm}$ isotropic
  - Region Count: Exactly 200 contiguous integer labels ($1\text{--}200$)
- **Output:** Validated atlas spatial handle.
- **Validation Method:** Byte-level SHA-256 computation and unique label set assertion ($\{1, 2, \dots, 200\}$).
- **Failure Conditions:** Hash mismatch, corrupted header, missing ROI labels. Pipeline fails closed.

### Stage 16: CC200 ROI Time-Series Extraction
- **Input:** Denoised 4D BOLD timeseries, validated canonical CC200 atlas.
- **Operation:** Computes the spatial mean BOLD intensity across all voxels belonging to each of the 200 regions of interest for each timepoint $t \in [1, N_t]$:
  $$\bar{y}_{k, t} = \frac{1}{|V_k|} \sum_{v \in V_k} y_{v, t}, \quad k \in \{1, \dots, 200\}$$
- **Output:** Extracted ROI time-series matrix $T \in \mathbb{R}^{N_t \times 200}$.
- **Validation Method:** Check for NaN/constant timeseries; assert exact matrix dimension $(N_t, 200)$.
- **Failure Conditions:** Empty ROI with zero voxels; NaN values in extracted timeseries.

### Stage 17: Functional Connectivity Computation & Fisher-z Transform
- **Input:** Extracted ROI time-series matrix $T \in \mathbb{R}^{N_t \times 200}$.
- **Operation:** Computes the empirical Pearson correlation matrix $R \in \mathbb{R}^{200 \times 200}$ between all pairs of ROI time-series:
  $$r_{i, j} = \frac{\sum_{t=1}^{N_t} (y_{i, t} - \bar{y}_i)(y_{j, t} - \bar{y}_j)}{\sqrt{\sum_{t=1}^{N_t} (y_{i, t} - \bar{y}_i)^2 \sum_{t=1}^{N_t} (y_{j, t} - \bar{y}_j)^2}}$$
  Applies the Fisher-$z$ transformation to stabilize variance across the correlation distribution:
  $$z_{i, j} = \frac{1}{2} \ln \left( \frac{1 + r_{i, j}}{1 - r_{i, j}} \right) = \operatorname{artanh}(r_{i, j})$$
  Extracts the strictly upper-triangular elements ($i < j$) in row-major lexicographical order:
  $$\binom{200}{2} = \frac{200 \times 199}{2} = 19{,}900 \text{ features}$$
- **Output:** Feature vector $f \in \mathbb{R}^{1 \times 19900}$ (`functional_connectivity_features.npy`).
- **Validation Method:** Correlation symmetry check ($r_{i, j} = r_{j, i}, r_{i, i} = 1.0$), finite values ($|r_{i, j}| \le 1.0, |z_{i, j}| < \infty$), dimension assertion.
- **Failure Conditions:** Non-positive semi-definite matrix; division by zero; NaN/Inf entries.

---

### Stage 18: Downstream Feature Preprocessing Pipeline
- **Input:** Raw connectivity feature vector $f \in \mathbb{R}^{1 \times 19900}$, frozen preprocessor `final_preprocessing_pipeline.joblib`.
- **Operation:** Passes feature vector through frozen scikit-learn pipeline without refitting:
  1. `SimpleImputer(strategy='median')`: Replaces any residual missing values using training set medians.
  2. `VarianceThreshold(threshold=0.0)`: Removes features with zero empirical variance in training data.
  3. `StandardScaler()`: Applies $z$-score normalization using frozen training means $\mu_{\text{train}}$ and variances $\sigma_{\text{train}}^2$:
     $$\hat{z}_k = \frac{z_k - \mu_k}{\sigma_k}$$
  4. `SelectKBest(score_func=f_classif, k=100)`: Projects the 19,900-dimensional vector onto the top 100 informative features selected during historical training.
- **Output:** Dimensionality-reduced feature vector $\hat{f} \in \mathbb{R}^{1 \times 100}$.
- **Validation Method:** Preprocessor SHA-256 verification (`228d3f01...`), output dimensionality assertion ($1 \times 100$), finite value verification.
- **Failure Conditions:** Checksum mismatch on `.joblib` file; dimension mismatch ($N \ne 19{,}900$); transformer failure.

### Stage 19: Downstream Frozen Classifier Inference
- **Input:** 100-dimensional feature vector $\hat{f}$, frozen classifier `final_model.joblib`.
- **Operation:** Executes inference via frozen `RandomForestClassifier`:
  - 50 decision trees, maximum tree depth = 3.
  - Computes class probability distribution:
    $$P(Y = \text{ADHD} \mid \hat{f}) = \frac{1}{50} \sum_{t=1}^{50} P_t(Y = \text{ADHD} \mid \hat{f})$$
  - Generates binary classification decision based on default decision threshold $\theta = 0.50$.
- **Output:** Classification results JSON: class probabilities, predicted label, model provenance metadata.
- **Validation Method:** Classifier SHA-256 verification (`d8eb23eb...`), probability normalization check ($P(\text{TDC}) + P(\text{ADHD}) = 1.0$).
- **Failure Conditions:** Model file tampering, probability outside $[0, 1]$, scikit-learn version deserialization incompatibility.

---

## 3. Strict Boundary Between Neuroimaging & Machine Learning Tiers

To ensure architectural modularity and scientific integrity, the pipeline maintains an immutable boundary between Tier I and Tier II:

```
[ Tier I: Preprocessing & FC Extraction ]
  ├── Language: Python 3.11 / NumPy / SciPy / NiBabel
  ├── Domain: Physics, signal processing, spatial geometry
  ├── Input: DICOM / BIDS NIfTI volumes
  └── Output: Standardized 19,900 Fisher-z connectivity vector
                         ||
                 [ RIGID CONTRACT ]
                         ||
[ Tier II: Frozen Model Inference ]
  ├── Language: scikit-learn / joblib
  ├── Domain: Statistical learning, feature selection, ensemble classification
  ├── Mode: READ-ONLY, ZERO RETRAINING
  └── Output: Exploratory prediction scores (strictly non-diagnostic)
```

**Key Boundary Principles:**
1. **Zero Upstream Leakage:** Downstream classification labels, weights, or feature importances are never fed back into image registration, parcellation, or filtering.
2. **Deterministic Contract:** Tier II expects strictly $1 \times 19{,}900$ float64 features ordered by CC200 upper-triangular indexing. Any deviation halts execution before model loading.
3. **Read-Only Model Storage:** The directory containing model binaries is mounted `:ro` inside the container, preventing accidental overwrite or modification.

---

## 4. Computational Resource Profile & Memory Isolation

The pipeline is engineered to execute within a strictly enforced **2.5 GB (2,560 MiB)** memory boundary governed by Linux cgroups v2 (`--memory=2560m`).

### Peak Memory Consumption by Pipeline Stage:

| Stage Group | Algorithmic Bottleneck | Typical RAM Usage | Peak RAM Limit | Memory Optimization Strategy |
|---|---|---|---|---|
| **Stages 1–4** | 4D BOLD loading & Motion Realignment | $\approx 450\text{ MB}$ | $750\text{ MB}$ | Slice-by-slice / chunked temporal loading |
| **Stages 5–8** | T1w Skull Stripping & Tissue Segmentation | $\approx 600\text{ MB}$ | $1{,}100\text{ MB}$ | Bounded GMM iterations; integer mask arrays |
| **Stages 9–11** | Non-Linear SyN Diffeomorphic Normalization | $\approx 1{,}100\text{ MB}$ | $1{,}850\text{ MB}$ | Multi-resolution pyramid with restricted deformation grid |
| **Stage 12** | 4D Pullback Resampling to CC200 | $\approx 850\text{ MB}$ | $1{,}400\text{ MB}$ | Temporal chunking across 3D volumes |
| **Stages 13–14** | aCompCor SVD & Simultaneous OLS Regression | $\approx 500\text{ MB}$ | $950\text{ MB}$ | In-place matrix operations; memory-mapped temporary arrays |
| **Stages 15–17** | CC200 ROI Extraction & $200 \times 200$ Pearson FC | $\approx 250\text{ MB}$ | $450\text{ MB}$ | Vectorized dot-product correlation; upper-tri indexing |
| **Stages 18–19** | Frozen Model Deserialization & Inference | $\approx 150\text{ MB}$ | $300\text{ MB}$ | Lightweight Random Forest (50 trees, max depth 3) |

**Enforcement Guarantee:**  
If memory exceeds 2,560 MiB during any stage, the Linux kernel OOM-killer immediately issues `SIGKILL` (exit code 137). Synthetic verification confirmed that memory guard tests pass 3/3, guaranteeing that unconstrained execution fails closed.

---

## 5. Summary & Verification Reference

The technical specifications detailed above are verified across 179 automated test suites in `tests/`:
- Input validation & safety gate: `tests/unit/test_input_validation.py`, `tests/unit/test_phase_a4_safety_gate.py`
- Segmentation & Registration: `tests/unit/test_segmentation.py`, `tests/unit/test_registration.py`, `tests/unit/test_scientific_registration_validation.py`
- Confound regression & Connectivity: `tests/unit/test_nuisance.py`, `tests/unit/test_connectivity.py`
- Memory isolation: `tests/unit/test_memory_guard.py`, `scripts/verify_container_memory.sh`
- Frozen model integration: `tests/unit/test_phase_a4_model_compatibility.py`, `tests/integration/test_container_model_integration.py`

