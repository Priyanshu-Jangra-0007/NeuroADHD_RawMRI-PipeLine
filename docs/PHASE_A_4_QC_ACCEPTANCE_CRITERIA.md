# PHASE A.4: QUALITY CONTROL (QC) ACCEPTANCE CRITERIA MATRIX

**Document Version:** 1.0.0  
**Phase:** Phase A.4 — Pre-Pilot Readiness Gate  
**Status:** DRAFT — PENDING SUPERVISOR REVIEW & APPROVAL  
**Policy Level:** MANDATORY SCIENTIFIC GATING SPECIFICATION  
**Execution Environment:** Strictly Synthetic-Only

---

## 1. Overview & Quality Assurance Philosophy

To ensure that real MRI participant scans are safe to process and do not inject corrupted, misaligned, or artifact-laden signals into downstream models, the **NeuroADHD RawMRI Pipeline** establishes an automated multi-stage Quality Control (QC) gating protocol.

Every scan must pass through 9 hierarchical quality checkpoints. Scans failing any mandatory gate are automatically rejected, and intermediate staging directories are quarantined.

> [!IMPORTANT]
> To prevent arbitrary or unscientific cutoff choices, all thresholds that have not been empirically validated across the ADHD-200 benchmark cohort are formally designated as `PENDING_SUPERVISOR_APPROVAL`.

---

## 2. Comprehensive QC Acceptance Criteria Matrix

| # | QC Category & Metric | Calculation Method / Formula | Interpretation | Proposed Rule | Approval Status |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Input File Integrity** | File existence, non-empty size, readable gzip/NIfTI header. | Ensures file is not truncated or corrupted during transfer. | Pass if valid NIfTI header; reject otherwise. | **ESTABLISHED** |
| **2** | **Signal Finiteness** | Count of NaN, Inf, or null voxel values inside image array: $\sum (\text{isnan}(V) \lor \text{isinf}(V))$. | Unphysical values corrupt convolution and matrix operations. | Strictly 0 NaNs/Infs within brain volume; reject if $>0$. | **ESTABLISHED** |
| **3** | **BOLD Temporal Length** | Count of 3D temporal volumes: $T = \text{shape}[3]$. | Sufficient sample size for covariance estimation ($T \ge 50$). | Reject if $T < 50$ volumes. | **ESTABLISHED** |
| **4** | **Repetition Time ($T_R$)** | Parsed from BIDS JSON sidecar `RepetitionTime`. | Temporal sampling rate for filter cutoffs and Nyquist limit. | $0.5\text{ s} \le T_R \le 3.5\text{ s}$; reject if missing or $\le 0$. | **ESTABLISHED** |
| **5** | **Affine Invertibility** | Determinant of upper $3 \times 3$ affine matrix: $\det(A_{3 \times 3})$. | Invertible affine required for spatial transformations. | Reject if $\det(A) = 0$ or condition number $\kappa(A) > 10^5$. | **ESTABLISHED** |
| **6** | **T1w Brain Volume** | Total intracranial volume (TIV) from brain mask: $V = N_{\text{voxels}} \times \Delta x \Delta y \Delta z$. | Verifies skull-stripping did not severely crop or under-strip brain. | $600\text{ cm}^3 \le V \le 2200\text{ cm}^3$; flag for review if outside. | `PENDING_SUPERVISOR_APPROVAL` |
| **7** | **EPI-T1w Coregistration Mutual Information** | Normalized Mutual Information (NMI) between mean EPI and skull-stripped T1w. | Measures spatial alignment between functional and structural volumes. | NMI $\ge 0.55$; reject scan if $\text{NMI} < 0.50$. | `PENDING_SUPERVISOR_APPROVAL` |
| **8** | **T1w-MNI SyN Overlap (Dice)** | Dice coefficient between warped T1w brain and MNI template mask: $\frac{2\|A \cap B\|}{\|A\| + \|B\|}$. | Measures non-linear registration fidelity into standard MNI space. | Dice $\ge 0.85$; reject scan if $\text{Dice} < 0.80$. | `PENDING_SUPERVISOR_APPROVAL` |
| **9** | **Mean Framewise Displacement (FD)** | Power et al. (2012) formulation: $\text{FD}_t = \sum \|\Delta d_i\| + \sum r \left\|\frac{\pi}{180}\Delta \theta_i\right\|$. | Overall subject head movement across the scan duration. | Mean $\text{FD} \le 0.50\text{ mm}$; reject if $> 0.50\text{ mm}$. | `PENDING_SUPERVISOR_APPROVAL` |
| **10**| **Maximum Motion Spike** | $\max_t(\text{FD}_t)$ across all timepoints. | Transient catastrophic movement (coughing, shifting). | Flag if $\max(\text{FD}) > 3.0\text{ mm}$; reject if $> 5.0\text{ mm}$. | `PENDING_SUPERVISOR_APPROVAL` |
| **11**| **Scrubbing Proportion** | Fraction of volumes with $\text{FD}_t > 0.50\text{ mm}$: $\frac{N_{\text{scrubbed}}}{T}$. | Loss of temporal degrees of freedom from censoring. | Reject if $> 30\%$ of volumes exceed $0.5\text{ mm}$. | `PENDING_SUPERVISOR_APPROVAL` |
| **12**| **Temporal SNR (tSNR)** | Median voxel-wise tSNR: $\text{tSNR} = \frac{\mu_t}{\sigma_t}$ within brain mask. | Measures raw BOLD signal quality and scanner stability. | Median $\text{tSNR} \ge 40$; flag for review if $< 30$. | `PENDING_SUPERVISOR_APPROVAL` |
| **13**| **DVARS Outliers** | Root mean square BOLD intensity change: $\text{DVARS}_t = \sqrt{\frac{1}{V}\sum (\Delta I_t)^2}$. | Detects global signal intensity jumps. | Flag volumes with $\text{DVARS} > 1.5 \times \text{median(DVARS)}$. | `PENDING_SUPERVISOR_APPROVAL` |
| **14**| **Tissue Mask Coverage** | Eroded WM and CSF mask voxel counts. | Guarantees non-empty compartments for CompCor / nuisance regression. | $N_{\text{WM}} \ge 500$ voxels, $N_{\text{CSF}} \ge 100$ voxels; reject if empty. | **ESTABLISHED** |
| **15**| **Canonical Grid Match** | Resampled BOLD spatial dimensions and affine match CC200 template. | Necessary for voxel-to-ROI lookup without implicit resampling. | Strictly $(63, 75, 61)$, voxel $(3.0, 3.0, 3.0)\text{ mm}$, affine match. | **ESTABLISHED** |
| **16**| **CC200 ROI Coverage** | Count of CC200 parcels containing $\ge 5$ functional voxels. | Confirms no ROIs fell outside the functional field of view. | Exactly 200 parcels with non-zero coverage; reject if $<200$. | **ESTABLISHED** |
| **17**| **Constant ROI Signals** | Count of ROIs with temporal standard deviation $\sigma_t = 0.0$. | Zero-variance signals make Pearson correlation undefined ($0/0$). | Strictly 0 constant ROIs; reject if $\ge 1$ constant signal. | **ESTABLISHED** |
| **18**| **Connectivity Finiteness** | Check $\text{all}(\text{isfinite}(R_{\text{sym}}))$ and $\text{all}(\text{isfinite}(z))$. | Prevents invalid floating-point values entering ML classifier. | Strictly 100% finite across all 19,900 edges; reject otherwise. | **ESTABLISHED** |
| **19**| **Fisher-$z$ Dynamic Range** | Min and max values in Fisher-$z$ feature vector. | Extreme values indicate unclipped singularity ($r = \pm 1.0$). | Min $\ge -5.0$, Max $\le +5.0$; strictly $|z| \le \operatorname{arctanh}(1 - 10^{-7}) \approx 8.4$. | **ESTABLISHED** |
| **20**| **Model Feature Dimension** | Length of output feature vector: $\text{len}(z) = \frac{200 \times 199}{2}$. | Structural compatibility with downstream scikit-learn pipeline. | Strictly 19,900 features; reject otherwise. | **ESTABLISHED** |

---

## 3. Visual QC Overlays Protocol

In addition to numerical metric gating, four orthogonal visual QC overlay plots are automatically generated for human inspection:
1. **Coregistration Overlay (`qc_registration_epi_to_t1w.png`)**: Mean EPI contours overlaid on high-resolution T1w anatomical slice. Evaluated for alignment of ventricles, cortical mantle, and corpus callosum.
2. **Normalization Overlay (`qc_registration_t1w_to_mni.png`)**: Warped T1w brain overlaid on MNI152 template. Evaluated for skull-to-cortex boundary and anterior/posterior commissure alignment.
3. **Tissue Masks Overlay (`qc_tissue_masks.png`)**: Eroded white matter (red) and CSF (blue) masks overlaid on T1w anatomical slice. Evaluated for erosion depth and lack of gray-matter encroachment.
4. **CC200 Parcellation Overlay (`qc_parcellation_cc200.png`)**: Parcellation boundary contours overlaid on cleaned BOLD volume. Evaluated for cerebellar, cortical, and subcortical coverage.

---

## 4. Summary of Gating Status

- **Established Rules (Enforced by Code)**: 12 metrics (Items 1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19, 20).
- **Rules Requiring Supervisor Review**: 8 metrics (Items 6, 7, 8, 9, 10, 11, 12, 13) marked `PENDING_SUPERVISOR_APPROVAL`.

