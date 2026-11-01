# RISKS, ASSUMPTIONS, AND OPEN QUESTIONS

## 1. Scientific & Methodological Risks

### 1.1 CC200 Atlas Identification & Label Ordering
- **Risk**: The Craddock 200 (CC200) functional clustering atlas exists in multiple spatial resolutions (e.g. 1mm, 2mm, 4mm) and parcellation clusterings (spectral vs two-level clustering, 200 vs 400 regions). If the parcellation mask applied to new scans does not match the exact spatial boundaries and label indices ($1 \dots 200$) of the historical C-PAC mask, the extracted 19,900 edges will represent biologically different regional pairings.
- **Mitigation**: The NIfTI CC200 mask must be verified against an official ADHD-200 / FCP-INDI C-PAC release resource before running Phase 2.

### 1.2 Constant / Zero-Variance Signal Dropout
- **Risk**: Resting-state fMRI frequently experiences susceptibility-induced BOLD signal dropout in the orbitofrontal cortex and inferior temporal lobes. If an ROI falls entirely within a dropout zone or outside the field of view (FOV), its signal variance is zero. As observed in historical subject `0015038` (which was discarded because 83 ROIs were constant), Pearson correlation with a zero-variance signal is mathematically undefined ($0/0$).
- **Mitigation**: The validation module enforces strict pre-extraction coverage checks and rejects any scan with $\sigma_k == 0.0$ before computing correlations.

### 1.3 Confound Regression Discrepancies
- **Risk**: Historical C-PAC used a specific 2015 Nipype implementation of aCompCor (5 components, 0.96 CSF threshold, 0.96 WM threshold). Subtle differences in tissue segmentation algorithms (e.g. FAST vs modern deep-learning segmenters) or erosion kernels will slightly alter nuisance regressors.
- **Mitigation**: In Phase 2, feature concordance will be benchmarked on a dual-processed scan (`1017176_session_1`), establishing an empirical tolerance window for edge correlation.

---

## 2. Engineering & Operational Risks

### 2.1 Host Memory (8 GB RAM) Exhaustion
- **Risk**: Apple Silicon Unified Memory is shared between GPU and system RAM. Processing full 4D BOLD scans in memory alongside spatial registration algorithms can easily exceed 4 GB if arrays are duplicated.
- **Mitigation**: 
  - Execute pipelines strictly one subject at a time.
  - Delete 4D intermediates immediately after extracting 2D ROI timeseries.
  - Employ in-place array operations and explicitly call `gc.collect()`.

### 2.2 Inactive Docker Daemon
- **Risk**: Docker daemon is currently stopped. Even if started, running x86_64 C-PAC 0.3.9.1 via Rosetta emulation on ARM64 poses performance penalties and stability hazards.
- **Mitigation**: Adopt a native Python 3 pipeline using standard scientific neuroimaging packages (`nibabel`, `nilearn`, `scipy`) that execute natively on ARM64.

### 2.3 Accidental Test Set Contamination
- **Risk**: The 20 locked test subjects from Peking_2 must remain strictly unseen. Accidental inclusion in preprocessing calibration would invalidate external test generalization.
- **Mitigation**: Hardcode explicit exclusion of all Peking_2 subject IDs from calibration and validation scripts.

---

## 3. Open Technical Questions

1. **CC200 Template Origin**: Can the exact historical NIfTI atlas file used by C-PAC 0.3.9.1 be retrieved from FCP-INDI S3 resources (`s3://fcp-indi/resources/`) or the original Craddock lab distribution?
2. **Slice Timing Correction**: Was slice timing correction applied uniformly across historical ADHD-200 sites, or omitted due to varying multi-site inter-slice acquisition times?
3. **Motion Outlier Scrubbing**: Did C-PAC apply scrubbing/censoring before computing ROI time series, or was confound regression alone relied upon?
