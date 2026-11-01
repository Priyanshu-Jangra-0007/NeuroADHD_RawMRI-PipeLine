# HISTORICAL PREPROCESSING AUDIT

## 1. Objective & Audit Scope

To ensure scientific compatibility, raw fMRI scans processed by the new pipeline must match the preprocessing characteristics of the historical derivatives upon which the NeuroADHD machine learning model was trained.

This audit catalogs every known, provisional, and unresolved parameter from the historical C-PAC pipeline used to generate the ADHD-200 CC200 derivatives. Findings are supported by read-only inspection of the local project repositories and S3 manifests in `../down/ADHD200_Official/` and `../down/NeuroADHD/`.

---

## 2. Parameter Catalog & Status Classification

Every parameter is categorized under one of four strict status labels:
- **`VERIFIED`**: Directly evidenced by local source files, manifests, or S3 execution logs.
- **`PROVISIONAL`**: Documented in historical project notes or literature, but lacking primary file evidence.
- **`NOT VERIFIED`**: Ambiguous, missing, or contradictory across accessible records.
- **`NOT APPLICABLE`**: Irrelevant to the resting-state fMRI BOLD workflow.

---

### Detailed Parameter Matrix

| Category | Parameter | Value / Implementation | Status | Evidence Source |
|---|---|---|---|---|
| **Software Stack** | Pipeline Tool | Configurable Pipeline for the Analysis of Connectomes (C-PAC) | **`VERIFIED`** | `ADHD200_Official/experiments/raw_pilot_18/qc/cpac_recovery_report.md:L30-L37` |
| **Software Stack** | Exact Version | `C-PAC 0.3.9.1` | **`VERIFIED`** | S3 execution stdout logs dated June 7, 2015 (`cpac_recovery_report.md:L33`) |
| **Software Stack** | Python Version | Python 2.7 (Historical runtime) | **`VERIFIED`** | S3 execution stdout logs cited in `cpac_recovery_report.md:L35` |
| **Software Stack** | Registration Tools | FSL 5.0 (`/usr/share/fsl/5.0`) & AFNI 2011 (`AFNI_2011_12_21_1014`) | **`VERIFIED`** | S3 execution stdout logs cited in `cpac_recovery_report.md:L35` |
| **Software Stack** | Container Digest | None recorded / Unpinned | **`NOT VERIFIED`** | `cpac_recovery_report.md:L42-L46` confirms no Docker/Singularity image digest in S3 |
| **Pipeline ID** | Execution Prefix | `pipeline_adhd200-benchmark__freq-filter` | **`VERIFIED`** | `NeuroADHD/data/adhd200/manifest.csv:L8`, S3 object keys |
| **Global Signal** | Global Signal Regression (GSR) | **DISABLED (`global0`)** | **`VERIFIED`** | Derivative S3 URI token `.../global0/...` in `official_cc200_inventory.csv:L10` and `manifest.csv:L8` |
| **Temporal Filter** | Bandpass Frequencies | **0.01 – 0.10 Hz** | **`VERIFIED`** | Derivative S3 URI token `_bandpass_freqs_0.01.0.1` in `manifest.csv:L8` |
| **Nuisance Model** | Linear Detrending | **Enabled (`linear1`)** | **`VERIFIED`** | S3 URI token `...selector_pc10.linear1...` |
| **Nuisance Model** | Quadratic Detrending | **Enabled (`quadratic1`)** | **`VERIFIED`** | S3 URI token `...quadratic1...` |
| **Nuisance Model** | Motion Regressors | **Enabled (`motion1`)** | **`VERIFIED`** | S3 URI token `...motion1...` and presence of `motion_parameters.txt` in inventory |
| **Nuisance Model** | CompCor | **Enabled (`compcor1`)** | **`VERIFIED`** | S3 URI token `...compcor1...` |
| **Nuisance Model** | CompCor Components | **5 components** | **`VERIFIED`** | S3 URI token `_compcor_ncomponents_5_selector_pc10` |
| **Tissue Masks** | Tissue Thresholds | CSF: 0.96, GM: 0.70, WM: 0.96 | **`VERIFIED`** | S3 URI token `_csf_threshold_0.96/_gm_threshold_0.7/_wm_threshold_0.96` |
| **Tissue Masks** | Mean Tissue Regression | **Disabled (`wm0.gm0.csf0`)** | **`VERIFIED`** | S3 URI token `wm0...gm0...csf0` (tissue signals handled by CompCor) |
| **Motion Scrubbing**| Censoring Threshold | Framewise Displacement (FD) generated, scrubbing threshold unverified | **`NOT VERIFIED`** | `FD.1D` files exist in S3, but scrubbing settings prior to extraction are absent from logs |
| **Slice Timing** | Slice Timing Correction | Unverified / site-dependent | **`NOT VERIFIED`** | Acquisition slice orders vary across the 8 ADHD-200 sites; not recorded in C-PAC logs |
| **Initial Volumes** | Dummy Scan Removal | Site-dependent / not uniform | **`NOT VERIFIED`** | Preprocessing logs do not specify universal volume discarding |
| **Parcellation** | Atlas Identifier | Craddock 200 (CC200) functional clustering atlas | **`VERIFIED`** | S3 path `_mask_CC200/roi_CC200.csv`, `NeuroADHD/src/features/extract_cc200_connectivity.py:L18` |
| **Parcellation** | Atlas Resolution | MNI standard space (likely 3mm or 2mm) | **`PROVISIONAL`** | C-PAC default CC200 template is typically 3mm or 2mm MNI152 |
| **Parcellation** | Atlas Mask NIfTI | Exact historical NIfTI file not locally stored | **`UNRESOLVED`** | Must acquire verified Craddock 200 NIfTI parcellation with 1..200 label matching |
| **ROI Output** | Format & Header | 200 rows, header `node/volume,0,1,2,...,T-1` | **`VERIFIED`** | Direct file inspection of `1017176_session_1/roi_CC200.csv` and `0016015_session_1/roi_CC200.csv` |
| **ROI Output** | ROI Label Indexing | Strictly sequential 1 to 200 | **`VERIFIED`** | Checked via `np.arange(1, 201)` in `extract_cc200_connectivity.py:L35` |
| **ROI Output** | Constant Signals | Zero-variance ROIs lead to immediate rejection | **`VERIFIED`** | Subject `0015038` skipped due to 83 constant ROIs (`extract_cc200_connectivity.py:L82`) |

---

## 3. Analysis of Exact Reproduction Blockers

As documented by the prior technical investigation in `ADHD200_Official/experiments/raw_pilot_18/qc/cpac_recovery_report.md`, **exact bitwise reproduction** of the historical 2015 C-PAC execution is impossible due to:
1. **Unrecoverable Run Configuration**: The exact runtime YAML configuration and custom submission environment variables are absent from official S3 logs.
2. **Missing Container Digest**: C-PAC 0.3.9.1 from 2015 has no surviving immutable Docker digest or Apptainer recipe.
3. **Obsolete Execution Environment**: The 2015 stack depends on Python 2.7, FSL 5.0, and AFNI 2011, which cannot run natively on macOS Apple Silicon.

---

## 4. Engineering Strategy: Parameter-Calibrated Modern Implementation

Rather than attempting to resurrect a fragile 2015 Python 2.7 environment, the recommended path is:
1. **Adopt a Modern, Standardized Neuroimaging Framework**: Utilize standard containerized or Python 3 pipelines (e.g. Nipype / nilearn / fMRIPrep) calibrated to match the verified historical parameter set.
2. **Strict Calibration Target**: Configure the pipeline to enforce:
   - Bandpass filter: 0.01 – 0.10 Hz.
   - Nuisance regression: 24 motion regressors + linear/quadratic trends + 5 aCompCor components (CSF 0.96 / WM 0.96 threshold).
   - Global Signal Regression: OFF (`global0`).
   - Normalization: MNI152 standard space.
   - Parcellation: CC200 atlas yielding exactly 200 ROI time series.
3. **One-Scan Validation**: Validate feature distributions and edge correlations against historical CC200 outputs for the same subject in Phase 2.
