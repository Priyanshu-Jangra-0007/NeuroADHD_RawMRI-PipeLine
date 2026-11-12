# PROJECT OVERVIEW: NeuroADHD_RawMRI

## 1. Executive Summary & Objective

**NeuroADHD_RawMRI** is an isolated, research-grade engineering initiative designed to extend the existing **NeuroADHD** machine-learning system. 

The existing NeuroADHD system operates on pre-extracted regional time-series data: it transforms Craddock 200 (CC200) resting-state functional MRI (fMRI) regions of interest (ROIs) into pairwise Pearson correlation features (Fisher $r$-to-$z$ transformed), selects 100 features via an ANOVA $F$-statistic pipeline, and executes an exploratory classification using a frozen Random Forest model.

However, the existing model cannot ingest raw resting-state fMRI scans (`.nii` or `.nii.gz`). The objective of this project is to develop, validate, and document a 22-stage raw resting-state fMRI preprocessing pipeline (`scripts/run_poc_v2.py`) capable of:
1. Ingesting raw 4D BOLD resting-state scans (BIDS standard format) and validating sidecar metadata.
2. Estimating 6-DOF rigid motion parameters on raw uncorrected BOLD prior to temporal interpolation, then executing metadata-driven slice-timing correction (`DEC-STC-01`), B0 susceptibility distortion correction discovery/application (`DEC-SDC-02`), single-step composed spatial pullback resampling (Motion + SDC + EPI-to-T1w Coregistration + T1w-to-MNI SyN), anatomical tissue masking, nuisance regression, bandpass filtering, and standard-space registration compatible with the historical ADHD-200 CC200 derivative release.
3. Extracting 200 CC200 ROI mean BOLD time series adhering strictly to the schema expected by `roi_CC200.csv`.
4. Supplying these extracted time series to the frozen NeuroADHD inference workflow without modifying, refitting, or retraining any existing model.

### 22-Stage Preprocessing & Inference Flow (`scripts/run_poc_v2.py`)

```
RAW 4D RESTING-STATE fMRI (BIDS: *_bold.nii.gz + *.json + *_T1w.nii.gz [+ fmap/])
               ↓
 [Stage 1]  BIDS Input & Sidecar Metadata Validation
               ↓
 [Stage 2]  3D Rigid Motion Parameter Estimation (Raw Uncorrected BOLD)
               ↓
 [Stage 3]  Metadata-Driven Slice-Timing Correction (DEC-STC-01; Original Slice Geometry)
            - Default: SKIPPED_MISSING_TIMINGS / SKIPPED_UNKNOWN_SLICE_AXIS with logged flag if missing SliceTiming/axis
            - Strict: --strict-slice-timing fails closed on missing timing/axis (REQUIRED for real-data mode)
               ↓
 [Stage 4]  B0 Susceptibility Distortion Correction (DEC-SDC-02; Fieldmap Discovery & Estimation)
            - Default: SKIPPED_NO_ASSOCIATED_FIELDMAP / SKIPPED_INSUFFICIENT_METADATA / SKIPPED_TOOL_UNAVAILABLE
            - Strict: --strict-sdc fails closed when SDC cannot be applied (REQUIRED for real-data mode)
            - Limitation: Only direct Hz (or rad/s) fieldmaps are applied; FSL topup/prelude/fugue were never run
               ↓
 [Stage 5-6] Motion Transform (+ Optional SDC) Application & 3D Functional Reference Generation
               ↓
 [Stage 7-10] T1w Brain Extraction, Bias-Field Correction, 3-Class GMM Segmentation & Eroded WM/CSF Masks
               ↓
 [Stage 11-12] EPI-to-T1w 6-DOF Coregistration & T1w-to-MNI152 SyN Nonlinear Normalization
               ↓
 [Stage 13-14] Single-Step Composed Pullback Resampling (Motion + SDC + Coreg + MNI Affine + SyN on STC BOLD) & Mask Resampling
               ↓
 [Stage 15-18] 5 aCompCor + Friston-24 + 31-Confounds Assembly, OLS Nuisance Regression & 0.01-0.10 Hz Bandpass (GSR OFF)
               ↓
 [Stage 19-20] CC200 Atlas Parcellation (roi_CC200.csv: 200 ROIs) & 19,900 Fisher-z Connectivity Features
               ↓
 [Stage 21-22] Orthogonal QC Overlays & Schema-Compliant Provenance JSON Generation (provenance.json, output_manifest.json)
               ↓
 EXISTING FROZEN INFERENCE (Median Imputer -> Variance Filter -> Scaler -> SelectKBest 100 -> 50-Tree Random Forest)
```

### Key Stage Behavior & Scientific Limitations
- **Default vs. Strict Correction Modes (`DEC-STC-01`, `DEC-SDC-02`):** By default, missing `SliceTiming` or unknown slice axis causes Stage 3 STC to log a warning and enter a `SKIPPED_*` status with a provenance flag while preserving raw timings; similarly, missing associated fieldmaps, `PhaseEncodingDirection`, `TotalReadoutTime`, or external FSL binaries causes Stage 4 SDC to log a `SKIPPED_*` status with a provenance flag rather than inventing timings or synthetic fieldmaps. Passing `--strict-slice-timing` and `--strict-sdc` forces fail-closed errors instead of skipping; **real-data mode must use strict mode**.
- **SDC External Tool Limitation:** Only calibrated direct `Hz` (and `rad/s` converted to `Hz`) fieldmaps are applied analytically in Python/SciPy. FSL `topup`, `prelude`, and `fugue` were never run (`SKIPPED_TOOL_UNAVAILABLE` if absent), so PEPOLAR and phase-difference execution paths remain unverified against live FSL binaries.

---

## 2. Strict Non-Clinical Scope & Regulatory Disclaimer

> [!CAUTION]
> **RESEARCH USE ONLY — NOT A CLINICAL DIAGNOSTIC DEVICE**
> 
> Attention-Deficit/Hyperactivity Disorder (ADHD) is a heterogeneous neurodevelopmental condition diagnosed exclusively through structured clinical assessments, developmental histories, behavioral rating scales, and clinical observation by qualified healthcare providers.
> 
> The pipeline developed here and the downstream model it interfaces with are exploratory scientific research tools. They are **NOT** medical devices, are not cleared by the FDA, CE, or any regulatory authority, and must never be used for clinical diagnosis, patient triage, screening, treatment planning, or risk stratification.

---

## 3. Strict Isolation Rules & Safety Principles

During this phase (Phase 1) and until explicit authorization is granted:
1. **Workspace Boundary**: All modifications, scripts, documentation, and configs reside strictly within `<your-project-folder>`.
2. **Read-Only Upstream**: The neighboring projects (`NeuroADHD`, `NeuroADHD_ModelBenchmark`, `NeuroADHD_ModelDashboard`, `ADHD200_Official` located in `../down`) are treated as strictly **READ-ONLY**.
3. **No Retraining / Refitting**: The model checkpoint (`final_model.joblib`) and preprocessing artifact (`final_preprocessing_pipeline.joblib`) remain completely frozen.
4. **No Data Leaks or Test Contamination**: The 20 locked test subjects from Peking_2 will never be accessed or used to tune any preprocessing parameter.
5. **No Premature Execution**: No heavy container downloads, massive raw dataset transfers, or uncalibrated batch processing will occur in this phase.
