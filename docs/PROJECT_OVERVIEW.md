# PROJECT OVERVIEW: NeuroADHD_RawMRI

## 1. Executive Summary & Objective

**NeuroADHD_RawMRI** is an isolated, research-grade engineering initiative designed to extend the existing **NeuroADHD** machine-learning system. 

The existing NeuroADHD system operates on pre-extracted regional time-series data: it transforms Craddock 200 (CC200) resting-state functional MRI (fMRI) regions of interest (ROIs) into pairwise Pearson correlation features (Fisher $r$-to-$z$ transformed), selects 100 features via an ANOVA $F$-statistic pipeline, and executes an exploratory classification using a frozen Random Forest model.

However, the existing model cannot ingest raw resting-state fMRI scans (`.nii` or `.nii.gz`). The objective of this project is to develop, validate, and document a raw resting-state fMRI preprocessing pipeline capable of:
1. Ingesting raw 4D BOLD resting-state scans (BIDS standard format).
2. Applying motion correction, nuisance regression, bandpass filtering, and standard-space registration compatible with the historical ADHD-200 CC200 derivative release.
3. Extracting 200 CC200 ROI mean BOLD time series adhering strictly to the schema expected by `roi_CC200.csv`.
4. Supplying these extracted time series to the frozen NeuroADHD inference workflow without modifying, refitting, or retraining any existing model.

```
RAW 4D RESTING-STATE fMRI (BIDS)
               ↓
 RAW DATA & METADATA VALIDATION
               ↓
 STANDARDIZED fMRI PREPROCESSING (Motion, CompCor, 0.01-0.1Hz, MNI)
               ↓
 CLEANED BOLD IMAGE IN STANDARD SPACE
               ↓
 CC200 ATLAS PARCELLATION
               ↓
 200 ROI MEAN TIME SERIES (roi_CC200.csv)
               ↓
 19,900 PAIRWISE PEARSON CORRELATIONS (Fisher z)
               ↓
 EXISTING FROZEN PREPROCESSING PIPELINE (Median Imputer -> Variance -> Scaler -> SelectKBest 100)
               ↓
 EXISTING TRAINED RANDOM FOREST (50 trees, max_depth 3)
               ↓
 EXPLORATORY RESEARCH PREDICTION & METRICS
```

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
1. **Workspace Boundary**: All modifications, scripts, documentation, and configs reside strictly within `/Users/apple/Documents/Codex/2026-09-29/NeuroADHD_RawMRI`.
2. **Read-Only Upstream**: The neighboring projects (`NeuroADHD`, `NeuroADHD_ModelBenchmark`, `NeuroADHD_ModelDashboard`, `ADHD200_Official` located in `../down`) are treated as strictly **READ-ONLY**.
3. **No Retraining / Refitting**: The model checkpoint (`final_model.joblib`) and preprocessing artifact (`final_preprocessing_pipeline.joblib`) remain completely frozen.
4. **No Data Leaks or Test Contamination**: The 20 locked test subjects from Peking_2 will never be accessed or used to tune any preprocessing parameter.
5. **No Premature Execution**: No heavy container downloads, massive raw dataset transfers, or uncalibrated batch processing will occur in this phase.
