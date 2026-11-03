# NeuroADHD_RawMRI

**Automated Resting-State fMRI Preprocessing & CC200 Feature Extraction Pipeline for the NeuroADHD Research Framework**

---

## 1. Project Purpose & Scope

The **NeuroADHD** machine-learning classifier was trained on resting-state functional MRI (fMRI) functional connectivity features derived from the **Craddock 200 (CC200)** atlas. The current production model accepts pre-extracted regional time-series CSV tables (`roi_CC200.csv`), transforms them into 19,900 Fisher-$z$ Pearson correlation features, selects 100 features via an ANOVA $F$-statistic pipeline, and executes inference using a frozen Random Forest classifier.

**NeuroADHD_RawMRI** is an isolated, specialized project designed to build the missing upstream capability:
Ingesting raw 4D resting-state fMRI scans in BIDS format, executing standardized spatial and temporal preprocessing compatible with the historical training data, and generating valid `roi_CC200.csv` tables to feed into the frozen downstream model.

> [!CAUTION]
> **RESEARCH NOTICE — NOT A CLINICAL DIAGNOSTIC TOOL**
> 
> Attention-Deficit/Hyperactivity Disorder (ADHD) is a complex neurodevelopmental condition diagnosed exclusively by licensed clinicians through formal clinical evaluations, developmental histories, and behavioral assessments.
> 
> This software is an exploratory neuroscience research tool. It is **NOT** a medical diagnostic device, has not been cleared or approved by any regulatory authority (e.g. FDA or CE), and must **NEVER** be used for clinical decision-making, patient diagnosis, risk assessment, or medical triage.

---

## 2. Current Status: SYNTHETIC PIPELINE + LOCAL MRI INTAKE

> [!IMPORTANT]
> **STATUS: Synthetic preprocessing demonstration and local upload validation are implemented; real-participant processing remains locked.**
>
> The local dashboard accepts complete BIDS-style MRI session ZIP files, individual NIfTI/DICOM files, and separate PDF/JPEG/PNG report uploads. It validates scan structure, NIfTI dimensions and BOLD JSON metadata. Report text extraction uses optional local PDF/OCR tools.
>
> **A successful upload validation is not a successful preprocessing run.** Real-scan preprocessing, DICOM conversion, and real-participant model inference remain disabled by the research authorization gate. The synthetic runner continues to process generated synthetic data only.

---

## 3. Workflow Architecture

```
RAW RESTING-STATE fMRI SCAN (BIDS format: *_bold.nii.gz + *.json)
                           ↓
             RAW DATA & METADATA VALIDATION
                           ↓
             STANDARDIZED fMRI PREPROCESSING
  - Motion Correction & Friston 24-parameter matrix
  - Non-linear MNI152 Standard Space Normalization
  - Linear & Quadratic Detrending
  - 5 aCompCor Confound Regressors (CSF/WM masks)
  - 0.01 - 0.10 Hz Temporal Bandpass Filtering
  - Global Signal Regression: OFF (global0)
                           ↓
             CC200 ATLAS PARCELLATION
  - Mean BOLD extraction across 200 Craddock ROIs
                           ↓
                     roi_CC200.csv
  - Exactly 200 ROI rows, header: node/volume,0,1,2,...,T-1
                           ↓
        19,900 PAIRWISE PEARSON CORRELATIONS (Fisher z)
                           ↓
        EXISTING FROZEN PREPROCESSING PIPELINE
  - Median Imputation -> Variance Filter -> Scaler -> SelectKBest 100
                           ↓
          EXISTING TRAINED RANDOM FOREST
                           ↓
     EXPLORATORY ADHD-PATTERN RESEARCH PREDICTION
```

---

## 4. Setup & Prerequisites

### 4.1 System Environment (Verified on Host)
- **OS**: macOS Darwin (Apple Silicon ARM64 / M2)
- **Memory**: 8 GB Physical RAM (Peak memory limit per scan: 2.5 GB)
- **Disk Space**: 274 GiB available SSD storage
- **Python**: Python 3.10+ / 3.11+

### 4.2 Installation

```bash
# Create and activate an isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements-lock.txt
```

### 4.3 Local Dashboard and MRI Intake

Start the local dashboard:

```bash
python scripts/serve_dashboard.py
```

Open http://127.0.0.1:8765. The preferred upload is one complete BIDS-style session ZIP containing a 4D resting-state BOLD NIfTI, its JSON sidecar, and a 3D T1w NIfTI. Individual NIfTI/DICOM files and separate PDF/JPEG/PNG reports are also accepted. See [UPLOAD_AND_INTAKE.md](docs/UPLOAD_AND_INTAKE.md) for input structure, validation rules, report extraction prerequisites, limits, and safety boundaries.

Uploads are staged locally under outputs/intake_sessions/. This local intake feature does not authorize or launch real-scan preprocessing.

---

## 5. Input and Output Specifications

### 5.1 Intended Inputs
- BIDS-formatted directory containing:
  - 4D functional BOLD scan (`*_task-rest_bold.nii.gz`) with $T \ge 50$ timepoints.
  - JSON sidecar (`*_task-rest_bold.json`) containing `RepetitionTime` ($0.5 \le T_R \le 5.0\text{ s}$).
  - Required anatomical scan (`*_T1w.nii.gz`) for anatomical processing and registration.

### 5.2 Intended Outputs
- Cleaned ROI timeseries: `roi_CC200.csv` (200 rows $\times$ $T$ columns).
- Motion & confound metrics: `FD.1D`, `motion_parameters.txt`, `confounds.tsv`.
- Processing audit report: `processing_manifest.json` with SHA-256 checksums and parameter records.

---

## 6. Commands Planned for Future Phases

The following commands are planned for execution upon explicit user approval of subsequent phases:

```bash
# PHASE 2: Execute single-scan proof of concept on verified test scan
python scripts/run_single_scan_poc.py --bids-dir /path/to/sub-01 --output-dir outputs/sub-01

# PHASE 3: Verify feature-level concordance against historical CC200 output
python scripts/verify_feature_compatibility.py --new-roi outputs/sub-01/roi_CC200.csv --historical-roi /path/to/historical/roi_CC200.csv

# Run automated validation test suite
pytest tests/unit -v
```

---

## 7. Project Documentation Sitemap

Detailed documentation is available in the `docs/` folder:
- [UPLOAD_AND_INTAKE.md](docs/UPLOAD_AND_INTAKE.md): Supported scan uploads, BIDS session structure, report extraction, validation checks, and remaining authorization gates.
- [PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md): Executive summary, scope, and non-clinical disclaimer.
- [EXISTING_PROJECT_ARCHITECTURE.md](docs/EXISTING_PROJECT_ARCHITECTURE.md): Audit of existing code, models, and endpoints.
- [HISTORICAL_PREPROCESSING_AUDIT.md](docs/HISTORICAL_PREPROCESSING_AUDIT.md): Catalog of historical C-PAC parameters and evidence.
- [DATA_INPUT_SPECIFICATION.md](docs/DATA_INPUT_SPECIFICATION.md): Strict BIDS requirements and metadata rules.
- [RAW_FMRI_PIPELINE_DESIGN.md](docs/RAW_FMRI_PIPELINE_DESIGN.md): Step-by-step pipeline design and toolchain analysis.
- [FEATURE_COMPATIBILITY_PLAN.md](docs/FEATURE_COMPATIBILITY_PLAN.md): Mathematical formulations and frozen pipeline interface.
- [VALIDATION_AND_TEST_PLAN.md](docs/VALIDATION_AND_TEST_PLAN.md): Comprehensive test matrix and error conditions.
- [REJECTED_POC_FORENSIC_AUDIT.md](docs/REJECTED_POC_FORENSIC_AUDIT.md): Forensic audit of the rejected PoC v1 prototype.
- [CORRECTED_POC_V2_DESIGN.md](docs/CORRECTED_POC_V2_DESIGN.md): Architectural design specification for Corrected PoC V2.
- [POC_V2_IMPLEMENTATION_REPORT.md](docs/POC_V2_IMPLEMENTATION_REPORT.md): Implementation details, synthetic tests, and memory benchmarks.
- [HARDWARE_AND_RESOURCE_ESTIMATES.md](docs/HARDWARE_AND_RESOURCE_ESTIMATES.md): RAM budgeting and disk pruning policy.
- [RISKS_AND_OPEN_QUESTIONS.md](docs/RISKS_AND_OPEN_QUESTIONS.md): Technical blockers, open items, and risks.
- [MERGE_AND_ROLLBACK_PLAN.md](docs/MERGE_AND_ROLLBACK_PLAN.md): Phase 5 integration and rollback strategy.

---

## 8. Changelog

- **2026-11-03 (Local MRI Intake UI & Validation)**:
  - Added the complete BIDS-style MRI session ZIP as the primary dashboard upload.
  - Added individual NIfTI, JSON and DICOM file staging.
  - Added safe ZIP extraction, NIfTI dimensionality checks, BOLD time-point checks and RepetitionTime validation.
  - Added a separate PDF/JPEG/PNG report extraction path with optional local PDF/OCR engines.
  - Kept real-participant preprocessing and model inference disabled behind the existing research authorization gate.

- **2026-11-01 (Phase 2 Corrected PoC V2 Implementation & Synthetic Validation)**:
  - Implemented spatial coregistration and MNI SyN normalization in `src/preprocessing/registration.py`.
  - Implemented single-step pullback resampling to eliminate multi-stage interpolation blurring.
  - Implemented anatomical brain extraction, 3D Legendre polynomial bias-field correction, and tissue segmentation (CSF/GM/WM) in `src/preprocessing/segmentation.py`.
  - Enforced strict historical 0.96 tissue probability thresholding and 1-voxel 3D morphological erosion.
  - Updated nuisance model in `src/preprocessing/nuisance.py` with 5 aCompCor components extracted strictly from eroded anatomical tissue masks (31 confound columns total, GSR OFF, mean tissue signals OFF).
  - Implemented orthogonal slice contour inspection figure generator in `src/preprocessing/qc.py`.
  - Created comprehensive test suite (34 tests, 89% coverage) covering registration, segmentation, nuisance regression, and an end-to-end synthetic pipeline.
  - Benchmarked peak RAM at 0.29 GB (well below the 2.50 GB ceiling).
  - Single-subject raw scan execution paused awaiting code review.
- **2026-10-21 (Phase 1 Completed)**:
  - Established isolated `NeuroADHD_RawMRI` project workspace.
  - Completed read-only audit of `NeuroADHD`, `NeuroADHD_ModelBenchmark`, `NeuroADHD_ModelDashboard`, and `ADHD200_Official`.
  - Cataloged historical C-PAC preprocessing parameters with strict status classifications (`VERIFIED`, `PROVISIONAL`, `NOT VERIFIED`).
  - Generated full architectural documentation suite and configuration schemas.
  - Completed hardware audit and established single-scan memory constraints for 8 GB RAM.
