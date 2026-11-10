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

## 2. Current Status: SYNTHETIC PIPELINE + AUTOMATED DICOM-TO-BIDS INTAKE

> [!IMPORTANT]
> **STATUS: Synthetic preprocessing demonstration and automated DICOM-to-BIDS ingestion are implemented; real-participant preprocessing remains locked.**
>
> The local dashboard accepts complete BIDS-style MRI session ZIP files, raw DICOM studies, and separate radiology reports. It automates DICOM series inspection and de-identification via `dcm2niix`, enables explicit T1w and BOLD series selection (with multi-run support), generates standardized BIDS NIfTI datasets, and provides downloadable validated BIDS session ZIPs.
>
> **All MRI data and conversion artifacts are stored strictly outside the repository** in `NEUROADHD_DATA_ROOT` (default: `~/NeuroADHD_Data`).
>
> **A successful BIDS conversion is an ingestion stage, not a preprocessing run.** Real-scan preprocessing, CC200 feature extraction, and real-participant model inference remain disabled by the research authorization gate (`processing_enabled = false`, `real_data_authorized = false`). The synthetic runner continues to process generated synthetic data only.

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

### 4.3 External MRI Storage Configuration

All real MRI data, staged uploads, converted BIDS datasets, downloadable ZIPs, and conversion logs are strictly saved outside the project code tree (default: `~/NeuroADHD_Data` when `NEUROADHD_DATA_ROOT` is unset):

```bash
export NEUROADHD_DATA_ROOT=~/NeuroADHD_Data
```

The external directory structure consists of:
- `dicom_input/`: Staged DICOM studies and uploads.
- `staged_sessions/`: Intake staging sessions.
- `neuroadhd_bids/`: Converted BIDS datasets (`sub-<id>/ses-<id>/...`).
- `validated_sessions/`: Downloadable validated BIDS session ZIPs.
- `conversion_logs/`: Anonymized conversion audit logs.

### 4.4 Local Dashboard and Automated DICOM Intake

Start the local dashboard:

```bash
export NEUROADHD_DATA_ROOT=~/NeuroADHD_Data
python scripts/serve_dashboard.py
```

Open http://127.0.0.1:8765. The dashboard supports:
1. Automated DICOM inspection (detects series, dimensions, TR, and modality without image conversion).
2. De-identification (strips patient names, IDs, DOB, etc.).
3. Explicit series selection (T1w and single/multi-run BOLD).
4. Automated `dcm2niix` conversion to BIDS format (`.nii.gz` and `.json`).
5. BIDS session validation and downloadable validated session ZIPs.

See [UPLOAD_AND_INTAKE.md](docs/UPLOAD_AND_INTAKE.md) for detailed workflow instructions, BIDS layout, validation rules, and safety boundaries.

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

## 6. Pipeline Execution & Automated Tests

### 6.1 Production Pipeline Entry Points
The production synthetic pipeline entry points are:

```bash
# Corrected PoC V2 single-scan synthetic execution
python scripts/run_poc_v2.py --output-dir outputs/synthetic_run --t-vol 60 --seed 42

# Synthetic cohort runner (batch execution)
python scripts/run_synthetic_cohort.py --output-dir outputs/dashboard_cohorts/test_cohort --n-subjects 2 --t-vol 60 --seed 42
```

> [!WARNING]
> **REJECTED V1 PROTOTYPE:**
> `scripts/run_single_scan_poc.py` is the **REJECTED V1 prototype** (see [REJECTED_POC_FORENSIC_AUDIT.md](docs/REJECTED_POC_FORENSIC_AUDIT.md)). It suffered from multi-stage interpolation blurring, incorrect affine composition order, and unverified tissue masks. Do **NOT** use it as an entry point.

### 6.2 Automated Test Suite
Run the automated test suite with the HMAC integrity key:

```bash
NEUROADHD_HMAC_KEY="local_test_key" pytest tests -q
```

**Test Suite Metrics (Measured November 2026):**
- **Collected:** 343 tests
- **Passed:** 336 passed
- **Skipped:** 7 skipped
- **Statement Coverage:** 80% across `src/` (`pytest --cov=src`)

**Running the model-dependent tests:**
- **2 skipped model tests** (`test_container_require_model_inference_fails_closed_without_mount`, etc.) require setting `NEUROADHD_MODEL_DIR` pointing to the trusted frozen model directory (e.g. `NeuroADHD_ModelBenchmark/results/final_locked`).
- **5 skipped Docker tests** require a running Docker daemon and container environment (`docker_run_synthetic.sh`).

---

## 7. Known Scientific Limitation

> [!WARNING]
> **UNPROVEN FEATURE COMPATIBILITY ON REAL PARTICIPANT DATA**
>
> The V2 preprocessing pipeline has been verified against synthetic phantoms and unit tests, but has **never been compared against historical C-PAC CC200 features on a real participant scan** (provenance records `historical_concordance_status: NOT_ASSESSED`). Because real participant execution remains locked under the research authorization gate, feature compatibility between the newly engineered pipeline outputs (`roi_CC200.csv` and resulting Fisher-$z$ connectivity matrices) and the historical training data used to fit the downstream frozen model remains empirically unproven. Investigators must not assume that the frozen model will produce identical probability scores without a formal, authorized concordance audit on real data. See [PHASE_A_9_KNOWN_LIMITATIONS.md](docs/PHASE_A_9_KNOWN_LIMITATIONS.md).

---

## 8. Project Documentation Sitemap

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

## 9. Changelog

- **2026-11-10 (Pipeline Clean-up & Documentation Audit)**:
  - Added automated stale `RUNNING` cohort detection and atomic recovery to `TERMINATED` state in `scripts/serve_dashboard.py`.
  - Added UI indicator in `frontend/index.html` clarifying execution duration for canonical CC200 atlas mode on CPU.
  - Added synthetic phantom data disclaimer to model output panel in `frontend/app.js`.
  - Added formal documentation of unproven feature concordance on real participant data.
  - Expanded test suite to 343 collected tests (336 passed, 7 skipped; 80% statement coverage across `src/`).

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
  - Created comprehensive test suite (initially 34 tests, now expanded to 343 collected tests: 336 passed, 7 skipped with 80% statement coverage as of November 2026).
  - Benchmarked peak RAM at 0.29 GB (well below the 2.50 GB ceiling).
  - Single-subject raw scan execution paused awaiting code review.
- **2026-10-21 (Phase 1 Baseline Audit & Architecture)**:
  - Established isolated `NeuroADHD_RawMRI` project workspace.
  - Completed read-only audit of `NeuroADHD`, `NeuroADHD_ModelBenchmark`, `NeuroADHD_ModelDashboard`, and `ADHD200_Official`.
  - Cataloged historical C-PAC preprocessing parameters with strict status classifications (`VERIFIED`, `PROVISIONAL`, `NOT VERIFIED`).
  - Generated full architectural documentation suite and configuration schemas.
  - Completed hardware audit and established single-scan memory constraints for 8 GB RAM.
