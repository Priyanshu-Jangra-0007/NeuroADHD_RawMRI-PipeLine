# MERGE AND ROLLBACK PLAN (PHASE 5 INTEGRATION STRATEGY)

## 1. Overview & Isolation Principles

This document defines the formal integration roadmap for Phase 5. Under current rules (Phase 1), **no code will be merged or modified in the existing NeuroADHD ecosystem**.

When Phase 5 is explicitly authorized, integration will follow strict non-destructive protocols:
1. The existing CSV upload workflow (`POST /api/predict/subject`) must continue functioning without alteration.
2. The raw scan pipeline will be introduced as an independent, modular service.
3. A complete, instantaneous rollback mechanism will be maintained throughout integration.

---

## 2. API Extension Architecture

```
                       +----------------------------------+
                       |   NeuroADHD Dashboard Backend    |
                       +----------------------------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
     [Existing CSV Endpoint]                         [New Raw BIDS Endpoint]
     POST /api/predict/subject                       POST /api/predict/raw
                 |                                             |
       Uploads roi_CC200.csv                         Uploads BIDS Zip / Folder
                 |                                             |
                 v                                             v
     Input Validation (200xT)                        Raw Validation & Preproc
                 |                                             |
                 |                                    Extracts roi_CC200.csv
                 |                                             |
                 +----------------------+----------------------+
                                        |
                                        v
                 +---------------------------------------------+
                 |       FROZEN NEUROADHD INFERENCE CORE       |
                 |  1. extract_connectivity_vector()           |
                 |  2. final_preprocessing_pipeline.joblib     |
                 |  3. final_model.joblib                      |
                 +---------------------------------------------+
```

### 2.1 Proposed New Endpoint Specification
- **Endpoint**: `POST /api/predict/raw`
- **Payload**: `multipart/form-data` with zipped BIDS subject directory or direct `.nii.gz` + `.json`.
- **Enforced Limits**:
  - Max upload size: 250 MB per request.
  - Processing timeout: 10 minutes.
  - Concurrency limit: 1 scan at a time (mutex locked to protect host 8 GB RAM).

---

## 3. Sandboxed Execution & Safe Cleanup

To prevent disk saturation and temporary file leakage:
1. Each incoming raw upload is unpacked into an isolated temporary folder under `/tmp/neuroadhd_raw_<uuid>/` or a dedicated workspace scratch path.
2. If preprocessing succeeds: `roi_CC200.csv` and validation metrics are cached, inference is executed, and all 4D intermediate files are removed immediately in a `finally` block.
3. If preprocessing fails: the exact exception is logged to a structured JSON error response, and the entire temporary folder is purged to free disk space.

---

## 4. Rollback Strategy & Git Reversion

Before modifying any file in `NeuroADHD_ModelDashboard` or `NeuroADHD` during Phase 5:
1. A git tag `pre-raw-integration` will be created in the target repositories.
2. All new raw-scan modules will reside in standalone files (e.g. `src/preprocessing/raw_pipeline.py`).
3. If any regression occurs in existing unit tests or the `/api/predict/subject` endpoint:
   ```bash
   # Instantaneous rollback command
   git checkout pre-raw-integration
   ```
4. Dashboard frontend will feature a feature-flag toggle (`ENABLE_RAW_SCAN_UPLOAD=false`) that can disable raw uploads instantaneously without rebuilding the application.
