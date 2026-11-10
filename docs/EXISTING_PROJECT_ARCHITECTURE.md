# EXISTING PROJECT ARCHITECTURE: AUDIT & VERIFICATION

This document details the architecture, code implementations, model artifacts, and evaluation parameters of the existing NeuroADHD ecosystem, verified via read-only inspection of the local filesystem.

---

## 1. Local Ecosystem Layout

The existing repositories are located adjacent to the current workspace in `<external-reference-folder>/`:

```
<your-project-folder>/
├── down/
│   ├── NeuroADHD/                  # Core feature extraction, ML preprocessing & inference
│   ├── NeuroADHD_ModelBenchmark/   # Model training runs, frozen artifacts & evaluation reports
│   ├── NeuroADHD_ModelDashboard/   # Full-stack dashboard (FastAPI + React/Vite)
│   └── ADHD200_Official/           # S3 manifests, C-PAC execution audits & metadata
└── NeuroADHD_RawMRI/               # ISOLATED NEW WORKSPACE (Current Root)
```

---

## 2. Core Components & Verified Code Inspection

### 2.1 NeuroADHD Core Inference & Feature Pipeline

- **Feature Extraction Engine**: `NeuroADHD/src/features/extract_cc200_connectivity.py`
  - **Status**: `VERIFIED`
  - **Input Format** (Lines 26–39): Accepts `roi_CC200.csv` with exactly 200 rows. Column 0 is `node/volume` with strictly sorted 1-based integer labels `1, 2, ..., 200`. Remaining columns are BOLD timepoints $0, \dots, T-1$.
  - **Transposition** (Lines 41–44): Values array transposed to shape `(T, 200)` where rows represent timepoints and columns represent ROIs.
  - **Signal Validity & Constant Check** (Lines 77–83): Computes standard deviation per ROI (`np.std(time_by_roi, axis=0, ddof=0)`). If any ROI has $\sigma = 0.0$ (constant signal), the scan is rejected with `ValueError` because Pearson correlation is undefined.
  - **Correlation Matrix** (Lines 85–92): Computes $200 \times 200$ Pearson correlation matrix via `np.corrcoef(..., rowvar=False)`. Enforces symmetry (`(corr + corr.T) / 2.0`) and sets diagonal elements to 1.0.
  - **Upper-Triangle Edge Extraction** (Lines 93–94): Extracts upper triangle without diagonal ($k=1$) via `np.triu_indices(200, k=1)`, generating exactly $\frac{200 \times 199}{2} = 19,900$ edges.
  - **Fisher $r$-to-$z$ Transformation** (Lines 95–97): Clips edge correlations to $[-1 + 10^{-7}, 1 - 10^{-7}]$ (`FISHER_EPSILON = 1e-7`) to prevent infinities, then evaluates $z = \operatorname{arctanh}(r_{\text{clipped}})$.
  - **Feature Naming** (Lines 57–60): Lexicographically ordered pair names `ROI_001__ROI_002` through `ROI_199__ROI_200`.

- **ML Preprocessing Definition**: `NeuroADHD/src/preprocessing/cc200_ml_preprocessing.py`
  - **Status**: `VERIFIED`
  - **Pipeline Steps** (Lines 60–73):
    1. `imputer`: `SimpleImputer(strategy="median", keep_empty_features=True)`
    2. `variance`: `VarianceThreshold(threshold=0.0)`
    3. `scaler`: `StandardScaler()`
    4. `select`: `SelectKBest(score_func=f_classif, k=100)`
  - **Leakage Prevention** (Lines 111–178): Pipeline is fit strictly on training splits; validation and test splits are transformed without fitting.

- **Single-Subject Inference Engine**: `NeuroADHD/src/inference/predict_subject.py`
  - **Status**: `VERIFIED`
  - **Function**: `predict_single_subject(roi_file, pipeline_path, model_path)`
  - **Workflow**:
    1. Validates input CSV (`load_roi_timeseries`).
    2. Computes 19,900 Fisher-z connectivity vector (`extract_connectivity_vector`).
    3. Loads frozen `final_preprocessing_pipeline.joblib` and executes `transform(X_raw)` $\to (1, 100)$.
    4. Loads frozen `final_model.joblib` and calls `predict_proba(X_selected)`.
    5. Returns prediction dictionary with probabilities, classification status, and clinical disclaimer.

- **Inference Unit Tests**: `NeuroADHD/tests/test_inference.py`
  - **Status**: `VERIFIED`
  - Validates full execution on verified test subject `1017176_session_1/roi_CC200.csv`.
  - Confirms rejection of non-existent files and invalid ROI counts.

---

## 3. Frozen Benchmark Artifacts & Evaluation Metrics

- **Artifact Directory**: `NeuroADHD_ModelBenchmark/results/final_locked/`
  - **Status**: `VERIFIED`
  - `final_preprocessing_pipeline.joblib`: 1,115,825 bytes. Pre-fitted 4-step pipeline yielding 100 features.
  - `final_model.joblib`: 74,409 bytes. Fitted `RandomForestClassifier` (50 estimators, `max_depth=3`, `class_weight='balanced'`, `max_features='sqrt'`).
  - `final_test_results.json`: Peking_2 locked test evaluation results:
    ```json
    {
      "run_id": "run_20261004_182318_final_locked",
      "test_site": "Peking_2",
      "test_subject_count": 20,
      "metrics": {
        "accuracy": 0.6,
        "balanced_accuracy": 0.6,
        "precision": 0.6,
        "recall_sensitivity": 0.6,
        "specificity": 0.6,
        "f1_score": 0.6,
        "roc_auc": 0.56,
        "pr_auc": 0.634112926547137,
        "confusion_matrix": {
          "tp": 6, "tn": 6, "fp": 4, "fn": 4
        }
      }
    }
    ```

---

## 4. Web Dashboard Architecture & Endpoint Verification

- **Dashboard Backend**: `NeuroADHD_ModelDashboard/backend/server.py`
  - **Status**: `VERIFIED`
  - **Single-Subject Prediction API**:
    - Route: `POST /api/predict/subject` (Lines 173–242)
    - Input: `multipart/form-data` with `file` containing `roi_CC200.csv` (Limit: 10 MB).
    - Internal handler: Spools to temporary CSV and invokes `predict_single_subject(temp_path)`.
    - Returns JSON: `input_validation`, `model_architecture`, `prediction`, `disclaimer`.
  - **Sample Verification Endpoint**:
    - Route: `GET /api/predict/sample` (Lines 251–257)
    - Returns verified sample file `NeuroADHD/data/adhd200/raw/cc200/1017176_session_1/roi_CC200.csv`.

---

## 5. Architectural Interface Summary

The existing system exhibits clean functional modularity. The future raw pipeline will interface directly at the boundary between raw scan preprocessing and `roi_CC200.csv`:

```
+---------------------------------------------------------------+
|                    NEW PIPELINE (NeuroADHD_RawMRI)            |
| Raw fMRI (BIDS) -> Preprocessing -> CC200 Atlas -> 200 ROIs   |
+---------------------------------------------------------------+
                                 |
                        Produces roi_CC200.csv
                                 |
                                 v
+---------------------------------------------------------------+
|                EXISTING FROZEN SYSTEM (NeuroADHD)             |
| 1. extract_connectivity_vector()      -> 19,900 Fisher-z edges|
| 2. final_preprocessing_pipeline.joblib -> 100 selected features|
| 3. final_model.joblib                  -> Class probabilities  |
+---------------------------------------------------------------+
```
