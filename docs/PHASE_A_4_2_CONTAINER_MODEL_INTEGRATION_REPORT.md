# NeuroADHD RawMRI PoC V2 — Phase A.4.2 Deployment Report: Container Model Integration & Verification

**Document Version:** 1.0.0  
**Phase:** Phase A.4.2 — Container Model Integration & Deployment Verification  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Baseline Commit:** `26b944106407ad7d343ae74f9f92575c45d0740d`  
**Phase A.4.2 Verified Commit:** `d6673c4ff7a1ac5f1116722acbca24783034fd65`  
*(Note: Initial pre-amend working commit `7282eea173f97118ee395a2d15eec49c7e953876` was superseded during test permission hardening; `d6673c4ff7a1ac5f1116722acbca24783034fd65` is the authoritative verified commit on `main`.)*  
**GitHub Actions CI Run:** [#41431929861](https://github.com/Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine/actions/runs/41431929861)  
**Deployment Decision:** `DEPLOYMENT_VERIFIED_WITH_FROZEN_MODEL`  

---

## Executive Summary

Phase A.4.2 completes deployment-level integration and cryptographic verification of the authentic frozen ADHD classification model inside the isolated Docker container (`neuro_adhd_pipeline:v2`).

All execution remains **100% synthetic-only**. No real clinical or participant scans were processed. The frozen model artifacts were mounted strictly read-only (`:ro`), leaving host binaries bitwise unchanged. Continuous Integration (CI) test reporting has been upgraded from static placeholders to dynamic, machine-readable JUnit XML parsing that accurately details passed, failed, and explicitly skipped tests.

---

## 1. Cryptographic Model Artifact Verification

The frozen model artifacts originate from the validated NeuroADHD benchmark training session (`run_20261004_182318_final_locked`) and were cryptographically verified using SHA-256 chunked hashing prior to deserialization:

| Artifact | File Name | Target & Verified SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| **Preprocessing Pipeline** | `final_preprocessing_pipeline.joblib` | `228d3f0180e4855d49fd732327ab1fc4f255beb7bea43dd4fd1db3064aa86249` | **VERIFIED** |
| **Classifier Model** | `final_model.joblib` | `d8eb23eb8c6d25ffc12b2bab515a2a7112daeeb0afa8f4e9eaa499dbe5b5f2ef` | **VERIFIED** |

### Verified Model Architecture & Parameters
- **Pipeline Structure:** `Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('variance', VarianceThreshold(threshold=0.0)), ('scaler', StandardScaler()), ('select', SelectKBest(k=100))])`
- **Input Dimension:** Exactly 19,900 Fisher-z transformed pairwise connectivity features ($\binom{200}{2}$).
- **Feature Selection:** Frozen `SelectKBest(score_func=f_classif, k=100)` retaining the exact historical 100 feature indices without recomputation.
- **Classifier Architecture:** `RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=42)`
- **Output Classes:** `[0, 1]` (0 = Typical Control Pattern, 1 = ADHD Pattern).

---

## 2. Container Volume Mount Hardening & Read-Only Enforcement

The model directory is mounted into the container using Linux kernel-level read-only protection:
```bash
docker run --rm \
    --memory=2560m --memory-swap=2560m \
    -v "${MODEL_DIR_RESOLVED}:/opt/neuro_adhd/models:ro" \
    -e NEUROADHD_MODEL_DIR="/opt/neuro_adhd/models" \
    ...
```

### Verified Security Controls
1. **Read-Only Mount Enforcement (`:ro`):**
   - Attempting to create new files inside `/opt/neuro_adhd/models` raises `OSError: [Errno 30] Read-only file system` (exit code 1).
   - Attempting to overwrite or append to `.joblib` files inside `/opt/neuro_adhd/models` is rejected by kernel filesystem protection (exit code 2).
   - Attempting to delete files inside `/opt/neuro_adhd/models` is rejected (exit code 1).
2. **Symlink Prohibition:**
   - Host directory and artifact paths are verified with `Path.is_symlink()`. Any symlink substitution triggers immediate fail-closed termination.
3. **Bitwise Host Immutability:**
   - Host SHA-256 digests of `final_preprocessing_pipeline.joblib` and `final_model.joblib` were recorded immediately before container execution and recomputed immediately after completion:
     - Pre-execution Pipeline SHA-256: `228d3f0180e4855d49fd732327ab1fc4f255beb7bea43dd4fd1db3064aa86249`
     - Post-execution Pipeline SHA-256: `228d3f0180e4855d49fd732327ab1fc4f255beb7bea43dd4fd1db3064aa86249`
     - Pre-execution Classifier SHA-256: `d8eb23eb8c6d25ffc12b2bab515a2a7112daeeb0afa8f4e9eaa499dbe5b5f2ef`
     - Post-execution Classifier SHA-256: `d8eb23eb8c6d25ffc12b2bab515a2a7112daeeb0afa8f4e9eaa499dbe5b5f2ef`
     - **Verification:** Bitwise identical ($0\text{ bytes modified}$).
4. **Fail-Closed Execution Policy (`--require-model-inference`):**
   - When `--require-model-inference` is passed without configuring `NEUROADHD_MODEL_DIR` or mounting `/opt/neuro_adhd/models`, execution terminates immediately at startup with exit code 1 and `RuntimeError: SAFETY VIOLATION: Model inference was explicitly requested (--require-model-inference), but neither --model-dir was provided nor NEUROADHD_MODEL_DIR environment variable is configured.`

---

## 3. Synthetic Pipeline & Container Inference Execution

The container completed the full 19-stage pipeline in canonical CC200 atlas mode:

```
Synthetic BOLD (60 vols, 3mm)
      ↓
Stages 1-15: Motion realignment, brain extraction, N4 bias correction, 
             Friston-24 + aCompCor regression, Butterworth bandpass [0.01-0.10 Hz]
      ↓
Stage 16: Canonical CC200 ROI time series extraction (200 parcels x 60 timepoints)
      ↓
Stage 17: Upper-triangle Pearson correlation & Fisher-z transformation (19,900 edges)
      ↓
Stage 18: Quality Control orthogonal visual overlays
      ↓
Stage 19: Export & Frozen Model Inference:
          - pipeline.transform(1, 19900) → (1, 100) selected features
          - classifier.predict_proba() & classifier.predict()
          - Zero refitting performed
```

### Generated Inference Artifact (`model_prediction.json`)
```json
{
  "status": "SUCCESS",
  "predicted_class": 0,
  "class_label": "Typical Control Pattern",
  "adhd_probability": 0.4154,
  "control_probability": 0.5846,
  "decision_threshold": 0.5,
  "features_in": 19900,
  "features_selected": 100,
  "refit_performed": false
}
```

### Verification Checks on Prediction Payload
- `status`: `"SUCCESS"`
- `predicted_class`: `0` (`int`, finite)
- `class_label`: `"Typical Control Pattern"`
- Probabilities: $\text{ADHD} = 0.4154$, $\text{Control} = 0.5846$, $\sum = 1.0000$ (all finite, non-NaN)
- Feature Dimensionality: Ingested 19,900 features, selected 100 features
- Model Refitting: `refit_performed: false` (guarantees zero parameter alteration).

---

## 4. Container Resource Isolation & Memory Budget

| Metric | Target Specification | Observed Container Value | Status |
| :--- | :--- | :--- | :--- |
| **Kernel Memory Limit** | 2560 MiB (2,684,354,560 bytes) | `2684354560 bytes` via cgroup | **VERIFIED** |
| **Swap Ceiling** | 2560 MiB (`--memory-swap=2560m`) | Enforced | **VERIFIED** |
| **OOM Killer** | Exit code 137 on breach | Verified in `scripts/verify_container_memory.sh` | **VERIFIED** |
| **Synthetic Peak Memory** | $< 2.50\text{ GB}$ | **0.77 GB** (~788 MiB) | **PASSED (30.8% of ceiling)** |
| **Execution Duration** | $< 300\text{ s}$ | **184 seconds** | **PASSED** |

---

## 5. Dynamic CI Test Reporting & Missing-Artifact Handling

CI test reporting has been corrected to dynamically extract test results directly from JUnit XML test outputs via `scripts/ci_test_summary.py`.

### Test Architecture Breakdown (179 Total Collected Tests)
- **Core Unit Tests (`tests/unit/`):** 145 tests total (143 passed, 2 skipped in CI / 145 passed locally with model directory).
- **Synthetic Pipeline & Runner Integrations (`tests/integration/`):** 29 tests total (29 passed in CI and locally).
- **Container Model Integrations (`tests/integration/test_container_model_integration.py`):** 5 tests total (4 passed, 1 skipped in CI / 5 passed locally with model directory).
- **Total Local Suite:** **179 passed, 0 failed, 0 errors, 0 skipped.**
- **Total CI Suite:** **176 passed, 0 failed, 0 errors, 3 skipped.**
- *(Note on historical documentation: An earlier draft referenced 170 passed tests based on a partial local test run before all 145 unit tests and 5 container tests were consolidated. The authoritative total across the entire codebase is 179 collected tests.)*

### Explicit CI Skip Policy
In GitHub Actions CI where model binaries (`*.joblib`) are excluded from Git per security/licensing rules:
1. `test_case_h_authentic_model_artifacts_mounted_correctly` is explicitly skipped with `skipTest`.
2. `test_case_i_actual_synthetic_inference_completes_successfully` is explicitly skipped with `skipTest`.
3. `test_container_full_synthetic_inference_leaves_host_checksums_identical` is explicitly skipped with `skipTest`.
4. The remaining 4 container tests (read-only mount enforcement, missing mount fail-closed, corrupted artifact fail-closed, symlink substitution fail-closed) run and pass inside CI against the built `neuro_adhd_pipeline:v2` container.
5. `scripts/ci_test_summary.py` explicitly documents why model verification was skipped in CI and references local container verification evidence.

---

## 6. Files Created and Modified

| File | Status | Description |
| :--- | :--- | :--- |
| `scripts/run_poc_v2.py` | Modified | Added `--require-model-inference` flag, early fail-closed safety check, zero-refit model inference, and `model_prediction.json` export. |
| `scripts/docker_run_synthetic.sh` | Modified | Hardened model mount checks (non-empty, non-symlink), mounted `/opt/neuro_adhd/models:ro`, passed `--require-model-inference`. |
| `src/integration/model_verifier.py` | Modified | Cryptographic chunked SHA-256 verification, strict model architecture checks, zero-refit inference runner. |
| `tests/unit/test_phase_a4_model_compatibility.py` | Modified | Added Cases A-I unit tests, fail-closed tests, and explicit `skipTest` handling when model directory is unset. |
| `tests/integration/test_container_model_integration.py` | Created | Added container integration test suite validating `:ro` mount, bitwise checksum preservation, fail-closed handling, and synthetic inference. |
| `scripts/ci_test_summary.py` | Created | Dynamic JUnit XML summary reporter categorizing unit, integration, and container model test outcomes. |
| `.github/workflows/ci.yml` | Modified | Added container build & memory verification prior to tests, added `--junitxml` flags, and replaced hardcoded counts with `scripts/ci_test_summary.py`. |
| `docs/PHASE_A_4_2_CONTAINER_MODEL_INTEGRATION_REPORT.md` | Created | Comprehensive deployment and verification report. |

---

## 7. Safety, Compliance, and Ethical Limitations

> [!IMPORTANT]
> **Strict Clinical and Scientific Limitations:**
> - The NeuroADHD pipeline remains **strictly synthetic-only**.
> - Real MRI processing and participant data ingestion remain **permanently disabled**.
> - The classifier evaluated is an **exploratory research model**, not a clinical diagnostic instrument.
> - Diagnostic accuracy, sensitivity, and clinical utility cannot be claimed from synthetic phantom data.
> - Execution on real human MRI data remains strictly blocked until:
>   1. Institutional Review Board (IRB) / Research Ethics Board (REB) protocol approval is documented.
>   2. Fully executed Data Use Agreement (DUA) is verified.
>   3. Principal Investigator and Project Supervisor approve clinical QC thresholds.
>   4. A formally audited, independent real-data enablement phase is authorized.

---

## 8. Final Deployment Decision

```
================================================================================
FINAL DEPLOYMENT STATUS: DEPLOYMENT_VERIFIED_WITH_FROZEN_MODEL
================================================================================
- Docker Container: Hardened Linux Bookworm, unprivileged user, 2560 MiB cgroup isolation
- Atlas Contract: Canonical CC200 (63x75x61, 3.0mm isotropic, verified SHA-256)
- Model Artifacts: Authentic benchmark frozen pipeline & classifier verified via SHA-256
- Volume Isolation: Read-only mount (/opt/neuro_adhd/models:ro) with bitwise immutability
- Fail-Closed Policy: Mandatory HMAC verification, fail-closed model inference policy
- CI Reporting: Dynamic JUnit XML summary with transparent skip reporting
- Real Data Execution: Permanently disabled (synthetic-only)
================================================================================
```
