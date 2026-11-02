# NeuroADHD RawMRI PoC V2 — Phase A.9
# Synthetic End-to-End Pipeline Execution & Validation Report

**Document ID:** `DOC-NA-A9-SER-001`  
**Version:** 1.0.0  
**Phase:** Phase A.9 — Existing Pipeline Validation & Scientific Hardening  
**Scope:** Complete End-to-End Execution Evidence under Synthetic Testing Conditions  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Baseline Commit:** [`4be77d03758983a65d7f9582011db8da11d20fd7`](https://github.com/Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine/commit/4be77d03758983a65d7f9582011db8da11d20fd7)  
**Execution Policy:** `STRICTLY_SYNTHETIC_ONLY`

---

## 1. Executive Summary

This report documents the end-to-end execution of the complete 19-stage pipeline using the functional orchestrator ([`scripts/run_poc_v2.py`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/scripts/run_poc_v2.py)).

Two comprehensive execution runs were performed and verified:
1. **Synthetic Phantom Mode:** Executed with authenticated synthetic BIDS inputs and synthetic atlas phantoms.
2. **Canonical CC200 & Frozen Model Inference Mode:** Executed with the authentic canonical CC200 atlas (`(63, 75, 61)`, 3.0 mm isotropic) and downstream frozen Random Forest classification.

Both executions completed with 100% stage success, generating all 12 verified output artifacts within the strict 2.5 GB memory ceiling.

---

## 2. Execution Run Profiles & Performance Metrics

| Execution Profile | CLI Command & Parameters | Total Runtime | Peak Memory (RSS) | Memory Headroom (vs 2.5 GB) | Exit Status | Output Directory |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Profile 1: Synthetic Phantom** | `--mode synthetic --output-dir outputs/synthetic_poc_v2` | **4.10 seconds** | **0.29 GB** | 2.21 GB (88.4% free) | `SUCCESS` | `outputs/synthetic_poc_v2/sub-synthetic001/run_20261102_165603/` |
| **Profile 2: Canonical CC200 + Frozen Model** | `--mode synthetic --atlas-mode canonical --model-dir <benchmark_dir> --require-model-inference` | **180.09 seconds** | **0.50 GB** | 2.00 GB (80.0% free) | `SUCCESS` | `outputs/synthetic_poc_v2_canonical/sub-synthetic001/run_20261102_165627/` |

---

## 3. Output Artifact Suite & Cryptographic Manifest Verification

Each successful execution generates twelve standardized output artifacts in an isolated run directory. After generation, the orchestrator computes SHA-256 digests on disk and asserts exact correspondence against `output_manifest.json`:

```json
{
  "status": "SUCCESS",
  "subject_id": "sub-synthetic001",
  "timestamp_utc": "2026-11-02T16:56:27.612479+00:00",
  "artifacts": {
    "roi_csv": "roi_CC200.csv",
    "features_csv": "features_19900_fisher_z.csv",
    "features_npy": "features_19900_fisher_z.npy",
    "motion_tsv": "motion_parameters.tsv",
    "fd_tsv": "framewise_displacement.tsv",
    "confounds_tsv": "confounds.tsv",
    "confound_report_json": "confound_report.json",
    "qc_coreg_png": "qc_registration_epi_to_t1w.png",
    "qc_norm_png": "qc_registration_t1w_to_mni.png",
    "qc_tissue_png": "qc_tissue_masks.png",
    "qc_parcellation_png": "qc_parcellation_cc200.png",
    "provenance_json": "provenance.json"
  }
}
```

### 3.1 Artifact Verification Summary
1. **`roi_CC200.csv`:** Validated $200 \times 61$ matrix (200 ROIs by 60 timepoints + 1 label column) matching historical C-PAC format.
2. **`features_19900_fisher_z.npy` & `features_19900_fisher_z.csv`:** Exactly 19,900 float64 values, 100% finite, range $[-4.701, +5.109]$ (clipped within $[-8.41, +8.41]$).
3. **`motion_parameters.tsv` & `framewise_displacement.tsv`:** 6 motion columns ($t_x, t_y, t_z$ in mm, $r_x, r_y, r_z$ in deg); Mean FD = $0.0982\text{ mm}$, Max FD = $0.3644\text{ mm}$, Scrubbed volumes ($>0.5\text{ mm}$) = 0.
4. **`confounds.tsv`:** 31-column nuisance matrix (24 Friston + 5 aCompCor + 2 polynomial trends); verified full column rank ($31/31$, condition number $\approx 3.1 \times 10^5$).
5. **Quality Control Visual Overlays (4 PNGs):** Multi-planar orthogonal contour plots verifying functional-to-structural alignment, MNI152 normalization, tissue masks, and CC200 parcellation.
6. **`provenance.json`:** Validated against [`configs/provenance_schema.json`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/configs/provenance_schema.json) with software versions, hardware memory profiles, and validation gates marked `PASSED`.

---

## 4. Downstream Frozen Model Inference Verification

In Profile 2, the extracted 19,900 Fisher-z vector was consumed by the frozen machine learning model:
- **Feature Selector Checksum:** `228d3f0180e4855d49fd732327ab1fc4f255beb7bea43dd4fd1db3064aa86249`
- **Classifier Checksum:** `d8eb23eb8c6d25ffc12b2bab515a2a7112daeeb0afa8f4e9eaa499dbe5b5f2ef`
- **Input Dimension:** $\mathbb{R}^{19,900}$
- **Selected Dimension:** $\mathbb{R}^{100}$ (matching `EXPECTED_SELECTED_FEATURE_INDICES`)
- **Zero-Refit Verification:** Model parameters and weights were untouched (`refit_performed = False`).
- **Prediction Payload:**
  ```json
  {
    "status": "SUCCESS",
    "evaluation_mode": "SYNTHETIC_DEMONSTRATION_ONLY",
    "diagnostic_disclaimer": "SYNTHETIC_DEMONSTRATION_ONLY: Not approved or validated for clinical diagnostic use.",
    "predicted_class": 1,
    "class_label": "ADHD Pattern",
    "adhd_probability": 0.5115,
    "control_probability": 0.4885,
    "decision_threshold": 0.5,
    "features_in": 19900,
    "features_selected": 100,
    "refit_performed": false
  }
  ```

---

## 5. Quality Control Gating Status

All eight quantitative quality control thresholds remain formally marked:
$$\mathbf{QC\_THRESHOLDS\_PENDING\_SUPERVISOR\_APPROVAL}$$
The numerical calculations were verified, and values are tracked as empirical measurements rather than approved clinical cutoffs.

---

## 6. Execution Verdict

$$\mathbf{PHASE\_A\_9\_SYNTHETIC\_EXECUTION\_SUCCESSFUL}$$
The pipeline executes reproducibly end-to-end, adheres strictly to memory isolation boundaries ($< 0.50\text{ GB} \ll 2.50\text{ GB}$), generates schema-compliant provenance, and integrates seamlessly with downstream frozen classifiers.
