# NEUROADHD — RESEARCH PROTOCOL (DRAFT)

> **`DRAFT — NOT SUBMITTED — NOT APPROVED`**
>
> **Status:** `DRAFT_FOR_SUPERVISOR_AND_IEC_REVIEW`  
> **Document ID:** `DOC-GOV-PROT-001`  
> **Pipeline Version:** NeuroADHD RawMRI PoC V2 (**23 stages**; Steps 1–6 synthetic-only complete)  
> **Execution Policy:** `STRICTLY_SYNTHETIC_ONLY` | `real_data_authorized = false` | `overall_authorization_state = NOT_AUTHORIZED`  
> **Sub-01 Register Status:** `PENDING` (`cryptographic_signature = null`)  
> **Principal Investigator:** `[PI_NAME]` (`[PI_AFFILIATION_PLACEHOLDER]`)  
> **Academic Supervisor / Guide:** `[SUPERVISOR_NAME]` (`[SUPERVISOR_DEPARTMENT_PLACEHOLDER]`)  
> **Ethics Committee:** `[IEC_NAME]` (DHR/NECRBHR Reg. No.: `[IEC_REGISTRATION_NUMBER]`)  
> **Protocol Reference:** `[IEC_PROTOCOL_NUMBER]` (Approval Date: `[APPROVAL_DATE]`)

---

## A. Research Title

**NeuroADHD — ADHD Classification Using Resting-State Functional MRI Connectivity and Machine Learning**

---

## B. Background and Rationale

1. **Neurodevelopmental Context (Non-Clinical Scope):**  
   Attention-Deficit/Hyperactivity Disorder (ADHD) is a heterogeneous neurodevelopmental condition typically assessed in childhood and adolescence through structured clinical interviews, developmental histories, behavioral rating scales, and direct observation by qualified clinicians. Because behavioral presentations overlap across developmental conditions and vary by informant, neuroimaging researchers investigate objective brain-connectivity correlates to better understand functional network organization.  
   **Non-Clinical Boundary:** This project is strictly an exploratory computational neuroscience research study. Neither the preprocessing pipeline nor the machine-learning models are intended, cleared, or validated as medical diagnostic devices, screening tools, or clinical decision-support instruments.

2. **Resting-State Functional MRI (rs-fMRI) and Connectivity:**  
   Resting-state BOLD functional MRI records spontaneous low-frequency ($0.01\text{--}0.10\text{ Hz}$) blood-oxygenation-level-dependent fluctuations across intrinsic brain networks without requiring an explicit task paradigm, making it practical for pediatric neuroimaging cohorts. Pairwise temporal correlation across regional time series captures functional connectivity architecture across distributed cortico-striato-thalamo-cortical and default-mode circuits.

3. **Craddock 200 (CC200) Parcellation and Feature Representation:**  
   Spatially standardized BOLD time series aligned to MNI152 space are summarized into 200 functionally coherent regions of interest using the Craddock 200 (CC200) parcellation (`roi_CC200.csv`). Symmetric pairwise Pearson correlations across all 200 parcels produce $\binom{200}{2} = 19{,}900$ upper-triangle ($k=1$) edges, stabilized via boundary clipping ($\varepsilon = 10^{-7}$) and Fisher $r$-to-$z$ transformation ($z = \operatorname{arctanh}(r_{\text{clipped}})$).

4. **Motivation for Evaluating Machine-Learning Pipelines End-to-End:**  
   Prior model development (`NeuroADHD` / `NeuroADHD_ModelBenchmark`) produced a frozen inference pipeline operating on pre-extracted historical C-PAC `roi_CC200.csv` files, achieving balanced accuracy `0.60` (`ROC-AUC = 0.56`, `PR-AUC = 0.634`) on a locked 20-subject `Peking_2` holdout set. However, that frozen classifier cannot ingest raw DICOM or BIDS NIfTI scans directly. Evaluating a reproducible raw-to-feature preprocessing pipeline—while quantifying inter-site variance, motion confounding, and feature concordance prior to any real-data deployment—is essential before scientifically interpreting downstream predictions.

---

## C. Research Objectives

### Primary Objective
Develop and evaluate a reproducible, memory-bounded research pipeline for ADHD-versus-control classification using resting-state fMRI functional connectivity extracted via the CC200 atlas.

### Secondary Objectives
1. Validate gated DICOM-to-BIDS ingestion (`dcm2niix` inspection vs. authorization-gated conversion with metadata redaction).
2. Validate standardized 23-stage resting-state fMRI preprocessing (raw motion estimation, slice-timing correction, B0 susceptibility distortion correction, anatomical defacing, single-step spatial normalization, tissue segmentation, and confound regression).
3. Extract verified 200-region CC200 ROI time series conforming strictly to `roi_CC200.csv` (`200` rows $\times$ `[node/volume + T timepoints]`).
4. Generate symmetric $200 \times 200$ functional connectivity matrices with explicit zero-variance ROI rejection.
5. Extract lexicographically ordered **19,900** Fisher-$z$ connectivity features.
6. Evaluate exploratory research classification models using frozen downstream artifacts (`final_preprocessing_pipeline.joblib` selecting `100` ANOVA $F$-features into `final_model.joblib`) and leakage-free validation splits.
7. Assess software determinism, peak memory compliance ($\le 2.50\text{ GB}$), and cross-site generalization behavior.
8. Compare newly preprocessed features against historical C-PAC 0.3.9.1 derivatives (`historical_concordance_status`) **only where formally authorized**.

---

## D. Research Design

The overall program separates **three distinct workstreams** that carry fundamentally different ethical, legal, and regulatory obligations—they must **not** be conflated or assumed to share a single exemption:

| Workstream | Description & Scope | Current Execution State | Distinct Ethics / Regulatory Requirement |
| :--- | :--- | :--- | :--- |
| **1. Synthetic Software Engineering Validation** | Mathematical phantom generation (`generate_synthetic_phantom_session`), unit/integration testing (`481 passed, 7 skipped`, `82%` `src/` coverage), determinism, memory guards, and fail-closed safety gating. | **COMPLETED (Synthetic-Only Baseline)** | Involves zero human participants, zero biological tissue, and zero personal data; exempt from human-participant review, though documentation must clearly mark findings as synthetic-only. |
| **2. Retrospective Secondary Analysis of Existing Dataset (ADHD-200)** | Computational reprocessing and feature-concordance evaluation of existing de-identified multi-site ADHD-200 scans accessed under NITRC/INDI terms, including proposed $N=6$ technical feasibility pilot (`DOC-NA-A5-FRP-001`). | **BLOCKED (`NOT_AUTHORIZED`; 23 items `PENDING`)** | Requires Data Use Agreement verification (`AUTH-DUA-02`), local IEC secondary-use review or formal administrative exemption determination (never self-declared), and DPDP Act 2023 / Rules 2025 applicability review. |
| **3. Prospective or Newly Collected $N=6$ Pilot (Conditional)** | Acquisition of new resting-state fMRI + 3D T1w scans from newly recruited participants (`N=6`), **only if formally planned, funded, and approved**. | **NOT INITIATED / CONDITIONAL** | Requires full prospective IEC protocol approval prior to enrollment, written parental/guardian informed consent + pediatric/adolescent assent (where minors are enrolled), DICOM PHI scrubbing, scanner safety screening, and prospective institutional data governance sign-off. |

---

## E. Study Population

All cohort counts below are drawn strictly from verified repository documentation without fabricating participant records, recruitment logs, consent records, or clinical metadata:

1. **Existing ADHD-200 Engineering Cohort (Documented Reference):**  
   - Project resource/scaling records cite **153 usable subjects**: **50 ADHD** and **103 Control** (`docs/HARDWARE_AND_RESOURCE_ESTIMATES.md`).  
   - **Documented Accounting Note:** Model-training artifact records (`docs/FEATURE_COMPATIBILITY_PLAN.md`) separately reference **114 training subjects + 20 locked `Peking_2` test subjects = 134 subjects**. Reconciling the `153`-subject engineering cohort against the `134`-subject training/test split requires checking the original split manifest once authorized (`REQUIRES_DATA_OWNER_CONFIRMATION` / audit milestone; never guessed).
2. **Future Proposed $N=6$ Pilot (`DOC-NA-A5-FRP-001`):**  
   - **Sample Size:** $N = 6$ (e.g., 3 ADHD, 3 Control proposed for technical feasibility benchmarking only), strictly contingent on academic supervisor sign-off (`SAFE-PILOT-06`) and formal IEC approval (`AUTH-IEC-01`).  
   - **Prospective Criteria Status:** Detailed clinical inclusion/exclusion criteria, recruitment site(s), language translations, and clinician screening procedures remain `[TO_BE_DEFINED_IF_PROSPECTIVE_PILOT_APPROVED]` and must never be fabricated in advance.

---

## F. Research Methodology (23-Stage PoC V2 Workflow)

End-to-end conceptual workflow:

```text
DICOM (Inspect-Only vs. Gate-Authorized Conversion) -> BIDS -> Raw Motion Estimation ->
Slice-Timing Correction (STC) -> B0 Susceptibility Distortion Correction (SDC) ->
Anatomical Defacing & Metadata Sanitization (Stage 7/23) -> Brain Extraction / Segmentation ->
Single-Step Pullback Registration & MNI152 Normalization -> Confound Regression & Bandpass ->
CC200 Parcellation (200 ROIs) -> Functional Connectivity (19,900 Fisher-z) ->
Frozen Feature Selection (100 features) -> Frozen ML Classifier (Exploratory Only)
```

### Current Verification Separation (Strict Scientific Honesty)

| Component | Implementation Status | Validation Basis | Current Limitation / Unproven Scope |
| :--- | :--- | :--- | :--- |
| **DICOM-to-BIDS Intake & Gating** | Implemented (`src/intake/service.py`) | Unit/integration tests via synthetic/mocked DICOM | Conversion fail-closed behind `ResearchAuthorizationGate` (`sub-01 = PENDING`). |
| **23-Stage fMRI Preprocessing (`scripts/run_poc_v2.py`)** | Implemented (Stages 1–23) | Synthetic phantom validation (`<= 2.50 GB` RAM, deterministic $< 10^{-7}$) | **Not validated on real human MRI scans** (`historical_concordance_status = NOT_ASSESSED`). |
| **Stage 3 STC (`DEC-STC-01`)** | Implemented (`src/preprocessing/slice_timing.py`) | Synthetic $64 \times 64 \times 37 \times 240$ tests & Step 4.2 audit | Skips cleanly if `SliceTiming` or slice axis unknown (`--strict-slice-timing` fails closed). |
| **Stage 4 SDC (`DEC-SDC-02`)** | Implemented (`src/preprocessing/susceptibility_distortion.py`) | Synthetic direct-Hz solver & QC tests | `topup`/`prelude`/`fugue` absent on host -> PEPOLAR/phasediff emit `SKIPPED_TOOL_UNAVAILABLE`. |
| **Stage 7 Defacing & Privacy Audit (`DEC-DEF-03`)** | Implemented (`src/preprocessing/defacing.py`) | Synthetic $64 \times 64 \times 64$ & injected runner tests | `pydeface`/`FSL` absent on host -> emits `SKIPPED_TOOL_UNAVAILABLE`; manual visual QC required. |
| **Stages 20–21 CC200 & 19,900 Fisher-$z$** | Implemented (`extractor.py`, `connectivity.py`) | Synthetic recovery & schema parity | Real-participant concordance against C-PAC 0.3.9.1 derivatives not yet measured. |
| **Downstream Frozen ML Inference** | Read-only interface (`src/integration/model_verifier.py`) | Verified on historical `roi_CC200.csv` & synthetic tests | Frozen model trained on historical C-PAC data; clinical validity unestablished. |

---

## G. Statistical Analysis Plan

1. **Subject-Level Splitting:** All cross-validation folds and holdout partitions operate strictly at the subject level (`subject_id`) so scans/runs from one participant never span training and test folds.
2. **Site-Disjoint Evaluation & Locked Holdout:** Cross-site generalization must be assessed via leave-one-site-out / site-disjoint evaluation, keeping the **20 locked `Peking_2` test subjects** strictly isolated from preprocessing calibration or threshold tuning.
3. **Leakage-Free Feature Selection:** All imputation medians (`SimpleImputer`), variance filters (`VarianceThreshold`), scaling (`StandardScaler`), and univariate ANOVA $F$-selection (`SelectKBest(k=100)`) are fitted exclusively inside training folds (or loaded read-only from the frozen `final_preprocessing_pipeline.joblib` without refitting).
4. **Class Imbalance & Evaluation Metrics:** Report class prevalence alongside **balanced accuracy**, **sensitivity (recall)**, **specificity**, **precision**, **F1-score**, **ROC-AUC**, and **PR-AUC** (`class_weight='balanced'` in the baseline Random Forest due to ~`50:103` cohort imbalance).
5. **Reproducibility & Model Freezing:** Enforce fixed random seeds, deterministic single-step spatial resampling tolerances ($< 10^{-7}$), SHA-256 checksum verification of model artifacts prior to deserialization, and provenance schema validation (`configs/provenance_schema.json`).
6. **$N=6$ Pilot Statistical Boundary:** **The proposed $N=6$ pilot is strictly a technical feasibility check** (runtime, peak RAM $\le 2.50\text{ GB}$, BIDS/QC pass rates, numerical stability, and historical C-PAC concordance where paired data exist). **It is NOT statistically powered for diagnostic classification accuracy, sensitivity/specificity confidence intervals, or clinical performance claims.**

---

## H. Ethical & Governance Summary

- **Unresolved Decisions:** Formal IEC approval (`AUTH-IEC-01`), NITRC/INDI DUA verification (`AUTH-DUA-02`), informed consent / waiver decision (`AUTH-DSU-03`), statutory secondary-use & privacy sign-off (`AUTH-SUA-04`, `AUTH-DPP-05`), and operational safeguards (`SAFE-DEID-01` through `SAFE-PILOT-06`) remain `PENDING` in `configs/research_authorization_register.json`.
- **Non-Bypassable Lock:** Governance readiness documents prepare the submission dossier; real-data execution remains programmatically locked (`STRICTLY_SYNTHETIC_ONLY`) until verified institutional authorization is recorded.
