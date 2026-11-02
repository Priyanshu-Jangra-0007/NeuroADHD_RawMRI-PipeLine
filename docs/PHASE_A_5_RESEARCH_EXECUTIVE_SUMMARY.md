# NeuroADHD RawMRI PoC V2 — Phase A.5
# Research Executive Summary

**Document ID:** `DOC-NA-A5-RES-001`  
**Version:** 1.0.0  
**Status:** `READY_FOR_SUPERVISOR_DECISION`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Target Audience:** Academic Research Supervisor, Departmental Review Committee, Institutional Ethics Committee  

---

## 1. Executive Statement

This executive summary presents the research architecture, implementation maturity, and verification status of the **NeuroADHD RawMRI Proof-of-Concept Version 2 (PoC V2)** pipeline. The system provides an end-to-end, reproducible, containerized pipeline transforming raw structural (T1-weighted) and functional (resting-state BOLD) magnetic resonance imaging (MRI) data into CC200 functional connectivity features, integrated with a frozen machine learning classifier for Attention-Deficit/Hyperactivity Disorder (ADHD) research.

As of Phase A.5, the pipeline has completed full synthetic implementation, scientific registration validation, containerized deployment verification, and rigorous evidence auditing. **All execution to date has been conducted under strictly synthetic conditions.** Real participant MRI data processing remains strictly disabled. This document and its accompanying review package are submitted to the Research Supervisor to provide the scientific and governance foundation required for informed institutional review.

---

## 2. Research Problem & Clinical Context

Attention-Deficit/Hyperactivity Disorder (ADHD) is a heterogeneous neurodevelopmental condition characterized by persistent patterns of inattention, hyperactivity, and impulsivity. While clinical diagnosis relies on behavioral observations and psychometric evaluations (e.g., DSM-5, Conners scales), neuroimaging research—particularly resting-state functional magnetic resonance imaging (rs-fMRI)—has revealed disruptions in large-scale functional brain networks, including the default mode network (DMN), frontoparietal control network (FPN), and ventral attention network (VAN).

Despite substantial academic literature, clinical translation and reproducible research in neuroimaging-based ADHD biomarker discovery face critical methodological bottlenecks:
1. **Pipeline Heterogeneity and Brittle Tooling:** Neuroimaging pipelines frequently depend on sprawling, non-containerized toolchains (FSL, AFNI, FreeSurfer, ANTs) that suffer from installation brittleness, operating system version drift, and library incompatibilities.
2. **Motion Sensitivity in Pediatric Cohorts:** Head motion is significantly elevated in children and adolescents with ADHD, introducing systematic spurious correlations into functional connectivity matrices if motion parameters are not rigorously estimated, scrubbed, or regressed out.
3. **Registration and Geometric Distortions:** Transforming heterogeneous subject anatomy into standardized stereotaxic space (MNI152) requires robust non-linear diffeomorphic transformations (e.g., SyN) and verified tissue segmentation to avoid severe parcellation misalignment.
4. **Machine Learning Integration Gaps:** Downstream machine learning models are frequently trained on ad-hoc feature subsets with data leakage (e.g., scaling or feature selection prior to cross-validation splits), or fail when interfaced with upstream raw data pipelines due to unstandardized parcellation and feature orderings.
5. **Data Governance and Ethics Ambiguity:** Complex legal and regulatory frameworks governing health data—specifically in academic research contexts—require formal governance regarding de-identification, facial defacing, and compliance with institutional and statutory privacy regulations.

The NeuroADHD RawMRI PoC V2 project addresses these challenges by establishing an automated, containerized, mathematically validated pipeline governed by fail-closed safety controls.

---

## 3. Proposed Approach & Technical Architecture

NeuroADHD RawMRI PoC V2 implements a unified 19-stage pipeline operating within an isolated Linux container with operating-system-enforced resource constraints:

```
[ Raw Structural & rs-fMRI (BIDS) ]
                 │
                 ▼
 ┌────────────────────────────────────────────────────────┐
 │ Stage 1–2:  Cryptographic Manifest & BIDS Validation  │
 │ Stage 3–4:  BOLD Motion Estimation & Reference Vol     │
 │ Stage 5–8:  T1w Skull Stripping, N4, Tissue Seg/Erode  │
 │ Stage 9–11: EPI-T1w Rigid & T1w-MNI SyN Registration   │
 │ Stage 12:   BOLD Pullback Resampling to CC200 Space    │
 │ Stage 13–14: aCompCor (5+5), Friston-24 & Bandpass    │
 │ Stage 15–17: Canonical CC200 Timeseries & Fisher-z FC  │
 └────────────────────────────────────────────────────────┘
                 │
                 ▼  [ 19,900 Functional Connectivity Features ]
 ┌────────────────────────────────────────────────────────┐
 │ Stage 18:   Frozen Preprocessing Pipeline              │
 │             - SimpleImputer(strategy='median')         │
 │             - VarianceThreshold(0.0)                   │
 │             - StandardScaler()                         │
 │             - SelectKBest(f_classif, k=100)            │
 │ Stage 19:   Frozen Classifier Inference                │
 │             - RandomForestClassifier(50 trees, depth 3)│
 └────────────────────────────────────────────────────────┘
                 │
                 ▼
 [ Exploratory Research Predictions & Multi-Metric QC Report ]
```

### Key Architectural Pillars:
- **Strict BIDS Compliance:** All input datasets must adhere to the Brain Imaging Data Structure (BIDS) specification with complete JSON sidecar metadata.
- **Fail-Closed Input Verification:** Mandatory HMAC-SHA256 integrity verification prevents execution on tampered, unverified, or out-of-boundary paths.
- **Containerized Memory Isolation:** Hard Linux cgroup memory limit of 2,560 MiB (2.5 GiB) enforced at the kernel level, ensuring operation on constrained laboratory hardware.
- **Mathematical Registration Rigor:** Verified pullback resampling composition from BOLD native space directly into the canonical CC200 stereotaxic grid ($63 \times 75 \times 61$, 3.0 mm isotropic) in a single interpolation step to minimize spatial blurring.
- **Confound Mitigation:** Anatomic CompCor (aCompCor with 5 CSF and 5 WM components) combined with Friston-24 motion parameter expansion and simultaneous bandpass filtering ($0.01\text{--}0.10\text{ Hz}$).

---

## 4. Existing Frozen Model Specification

The downstream statistical classifier integrated into RawMRI PoC V2 is an authentic, frozen reference model trained on historical ADHD-200 CC200 functional connectivity benchmarks.

### Model Integration Contract:
- **Feature Vector:** Exactly 19,900 Fisher-$z$ transformed Pearson correlation coefficients derived from the upper triangle of the $200 \times 200$ CC200 parcellation matrix ($\binom{200}{2} = 19{,}900$).
- **Preprocessing Pipeline (`final_preprocessing_pipeline.joblib`):**
  - Cryptographic Hash (SHA-256): `228d3f0180e4855d49fd732327ab1fc4f255beb7bea43dd4fd1db3064aa86249`
  - Step 1: `SimpleImputer(strategy='median')`
  - Step 2: `VarianceThreshold(threshold=0.0)`
  - Step 3: `StandardScaler(copy=True, with_mean=True, with_std=True)`
  - Step 4: `SelectKBest(score_func=f_classif, k=100)`
- **Classification Model (`final_model.joblib`):**
  - Cryptographic Hash (SHA-256): `d8eb23eb8c6d25ffc12b2bab515a2a7112daeeb0afa8f4e9eaa499dbe5b5f2ef`
  - Architecture: `RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=42)`
- **Immutability Contract:**
  - The model artifacts are mounted into the container as **read-only** (`:ro`).
  - Cryptographic checksums are validated against hardcoded expectations prior to deserialization.
  - Zero parameter refitting or retraining occurs during pipeline execution.

---

## 5. Summary of Project Achievements (Phases A.1 through A.4.2.1)

Through systematic engineering and scientific validation phases, the project has established the following baseline:

| Phase | Milestone | Primary Output / Verification |
|---|---|---|
| **Phase A.1 / A.1.1** | Manifest Verification Hardening | Enforced mandatory HMAC-SHA256 signature verification, rejected empty/fallback keys, and implemented strict path traversal boundary checks. |
| **Phase A.2 / A.2.1** | Scientific Registration Validation | Validated coordinate frames, affine matrices, SyN pullback composition, and reconciled canonical CC200 atlas geometry ($63 \times 75 \times 61$, 3.0 mm isotropic, SHA-256: `9467afce...`). |
| **Phase A.3 / A.3.1** | Deployment & Memory Isolation | Implemented Docker containerization with 2.5 GB Linux cgroup memory limits; verified kernel OOM-killer termination and fail-closed host guards. |
| **Phase A.4** | Pre-Pilot Readiness Gate | Designed multi-metric QC framework (8 acceptance criteria), real MRI input contract, and comprehensive data governance specification. |
| **Phase A.4.1** | Frozen Artifact Verification | Validated discovery via `NEUROADHD_MODEL_DIR`, verified SHA-256 checksums, and confirmed feature extraction compatibility ($19{,}900 \to 100 \to \text{RF}$). |
| **Phase A.4.2** | Container Model Integration | Verified read-only container mounting, non-root user execution (`UID 10001`), and deterministic synthetic inference inside Docker. |
| **Phase A.4.2.1** | Evidence Audit & CI Reconciliation | Upgraded JUnit XML summary script to fail closed on malformed XMLs; confirmed 100% test pass rate in CI (176 passed / 3 skipped without model; 179 passed with model). |

---

## 6. Scientific, Clinical & Methodological Limitations

Prior to any consideration of real participant data, the supervisor must review the inherent scientific and translational limitations of this research tool:

> [!WARNING]
> **Non-Clinical Research Disclaimer:**
> NeuroADHD RawMRI PoC V2 is an **investigational computational research pipeline**. It is **NOT** a certified medical diagnostic device, clinical decision support system, or Software as a Medical Device (SaMD). Its classification outputs must never be used for clinical diagnosis, treatment planning, or participant risk stratification.

### Specific Research Limitations:
1. **Synthetic-Only Validation:** All automated tests and pipeline executions to date have operated exclusively on procedurally generated synthetic phantom datasets. Real MRI data exhibits physiological artifacts, scanner acoustic noise, ghosting, and anatomical abnormalities not fully represented in synthetic phantoms.
2. **Exploratory Model Nature:** The downstream frozen Random Forest classifier is a shallow benchmark model (50 trees, max depth 3) trained on historical, multi-site open-source data (ADHD-200). While useful as a fixed software contract, its diagnostic sensitivity and specificity have not been established in prospective cohorts.
3. **Cross-Site Protocol Domain Shift:** The historical model was trained on data acquired with specific scanner hardware and sequence parameters (predominantly 1.5T/3.0T, repetition time $\text{TR} \approx 2.0\text{ s}$, no multiband acceleration). Running the model on modern acquisitions (e.g., multiband accelerated rs-fMRI, high-channel head coils) may induce domain shift.
4. **Motion-Induced Functional Spuriousness:** Despite aCompCor and Friston-24 regression, severe pediatric motion can corrupt functional connectivity estimates. The pipeline's scrubbing thresholds must be empirically validated against real motion traces.
5. **Parcellation Scale Constraints:** The Craddock CC200 parcellation comprises 200 functionally homogeneous ROIs. While widely cited, it does not capture fine-grained subcortical nuclei or individualized functional boundary reconfigurations.

---

## 7. Supervisor Decisions Required in Phase A.5

To transition the pipeline toward future real-data readiness, the Research Supervisor's formal review and direction are requested on the following key areas:

1. **MRI Preprocessing Decisions:**
   - Formal policy on slice-timing correction for prospective acquisitions.
   - Strategy for susceptibility distortion correction (fieldmap requirement vs synthetic fallback).
   - Defacing protocol sign-off ensuring participant facial privacy while preserving frontal pole cortical tissue.
2. **Quality Control Acceptance Thresholds:**
   - Formal review and provisional approval of the 8 predefined QC acceptance criteria (EPI-T1w NMI, T1w-MNI Dice, Mean FD, Max Spike, Scrubbing %, Brain Volume, tSNR, DVARS).
3. **Institutional Ethics & Data Governance:**
   - Alignment with Indian Council of Medical Research (ICMR) Ethical Guidelines (2017) and the Digital Personal Data Protection (DPDP) Act 2023.
   - Review of proposed Data Use Agreement (DUA) requirements for retrospective ADHD-200 benchmark data.
4. **First Research Pilot Protocol:**
   - Review of the proposed minimal feasibility pilot protocol ($N=5\text{--}10$ retrospective scans) to assess technical throughput without clinical claims.

---

## 8. Master Document Cross-References

This summary introduces the complete Phase A.5 review package available in the `docs/` repository:

| Document ID | File Path | Scope and Purpose |
|---|---|---|
| `DOC-NA-A5-RES-001` | [PHASE_A_5_RESEARCH_EXECUTIVE_SUMMARY.md](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_RESEARCH_EXECUTIVE_SUMMARY.md) | High-level research summary for supervisors and reviewers (this document). |
| `DOC-NA-A5-TR-001` | [PHASE_A_5_TECHNICAL_REVIEW.md](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_TECHNICAL_REVIEW.md) | Comprehensive stage-by-stage technical, mathematical, and algorithmic documentation. |
| `DOC-NA-A5-SDR-001` | [PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md) | Formal register of preprocessing decisions and 8 QC thresholds awaiting supervisor approval. |
| `DOC-NA-A5-EDG-001` | [PHASE_A_5_ETHICS_AND_DATA_GOVERNANCE_REVIEW.md](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_ETHICS_AND_DATA_GOVERNANCE_REVIEW.md) | Detailed 15-point review of research ethics, ICMR 2017 guidelines, DPDP Act 2023, and DUAs. |
| `DOC-NA-A5-FRP-001` | [PHASE_A_5_FIRST_RESEARCH_PILOT_PROTOCOL.md](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_FIRST_RESEARCH_PILOT_PROTOCOL.md) | 10-stage execution protocol for an initial feasibility pilot on retrospective scans. |
| `DOC-NA-A5-SPO-001` | [PHASE_A_5_SUPERVISOR_PRESENTATION_OUTLINE.md](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_SUPERVISOR_PRESENTATION_OUTLINE.md) | 12-slide presentation deck outline for departmental thesis committee review. |
| `DOC-NA-A5-AC-001` | [PHASE_A_5_APPROVAL_CHECKLIST.md](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_APPROVAL_CHECKLIST.md) | Actionable 10-point governance checklist tracking mandatory prerequisites prior to real data. |
| `DOC-NA-A5-SRP-001` | [PHASE_A_5_SUPERVISOR_REVIEW_PACKAGE.md](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_SUPERVISOR_REVIEW_PACKAGE.md) | Master review package consolidating all findings, evidence, and formal decision blocks. |

---

## 9. Conclusion & Phase A.5 Final Status

Phase A.5 has successfully produced a complete, professional, supervisor-ready review package that transparently documents technical rigor, computational safety, scientific limitations, and governance requirements.

**Operational Decision:**  
`READY_FOR_SUPERVISOR_DECISION`  
*(Real MRI processing remains strictly disabled. Zero real participant scans may be ingested until formal institutional and supervisor authorizations are executed.)*

