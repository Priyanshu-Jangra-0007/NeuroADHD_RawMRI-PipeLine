# NeuroADHD RawMRI PoC V2 — Phase A.5
# Master Supervisor Review Package & Research Authorization Dossier

**Document ID:** `DOC-NA-A5-SRP-001`  
**Version:** 1.0.0  
**Phase:** Phase A.5 — Supervisor Review & Research Authorization  
**Final Status:** `READY_FOR_SUPERVISOR_DECISION`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Baseline Git Commit:** `f0c093388862dc1cd2064860b8d4059189030e3a`  
**Target Reviewers:** Primary Research Supervisor, Thesis Committee, Departmental Review Board  

---

## 1. Executive Synthesis & Review Dossier Overview

This Master Review Dossier consolidates the complete scientific, engineering, and ethical evidence for the **NeuroADHD RawMRI Proof-of-Concept Version 2 (PoC V2)** pipeline. It serves as the single authoritative entrypoint for the Research Supervisor to evaluate the readiness of the codebase, review methodological decisions, and determine whether the project may advance to formal institutional review.

Over development Phases A.1 through A.4.2.1, the pipeline has achieved complete synthetic-only verification, including cryptographic manifest verification, mathematical validation of non-linear spatial normalization, canonical CC200 atlas reconciliation, kernel-enforced 2.5 GB memory isolation, and read-only integration with a frozen ADHD classification model.

> [!IMPORTANT]
> **Strict Phase A.5 Policy: Synthetic-Only Scope**
> - **Execution Status:** 100% strictly synthetic-only.
> - **Real Participant Data Status:** Strictly prohibited. Zero real MRI scans have been processed or staged.
> - **Action Decision:** This dossier requests a determination of `READY_FOR_SUPERVISOR_DECISION`.
> - **Crucial Condition:** A supervisory decision of `APPROVED_FOR_NEXT_REVIEW_STAGE` authorizes advancing to formal institutional ethics (IEC) submission and Data Use Agreement (DUA) applications; it **DOES NOT** authorize running real MRI scans.

---

## 2. Dossier Document Architecture

The Phase A.5 package comprises eight interconnected, specialized documents within `docs/`:

```
                           +----------------------------------------------------+
                           |    DOC-NA-A5-SRP-001: MASTER REVIEW PACKAGE        |
                           |   (PHASE_A_5_SUPERVISOR_REVIEW_PACKAGE.md)         |
                           +----------------------------------------------------+
                                      |                 |
          +---------------------------+                 +---------------------------+
          |                                                                         |
          v                                                                         v
+-----------------------------+                                           +-----------------------------+
| DOC-NA-A5-RES-001           |                                           | DOC-NA-A5-TR-001            |
| Executive Summary           |                                           | Technical Review            |
| (High-level scientific      |                                           | (19-stage mathematical and  |
| overview for leadership)    |                                           | algorithmic specification)  |
+-----------------------------+                                           +-----------------------------+
          |                                                                         |
          v                                                                         v
+-----------------------------+                                           +-----------------------------+
| DOC-NA-A5-SDR-001           |                                           | DOC-NA-A5-EDG-001           |
| Decision Register           |                                           | Ethics & Data Governance    |
| (4 technical decisions &    |                                           | (15-point review: ICMR 2017,|
| 8 QC thresholds awaiting OK)|                                           | DPDP Act 2023, DUA rules)   |
+-----------------------------+                                           +-----------------------------+
          |                                                                         |
          v                                                                         v
+-----------------------------+                                           +-----------------------------+
| DOC-NA-A5-FRP-001           |                                           | DOC-NA-A5-AC-001            |
| First Research Pilot        |                                           | Approval Checklist          |
| (10-stage execution plan on |                                           | (Actionable 10-point gating |
| N=6 retrospective scans)    |                                           | table; all marked PENDING)  |
+-----------------------------+                                           +-----------------------------+
          |
          v
+-----------------------------+
| DOC-NA-A5-SPO-001           |
| Presentation Outline        |
| (12-slide presentation deck |
| for departmental defense)   |
+-----------------------------+
```

---

## 3. 19-Stage Pipeline Technical Summary

The pipeline bridges raw neuroimaging data and machine learning via a strictly separated, two-tier architecture:

| Tier | Stage # | Stage Name | Algorithmic Mechanism | Output Artifact |
|---|---|---|---|---|
| **Tier I: Input** | 1 | Manifest Verification | HMAC-SHA256 signature evaluation (fail-closed) | Verified manifest |
| | 2 | BIDS Validation | Directory syntax & JSON sidecar parsing | Validated BIDS tree |
| **Tier I: Structural** | 5 | Skull Stripping | Adaptive Otsu thresholding + morphological opening | Brain volume + mask |
| | 6 | N4 Bias Correction | Non-parametric non-uniform intensity normalization | Corrected T1w volume |
| | 7 | Tissue Segmentation | 3-class GMM with spatial MRF regularization | CSF, WM, GM masks |
| | 8 | Mask Erosion | Spherical erosion: CSF ($r=1$), WM ($r=2$) | Eroded tissue masks |
| **Tier I: Functional** | 3 | Motion Estimation | 6-DOF Gauss-Newton rigid realignment | 6 motion params, FD, DVARS |
| | 4 | Reference Vol Gen | Median volume across low-motion timepoints ($\text{FD} < 0.2$) | 3D BOLD reference |
| | 9 | EPI-to-T1w Coreg | 6-DOF rigid Normalized Mutual Information | Rigid affine $T_{\text{EPI}\to\text{T1w}}$ |
| **Tier I: Normalization** | 10 | T1w-to-MNI SyN | Affine + Symmetric Diffeomorphic Normalization | Diffeomorphic warp $\Phi$ |
| | 11 | Transform Composition| $\Phi_{\text{total}} = \Phi_{\text{T1w}\to\text{MNI}} \circ T_{\text{EPI}\to\text{T1w}} \circ T_{\text{motion}}$ | Composite pullback field |
| | 12 | CC200 Resampling | Single-step 4th-order B-spline interpolation | BOLD in CC200 ($63 \times 75 \times 61$) |
| **Tier I: Denoising** | 13 | Nuisance Extraction | aCompCor (5 CSF + 5 WM) + Friston-24 motion | Design matrix ($N_t \times 34$) |
| | 14 | Regression & Filtering | Simultaneous OLS regression + bandpass ($0.01\text{--}0.10\text{ Hz}$) | Denoised 4D BOLD |
| **Tier I: Features** | 15 | CC200 Atlas Verify | SHA-256 check (`9467afce...`, shape $63 \times 75 \times 61$) | Validated atlas handle |
| | 16 | ROI Time-Series | Spatial mean within each of 200 CC200 clusters | Time-series ($N_t \times 200$) |
| | 17 | FC Matrix Extraction | Pearson correlation $\to$ Fisher-$z$ $\to$ Upper triangle | $1 \times 19{,}900$ vector |
| **Tier II: Frozen ML** | 18 | Feature Preprocessing | Read-only frozen `Pipeline(Imputer, VarThresh, Scaler, SelectKBest-100)` | $1 \times 100$ feature vector |
| | 19 | Classifier Inference | Read-only frozen `RandomForestClassifier(50 trees, depth 3)` | Class probabilities & call |

---

## 4. Verification Evidence & Provenance Synthesis

The pipeline is verified through automated test suites, container memory tests, and cryptographic integrity checks:

### Automated Test Suite Outcomes:
- **Baseline CI Suite (GitHub Actions Run #41433992271):**
  - **176 Passed, 3 Skipped** (skips occur strictly when `NEUROADHD_MODEL_DIR` is unprovisioned in public CI).
  - Test run duration: 46.38 seconds. Statement coverage: 83%.
- **Local Model-Enabled Suite:**
  - **179 Passed, 0 Skipped, 0 Failed** when authentic frozen model directory is provisioned.
- **Container Memory & Cgroup Isolation Suite (`scripts/verify_container_memory.sh`):**
  - Test 1: Kernel cgroup memory limit confirmed at $2{,}684{,}354{,}560\text{ bytes}$ ($2.5\text{ GiB}$) — `PASSED`.
  - Test 2: Linux kernel OOM-killer termination verified (exit code 137 / SIGKILL) — `PASSED`.
  - Test 3: Fail-closed behavior on unconstrained host environment verified — `PASSED`.

### Cryptographic Baseline Provenance:
- **Canonical CC200 Atlas:** `9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650`
- **Frozen Preprocessing Pipeline (`final_preprocessing_pipeline.joblib`):** `228d3f0180e4855d49fd732327ab1fc4f255beb7bea43dd4fd1db3064aa86249`
- **Frozen Classifier (`final_model.joblib`):** `d8eb23eb8c6d25ffc12b2bab515a2a7112daeeb0afa8f4e9eaa499dbe5b5f2ef`

---

## 5. Non-Clinical Disclaimer & Methodological Limitations

> [!WARNING]
> **Essential Non-Clinical Disclaimer:**
> NeuroADHD RawMRI PoC V2 is an investigational research pipeline engineered for computational neuroimaging exploration. It is **not** a cleared Medical Device (SaMD), clinical diagnostic tool, or clinical decision support system. Its predictions must never be utilized to influence clinical management, diagnosis, or patient care.

### Primary Research Limitations:
1. **Synthetic-Only Validation:** Biological variations, scanner phase drift, and real anatomical anomalies have not yet been evaluated in this pipeline.
2. **Exploratory Classifier Architecture:** The downstream Random Forest classifier is a shallow historical benchmark (50 trees, depth 3) intended to test software interfaces, not to serve as an optimized diagnostic engine.
3. **Hardware & Protocol Domain Shift:** Differences between historical training data (ADHD-200) and modern MRI scanners (multiband, 32-channel coils, advanced pulse sequences) may induce systematic feature distribution shift.
4. **Pediatric Motion Susceptibility:** Extreme motion in hyperactive pediatric cohorts can compromise covariance estimation despite aCompCor and scrubbing.

---

## 6. Synthesis of Supervisor Decisions Required

Full details and options are presented in [PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md). The four key technical decisions and eight QC thresholds are summarized below:

### Technical Decisions Awaiting Direction:
1. **Slice-Timing Correction:** Option B (BIDS-driven for $TR \ge 1.5\text{s}$) vs Option C (Adaptive fallback).
2. **Susceptibility Distortion Correction:** Option C (Fieldmaps preferred; synthetic syn-b0 fallback for historical data).
3. **Facial Defacing Approach:** Option C (`pydeface` coupled with automated 0% brain mask erosion audit).
4. **Prospective Acquisition Protocol:** Review of proposed 3.0T, 32-channel, 1mm T1w, 3mm rs-fMRI standard.

### 8 QC Acceptance Criteria (All Marked `PENDING_SUPERVISOR_APPROVAL`):
1. T1w Brain Volume: $600\text{--}2200\text{ cm}^3$
2. EPI-to-T1w NMI: $\ge 0.55$
3. T1w-to-MNI SyN Dice: $\ge 0.85$
4. Mean Framewise Displacement: $\le 0.50\text{ mm}$
5. Maximum Motion Spike: Flag $> 3.0\text{ mm}$, Reject $> 5.0\text{ mm}$
6. Scrubbing Proportion: $\le 30\%$ of total run
7. Temporal SNR: Median $\ge 40$, Flag $< 30$
8. DVARS Outliers: $> 1.5 \times \text{median}(\text{DVARS})$

---

## 7. Ethics, Privacy & Indian Regulatory Governance Synthesis

Full details are presented in [PHASE_A_5_ETHICS_AND_DATA_GOVERNANCE_REVIEW.md](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_ETHICS_AND_DATA_GOVERNANCE_REVIEW.md):
- **Indian Institutional Context:** Compliance with **ICMR 2017 Guidelines** (pediatric vulnerability, essentiality, non-exploitation) and the **DPDP Act 2023** (Section 9 children's data obligations, fiduciary responsibilities).
- **Public Data Governance:** Debunking the notion that ADHD-200 is public domain; a formal institutional Data Use Agreement (DUA) through NITRC/INDI is mandatory.
- **De-Identification & Security:** 18 HIPAA Safe Harbor identifiers and Indian statutory identifiers stripped; AES-256 encryption at rest; air-gapped participant linkage key storage.
- **Current Operational Reality:** No active IEC clearance or DUA exists. Real data ingestion remains disabled.

---

## 8. Proposed First Research Pilot Synthesis

Full protocol is presented in [PHASE_A_5_FIRST_RESEARCH_PILOT_PROTOCOL.md](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_FIRST_RESEARCH_PILOT_PROTOCOL.md):
- **Cohort:** Minimal feasibility cohort of $N=6$ retrospective scans from ADHD-200 (3 TDC, 3 ADHD).
- **Goal:** Stress-test memory isolation, SyN registration convergence, and 19-stage continuity under real biological data.
- **Success Criteria:** 0% crashes, RAM $< 2.3\text{ GB}$, SyN Dice $\ge 0.85$, valid $19{,}900$ FC vector. Classification accuracy is explicitly not an evaluation metric.
- **Gate:** Requires prior IEC clearance, DUA execution, and supervisor sign-off.

---

## 9. Master Governance Checklist Status

Full checklist is presented in [PHASE_A_5_APPROVAL_CHECKLIST.md](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_APPROVAL_CHECKLIST.md):
- All 10 mandatory governance items (`REQ-01` through `REQ-10`) are strictly marked:
  $$\mathbf{PENDING}$$
- No fake approvals or dates are recorded. Real MRI processing remains blocked.

---

## 10. Master Supervisor Review & Formal Action Block

> [!IMPORTANT]
> **Instructions for the Research Supervisor:**
> Please review the accompanying documentation and indicate your formal determination below.
> 
> Selecting **`APPROVED_FOR_NEXT_REVIEW_STAGE`** certifies that the technical, mathematical, and ethical documentation is complete and approves proceeding to formal Institutional Ethics Committee (IEC) submission and Data Use Agreement (DUA) execution.
> 
> **Notice:** Selecting `APPROVED_FOR_NEXT_REVIEW_STAGE` **DOES NOT** authorize running real MRI scans or participant data. Real MRI ingestion remains strictly disabled until all items in `DOC-NA-A5-AC-001` are certified.

### Formal Action Determination:

```
[ ] APPROVED_FOR_NEXT_REVIEW_STAGE
    The Phase A.5 review package is accepted. The research team is authorized
    to prepare formal submissions for Institutional Ethics Committee (IEC) review
    and to initiate the Data Use Agreement (DUA) process for retrospective data.
    Real participant scan processing remains disabled.

[ ] REVISION_REQUIRED
    Specific modifications to the technical architecture, QC criteria, or
    governance protocols are required as detailed in the Supervisor Comments below.

[ ] NOT_APPROVED
    The proposal is rejected in its current form.
```

### Supervisor Review Record:

**Supervisor Name:** `__________________________________________________`  
**Academic Title / Designation:** `__________________________________________________`  
**Department / Academic Unit:** `__________________________________________________`  
**Institution:** `__________________________________________________`  
**Date of Review:** `YYYY-MM-DD`  

**Supervisor Comments / Stipulations:**  
`___________________________________________________________________________________`  
`___________________________________________________________________________________`  
`___________________________________________________________________________________`  

**Supervisor Signature:**  
`[ PENDING SUPERVISOR REVIEW AND SIGNATURE ]`

---

## 11. Dossier Final Status

**Operational Phase A.5 Decision:**  
$$\mathbf{READY\_FOR\_SUPERVISOR\_DECISION}$$  
*(Real MRI processing remains strictly disabled. Zero real participant data may be processed.)*

