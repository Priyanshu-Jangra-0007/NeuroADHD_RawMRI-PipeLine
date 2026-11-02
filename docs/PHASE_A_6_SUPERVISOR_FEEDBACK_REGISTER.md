# NeuroADHD RawMRI PoC V2 — Phase A.6
# Supervisor Feedback Register & Scientific Decision Log

**Document ID:** `DOC-NA-A6-SFR-001`  
**Version:** 1.0.0  
**Phase:** Phase A.6 — Supervisor Feedback Intake & Research Authorization Gate  
**Status:** `PENDING_SUPERVISOR_FEEDBACK`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Review Target:** Primary Research Supervisor & Thesis Evaluation Committee  
**Machine-Readable Twin:** [`configs/research_authorization_register.json`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/configs/research_authorization_register.json)  

---

## 1. Preamble & Intake Policy

This register provides the formal, structured intake log for capturing, tracking, and auditing the Primary Research Supervisor's decisions regarding MRI preprocessing, parcellation, and quality control (QC) acceptance criteria for the NeuroADHD RawMRI PoC V2 pipeline.

Under Phase A.6:
- Every decision item is actively tracked in both human-readable Markdown (this document) and machine-readable JSON ([`configs/research_authorization_register.json`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/configs/research_authorization_register.json)).
- **Current Operational Status:** All items below remain strictly:
  $$\mathbf{PENDING}$$
- No supervisor decision will be recorded as `APPROVED` without verified supporting evidence and formal sign-off.

---

## 2. Category A: Technical Preprocessing Decisions

### DEC-STC-01: Slice-Timing Correction (STC) Policy
- **Description:** Formally specify the policy for slice-timing correction across prospective acquisitions and retrospective cohorts with variable repetition time ($\text{TR}$) and multiband acceleration.
- **Evaluated Options:**
  - `OPTION_A`: Universal bypass across all acquisitions (prevents temporal blurring; retains slice phase delays).
  - `OPTION_B`: Mandatory BIDS-metadata-driven STC for $\text{TR} \ge 1.5\text{ s}$; bypass for $\text{TR} < 1.0\text{ s}$.
  - `OPTION_C`: Adaptive fallback: require BIDS `SliceTiming` when $\text{TR} \ge 2.0\text{ s}$; fail closed if absent unless explicit override is logged.
- **Engineering Recommendation:** Option B for prospective clinical scanning; Option C for mixed retrospective public repositories.
- **Current Status:** `PENDING`
- **Responsible Reviewer:** `[ PENDING SUPERVISOR ASSIGNMENT ]`
- **Decision Date:** `PENDING`
- **Selected Option:** `PENDING`
- **Evidence Reference:** `docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md`
- **Reviewer Comments:** `[ Awaiting supervisor review ]`
- **Revision History:**
  | Timestamp | Previous Status | New Status | Changed By | Reason / Evidence |
  |---|---|---|---|---|
  | 2026-11-02T19:30:00Z | `NONE` | `PENDING` | Phase A.6 Init | Initialized into structured feedback register. |

---

### DEC-SDC-02: Susceptibility Distortion Correction (SDC) Architecture
- **Description:** Establish the architecture for echo-planar imaging (EPI) B0 magnetic susceptibility distortion unwarping near tissue-air boundaries.
- **Evaluated Options:**
  - `OPTION_A`: Mandatory dual phase-encoding spin-echo EPI pairs (FSL TOPUP).
  - `OPTION_B`: Dual-echo gradient-echo fieldmaps ($\Delta\text{TE}$).
  - `OPTION_C`: Hierarchical fallback: require formal fieldmaps for prospective protocols; allow synthetic deformation estimation (syn-b0) with graceful fallback to rigid coregistration for historical public benchmarks.
- **Engineering Recommendation:** Option C to support both legacy ADHD-200 benchmarks and modern prospective scanner protocols.
- **Current Status:** `PENDING`
- **Responsible Reviewer:** `[ PENDING SUPERVISOR ASSIGNMENT ]`
- **Decision Date:** `PENDING`
- **Selected Option:** `PENDING`
- **Evidence Reference:** `docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md`
- **Reviewer Comments:** `[ Awaiting supervisor review ]`
- **Revision History:**
  | Timestamp | Previous Status | New Status | Changed By | Reason / Evidence |
  |---|---|---|---|---|
  | 2026-11-02T19:30:00Z | `NONE` | `PENDING` | Phase A.6 Init | Initialized into structured feedback register. |

---

### DEC-DEF-03: Facial Defacing Methodology & Anonymization
- **Description:** Authorize the anatomical facial defacing protocol ensuring complete participant biometric anonymization while mathematically guaranteeing zero cortical gray matter or frontal-pole tissue erosion.
- **Evaluated Options:**
  - `OPTION_A`: `pydeface` automated defacing.
  - `OPTION_B`: FreeSurfer `mri_deface` multi-template facial stripping.
  - `OPTION_C`: Automated `pydeface` coupled with an algorithmic spatial audit asserting strictly zero intersection ($\text{Voxels} = 0$) between the defacing mask and the structural brain mask.
- **Engineering Recommendation:** Option C (containerizable within 2.5 GB RAM with automated frontal-pole verification).
- **Current Status:** `PENDING`
- **Responsible Reviewer:** `[ PENDING SUPERVISOR ASSIGNMENT ]`
- **Decision Date:** `PENDING`
- **Selected Option:** `PENDING`
- **Evidence Reference:** `docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md`
- **Reviewer Comments:** `[ Awaiting supervisor review ]`
- **Revision History:**
  | Timestamp | Previous Status | New Status | Changed By | Reason / Evidence |
  |---|---|---|---|---|
  | 2026-11-02T19:30:00Z | `NONE` | `PENDING` | Phase A.6 Init | Initialized into structured feedback register. |

---

### DEC-ACQ-04: Prospective MRI Acquisition Protocol Parameters
- **Description:** Formally specify prospective scanner hardware and pulse sequence parameters to minimize domain shift relative to the CC200 training baseline.
- **Proposed Parameters:**
  - Field Strength: 3.0 Tesla; RF Head Coil: Minimum 32 channels.
  - T1w MP-RAGE: $1.0\text{ mm}$ isotropic, $\text{TI} = 900\text{ ms}$, $\text{FA} = 8^\circ$.
  - rs-fMRI BOLD: $3.0\text{ mm}$ isotropic, $\text{TR} = 2000\text{ ms}$, $\text{TE} = 30\text{ ms}$, $\text{FA} \approx 77^\circ$, duration $\ge 6\text{ minutes}$ (180 volumes), eyes open with central fixation.
- **Current Status:** `PENDING`
- **Responsible Reviewer:** `[ PENDING SUPERVISOR ASSIGNMENT ]`
- **Decision Date:** `PENDING`
- **Selected Option:** `PENDING`
- **Evidence Reference:** `docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md`
- **Reviewer Comments:** `[ Awaiting supervisor review ]`
- **Revision History:**
  | Timestamp | Previous Status | New Status | Changed By | Reason / Evidence |
  |---|---|---|---|---|
  | 2026-11-02T19:30:00Z | `NONE` | `PENDING` | Phase A.6 Init | Initialized into structured feedback register. |

---

## 3. Category A (Cont.): Quality Control (QC) Acceptance Thresholds

| Decision ID | Metric Name & Mathematical Target | Proposed Threshold | Current Status | Responsible Reviewer | Decision Date | Evidence Reference |
|---|---|---|---|---|---|---|
| **QC-01** | Total Intracranial Brain Volume ($V_{\text{brain}}$) | **$600\text{--}2200\text{ cm}^3$** | `PENDING` | `[ PENDING ]` | `null` | `docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md` |
| **QC-02** | EPI-to-T1w Coregistration Normalized Mutual Information | **$\text{NMI} \ge 0.55$** | `PENDING` | `[ PENDING ]` | `null` | `docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md` |
| **QC-03** | T1w-to-MNI SyN Normalization Dice Overlap | **$\text{Dice} \ge 0.85$** | `PENDING` | `[ PENDING ]` | `null` | `docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md` |
| **QC-04** | Mean Framewise Displacement ($\overline{\text{FD}}$) | **$\le 0.50\text{ mm}$** | `PENDING` | `[ PENDING ]` | `null` | `docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md` |
| **QC-05** | Maximum Motion Spike ($\text{FD}_{\max}$) | **Flag $> 3.0\text{ mm}$<br>Reject $> 5.0\text{ mm}$** | `PENDING` | `[ PENDING ]` | `null` | `docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md` |
| **QC-06** | Scrubbing / Censoring Proportion ($P_{\text{scrub}}$) | **$\le 30\%$ of total run** | `PENDING` | `[ PENDING ]` | `null` | `docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md` |
| **QC-07** | Temporal Signal-to-Noise Ratio (tSNR) | **Median $\ge 40$<br>Flag $< 30$** | `PENDING` | `[ PENDING ]` | `null` | `docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md` |
| **QC-08** | DVARS Outlier Metric Ceiling | **No volumes $> 1.5 \times \text{median}$** | `PENDING` | `[ PENDING ]` | `null` | `docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md` |

---

## 4. Feedback Intake Workflow & Synchronous Audit Procedure

When supervisor feedback is received, the following 4-step synchronization procedure must be executed:

1. **Step 1: Record Supervisor Action in Markdown:**
   Update the corresponding section above with the selected option, date, reviewer name, and notes.
2. **Step 2: Deposit Physical Evidence File:**
   Place the formal approval memo, signed decision block, or email confirmation in `docs/evidence/`. Compute its SHA-256 hash.
3. **Step 3: Update Machine-Readable JSON Register:**
   Update `configs/research_authorization_register.json` with the matching status, reviewer, date, option, evidence path, and SHA-256 hash.
4. **Step 4: Execute Programmatic Audit Trail Logging:**
   Call `ResearchAuthorizationGate.record_audit_event()` to append an immutable audit log record.
5. **Step 5: Run Automated Validation Suite:**
   Execute `pytest tests/unit/test_phase_a6_authorization_gate.py` to confirm that the updated register passes schema, evidence, and conflict checks.

---

## 5. Summary State

- Total Supervisor Decisions Tracked: 12
- Decisions Pending Approval: 12
- Decisions Approved: 0
- Authorization Gate Outcome: `NOT_AUTHORIZED`

