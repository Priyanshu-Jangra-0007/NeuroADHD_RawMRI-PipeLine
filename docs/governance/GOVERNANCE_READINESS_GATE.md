# GOVERNANCE READINESS GATE & REVIEW CHECKLIST

> **`DRAFT — NOT SUBMITTED — NOT APPROVED`**
>
> **Status:** `DRAFT_INCOMPLETE` *(Documentation templates complete; missing institutional metadata, supervisor signatures, DUA archive, DPO determination, and IEC decision)*  
> **Document ID:** `DOC-GOV-GATE-001`  
> **Tracker Separation Invariant:** This documentation readiness checklist and [`APPROVAL_TRACKER.csv`](APPROVAL_TRACKER.csv) are **strictly separate** from the runtime fail-closed gate in `configs/research_authorization_register.json` and `src/validation/authorization_gate.py`. Completing items here **never** unlocks real MRI data processing.

---

## 1. Allowed Governance Readiness State Machine

Every governance artifact and submission gate uses one of these exact seven statuses and **cannot** be automatically promoted to `APPROVED_BY_IEC`:

- `DRAFT_INCOMPLETE` — Required placeholders (`[PI_NAME]`, `[IEC_NAME]`, `[INSTITUTIONAL_RETENTION_YEARS]`, etc.) or mandatory forms/evidence remain unpopulated.
- `READY_FOR_SUPERVISOR_REVIEW` — Technical & draft governance documents internally complete and ready for `[SUPERVISOR_NAME]` academic/methodological audit.
- `READY_FOR_IEC_SUBMISSION` — All institutional placeholders populated, COI/undertakings signed, DUA & infosec checks attached, ready to lodge with `[IEC_NAME]`.
- `SUBMITTED_PENDING_REVIEW` — Formally submitted to `[IEC_NAME]` / `[DPO_NAME]` with dated submission receipt (`submission_date` populated).
- `APPROVED_BY_IEC` — **Requires genuine written approval letter** (protocol number, approval date, validity dates, authorized committee signature) verified by an authorized institutional official.
- `REJECTED_OR_REVISION_REQUIRED` — Returned by supervisor, DPO, or `[IEC_NAME]` for modifications prior to re-submission.
- `NOT_APPLICABLE_PENDING_AUTHORITY_CONFIRMATION` — Conditional workstream (e.g., Tier 3 prospective collection or HMSC / 2026 Single EC multicentre reliance) awaiting written determination on applicability.

---

## 2. Gate Evaluation Checklist (Current Posture: `2026-11-12`)

| Gate Criterion | Required Evidence (Genuine Human/Institutional Record Needed) | Automated Promotion Allowed? | Current Gate Status | Blocking Missing Details |
| :--- | :--- | :--- | :--- | :--- |
| **G-01: Scientific Protocol & 23-Stage Technical Documentation** | Signed protocol (`DOC-GOV-PROT-001`) + supervisor decision sign-off on `DEC-STC-01`, `DEC-SDC-02`, `DEC-DEF-03`, `DEC-ACQ-04`, `QC-01..08`. | **NO** | `READY_FOR_SUPERVISOR_REVIEW` *(for technical draft review only)* | Requires `[PI_NAME]` & `[SUPERVISOR_NAME]` names, affiliations, and dated signatures; host lacks `pydeface`/`FSL` & `topup`/`prelude`/`fugue`. |
| **G-02: Dataset Provenance & ADHD-200 DUA Archive** | Executed NITRC/INDI DUA proof (`AUTH-DUA-02`) + split-manifest check reconciling `153` (`50/103`) vs. `134` (`114+20`) subject counts. | **NO** | `DRAFT_INCOMPLETE` | Signed/confirmed DUA record (`[EXECUTED_DUA_PDF]`), named NITRC user roster, and derivative-sharing confirmation from provider/institution. |
| **G-03: Regulatory & DPDP Act 2023 / Rules 2025 Determination** | Written DPO / Institutional Legal memo answering Sections 9, 8(5), 17(2)(b) + Rule 16 Second Schedule, and DHR/HMSC screening. | **NO** | `DRAFT_INCOMPLETE` | Formal `[DPO_NAME]` / Legal applicability ruling (`AUTH-SUA-04`, `AUTH-DPP-05`) and `[INSTITUTIONAL_RETENTION_YEARS]`. |
| **G-04: Institutional Infrastructure & Defacing Operational Readiness** | Verified AES-256 volume encryption (`SAFE-STOR-03`), cloud-sync exclusion, backup/NIST 800-88 destruction SOP, and installed `pydeface` + FSL + visual QC SOP. | **NO** | `DRAFT_INCOMPLETE` | Signed `[IT_SECURITY_OFFICER]` / `[DATA_CUSTODIAN]` audit checklist + operational defacing binary setup. |
| **G-05: Local IEC Submission & Ethics Determination (Tier 2 Secondary Use)** | Formal written `[IEC_NAME]` determination (approval or formal administrative exemption + ICMR Sec 5.7 consent waiver) with DHR NECRBHR registration number verified. | **NO** | `DRAFT_INCOMPLETE` | `[IEC_NAME]`, `[IEC_REGISTRATION_NUMBER]`, institutional proforma, signed CVs/COI/undertakings, and genuine committee ruling. |
| **G-06: Conditional Prospective $N=6$ Pilot Package (Tier 3)** | Full IEC prospective protocol approval, parental consent (`DOC-GOV-ICF-001`), pediatric assent, PIS (`DOC-GOV-PIS-001`), recruitment script, and radiology incidental-findings SOP. | **NO** | `NOT_APPLICABLE_PENDING_AUTHORITY_CONFIRMATION` | Confirmation of whether any new human scans ($N=6$) will actually be acquired vs. strictly retrospective secondary ADHD-200 feasibility. |
| **G-07: Production Pipeline Runtime Unlock (`sub-01` / Real Data)** | External, authorized completion & cryptographic signing of all `23` items in `configs/research_authorization_register.json`. | **NEVER** | `NOT_AUTHORIZED` (`PENDING`) | **Intentionally locked (`STRICTLY_SYNTHETIC_ONLY`)**; Step 7 never alters production authorization state. |

---

## 3. Overall Step 7 Package Readiness Determination

- **Overall Readiness Status for Academic/Supervisor Review:** Technical & draft governance dossier documents are complete (`READY_FOR_SUPERVISOR_REVIEW` strictly as **unsubmitted drafts**); overall institutional ethics submission state remains **`DRAFT_INCOMPLETE`** until `[INSTITUTION_NAME]` personnel, `[IEC_NAME]` registration details, DUA archive, retention schedule, DPO ruling, and signed undertakings are supplied by authorized human officials.
- **No Unattended Promotion:** Transitioning any item to `APPROVED_BY_IEC` requires uploading/attaching an authentic, signed IEC clearance document verified by an authorized institutional official—automated scripts or AI agents must never mark an item approved.
