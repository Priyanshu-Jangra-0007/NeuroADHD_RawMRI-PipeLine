# ROLES AND INSTITUTIONAL RESPONSIBILITIES MATRIX

> **`DRAFT — NOT SUBMITTED — NOT APPROVED`**
>
> **Status:** `DRAFT_FOR_INSTITUTIONAL_CONFIRMATION`  
> **Document ID:** `DOC-GOV-ROLE-001`  
> **Project:** NeuroADHD — ADHD Classification Using Resting-State Functional MRI Connectivity and Machine Learning  
> **Rule:** Personnel names are intentionally left as placeholders (`[... ]`); **only competent institutional authorities may approve human-participant research or authorize real-data access.**

---

## 1. Governance Authority Matrix (Recommend / Approve / Authorize / Execute / Audit)

| Institutional Role | Placeholder Identity | May Recommend | May Approve (Governance / Ethics) | May Authorize Real-Data Execution | May Execute Work | May Audit / Suspend |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Principal Investigator (PI)** | `[PI_NAME]` (`[PI_AFFILIATION]`) | Protocol submission, consent waiver/forms, DUA filings, team access list | Internal protocol submission sign-off (**NOT** final IEC approval) | May request authorization after IEC + DUA + DPO clearance (**cannot self-grant IEC approval**) | Oversees study conduct, linkage-key custody (air-gapped), incident escalation | Audits team compliance; can immediately suspend study access |
| **Academic Supervisor / Guide** | `[SUPERVISOR_NAME]` (`[SUPERVISOR_DEPT]`) | Technical readiness (`DEC-STC-01`, `DEC-SDC-02`, `DEC-DEF-03`, `DEC-ACQ-04`, `QC-01..08`), pilot scope (`SAFE-PILOT-06`) | Endorses scientific protocol & supervisor checklist items in `REG-NA-A6-001` (**cannot replace IEC**) | Co-signs technical readiness & register review alongside institutional clearances | Reviews validation benchmarks & thesis progress | Audits scientific validity, reproducibility, and safety adherence |
| **Co-Investigator(s) / Site Clinician** *(if applicable)* | `[CO_INVESTIGATOR_NAMES]` *(e.g., clinical/radiology lead if Tier 3)* | Clinical screening rules, incidental-findings escalation pathway | Signs site feasibility / clinical supervision undertaking (if Tier 3 prospective) | **No** (cannot unlock pipeline unilaterally) | Conducts clinical screening / visual MRI safety review (if Tier 3) | Flags clinical safety concerns to PI & `[IEC_NAME]` |
| **Student Developer / Researcher** | `[STUDENT_DEVELOPER]` | Software improvements, synthetic test benchmarks, draft governance templates | **No** (cannot approve ethics, privacy, or self-unlock data) | **No** (strictly prohibited from bypassing `ResearchAuthorizationGate`) | Implements & tests code strictly on **synthetic** data; executes coded runs **only** after full authorization | Runs CI hygiene & automated synthetic test verification |
| **Institutional Ethics Committee (IEC)** | `[IEC_NAME]` (DHR Reg. No.: `[IEC_REGISTRATION_NUMBER]`) | Protocol revisions, risk mitigations, regional consent requirements | **YES — Sole authority** to grant/reject/waive IEC ethics approval (`AUTH-IEC-01`, `AUTH-DSU-03`) | Institutional ethics prerequisite for any Tier 2/Tier 3 human data | Independent ethical review & continuing review | Audits protocol adherence; may suspend or revoke ethical clearance |
| **Institutional Data Custodian / IT Security** | `[DATA_CUSTODIAN]` / `[IT_SECURITY_OFFICER]` | Encrypted storage architecture, backup/destruction SOPs | Approves infrastructure security (`SAFE-STOR-03`, `SAFE-ACC-04`, `SAFE-AUD-05` technical controls) | Enforces technical access provisioning **only** post-IEC/DUA/DPO sign-off | Provisions AES-256 volumes, OS permissions, backups, NIST 800-88 wiping | Audits access logs, cloud-sync blocks, and disk encryption |
| **Data Protection Officer (DPO) / Institutional Legal** | `[DPO_NAME]` / `[INSTITUTION_LEGAL_OFFICE]` | DPDP Act 2023 / Rules 2025 applicability, HMSC/DUA terms | Approves statutory privacy assessment (`AUTH-SUA-04`, `AUTH-DPP-05`) & legal DUA checks | Legal/privacy clearance prerequisite prior to real data staging | Reviews DUA, retention period, breach reporting (72h DPBI workflow) | Conducts privacy compliance audits & handles statutory breach notices |
| **Dataset Owner / Data Provider** | **INDI / NITRC / ADHD-200 Consortium** (Tier 2); Participating Institution (Tier 3) | Clarification of split manifests or derivative rules (`REQUIRES_DATA_OWNER_CONFIRMATION`) | Sets & interprets upstream DUA / CC-BY-NC non-commercial licensing conditions | Grants portal/DUA account access under provider terms (**never substitutes for local IEC**) | Maintains upstream repository notices & citation rules | May revoke DUA access upon redistribution or re-identification breach |

---

## 2. Explicit Separation of Powers Invariant

1. **Developer / AI Coding Agent Separation:** Neither the student developer nor automated engineering tools may grant ethical clearance, claim DPDP exemption, generate cryptographic authorization signatures, or alter `configs/research_authorization_register.json` without verified institutional approval artifacts.
2. **Supervisor vs. Ethics Authority Separation:** The Academic Supervisor reviews and approves technical & methodological quality (`DEC-*`, `QC-*`, `SAFE-PILOT-06`), whereas **only `[IEC_NAME]`** (supported by `[DPO_NAME]` / Legal & `[DATA_CUSTODIAN]`) can approve human-subjects ethics, consent waivers, or statutory privacy compliance.
