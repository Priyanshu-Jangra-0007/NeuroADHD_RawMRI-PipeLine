# NeuroADHD RawMRI PoC V2 — Phase A.8
# External Research Governance & Institutional Approval Checklist

**Document ID:** `DOC-NA-A8-EAC-001`  
**Version:** 1.0.0  
**Phase:** Phase A.8 — Supervisor Handoff & External Research Authorization Preparation  
**Target Audience:** Institutional Review Board (IRB), Ethics Committee, Data Protection Officer, Research Supervisor  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Baseline Commit:** [`0547574a6db2eb40668046e065fe8dcf989f09b1`](https://github.com/Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine/commit/0547574a6db2eb40668046e065fe8dcf989f09b1)  
**Current Governance Posture:** `STRICTLY_PENDING_EXTERNAL_APPROVALS`  
**Real-Data Execution Gate:** `FAIL_CLOSED_BLOCKED`

---

## 1. Compliance Disclaimer & Purpose

> [!CAUTION]
> **Legal and Bioethical Disclaimer:**  
> This checklist tracks institutional, legal, and operational governance prerequisites. This document **does not constitute legal advice, statutory certification, or an official institutional approval letter**.  
> In compliance with ICMR 2017 Guidelines for Biomedical Research and the Digital Personal Data Protection (DPDP) Act 2023, **no real participant MRI data may be downloaded, ingested, transferred, or processed** until authentic evidence files and written institutional approvals are secured and cryptographically verified.

The checklist below consolidates all **eleven institutional and operational governance requirements** established across Phase A.5 and Phase A.6. Every item corresponds directly to an entry in the machine-readable authorization register ([`configs/research_authorization_register.json`](<your-project-folder>/configs/research_authorization_register.json)).

---

## 2. Master Institutional Governance Tracking Table

| Req ID | Governance Domain & Title | Required Document / Physical Evidence | Responsible Institutional Role | Current Status | Verification Method | Outstanding Action Before Real Data |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **AUTH-IEC-01** | Institutional Ethics Clearance | Formal written IEC/IRB approval letter with protocol number and validity window | Chair, Institutional Ethics Committee | `PENDING` | PDF existence + SHA-256 hash match against register | Submit protocol to IEC; obtain signed approval letter |
| **AUTH-DUA-02** | ADHD-200 Data Use Agreement | Counter-signed Data Use Agreement through NITRC / INDI platform | Research Supervisor & Institutional Signing Official | `PENDING` | Executed DUA document + SHA-256 match | Complete NITRC DUA application; secure institutional signature |
| **AUTH-DSU-03** | Pediatric Consent & Assent | Documented parental consent and child assent protocol, or formal secondary-use waiver | Principal Investigator / IEC | `PENDING` | Formal waiver or consent protocol + SHA-256 match | File for secondary-use waiver under ICMR 2017 Chapter 5 |
| **AUTH-SUA-04** | Statutory Compliance Review | Legal/regulatory memo certifying adherence to ICMR 2017 & DPDP Act 2023 Sec. 9 | University Legal Counsel / Bioethics Advisor | `PENDING` | Legal compliance memo + SHA-256 match | Request statutory review memo from legal counsel |
| **AUTH-DPP-05** | Data Protection Officer Review | Institutional DPO audit verifying purpose limitation, storage limits, and non-reidentification | Institutional Data Protection Officer | `PENDING` | DPO audit certification report + SHA-256 match | Submit Data Protection Impact Assessment (DPIA) to DPO |
| **SAFE-DEID-01** | De-Identification Protocol | DICOM scrubbing specification removing 18 HIPAA identifiers & Indian IDs | Lead Research Software Engineer | `PENDING` | Automated tag scrubbing test log + SHA-256 match | Verify DICOM header scrubbing script on test phantom |
| **SAFE-FACE-02** | Facial Defacing Verification | Post-defacing volumetric audit proving 0-voxel cortical gray matter erosion | Neuroimaging Pipeline Architect | `PENDING` | Algorithmic audit script passing with zero mask overlap | Execute pydeface + cortical mask audit on test subject |
| **SAFE-STOR-03** | Storage Encryption & Memory | Verification of LUKS AES-256 partition and 2.5 GB Linux cgroup memory limit | System Administrator / DevOps | `PENDING` | `cryptsetup status` & `scripts/verify_container_memory.sh` log | Configure encrypted partition; run verification script in production |
| **SAFE-ACC-04** | Least-Privilege & Non-Root | Linux group/file permissions audit and container non-root execution (UID 10001) | Security Engineer | `PENDING` | Container audit script confirming UID 10001 execution | Inspect production deployment host file permissions |
| **SAFE-AUD-05** | Breach Response Procedure | Approved Standard Operating Procedure (SOP) for breach response and 72-hr notification | Chief Information Security Officer / PI | `PENDING` | Signed SOP document + SHA-256 match | Author and obtain sign-off on 72-hour breach SOP |
| **SAFE-PILOT-06** | Pilot Protocol Sign-Off | Signed approval of First Research Pilot Protocol (`DOC-NA-A5-FRP-001`, $N=6$) | Primary Research Supervisor | `PENDING` | Signed protocol document + SHA-256 match | Secure supervisor signature on Phase A.5 protocol |

---

## 3. Detailed Requirements Breakdown

### 3.1 Requirement AUTH-IEC-01: Institutional Ethics Committee Clearance
- **Regulatory Framework:** ICMR National Ethical Guidelines for Biomedical and Health Research (2017).
- **Mandate:** Any computational research utilizing pediatric neuroimaging data—even anonymized or public repositories—requires institutional review to verify risk categorization and ethical justification.
- **Verification Criteria:** An official PDF approval letter on institutional letterhead bearing:
  1. Protocol Registration Number.
  2. Principal Investigator and Student names.
  3. Approval Date and Expiration Date.
  4. Specific authorization for computational neuroimaging analysis.
- **Fail-Closed Gatekeeper Check:** The authorization gate asserts that `AUTH-IEC-01` has status `APPROVED`, a valid document path in `evidence_reference`, and matching 64-character SHA-256 hash.

### 3.2 Requirement AUTH-DUA-02: Data Use Agreement Execution for ADHD-200
- **Regulatory Framework:** International Neuroimaging Data-sharing Initiative (INDI) / Neuro Bureau.
- **Mandate:** Users of the ADHD-200 Consortium dataset must execute a formal Data Use Agreement agreeing to:
  1. Use data strictly for scientific and educational purposes.
  2. Never attempt to re-identify participants or reverse-engineer anonymization.
  3. Acknowledge contributing centers in publications.
  4. Restrict raw data access to authorized project personnel.
- **Verification Criteria:** Executed agreement returned by INDI/NITRC coordinators.

### 3.3 Requirement AUTH-DSU-03: Secondary-Use Consent & Assent Protocols
- **Regulatory Framework:** ICMR 2017 Guidelines, Chapter 5 (Informed Consent in Secondary Data Use).
- **Mandate:** Retrospective analysis of pediatric datasets requires either:
  1. Verification that original parental consent and child assent encompassed downstream secondary computational research, OR
  2. A formal waiver of consent granted by the IEC on grounds of minimal risk and practical impossibility of re-contacting historical participants.

### 3.4 Requirement AUTH-SUA-04 & AUTH-DPP-05: Indian Statutory Privacy Compliance
- **Regulatory Framework:** Digital Personal Data Protection (DPDP) Act 2023 (Section 9: Processing of personal data of children) and Information Technology (SPDI) Rules 2011.
- **Mandate:**
  1. Processing of children's data must not cause harm to the child.
  2. Processing must strictly adhere to purpose limitation (biomarker discovery for ADHD).
  3. Data storage must be geographically and logically bounded with strict storage limitation.
- **Verification Criteria:** Formal sign-off memo from the institutional Data Protection Officer.

### 3.5 Operational Safeguards (SAFE-DEID-01 through SAFE-AUD-05)
- **SAFE-DEID-01:** Automated verification that all 18 HIPAA Safe Harbor identifiers (patient names, dates of scan, hospital accession numbers, physician names) and Indian personal identifiers are wiped from NIfTI/DICOM headers.
- **SAFE-FACE-02:** Structural T1w scans must undergo algorithmic facial defacing (`pydeface`), with mathematically verified 0-voxel overlap between defacing cut-planes and intracranial brain masks.
- **SAFE-STOR-03:** Storage at rest must reside on an AES-256 encrypted partition (LUKS/FileVault), and execution memory must be capped at 2.5 GB RAM via Linux cgroups.
- **SAFE-ACC-04:** Pipeline containers must execute under unprivileged non-root users (`neuro`, UID 10001), mounting host data directories as read-only.
- **SAFE-AUD-05:** Written SOP defining immediate containment, forensic logging, and mandatory 72-hour notification to the Data Protection Board of India and IEC in the event of an unauthorized data breach.

---

## 4. Evidence Incorporation Procedure

When an external approval or evidence document is obtained:
1. Place the final document into the repository's governance evidence directory: `docs/evidence/`.
2. Compute its SHA-256 digest: `sha256sum docs/evidence/<filename>`.
3. Update the corresponding entry in [`configs/research_authorization_register.json`](<your-project-folder>/configs/research_authorization_register.json):
   - Set `"status": "APPROVED"`
   - Set `"evidence_reference": "docs/evidence/<filename>"`
   - Set `"evidence_sha256": "<computed_64_hex_hash>"`
   - Set `"responsible_reviewer": "<reviewer_name>"`
   - Set `"decision_date": "<iso8601_date>"`
4. Run the gate validation engine to verify evidence integrity:
   ```bash
   .venv/bin/pytest tests/unit/test_phase_a6_authorization_gate.py tests/security/test_phase_a7_adversarial_authorization.py
   ```
5. Real data processing will remain blocked until **all 23 register items** are approved and verified.
