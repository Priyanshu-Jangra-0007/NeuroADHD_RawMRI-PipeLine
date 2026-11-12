# REAL-DATA AUTHORIZATION EVIDENCE MATRIX

> **`DRAFT — NOT SUBMITTED — NOT APPROVED`**
>
> **Document ID:** `DOC-GOV-AUTHMAT-001`  
> **Step / Milestone:** Step 8 — Formal Real-Data Authorization Readiness  
> **Current Authorization State:** `NOT_AUTHORIZED` | `execution_policy = STRICTLY_SYNTHETIC_ONLY` | `real_data_authorized = false`  
> **Target Register:** `configs/research_authorization_register.json` (`REG-NA-A6-001`, `23 / 23` items `PENDING`, `sub-01 = PENDING`)  
> **Non-Fabrication Rule:** All unknown institutional details remain explicit placeholders (`[...]`). Software artifacts, passing synthetic tests, and generated PDFs never constitute approval evidence.

---

## 1. Two-Phase Evidence-to-Runtime Authorization Flow

```text
+-------------------------------------------------------------------------+
| 1. INSTITUTIONAL DOCUMENTARY EVIDENCE (Genuine Human / Legal Artifacts) |
|    A. Institutional Sign-off | B. IEC Clearance / Waiver                |
|    C. ADHD-200 DUA / License | D. Participant Scope (N=6 / Coded ID)    |
|    E. Privacy & InfoSec      |                                          |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| 2. PHASE A — EVIDENCE READINESS ENGINE                                  |
|    `src/validation/authorization_evidence.py` (`AuthorizationEvidence`) |
|    Checks: existence, SHA-256, non-draft status, institution/authority, |
|    refs, dates/expiry, protocol version, dataset/ops, storage/sharing,  |
|    participant/session scope, and register alignment.                   |
|    States: PENDING_EVIDENCE / EVIDENCE_INCOMPLETE ->                    |
|            READY_FOR_AUTHORIZATION_REVIEW (NEVER unlocks real MRI)      |
+------------------------------------+------------------------------------+
                                     |  Human Authority Verification
                                     |  + Explicit User Activation Command
                                     v
+-------------------------------------------------------------------------+
| 3. PHASE B — RUNTIME AUTHORIZATION REGISTER & EXISTING HMAC GATE        |
|    `configs/research_authorization_register.json` +                     |
|    `src/validation/authorization_gate.py` (`ResearchAuthorizationGate`) |
|    Canonical HMAC-SHA256 verification (`hmac.compare_digest`),          |
|    23/23 items APPROVED + SHA-256 evidence digests + scoped dataset.    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| 4. RUNTIME EXECUTION ENFORCEMENT (Fail-Closed Before Any MRI Read)      |
|    `ResearchAuthorizationGate.check_execution_allowed()` +              |
|    `IntakeService.convert_dicom_to_bids()` + `check_safety_guards()`    |
|    Result: DENY (`NOT_AUTHORIZED`) -> Zero MRI Access | `APPROVED_SCOPED`|
+-------------------------------------------------------------------------+
```

---

## A. Institutional Authorization Evidence

| Evidence ID | Required Institutional Field / Record | Mandatory Verification Criteria | Current Placeholder / Repository State | Required Evidence Artifact & Register Link | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AUTH-INST-01** | **Official Institution Name & Address** | Exact legal entity name and physical campus address; must not contain placeholders or abbreviations only. | `[INSTITUTION_NAME]`, `[INSTITUTION_ADDRESS]` | Signed institutional forwarding / cover authorization letter | `PENDING_EVIDENCE` |
| **AUTH-INST-02** | **Approved Research Department** | Recognized academic/research department sponsoring the study. | `[APPROVED_RESEARCH_DEPARTMENT]` | Departmental forwarding memo signed by `[HEAD_OF_DEPARTMENT]` | `PENDING_EVIDENCE` |
| **AUTH-INST-03** | **Principal Investigator (PI) Identity** | Faculty/lead PI full name, designation, institutional affiliation, and dated investigator undertaking. | `[PI_NAME]` (`[PI_DESIGNATION]`, `[PI_AFFILIATION]`) | Signed PI CV, ethics/GCP certificate, & undertaking | `PENDING_EVIDENCE` |
| **AUTH-INST-04** | **Academic Supervisor Identity** | Supervisor/guide name, department, and dated methodological sign-off (`DEC-STC-01`, `DEC-SDC-02`, `DEC-DEF-03`, `DEC-ACQ-04`, `QC-01..08`). | `[SUPERVISOR_NAME]` (`[SUPERVISOR_DEPT]`) | Signed Supervisor Endorsement Memo (`GOV-APP-001`) | `PENDING_EVIDENCE` |
| **AUTH-INST-05** | **Institutional Data Custodian & Security Authority** | Named custodian and IT security official authorizing storage path (`NEUROADHD_DATA_ROOT`) and controls. | `[DATA_CUSTODIAN]`, `[IT_SECURITY_OFFICER]` | Signed Infrastructure & Custodian Appointment Record | `PENDING_EVIDENCE` |
| **AUTH-INST-06** | **Institutional Authorization Reference & Validity Dates** | Unique institutional approval reference, effective date (`ISO 8601`), and expiry/continuing-review date. | `[INSTITUTIONAL_APPROVAL_REFERENCE]`, `[EFFECTIVE_DATE]`, `[EXPIRY_DATE]` | Institutional approval letter + `REG-NA-A6-001` record binding | `PENDING_EVIDENCE` |

---

## B. Ethics Approval Evidence (`AUTH-IEC-01` & `AUTH-DSU-03`)

| Evidence ID | Required Ethics Record / Determination | Mandatory Verification Criteria | Current Repository Posture | Applicable Scope Rule | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ETH-IEC-01** | **IEC Identity & DHR NECRBHR Registration** | Official `[IEC_NAME]` and active DHR NECRBHR registration certificate/number & validity dates. | `[IEC_NAME]`, Reg. No.: `[IEC_REGISTRATION_NUMBER]` | Both Tier 2 & Tier 3 (`GOV-APP-002`) | `PENDING_EVIDENCE` |
| **ETH-IEC-02** | **Applicable IEC Determination Category** | Written committee ruling selecting **Full IEC Review**, **Expedited Review**, or **Formal Administrative Exemption** + written **Secondary-Use Determination** (or **Consent Waiver** under ICMR 2017 Sec. 5.7) vs. **Prospective Consent/Assent**. | `[PENDING_IEC_DETERMINATION]` (`REQUIRES_AUTHORITY_DECISION`) | **Strict Separation:** Retrospective secondary ADHD-200 approval **NEVER** authorizes prospective $N=6$ recruitment! | `PENDING_EVIDENCE` |
| **ETH-IEC-03** | **Approved Protocol Identity & Version** | Exact approved title (`NeuroADHD — ADHD Classification Using Resting-State Functional MRI Connectivity and Machine Learning`) and protocol version matching `DOC-GOV-PROT-001` (`v1.0` or signed final). | Draft only (`DOC-GOV-PROT-001`); `[APPROVED_PROTOCOL_VERSION]` | Version mismatch must block runtime activation | `PENDING_EVIDENCE` |
| **ETH-IEC-04** | **IEC Approval Reference, Approval Date & Continuing Review** | Official reference number (`[IEC_PROTOCOL_NUMBER]`), approval date, expiry date, and annual continuing-review schedule. | `[IEC_PROTOCOL_NUMBER]`, `[APPROVAL_DATE]`, `[EXPIRY_DATE]` | Must be non-expired (`expiry_date >= current_date`) | `PENDING_EVIDENCE` |
| **ETH-IEC-05** | **Approved Population, Data Types & Scope Conditions** | Explicitly approved age/group cohort, modalities (`3D T1w`, `rs-fMRI BOLD`, optional `fmap`), objectives, processing ops, and sharing limits (`AUTH-IEC-01`, `AUTH-DSU-03`). | `[APPROVED_POPULATION]`, `[APPROVED_DATA_TYPES]`, `[APPROVED_SHARING_CONDITIONS]` | Cohort/session scope enforced per protocol tier | `PENDING_EVIDENCE` |

---

## C. Dataset-Owner Permission and DUA (`AUTH-DUA-02`)

| Evidence ID | Required ADHD-200 / INDI / NITRC Record | Mandatory Verification Criteria | Current Repository State | Required Verification Guard | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DUA-ACC-01** | **Actual Dataset Registration / Portal Access Record** | Named NITRC / INDI portal account confirmation & registration timestamp for `[PI_NAME]` / authorized personnel (downloading files is **not** proof of institutional DUA). | `[PENDING_NITRC_ACCESS_RECORD]` | Reject inferred permission from local file presence | `PENDING_EVIDENCE` |
| **DUA-EXEC-02** | **Executed or Accepted DUA & Applicable Licence** | Archived, dated FCP/INDI/NITRC DUA acceptance (`AUTH-DUA-02`) and confirmed **non-commercial (CC-BY-NC)** licence terms (`DOC-GOV-DATA-001`). | `[PENDING_EXECUTED_DUA_RECORD]` (`AUTH-DUA-02 = PENDING`) | Reject fabricated/self-written DUA references | `PENDING_EVIDENCE` |
| **DUA-USE-03** | **Non-Commercial, Citation & Anti-Reidentification Terms** | Confirmed non-commercial research restriction, mandatory citations (*ADHD-200 Consortium 2012*; *Bellec et al. 2017*; INDI URL), and anti-reidentification covenant. | Policies documented in `DOC-GOV-DATA-001`; signed institutional undertaking `[PENDING]` | Cannot publish clinical or commercial claims | `PENDING_EVIDENCE` |
| **DUA-SHARE-04** | **Onward-Sharing, Derivative-Data & Cross-Border Rules** | Prohibition on onward raw MRI redistribution; written clarification on derived `roi_CC200.csv` / Fisher-$z$ sharing & DHR/HMSC international dataset determination (`REQUIRES_AUTHORITY_DECISION`), plus retention/deletion duties and **`153` (`50/103`) vs. `134` (`114+20`)** manifest reconciliation. | `[PENDING_DERIVATIVE_SHARING_RULING]`, `[PENDING_HMSC_DETERMINATION]` | `AUTH-DUA-02`, `AUTH-SUA-04`, `REQUIRES_DATA_OWNER_CONFIRMATION` | `PENDING_EVIDENCE` |

---

## D. Participant-Level Authorization (`N=6` & Scoped Access)

| Evidence ID | Required Participant-Scope Field | Mandatory Verification Rule | Current Repository State | Security Constraint | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PART-ID-01** | **Pseudonymous Participant Identifier** | Coded BIDS ID only (`sub-...`); real names, MRNs, and clinical site numbers prohibited in authorization logs and reports. | `sub-01 = PENDING` (`REG-NA-A6-001`); 6 pilot placeholders in `N6_PILOT_AUTHORIZATION_MATRIX.md` = `PENDING` | Approving 1 participant **never** authorizes the entire cohort | `PENDING_EVIDENCE` |
| **PART-CON-02** | **Applicable Consent or IEC Waiver** | Paired parental/LAR consent + child/adolescent assent (Tier 3 prospective) OR IEC written consent waiver under ICMR 2017 Sec. 5.7 (Tier 2 secondary). | `[PENDING_CONSENT_OR_WAIVER_REF]` | Prospective consent form drafts (`DOC-GOV-ICF-001`) are unapproved templates | `PENDING_EVIDENCE` |
| **PART-SCP-03** | **Approved Protocol, Purpose, Operations & Session Scope** | Explicit `protocol_version`, `study_tier`, `approved_purpose`, exact `permitted_operations`, `session_scope`, `effective_date`, and `expiry_date`. | `[PENDING_PARTICIPANT_SCOPE]` | Unlisted participant, unapproved session, unauthorized op, or wrong protocol version **fails closed** | `PENDING_EVIDENCE` |

---

## E. Privacy and Information Security (`SAFE-DEID-01` – `SAFE-AUD-05`, `AUTH-DPP-05`)

| Evidence ID | Control Area | Technical / Documentary Proof Required (No Secrets / Keys / PII Exposed) | Current Pipeline State | Institutional Evidence Reference | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SEC-DMP-01** | **Data Management Plan & Storage Approval** | Signed `DOC-GOV-DMP-001` confirming external `NEUROADHD_DATA_ROOT` (`~/NeuroADHD_Data`, outside Git repo) & approved path. | Draft DMP complete; external root implemented (`src/intake/service.py`) | `[PENDING_APPROVED_STORAGE_RECORD]` | `PENDING_EVIDENCE` |
| **SEC-ENC-02** | **Encryption at Rest & in Transit** | Host/external disk **AES-256** verification certificate (`SAFE-STOR-03`) + `TLS 1.3`/`SSH ed25519` transfer SOP; loopback dashboard `127.0.0.1:8765`, `--network none` container rule. | Loopback & `--network none` enforced in code; AES-256 host proof missing | `[PENDING_AES256_ATTESTATION]` | `PENDING_EVIDENCE` |
| **SEC-ACC-03** | **RBAC, Access Logging & Backup/Destruction** | Named RBAC user roster (`SAFE-ACC-04`), non-root container (`UID 10001`), cloud-sync exclusion proof, encrypted backup schedule, NIST SP 800-88 destruction SOP, & `[INSTITUTIONAL_RETENTION_YEARS]`. | Non-root container + SHA-256 audit logs active; institutional retention & backup SOP pending | `[PENDING_RBAC_BACKUP_RETENTION_SIGN_OFF]` | `PENDING_EVIDENCE` |
| **SEC-INC-04** | **Incident Response (`SAFE-AUD-05`)** | Approved breach containment & statutory notification SOP (including DPDP Act 2023 / Rules 2025 `G.S.R. 846(E)` 72-hour DPBI escalation alignment). | Draft SOP documented in `DOC-GOV-DMP-001` Sec. 7 | `[PENDING_INCIDENT_SOP_APPROVAL]` | `PENDING_EVIDENCE` |
| **SEC-DEF-05** | **Defacing Tooling, Metadata Sanitization & DICOM PHI** | Operational `pydeface` + FSL `flirt` verification + visual orthoslice QC SOP (`DEC-DEF-03`, `SAFE-FACE-02`); `dcm2niix -ba y` + 35-field BIDS sidecar scrubber (`SAFE-DEID-01`); `export_ready_for_review` gate. | Code active (`src/preprocessing/defacing.py` Stage 7/23); host lacks `pydeface` & `FSL` (`SKIPPED_TOOL_UNAVAILABLE`) | `[PENDING_PYDEFACE_FSL_INSTALL_AND_QC_SOP]` | `PENDING_EVIDENCE` |
| **SEC-REG-06** | **DPDP Act 2023 / Rules 2025 (`G.S.R. 846(E)`) DPO Sign-Off** | Formal written DPO/Legal determination covering Sec. 9 (children), Sec. 8(5) safeguards, Sec. 17(2)(b) + Rule 16 Second Schedule research compliance prior to `2027-05-13/14`. | Flagged `REQUIRES_AUTHORITY_DECISION` in `DOC-GOV-REG-001` | `[PENDING_DPO_DETERMINATION_MEMO]` (`AUTH-SUA-04`, `AUTH-DPP-05`) | `PENDING_EVIDENCE` |

