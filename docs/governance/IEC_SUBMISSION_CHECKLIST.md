# IEC SUBMISSION DOCUMENT CHECKLIST

> **`DRAFT — NOT SUBMITTED — NOT APPROVED`**
>
> **Status:** `DRAFT_CHECKLIST_PENDING_INSTITUTIONAL_DETAILS`  
> **Document ID:** `DOC-GOV-CHK-001`  
> **Target Committee:** `[IEC_NAME]` (DHR NECRBHR Reg. No.: `[IEC_REGISTRATION_NUMBER]`)  
> **Protocol Title:** NeuroADHD — ADHD Classification Using Resting-State Functional MRI Connectivity and Machine Learning  
> **Execution Policy:** `STRICTLY_SYNTHETIC_ONLY` | `real_data_authorized = false`

---

## 1. Pre-Submission Institutional Metadata Required (Placeholders Only)

| Field | Required Value | Current Placeholder / Status |
| :--- | :--- | :--- |
| **Institution Name & Address** | Official parent university / medical institute / department | `[INSTITUTION_NAME]`, `[INSTITUTION_ADDRESS]` |
| **IEC Name & Registration** | DHR-registered Institutional Ethics Committee | `[IEC_NAME]`, Reg. No.: `[IEC_REGISTRATION_NUMBER]` |
| **Principal Investigator (PI)** | Faculty/Lead Investigator name, designation, signature | `[PI_NAME]`, `[PI_DESIGNATION]`, `[PI_CONTACT]` |
| **Academic Supervisor / Guide** | Research supervisor name, review date, signature | `[SUPERVISOR_NAME]`, `[SUPERVISOR_APPROVAL_DATE]` |
| **Co-Investigator(s)** | Radiologist / clinician / biostatistician (if applicable) | `[CO_INVESTIGATOR_NAMES_AND_ROLES]` |
| **Assigned IEC Protocol No.** | Issued only by IEC Secretariat upon intake | `[PENDING_IEC_PROTOCOL_NUMBER]` |

---

## 2. Complete IEC Submission Dossier Checklist

| # | Required Submission Item | Applicability | Linked Repository Draft / Source | Completeness Status | Authority Action Needed |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | **IEC Application Form (Institutional Format)** | Mandatory for Tier 2/3 | `[INSTITUTIONAL_IEC_FORM_TEMPLATE]` | `PENDING_INSTITUTION_FORM` | Populate using `[IEC_NAME]` official proforma & sign. |
| 2 | **Full Research Protocol** | Mandatory | `docs/governance/NEUROADHD_RESEARCH_PROTOCOL_DRAFT.md` | `DRAFT_COMPLETE` | Review & sign by `[PI_NAME]` & `[SUPERVISOR_NAME]`. |
| 3 | **Executive Synopsis / Protocol Summary (1–2 pages)** | Mandatory | Extract from `DOC-GOV-PROT-001` Sections A–G | `DRAFT_COMPLETE` | Adapt to `[IEC_NAME]` word-limit template if specified. |
| 4 | **Principal Investigator CVs, Qualifications & Good Research Practice Declarations** | Mandatory | `[PI_CV_AND_TRAINING_RECORDS]` | `PENDING_PERSONNEL_DETAILS` | Attach dated CVs & ethics/GCP training certificates. |
| 5 | **Academic Supervisor / Guide Endorsement** | Mandatory (Student/Researcher project) | `[SUPERVISOR_ENDORSEMENT_LETTER]` | `PENDING_SIGNATURE` | Signed academic endorsement + 23-item register oversight plan. |
| 6 | **Co-Investigator Details & Site Clinician/Radiologist Assignment** | Required if prospective scan or incidental findings review | `[CO_INVESTIGATOR_ROSTER]` | `PENDING_IF_APPLICABLE` | Confirm qualified radiologist/clinician if prospective Tier 3 occurs. |
| 7 | **Institutional Affiliation & Departmental Forwarding Letter** | Mandatory | `[HEAD_OF_DEPARTMENT_ENDORSEMENT]` | `PENDING_SIGNATURE` | Obtain Head of Department forwarding signature. |
| 8 | **Detailed Technical & Preprocessing Methodology (23-Stage PoC V2)** | Mandatory | `DOC-GOV-PROT-001` Section F + `docs/STEP6_MRI_DEFACING_AND_ANONYMIZATION.md` | `DRAFT_COMPLETE` | Note current limitation: synthetic-verified only; real-data concordance unproven. |
| 9 | **Dataset Provenance Record (ADHD-200 / INDI / NITRC)** | Mandatory for Tier 2 | `docs/governance/DATASET_PROVENANCE_AND_PERMISSIONS.md` | `DRAFT_COMPLETE` | Attach NITRC account confirmation & original split manifest clarification. |
| 10 | **Data Use Agreement (DUA) & Licence Verification** | Mandatory for Tier 2 | `DOC-GOV-DATA-001` + `[EXECUTED_DUA_PDF]` | `PENDING_DOCUMENTATION` | Execute/archive institutional ADHD-200 DUA confirmation (`AUTH-DUA-02`). |
| 11 | **Privacy, Confidentiality & De-Identification Plan** | Mandatory | `docs/governance/DATA_MANAGEMENT_AND_PRIVACY_PLAN.md` | `DRAFT_COMPLETE` | Complete institutional retention period & `[DPO_NAME]` sign-off. |
| 12 | **Data Management, Storage & Retention Plan** | Mandatory | `DOC-GOV-DMP-001` (`AES-256`, `NEUROADHD_DATA_ROOT`, incident response) | `DRAFT_PARTIAL` | Fill `[INSTITUTIONAL_RETENTION_YEARS]`, backup encryption, & archival custodian. |
| 13 | **Risk-Benefit & AI/Biomarker Risk Register** | Mandatory | `docs/governance/ETHICS_RISK_REGISTER.md` | `DRAFT_COMPLETE` | Review residual risks (re-identification, motion bias, false predictions) with IEC. |
| 14 | **Statistical Analysis Plan (SAP)** | Mandatory | `DOC-GOV-PROT-001` Section G | `DRAFT_COMPLETE` | Confirm site-disjoint validation & explicit non-powered status of $N=6$ pilot. |
| 15 | **Participant Information Sheet (PIS)** | Conditional: **Tier 3 Prospective** or if IEC mandates | `docs/governance/PARTICIPANT_INFORMATION_SHEET_DRAFT.md` | `DRAFT_CONDITIONAL` | Translate into local/regional languages; populate clinician/IEC contacts. |
| 16 | **Informed Consent Form (ICF) & Pediatric Assent** | Conditional: **Tier 3 Prospective** | `docs/governance/INFORMED_CONSENT_FORM_DRAFT.md` | `DRAFT_CONDITIONAL` | Pair parental/LAR consent with age-appropriate pediatric/adolescent assent (`ICMR 2017`). |
| 17 | **Waiver of Informed Consent Request Form** | Conditional: **Tier 2 Retrospective (ADHD-200)** | `[IEC_CONSENT_WAIVER_FORM]` (rationale per `DOC-GOV-REG-001`) | `REQUIRES_AUTHORITY_DECISION` | Request IEC ruling on consent waiver / exemption under ICMR 2017 Section 5.7. |
| 18 | **Recruitment Advertisements / Flyers / Screening Scripts** | Conditional: **Tier 3 Prospective ONLY** | `[NOT_DRAFTED_PENDING_PILOT_SCOPE]` | `NOT_APPLICABLE_UNLESS_PROSPECTIVE` | Submit prior to any recruitment if Tier 3 clinical pilot is authorized. |
| 19 | **Data-Sharing & External Collaboration Agreements (MTA/DSA/HMSC check)** | Mandatory if sharing derivatives or collaborating | `DOC-GOV-DATA-001` + `DOC-GOV-REG-001` | `REQUIRES_AUTHORITY_DECISION` | Obtain HMSC/DHR screening determination & derivative-sharing ruling. |
| 20 | **Data Retention, Archival & Certified Destruction SOP** | Mandatory | `DOC-GOV-DMP-001` Section 5 | `PENDING_INSTITUTIONAL_POLICY` | Insert `[INSTITUTION_NAME]` official minimum retention schedule & NIST 800-88 wipe sign-off. |
| 21 | **Conflict of Interest (COI) Declarations** | Mandatory | `[SIGNED_COI_FORMS]` | `PENDING_SIGNATURES` | Individual signed COI forms from `[PI_NAME]`, `[SUPERVISOR_NAME]`, student/co-investigators. |
| 22 | **Investigator Undertakings (Non-Reidentification, Non-Clinical Use, Security)** | Mandatory | `[SIGNED_INVESTIGATOR_UNDERTAKINGS]` | `PENDING_SIGNATURES` | Sign undertakings prohibiting re-identification, unapproved real scans, and clinical reporting. |
| 23 | **Incidental Findings & Clinical Escalation Protocol (MRI)** | Conditional: **Tier 3 Prospective ONLY** | `[RADIOLOGY_INCIDENTAL_FINDINGS_SOP]` | `PENDING_IF_PROSPECTIVE` | Mandatory only if new human scans are acquired; specify neuroradiologist referral workflow. |
| 24 | **Regulatory & DPDP Act Applicability Memo** | Recommended / Institutional | `docs/governance/ETHICS_AND_REGULATORY_APPLICABILITY.md` | `DRAFT_COMPLETE` | Submit to `[DPO_NAME]` / `[INSTITUTION_LEGAL_OFFICE]` for `AUTH-SUA-04` & `AUTH-DPP-05`. |
