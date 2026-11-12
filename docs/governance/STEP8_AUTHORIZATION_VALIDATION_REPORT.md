# STEP 8 — FORMAL REAL-DATA AUTHORIZATION VALIDATION REPORT

> **`NOT AUTHORIZED — STRICTLY_SYNTHETIC_ONLY`**
>
> **Document ID:** `DOC-GOV-STEP8-001`  
> **Milestone:** Step 8 — Formal Real-Data Authorization Readiness  
> **Branch:** `feat/step8-real-data-authorization` (parent: Step 7 `986fc65` -> Step 6 `f5c4fcd` -> `origin/main` `d1e928e`)  
> **Pipeline Version:** 23-stage PoC V2 (`DOC-GOV-PROT-001`)  
> **Runtime Authorization State:** `overall_authorization_state = NOT_AUTHORIZED` | `execution_policy = STRICTLY_SYNTHETIC_ONLY` | `real_data_authorized = false` | `sub-01 = PENDING` (`cryptographic_signature = null`)  
> **Step 8 Readiness Verdict:** **`BLOCKED — NOT AUTHORIZED`** *(Software validation & two-phase readiness engine passed; genuine institutional approvals, DUA archive, DPO rulings, operational `pydeface`/FSL setup, and HMAC signatures remain missing / `PENDING`.)*

---

## 1. Existing Authorization Architecture Audit

- **Runtime Register (`configs/research_authorization_register.json` / `REG-NA-A6-001`):** Unchanged (`23 / 23` checklist items `PENDING`: `12` supervisor decisions, `5` institutional/legal items, `6` operational safeguards; `1` dataset entry `AUTH-DS-SUB01-001` for `sub-01` = `PENDING`, `cryptographic_signature = null`, `institutional_approval_reference = null`).
- **Cryptographic Gate (`src/validation/authorization_gate.py` -> `ResearchAuthorizationGate`):** Preserved intact. Enforces case-insensitive rejection of `8` prohibited override environment variables, validates `20` schema keys (`configs/authorization_register_schema.json`), checks SHA-256 digests and symlink/traversal boundaries on item evidence references, prevents `NOT_APPLICABLE` overrides on `NON_EXEMPT_REQUIREMENTS` (`AUTH-IEC-01`, `AUTH-DUA-02`, `SAFE-DEID-01`, `SAFE-FACE-02`, `SAFE-STOR-03`), and computes/verifies canonical JSON HMAC-SHA256 dataset signatures over sorted `CANONICAL_DATASET_FIELDS` via `hmac.compare_digest()`.
- **Pre-Access & Intake Enforcement (`scripts/run_poc_v2.py` -> `check_safety_guards()`, `src/intake/service.py`):** Preserved intact. `check_safety_guards()` denies `allow_real_scans=True`, clinical subject IDs, clinical filesystem paths, symlinks, and unauthenticated NIfTI inputs before reading image voxels; `IntakeService._enforce_dicom_conversion_authorization()` blocks `convert_dicom` unless an exact `APPROVED`, non-expired, HMAC-signed record covers `dicom_conversion`.
- **Documentary Gap Closed in Step 8 (`src/validation/authorization_evidence.py`):** Composes `ResearchAuthorizationGate` without duplicating or altering HMAC logic, bridging documentary verification (protocol version, study tier separation between retrospective vs. prospective studies, non-draft/non-request checks, institutional placeholders, participant/session scoping, and register consistency) into the runtime gate.

---

## 2. Verification & Evidence Status Summary

| Category | Verified Controls (Software / Synthetic Fixture) | Missing Genuine Evidence / External Requirements | Verification Authority Required | Status |
| :--- | :--- | :--- | :--- | :--- |
| **A. Institutional Authorization** | Draft dossier & evidence validator schema ready (`REAL_DATA_AUTHORIZATION_EVIDENCE_MATRIX.md`) | Populated institution name/address, HoD endorsement, signed PI/Supervisor identities, Data Custodian / Infosec appointment, approval ref & dates | `[INSTITUTION_ADMINISTRATION]`, `[PI_NAME]`, `[SUPERVISOR_NAME]` | `PENDING_EVIDENCE` |
| **B. IEC Ethics Determination (`AUTH-IEC-01`, `AUTH-DSU-03`)** | Validator separates retrospective secondary vs. prospective pilot tiers (`study_tier`) | Verified `[IEC_NAME]` DHR NECRBHR registration certificate, signed determination letter (exemption/expedited/full review + ICMR Sec 5.7 waiver or Tier 3 PIS/ICF/Assent), & approved protocol version | `[IEC_NAME]` | `PENDING_EVIDENCE` |
| **C. ADHD-200 / INDI / NITRC DUA (`AUTH-DUA-02`)** | Non-commercial flag & anti-download-inference guard enforced | Archived NITRC/INDI DUA execution record, named team roster, derivative-sharing confirmation, & **`153` (`50/103`) vs. `134` (`114+20`)** manifest reconciliation | `[INSTITUTION_LEGAL_OFFICE]`, `[PI_NAME]`, INDI/NITRC | `PENDING_EVIDENCE` |
| **D. Participant-Scoped Pilot (`N=6`, `SAFE-PILOT-06`)** | 6 placeholder slots initialized `PENDING`; rejects unlisted participant, wrong session, wrong op, cohort inference, wrong protocol | Signed pilot participant scope records + consent/waiver links once genuine protocol is cleared | `[IEC_NAME]`, `[SUPERVISOR_NAME]`, `[PI_NAME]` | `PENDING_EVIDENCE` (`0 / 6`) |
| **E. Privacy, InfoSec & Regulatory (`AUTH-SUA-04`, `AUTH-DPP-05`, `SAFE-*`)** | External `~/NeuroADHD_Data`, `--network none`, `0o700`/`0o600`, Stage 7 defacing/metadata sanitization, zero-secret dashboard enforced | Signed AES-256 disk attestation, `[INSTITUTIONAL_RETENTION_YEARS]`, backup/NIST 800-88 wipe SOP, `[DPO_NAME]` DPDP (`G.S.R. 846(E)`) & DHR/HMSC rulings, **`pydeface` + FSL runtime installation** | `[DPO_NAME]`, `[IT_SECURITY_OFFICER]`, `[DATA_CUSTODIAN]`, `[INSTITUTION_RESEARCH_OFFICE]` | `PENDING_EVIDENCE` |

---

## 3. DPDP Rules 2025 Gazette & Regulatory Reconciliation

- Corrected Gazette notification number across [`ETHICS_AND_REGULATORY_APPLICABILITY.md`](ETHICS_AND_REGULATORY_APPLICABILITY.md) and [`GOVERNANCE_READINESS_GATE.md`](GOVERNANCE_READINESS_GATE.md) from erroneous `G.S.R. 843(E)` to verified **`G.S.R. 846(E)`** (dated **November 13, 2025**; Official Gazette publication **November 13–14, 2025** / MeitY PIB announcement November 14, 2025).
- Explicitly mapped staggered commencement by Rule under **Rule 1**:
  - **Immediate (Nov 13/14, 2025):** Rules **1, 2, and 17 to 21** (short title, definitions, DPBI establishment).
  - **12 Months (Nov 13/14, 2026):** **Rule 4** (Consent Managers).
  - **18 Months (May 13/14, 2027):** Rules **3, 5 to 16, 22, and 23** (notices, Data Fiduciary duties, security safeguards, 72h breach reporting, children's data, and **Rule 16 / Section 17(2)(b) + Second Schedule research exemption standards**).
- Retained **`REQUIRES_AUTHORITY_DECISION`** on all unresolved legal/regulatory questions (`DOC-GOV-REG-001` Section 3).

---

## 4. Synthetic Security & Regression Test Summary

- **Step 8 Targeted Suite (`test_step8_authorization_evidence.py`, `test_dashboard_authorization_endpoint.py`, `test_phase_a4_safety_gate.py`, `test_phase_a6_authorization_gate.py`, `test_step2_research_authorization_gate.py`):** **55 passed in 2.62s**.
  - Covers: baseline pending lock, valid synthetic signed scoped fixture (temporary dir only), missing signature, invalid signature, post-sign tamper & evidence SHA-256 mismatch, expired/revoked documentary & register records, unlisted participant (`PENDING` init), unapproved session, unauthorized operation, mismatched protocol version, retrospective-for-prospective tier escalation rejection, mismatched dataset ID, missing DUA/download-inference rejection, missing IEC/draft/self-PDF/request-email rejection, Phase A `READY_FOR_AUTHORIZATION_REVIEW` fail-closed invariant, non-mutating Phase B `preview_activation_plan`, zero-MRI-read pre-access denial (`enforce_before_data_access`), and dashboard secret redaction.
- **Full Repository Regression (`pytest tests --cov=src -q -p no:cacheprovider`):** **494 passed, 7 skipped** (`501` collected; `82%` overall `src/` coverage; `77%` on [`authorization_evidence.py`](../../src/validation/authorization_evidence.py), `85%` on [`authorization_gate.py`](../../src/validation/authorization_gate.py)).
- **Frontend & Compile Checks:** `node --check frontend/app.js` passed; `python -m compileall` passed.

---

## 5. Runtime Authorization Register Before vs. After Step 8

- **`configs/research_authorization_register.json` Modified?:** **NO** (0 byte diff; verified untouched).
- **`overall_authorization_state`:** `NOT_AUTHORIZED` -> `NOT_AUTHORIZED` (unchanged).
- **`execution_policy`:** `STRICTLY_SYNTHETIC_ONLY` -> `STRICTLY_SYNTHETIC_ONLY` (unchanged).
- **`real_data_authorized`:** `false` -> `false` (unchanged).
- **`sub-01` (`AUTH-DS-SUB01-001`):** `PENDING`, `cryptographic_signature = null` (unchanged).
- **Dashboard Readiness State:** `/api/authorization` (`evidence_readiness`) and `/api/authorization/readiness` report `PENDING` / `PENDING_EVIDENCE`, `readiness_permits_execution = false`, and expose zero HMAC keys, raw signatures, or personal identifiers.
