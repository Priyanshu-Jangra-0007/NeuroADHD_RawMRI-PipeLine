# N=6 FEASIBILITY PILOT — PARTICIPANT-SCOPED AUTHORIZATION MATRIX

> **`DRAFT — NOT SUBMITTED — NOT AUTHORIZED`**
>
> **Document ID:** `DOC-GOV-PILOTMAT-001`  
> **Scope Limit:** Strictly **$N = 6$** technical feasibility participants (`SAFE-PILOT-06`, `DOC-NA-A5-FRP-001`)  
> **Execution Policy:** `STRICTLY_SYNTHETIC_ONLY` | `real_data_authorized = false` | `overall_authorization_state = NOT_AUTHORIZED`  
> **Placeholder & Anti-Fabrication Policy:** All 6 slots below use **placeholder / synthetic identifiers only** and remain strictly **`PENDING`**. No real ADHD-200 or clinical participant IDs are listed, and no approval has been granted.

---

## 1. Per-Participant Scoped Authorization Rules

Every real participant run in the future $N=6$ technical feasibility pilot is governed individually by `AuthorizationEvidenceValidator.verify_participant_authorization()` (`src/validation/authorization_evidence.py`) paired with `ResearchAuthorizationGate` (`src/validation/authorization_gate.py`).

### Mandatory Fail-Closed Rejection Conditions

Execution is immediately **REJECTED (`NOT_AUTHORIZED` / `AuthorizationDeniedError`)** before any MRI file is opened if **any** of the following holds:

1. **Unlisted Participant:** Requested `participant_id` is absent from the authorized scoped roster (unknown participants initialize as `PENDING` and fail closed).
2. **Unapproved Session:** Requested `session_id` (e.g., `ses-02`) is not explicitly listed in that participant's `session_scope`.
3. **Unauthorized Operation:** Requested operation (e.g., `clinical_diagnostic_inference`, `DICOM conversion` when only `preprocessing` was approved) is missing from `permitted_operations`.
4. **Revoked or Rejected Record:** `authorization_status` or linked documentary evidence is `REVOKED` or `REJECTED`.
5. **Expired Record or Invalid Date Window:** `current_date > expiry_date` or `effective_date > current_date` or `expiry_date < effective_date`.
6. **Protocol Version / Tier Mismatch:** `protocol_version` does not match the approved IEC protocol, OR a **retrospective secondary-analysis approval** is presented for a **prospective recruitment** (`prospective_pilot`) participant.
7. **Missing or Invalid HMAC Signature / Dataset Record:** Underlying `dataset_authorizations[]` record in `configs/research_authorization_register.json` lacks a valid `hmac.compare_digest` HMAC-SHA256 signature (`MISSING_SIGNATURE`, `INVALID_SIGNATURE`, or `MISSING_KEY`), or `real_data_authorized` is `false`.
8. **Cohort Generalization Attempt:** Approval of one participant record (e.g., Slot 1) **never** authorizes any other participant slot or the broader `153`-subject engineering cohort.

---

## 2. Individualized N=6 Pilot Authorization Records (All Initialized `PENDING`)

> **Notice:** Identifiers `[PILOT_SYNTH_01]`–`[PILOT_SYNTH_06]` are non-clinical placeholder handles strictly for readiness tracking. Do **not** populate with real participant names, hospital MRNs, or unauthorized subject codes.

| Pilot Slot | Pseudonymous Participant ID (Placeholder) | Study Tier & Dataset / Acquisition Scope | Permitted Research Purpose | Session Scope | Permitted Operations Scope | Required Approval Evidence References | Effective Date | Expiry / Review Date | Signature-Validation State | Current Authorization State |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **01 / 06** | `sub-pilot-placeholder-01` | `[PENDING_TIER_2_OR_3]`: `[PENDING_DATASET_OR_SCANNER_REF]` (`3D T1w` + `rs-fMRI BOLD`) | Technical feasibility & historical C-PAC concordance benchmark only (`DOC-GOV-PROT-001`; non-clinical) | `[PENDING_SESSION_SCOPE]` *(e.g. `ses-01` only)* | `bids_validation`, `preprocessing`, `cc200_extraction`, `quality_control` *(no clinical use)* | `AUTH-IEC-01`, `AUTH-DUA-02`, `AUTH-DSU-03`, `SAFE-PILOT-06`, `[PENDING_PARTICIPANT_RECORD_01]` | `[PENDING_EFFECTIVE_DATE]` | `[PENDING_EXPIRY_DATE]` | `MISSING_SIGNATURE` *(unlinked / unsigned)* | **`PENDING`** (`NOT_AUTHORIZED`) |
| **02 / 06** | `sub-pilot-placeholder-02` | `[PENDING_TIER_2_OR_3]`: `[PENDING_DATASET_OR_SCANNER_REF]` (`3D T1w` + `rs-fMRI BOLD`) | Technical feasibility & historical C-PAC concordance benchmark only (`DOC-GOV-PROT-001`; non-clinical) | `[PENDING_SESSION_SCOPE]` | `bids_validation`, `preprocessing`, `cc200_extraction`, `quality_control` *(no clinical use)* | `AUTH-IEC-01`, `AUTH-DUA-02`, `AUTH-DSU-03`, `SAFE-PILOT-06`, `[PENDING_PARTICIPANT_RECORD_02]` | `[PENDING_EFFECTIVE_DATE]` | `[PENDING_EXPIRY_DATE]` | `MISSING_SIGNATURE` | **`PENDING`** (`NOT_AUTHORIZED`) |
| **03 / 06** | `sub-pilot-placeholder-03` | `[PENDING_TIER_2_OR_3]`: `[PENDING_DATASET_OR_SCANNER_REF]` (`3D T1w` + `rs-fMRI BOLD`) | Technical feasibility & historical C-PAC concordance benchmark only (`DOC-GOV-PROT-001`; non-clinical) | `[PENDING_SESSION_SCOPE]` | `bids_validation`, `preprocessing`, `cc200_extraction`, `quality_control` *(no clinical use)* | `AUTH-IEC-01`, `AUTH-DUA-02`, `AUTH-DSU-03`, `SAFE-PILOT-06`, `[PENDING_PARTICIPANT_RECORD_03]` | `[PENDING_EFFECTIVE_DATE]` | `[PENDING_EXPIRY_DATE]` | `MISSING_SIGNATURE` | **`PENDING`** (`NOT_AUTHORIZED`) |
| **04 / 06** | `sub-pilot-placeholder-04` | `[PENDING_TIER_2_OR_3]`: `[PENDING_DATASET_OR_SCANNER_REF]` (`3D T1w` + `rs-fMRI BOLD`) | Technical feasibility & historical C-PAC concordance benchmark only (`DOC-GOV-PROT-001`; non-clinical) | `[PENDING_SESSION_SCOPE]` | `bids_validation`, `preprocessing`, `cc200_extraction`, `quality_control` *(no clinical use)* | `AUTH-IEC-01`, `AUTH-DUA-02`, `AUTH-DSU-03`, `SAFE-PILOT-06`, `[PENDING_PARTICIPANT_RECORD_04]` | `[PENDING_EFFECTIVE_DATE]` | `[PENDING_EXPIRY_DATE]` | `MISSING_SIGNATURE` | **`PENDING`** (`NOT_AUTHORIZED`) |
| **05 / 06** | `sub-pilot-placeholder-05` | `[PENDING_TIER_2_OR_3]`: `[PENDING_DATASET_OR_SCANNER_REF]` (`3D T1w` + `rs-fMRI BOLD`) | Technical feasibility & historical C-PAC concordance benchmark only (`DOC-GOV-PROT-001`; non-clinical) | `[PENDING_SESSION_SCOPE]` | `bids_validation`, `preprocessing`, `cc200_extraction`, `quality_control` *(no clinical use)* | `AUTH-IEC-01`, `AUTH-DUA-02`, `AUTH-DSU-03`, `SAFE-PILOT-06`, `[PENDING_PARTICIPANT_RECORD_05]` | `[PENDING_EFFECTIVE_DATE]` | `[PENDING_EXPIRY_DATE]` | `MISSING_SIGNATURE` | **`PENDING`** (`NOT_AUTHORIZED`) |
| **06 / 06** | `sub-pilot-placeholder-06` | `[PENDING_TIER_2_OR_3]`: `[PENDING_DATASET_OR_SCANNER_REF]` (`3D T1w` + `rs-fMRI BOLD`) | Technical feasibility & historical C-PAC concordance benchmark only (`DOC-GOV-PROT-001`; non-clinical) | `[PENDING_SESSION_SCOPE]` | `bids_validation`, `preprocessing`, `cc200_extraction`, `quality_control` *(no clinical use)* | `AUTH-IEC-01`, `AUTH-DUA-02`, `AUTH-DSU-03`, `SAFE-PILOT-06`, `[PENDING_PARTICIPANT_RECORD_06]` | `[PENDING_EFFECTIVE_DATE]` | `[PENDING_EXPIRY_DATE]` | `MISSING_SIGNATURE` | **`PENDING`** (`NOT_AUTHORIZED`) |

---

## 3. Baseline Runtime Authorization Record in `REG-NA-A6-001`

- **Registered Entry:** `AUTH-DS-SUB01-001` (`dataset_id = "sub-01"`, `authorization_status = "PENDING"`, `cryptographic_signature = null`, `institutional_approval_reference = null`, `approval_date = null`, `expiry_date = null`).
- **Authorized Pilot Count:** **`0 / 6`** (`NOT_AUTHORIZED`).
- **Statistical & Clinical Boundary:** Even if all 6 feasibility records are legitimately signed in a future authorized phase, $N=6$ remains strictly a **technical feasibility pilot** (RAM $\le 2.50\text{ GB}$, BIDS/QC pass rates, and C-PAC concordance), never a statistically powered diagnostic validation.

