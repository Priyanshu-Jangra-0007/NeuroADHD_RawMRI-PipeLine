# NeuroADHD RawMRI PoC V2 — Phase A.6
# Research Authorization Gate Architecture & Fail-Closed Specification

**Document ID:** `DOC-NA-A6-AGD-001`  
**Version:** 1.0.0  
**Phase:** Phase A.6 — Supervisor Feedback Intake & Research Authorization Gate  
**Status:** `ACTIVE_FAIL_CLOSED`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Execution Policy:** `STRICTLY_SYNTHETIC_ONLY`  
**Real-Data Access State:** `NOT_AUTHORIZED`  

---

## 1. System Objective & Problem Context

In neuroimaging biomarker discovery and clinical machine learning, premature execution of analytical pipelines on real participant data prior to full institutional review introduces severe regulatory, scientific, and privacy vulnerabilities. While Phase A.5 created comprehensive human-readable review dossiers, human-readable documents alone cannot enforce operational boundaries against accidental script execution, automated batch processing, or environment misconfiguration.

**Phase A.6 establishes an automated, machine-verifiable Research Authorization Gate.**
The gate enforces an immutable fail-closed security boundary:
1. All pipeline execution defaults strictly to `STRICTLY_SYNTHETIC_ONLY`.
2. Real participant MRI processing is programmatically blocked at the root entrypoint.
3. Authorization requires an authenticated, schema-validated, machine-readable register (`configs/research_authorization_register.json`).
4. Approval requires verifiable, unforgeable physical evidence references on disk (with SHA-256 integrity verification).
5. Environment variable overrides, CLI flag bypasses, symlink substitutions, and conflicting states are strictly detected and rejected.

---

## 2. Multi-Tiered Authorization Architecture

The authorization architecture operates across three synchronized tiers:

```
+==================================================================================================+
|                                    RESEARCH AUTHORIZATION ARCHITECTURE                            |
+==================================================================================================+

   [ Developer / Environment / CLI / CI ]
                     │
                     │  Execution Request (e.g., --mode synthetic | real)
                     ▼
  +─────────────────────────────────────────────────────────────+
  | TIER 1: ENVIRONMENT & OVERRIDE DEFENSE LAYER                |
  | - Scans for prohibited bypass env vars:                     |
  |   (NEUROADHD_ALLOW_REAL_SCANS, NEUROADHD_BYPASS_SAFETY, etc)|
  | - Any bypass attempt raises AuthorizationBypassAttemptError |
  +─────────────────────────────────────────────────────────────+
                     │ [PASSED]
                     ▼
  +─────────────────────────────────────────────────────────────+
  | TIER 2: MACHINE-READABLE AUTHORIZATION ENGINE               |
  | - Loads configs/research_authorization_register.json        |
  | - Validates JSON schema (authorization_register_schema.json)|
  | - Evaluates Category A: Supervisor Decisions (12 items)    |
  | - Evaluates Category B: Institutional/Legal (5 items)       |
  | - Evaluates Category C: Operational Safeguards (6 items)    |
  | - Cross-validates Physical Evidence References on Disk      |
  | - Verifies SHA-256 Checksums of Evidence Files              |
  | - Detects Status / Comment Inconsistencies & Conflicts     |
  +─────────────────────────────────────────────────────────────+
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼ [SYNTHETIC MODE]      ▼ [REAL DATA MODE / allow_real_scans=True]
  +──────────────────────+  +───────────────────────────────────────────────+
  | TIER 3A: PERMITTED   |  | TIER 3B: FAIL-CLOSED GATEKEEPER               |
  | Synthetic execution  |  | IF real_data_authorized == False:             |
  | proceeds under       |  |   -> Collect all PENDING / REJECTED /         |
  | HMAC integrity check |  |      MISSING EVIDENCE blockers                |
  | & 2.5 GB RAM bounds  |  |   -> Raise AuthorizationDeniedError           |
  |                      |  |      "SAFETY VIOLATION: Execution on real     |
  |                      |  |       MRI data is strictly denied..."         |
  +──────────────────────+  +───────────────────────────────────────────────+
```

---

## 3. Machine-Readable Register Specification

The authorization register is serialized in JSON at [`configs/research_authorization_register.json`](<your-project-folder>/configs/research_authorization_register.json) and governed by [`configs/authorization_register_schema.json`](<your-project-folder>/configs/authorization_register_schema.json).

### Core Register Fields:
- `schema_version`: Must equal `"1.0.0"`.
- `register_id`: Unique identifier (e.g., `"REG-NA-A6-001"`).
- `overall_authorization_state`: Enum `["NOT_AUTHORIZED", "CONDITIONALLY_AUTHORIZED", "FULLY_AUTHORIZED"]`.
- `execution_policy`: Enum `["STRICTLY_SYNTHETIC_ONLY", "AUTHORIZED_PILOT_ONLY", "AUTHORIZED_RESEARCH_ONLY"]`.
- `real_data_authorized`: Boolean flag (defaults strictly to `false`).
- `categories`: Contains three distinct arrays of items:
  1. `supervisor_decisions` (12 items: 4 technical preprocessing policies + 8 QC metrics).
  2. `institutional_legal` (5 items: IEC clearance, DUA execution, consent/assent, ICMR/DPDP review, privacy audit).
  3. `operational_safeguards` (6 items: De-identification, facial defacing audit, storage encryption, access control, breach response, pilot scope).
- `audit_trail`: Append-only array of governance event logs.

### Item State Machine & Approval Semantics:
Each item within the register must strictly occupy one of five discrete states:

```
                     ┌──────────────────┐
                     │     PENDING      │ <--- Default initial state
                     └────────┬─────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│REVISION_REQUIRED │ │     REJECTED     │ │  NOT_APPLICABLE  │
└──────────────────┘ └──────────────────┘ └────────┬─────────┘
                                                   │ Requires formal
                                                   │ justification;
                                                   │ Non-exempt items
                                                   │ forbidden.
                                                   │
                              ┌────────────────────┘
                              ▼
                     ┌──────────────────┐
                     │     APPROVED     │ <--- Requires:
                     └──────────────────┘      - Non-empty Reviewer
                                               - Valid ISO Date
                                               - Valid Selected Option
                                               - Physical Evidence on Disk
                                               - Matching Evidence SHA-256
                                               - Non-contradictory Notes
```

---

## 4. Verification Rules Enforced by the Validation Engine

The `ResearchAuthorizationGate` class ([`src/validation/authorization_gate.py`](<your-project-folder>/src/validation/authorization_gate.py)) enforces the following deterministic verification rules:

| Rule ID | Validation Rule Description | Enforcement Consequence on Failure |
|---|---|---|
| **RULE-01** | **Bypass Environment Variable Detection:** Scan environment for prohibited keys (`NEUROADHD_ALLOW_REAL_SCANS`, `NEUROADHD_BYPASS_SAFETY`, etc.). | Immediate `AuthorizationBypassAttemptError`. |
| **RULE-02** | **Register File Integrity:** Register file must exist, be non-empty, valid JSON, and cannot be a symlink. | Immediate `AuthorizationSchemaError`. |
| **RULE-03** | **JSON Schema Validation:** Register must validate against Draft-07 JSON schema (`authorization_register_schema.json`). | Immediate `AuthorizationSchemaError`. |
| **RULE-04** | **Item Approval Completeness:** Any item claiming `APPROVED` status must specify `responsible_reviewer`, valid `decision_date`, `selected_option`, and `evidence_reference`. | Item marked invalid; added to issues list; `real_data_authorized` blocked. |
| **RULE-05** | **Physical Evidence Verification:** The file referenced by `evidence_reference` must exist on the local filesystem and cannot be a symlink. | Item marked invalid; added to issues list; `real_data_authorized` blocked. |
| **RULE-06** | **Cryptographic Evidence Checksum:** If `evidence_sha256` is provided, the file's computed SHA-256 digest must match byte-for-byte. | Immediate hash mismatch error; `real_data_authorized` blocked. |
| **RULE-07** | **Semantic Conflict Detection:** If an item is `APPROVED`, `reviewer_comments` cannot contain contradictory rejection keywords (`"reject"`, `"denied"`, `"requires revision"`). | Conflict logged; approval invalidated. |
| **RULE-08** | **Non-Exempt Safeguards:** Core governance requirements (`AUTH-IEC-01`, `AUTH-DUA-02`, `SAFE-DEID-01`, `SAFE-FACE-02`, `SAFE-STOR-03`) can never be marked `NOT_APPLICABLE`. | Item marked invalid; added to issues list. |
| **RULE-09** | **Global Authorization Consistency:** `real_data_authorized` can be `True` if and only if: total items > 0, pending count == 0, revision required count == 0, rejected count == 0, zero missing evidence files, and `overall_authorization_state == "FULLY_AUTHORIZED"`. | Conflict logged; authorization denied. |
| **RULE-10** | **Audit Trail PHI Scrubbing:** Any attempt to record patient names, medical record numbers (MRNs), hospital names, Aadhaar numbers, or health IDs into the audit trail raises an immediate exception. | Immediate `ValueError` rejecting audit entry. |

---

## 5. Fail-Closed Integration with Runner Entrypoint

The authorization gate is integrated into `scripts/run_poc_v2.py` at two critical checkpoints:

1. **CLI Flag Parser Checkpoint:**
   In `main()`, `args.mode` and `args.allow_real_scans` are evaluated against `ResearchAuthorizationGate.check_execution_allowed()` prior to any file discovery or pipeline setup.
2. **Safety Guard Function Checkpoint:**
   In `check_safety_guards()`, `ResearchAuthorizationGate.check_for_bypass_attempts()` runs first. If `allow_real_scans=True` is passed to the function programmatically, it evaluates the register and raises `AuthorizationDeniedError` detailing all pending items and blocking issues.

Because `AuthorizationDeniedError` inherits from `RuntimeError`, existing tests expecting `RuntimeError` on safety violations remain 100% compliant while benefiting from deep governance verification.

---

## 6. Audit Trail & Change Management

Modifications to the authorization register must be performed via `ResearchAuthorizationGate.record_audit_event()`. Every change appends an immutable JSON record containing:
- ISO 8601 UTC timestamp.
- Action event identifier (e.g., `SUPERVISOR_DECISION_RECORDED`, `DUA_EXECUTED`).
- Target item ID.
- Previous state and new state.
- Authorizing reviewer / role.
- Formal scientific or administrative rationale.
- Physical evidence reference path.

Direct manual tampering with the JSON file without corresponding evidence results in schema invalidation or conflict detection upon next validation run.

---

## 7. Current System State

Under Phase A.6, the authorization gate reports:
- **Total Registered Items:** 23
- **Pending Items:** 23 (100%)
- **Approved Items:** 0 (0%)
- **Real-Data Execution Allowed:** `FALSE`
- **Effective Policy:** `STRICTLY_SYNTHETIC_ONLY`
- **Operational Decision:** `READY_FOR_SUPERVISOR_FEEDBACK_INTAKE`

