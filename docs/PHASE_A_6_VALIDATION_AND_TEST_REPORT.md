# NeuroADHD RawMRI PoC V2 — Phase A.6
# Authorization Gate Validation & Comprehensive Test Report

**Document ID:** `DOC-NA-A6-VTR-001`  
**Version:** 1.0.0  
**Phase:** Phase A.6 — Supervisor Feedback Intake & Research Authorization Gate  
**Report Date:** 2026-11-02  
**Test Suite Status:** `ALL_TESTS_PASSED`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Baseline Commit:** `22f2a7e2997f7a26ad32d395a0b422930c7790b8`  

---

## 1. Executive Summary

This report documents the validation methodology, test scenarios, coverage metrics, and verification results for Phase A.6 of NeuroADHD RawMRI PoC V2. Phase A.6 implemented an automated, fail-closed Research Authorization Gate ([`src/validation/authorization_gate.py`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/src/validation/authorization_gate.py)) and machine-readable authorization register ([`configs/research_authorization_register.json`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/configs/research_authorization_register.json)).

A dedicated test suite ([`tests/unit/test_phase_a6_authorization_gate.py`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/tests/unit/test_phase_a6_authorization_gate.py)) containing 16 rigorous unit tests was added, expanding the total automated test count from 179 to 195 tests.

### Overall Test Suite Results:
- **Phase A.6 Authorization Tests:** 16 Passed, 0 Failed, 0 Skipped (100% pass rate).
- **Core Library Test Suite:** 192 Passed, 3 Skipped (skips strictly limited to optional local model artifact tests in unprovisioned environments).
- **OS Memory Isolation Suite (`scripts/verify_container_memory.sh`):** 3/3 Passed (2.5 GB cgroup limits, OOM killer enforcement, unconstrained host guard).
- **Authorization Gate Statement Coverage:** 88% (`src/validation/authorization_gate.py`).

---

## 2. Test Methodology & Authorization Scenarios

The testing harness evaluates the authorization gate across fourteen adversarial and edge-case scenarios using isolated temporary filesystems to prevent cross-test state pollution:

| Scenario ID | Test Method Name | Injected Fault / Evaluated Condition | Expected Behavior | Observed Result |
|---|---|---|---|---|
| **SCN-01** | `test_default_register_all_pending_results_in_denied` | All 23 items set to `PENDING` in baseline register. | `real_data_authorized=False`; raises `AuthorizationDeniedError` on real execution request. | **PASSED** |
| **SCN-02** | `test_synthetic_only_execution_is_allowed` | Requested mode is `"synthetic"`, `allow_real_scans=False`. | Permitted; returns decision dictionary with `authorized_mode="synthetic"`. | **PASSED** |
| **SCN-03** | `test_supervisor_approval_missing_denies_real_data` | Institutional items approved, but supervisor decisions remain `PENDING`. | `real_data_authorized=False`; execution denied. | **PASSED** |
| **SCN-04** | `test_iec_approval_missing_denies_real_data` | All items approved except Institutional Ethics Committee approval (`AUTH-IEC-01`). | `real_data_authorized=False`; raises `AuthorizationDeniedError`. | **PASSED** |
| **SCN-05** | `test_dua_missing_denies_real_data` | Data Use Agreement (`AUTH-DUA-02`) set to `REJECTED`. | Execution denied; fail-closed termination. | **PASSED** |
| **SCN-06** | `test_missing_evidence_file_denies_real_data` | Item claims `APPROVED` but references non-existent evidence file path. | File existence check fails; item invalidated; authorization denied. | **PASSED** |
| **SCN-07** | `test_evidence_hash_mismatch_denies_real_data` | Evidence file modified on disk, causing SHA-256 mismatch against registered digest. | Hash mismatch detected; approval invalidated; authorization denied. | **PASSED** |
| **SCN-08** | `test_invalid_approval_status_denies_authorization` | Register item contains illegal status string (`"CONDITIONALLY_ALLOWED"`). | Raises `AuthorizationSchemaError`; fails closed. | **PASSED** |
| **SCN-09** | `test_malformed_json_fails_closed` | Authorization register corrupted into malformed JSON. | Raises `AuthorizationSchemaError`; fails closed. | **PASSED** |
| **SCN-10** | `test_conflicting_decision_comments_rejected` | Item status is `APPROVED` but comments state `"rejected due to safety concerns"`. | Conflict detected; approval revoked; authorization denied. | **PASSED** |
| **SCN-11** | `test_conflicting_root_claims_rejected` | Root claims `real_data_authorized=true` while items are `PENDING`. | Conflict detected; root override rejected; fails closed. | **PASSED** |
| **SCN-12** | `test_unauthorized_env_override_rejected` | Prohibited bypass environment variables set (`NEUROADHD_ALLOW_REAL_SCANS=1`, etc.). | Immediate `AuthorizationBypassAttemptError`. | **PASSED** |
| **SCN-13** | `test_allow_real_scans_in_check_safety_guards_denied_by_gate` | Programmatic invocation of `check_safety_guards(..., allow_real_scans=True)`. | Raises `AuthorizationDeniedError` (which is a `RuntimeError`). | **PASSED** |
| **SCN-14** | `test_core_requirements_cannot_be_marked_not_applicable` | Attempting to mark non-exempt items (`AUTH-IEC-01`) as `NOT_APPLICABLE`. | Rejection; non-exempt items cannot bypass review. | **PASSED** |
| **SCN-15** | `test_genuine_but_incomplete_approvals_denied` | 22 out of 23 items genuinely approved; 1 item remaining `PENDING`. | Authorization strictly denied (`real_data_authorized=False`). | **PASSED** |
| **SCN-16** | `test_audit_trail_logging_and_phi_rejection` | Logging audit events; attempting to inject PHI keywords (e.g., patient name, MRN). | Valid events appended; PHI attempts raise `ValueError`. | **PASSED** |

---

## 3. Code Coverage & Static Quality Metrics

```
================================ tests coverage ================================
Name                                   Stmts   Miss  Cover
----------------------------------------------------------
src/validation/authorization_gate.py     186     22    88%
src/validation/bids_validator.py          34      2    94%
src/validation/memory_guard.py            70     13    81%
src/preprocessing/motion.py               63      6    90%
src/preprocessing/nuisance.py            106     12    89%
src/preprocessing/qc.py                  234      4    98%
src/preprocessing/registration.py        178     43    76%
src/preprocessing/segmentation.py        179     31    83%
src/features/connectivity.py              26      0   100%
src/parcellation/extractor.py             61      8    87%
----------------------------------------------------------
TOTAL                                   1251    202    84%
================================================================================
```

---

## 4. Security & Isolation Verification

1. **Linux Cgroup v2 Isolation:**
   Verified using `scripts/verify_container_memory.sh`:
   - Kernel memory ceiling confirmed at $2{,}684{,}354{,}560\text{ bytes}$ ($2.5\text{ GiB}$).
   - Kernel OOM-killer termination verified on allocation stress (exit code 137).
   - Fail-closed behavior on unconstrained host environments verified.
2. **Read-Only Model Artifact Protection:**
   Model artifacts are mounted strictly as read-only (`:ro`). Checksums are verified prior to deserialization (`228d3f01...` for preprocessor, `d8eb23eb...` for Random Forest).
3. **Data Boundary Enforcement:**
   Zero real participant scans or clinical DICOM files exist in the repository. Synthetic test phantoms are procedurally generated in temporary sandboxes and destroyed post-execution.

---

## 5. Known Limitations & Research Boundaries

1. **Synthetic-Only Scope:** The authorization gate has been validated exclusively against synthetic test fixtures and simulated approval states. Real clinical scanning artifacts have not been processed.
2. **Evidence Storage Convention:** The gate assumes evidence files reside within the repository tree (e.g., `docs/evidence/`). For sensitive institutional agreements containing institutional signatures, encrypted archival or redaction of non-public contact details may be necessary before repository commit.
3. **External Reliance:** The gate cannot independently verify the authenticity of an external ethics committee letter beyond its cryptographic hash; human oversight by the Principal Investigator remains mandatory.

---

## 6. Recommended Next Steps

With Phase A.6 fully implemented and verified, the recommended progression is:
1. Conduct formal presentation to the Primary Research Supervisor using [`docs/PHASE_A_5_SUPERVISOR_PRESENTATION_OUTLINE.md`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_SUPERVISOR_PRESENTATION_OUTLINE.md).
2. Await supervisor review of [`docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md`](file:///Users/apple/Documents/Codex/2026-09-29/untitled%20folder/docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md).
3. Record supervisor decisions using `ResearchAuthorizationGate.record_audit_event()`.
4. Submit formal ethics application to the Institutional Ethics Committee (IEC).
5. Execute Data Use Agreement with NITRC/INDI for the ADHD-200 repository.

