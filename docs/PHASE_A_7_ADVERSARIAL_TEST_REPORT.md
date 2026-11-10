# NeuroADHD RawMRI PoC V2 — Phase A.7
# Adversarial Security Testing & Evidence Validation Report

**Document ID:** `DOC-NA-A7-ATR-001`  
**Version:** 1.0.0  
**Phase:** Phase A.7 — Independent Security & Scientific Reproducibility Audit  
**Report Date:** 2026-11-02  
**Test Suite Status:** `ALL_ADVERSARIAL_TESTS_PASSED`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Baseline Commit:** `de3150b93ebe6b79242b44e189866091794c83b5`  

---

## 1. Executive Summary

This report documents the execution results of the dedicated adversarial security and evidence integrity test suite engineered for Phase A.7:
[`tests/security/test_phase_a7_adversarial_authorization.py`](<your-project-folder>/tests/security/test_phase_a7_adversarial_authorization.py).

The adversarial testing harness evaluated the pipeline against environment variable injection, CLI flag manipulation, symlink substitution, path traversal, documentary evidence tampering, audit-log injection, and alternate runner bypasses.

### Test Execution Overview:
- **Dedicated Phase A.7 Adversarial Security Suite:** **14 Passed, 0 Failed, 0 Skipped** (100% pass rate in 1.99s).
- **Phase A.6 Core Authorization Suite:** **16 Passed, 0 Failed, 0 Skipped** (100% pass rate in 1.74s).
- **Combined Authorization & Security Tests:** **30 Passed, 0 Failed, 0 Skipped** (Statement coverage: 88%).
- **Operating-System Cgroup Memory Isolation Suite:** **3/3 Passed** (2560 MiB cgroup limit, kernel OOM kill, unconstrained host guard).
- **Full Repository Test Suite:** **206 Passed, 3 Skipped** (without model); **209 Passed** (with model).

---

## 2. Adversarial Scenarios & Observed Test Outcomes

| Scenario ID | Attack Vector / Adversarial Probe | Tested Method | Observed System Response | Security Property Verified | Test Status |
|---|---|---|---|---|---|
| **ADV-01** | Case-insensitive & non-standard truthy environment variables (`neuroadhd_allow_real_scans=1`, `NEUROADHD_BYPASS_SAFETY=force`, `NEUROADHD_FORCE_REAL_DATA=enabled`). | `test_case_insensitive_and_alternative_truthy_env_bypasses` | `AuthorizationBypassAttemptError` raised immediately on startup. | Prevents environment variable safety bypass regardless of casing or truthy format. | **PASSED** |
| **ADV-02** | Explicit falsy environment variables (`NEUROADHD_ALLOW_REAL_SCANS=0`, `false`, `no`, `off`). | `test_falsy_env_vars_do_not_trigger_false_positive` | No false positive; synthetic pipeline permitted. | Clean differentiation between bypass attempts and benign configuration. | **PASSED** |
| **ADV-03** | Path traversal via relative reference (`../../etc/passwd`). | `test_path_traversal_in_evidence_reference_rejected` | Boundary check detects traversal outside repo; authorization denied. | Evidence files cannot reference host files outside repository tree. | **PASSED** |
| **ADV-04** | Boundary escape via absolute path (`/etc/hosts`). | `test_absolute_path_outside_repository_rejected` | Absolute path detection raises path traversal issue; authorization denied. | Absolute filesystem paths strictly forbidden for documentary evidence. | **PASSED** |
| **ADV-05** | Symlink substitution of evidence file pointing to external file. | `test_symlink_evidence_file_rejected` | `is_symlink()` detected on candidate path; approval invalidated. | Symlink file substitution strictly prevented. | **PASSED** |
| **ADV-06** | Symlink substitution via intermediate parent directory. | `test_symlinked_parent_directory_rejected` | Directory component walk detects symlink; approval invalidated. | Symlink directory substitution strictly prevented. | **PASSED** |
| **ADV-07** | Directory path referenced as documentary evidence. | `test_directory_as_evidence_rejected` | `is_file()` assertion fails; approval invalidated. | Non-file paths (directories, FIFOs, devices) rejected. | **PASSED** |
| **ADV-08** | Unauthorized file extension (`approval_script.sh`, `.exe`, `.py`). | `test_unsupported_evidence_file_extension_rejected` | Extension whitelist check fails; approval invalidated. | Evidence restricted to document formats (`.pdf`, `.md`, `.txt`, `.json`, `.png`, `.jpg`). | **PASSED** |
| **ADV-09** | Approved item missing mandatory 64-character hex SHA-256 digest. | `test_approved_item_missing_mandatory_sha256_rejected` | Checksum presence check fails; approval invalidated. | Cryptographic verification mandatory on all approvals. | **PASSED** |
| **ADV-10** | Execution of unhardened alternate entrypoint (`scripts/run_single_scan_poc.py`). | `test_alternate_entrypoint_run_single_scan_poc_fails_closed` | `AuthorizationDeniedError` raised immediately inside `run_poc()`. | Alternate execution routes fail closed. | **PASSED** |
| **ADV-11** | Log injection via newline (`\n`, `\r`, `\x00`) in audit fields. | `test_audit_log_newline_injection_rejected` | `ValueError` raised rejecting control characters. | Audit log formatting integrity preserved. | **PASSED** |
| **ADV-12** | Log injection of clinical subject IDs (`sub-2344857`, `KKI`, `Peking`). | `test_audit_log_clinical_subject_keywords_rejected` | `ValueError` raised rejecting clinical identifiers. | Audit log privacy & de-identification preserved. | **PASSED** |
| **ADV-13** | Verification of canonical CC200 atlas cryptographic hash. | `test_canonical_atlas_hash_reproducibility` | Atlas digest matches authoritative SHA-256 (`9467afce...`). | Parcellation template immutability verified. | **PASSED** |
| **ADV-14** | Register JSON schema injection of unauthorized root backdoor keys. | `test_register_schema_rejects_unknown_root_properties` | `AuthorizationSchemaError` raised by Draft-07 validator. | Prevents silent register tampering. | **PASSED** |

---

## 3. Evidence-Integrity Verification Rigor

The audit verified the distinction between **cryptographic digest verification** and **formal institutional authorization**:
- **What SHA-256 Verifies:** The SHA-256 digest proves that a file on disk has not been altered, truncated, corrupted, or substituted since its hash was recorded in the register.
- **What SHA-256 Does NOT Prove:** A valid cryptographic checksum alone does not prove that an Institutional Ethics Committee letter is authentic, legally binding, or issued by an accredited authority. Human supervisory and institutional oversight remains essential.

---

## 4. Resource & Operating-System Memory Isolation Results

Re-execution of `scripts/verify_container_memory.sh` verified kernel-enforced isolation:
1. **Kernel Cgroup Memory Limit:** Confirmed hard ceiling at $2{,}684{,}354{,}560\text{ bytes}$ ($2{,}560\text{ MiB}$ / $2.5\text{ GiB}$) — `PASSED`.
2. **Linux Kernel OOM-Killer Enforcement:** Process allocation exceeding 2,560 MiB immediately terminated with exit code 137 (`SIGKILL`) — `PASSED`.
3. **Fail-Closed Unconstrained Host Policy:** Pipeline terminates if container isolation is missing and developer override is absent — `PASSED`.

---

## 5. Conclusion

Adversarial testing confirms that the Research Authorization Gate enforces strict, tamper-resistant, fail-closed boundaries across all supported execution paths.

