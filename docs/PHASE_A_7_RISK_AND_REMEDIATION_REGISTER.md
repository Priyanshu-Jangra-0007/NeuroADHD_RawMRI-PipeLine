# NeuroADHD RawMRI PoC V2 — Phase A.7
# Risk, Vulnerability & Remediation Register

**Document ID:** `DOC-NA-A7-RRR-001`  
**Version:** 1.0.0  
**Phase:** Phase A.7 — Independent Security & Scientific Reproducibility Audit  
**Status:** `ALL_CONFIRMED_VULNERABILITIES_REMEDIATED`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Review Target:** Primary Research Supervisor & Institutional Information Security Officer  

---

## 1. Preamble & Finding Classification Scheme

This register provides the authoritative log of all software vulnerabilities, architectural weaknesses, and scientific risks identified during the Phase A.7 Independent Security and Scientific Reproducibility Audit.

### Severity Classification Definitions:
- `CRITICAL`: A verified execution route that allows unauthorized real-data processing or bypasses a mandatory safety control.
- `HIGH`: A serious vulnerability in authorization, evidence validation, privacy, or security controls.
- `MEDIUM`: A meaningful reliability, boundary enforcement, or reproducibility weakness.
- `LOW`: A documentation, maintainability, or minor validation issue.
- `INFORMATIONAL`: A scientific limitation, observation, or domain constraint that does not represent an exploitable software flaw.

---

## 2. Comprehensive Master Findings Register

### Finding SEC-01: Symlink Detection Bypass in Evidence Verification
- **Finding ID:** `SEC-01`
- **Severity:** `HIGH`
- **Affected Component:** `src/validation/authorization_gate.py` (`verify_item`)
- **Ownership:** Security Engineering
- **Reproduction Steps:** Set `evidence_reference` to a symlink pointing to an existing file in `docs/evidence/`. Prior to fix, `Path.resolve()` followed the link before `.is_symlink()` was checked.
- **Observed Result:** Symlink was followed silently; `.is_symlink()` evaluated to `False`.
- **Expected Result:** Symlinks in candidate path or parent components must be detected and rejected.
- **Potential Impact:** An unauthorized user could symlink to a file outside the repository and have the gate treat external files as valid local evidence.
- **Remediation Implemented:** Inspected `candidate_path.is_symlink()` and traversed parent directories up to `base_dir` prior to `.resolve()`.
- **Verification Evidence:** `tests/security/test_phase_a7_adversarial_authorization.py::test_symlink_evidence_file_rejected` (`PASSED`).
- **Current Status:** **REMEDIATED (CLOSED)**

---

### Finding SEC-02: Path Traversal & Boundary Escape in Evidence Path Resolution
- **Finding ID:** `SEC-02`
- **Severity:** `HIGH`
- **Affected Component:** `src/validation/authorization_gate.py` (`verify_item`)
- **Ownership:** Security Engineering
- **Reproduction Steps:** Provide `evidence_reference` as `../../etc/passwd` or `/etc/hosts`. Prior to fix, `.resolve()` did not assert that the resolved path remained inside `self.base_dir`.
- **Observed Result:** Path resolved outside the repository root without raising a boundary error.
- **Expected Result:** Immediate path traversal error; rejection of any path escaping repository root.
- **Potential Impact:** An adversary could point evidence to arbitrary files on the host machine.
- **Remediation Implemented:** Strictly prohibited absolute paths (`raw_path.is_absolute() -> REJECT`) and asserted `resolved_path.relative_to(self.base_dir)`.
- **Verification Evidence:** `tests/security/test_phase_a7_adversarial_authorization.py::test_path_traversal_in_evidence_reference_rejected` (`PASSED`).
- **Current Status:** **REMEDIATED (CLOSED)**

---

### Finding SEC-03: Alternate Unrestricted Pipeline Entrypoint (`scripts/run_single_scan_poc.py`)
- **Finding ID:** `SEC-03`
- **Severity:** `CRITICAL`
- **Affected Component:** `scripts/run_single_scan_poc.py` (`run_poc`)
- **Ownership:** Pipeline Architecture
- **Reproduction Steps:** Execute `python scripts/run_single_scan_poc.py`. Prior to fix, the script executed preprocessing on real participant `sub-2344857` without calling `check_safety_guards` or `ResearchAuthorizationGate`.
- **Observed Result:** If historical input files were present, the script would process real MRI scans without checking authorization.
- **Expected Result:** Immediate `AuthorizationDeniedError` halting execution before any file access.
- **Potential Impact:** Direct execution of real MRI data bypassing Phase A.6 authorization controls.
- **Remediation Implemented:** Embedded mandatory `ResearchAuthorizationGate.check_execution_allowed()` and `check_safety_guards()` at the top of `run_poc()`.
- **Verification Evidence:** `tests/security/test_phase_a7_adversarial_authorization.py::test_alternate_entrypoint_run_single_scan_poc_fails_closed` (`PASSED`).
- **Current Status:** **REMEDIATED (CLOSED)**

---

### Finding SEC-04: Case-Sensitivity and Non-Standard Truthy Values in Environment Bypass Detection
- **Finding ID:** `SEC-04`
- **Severity:** `HIGH`
- **Affected Component:** `src/validation/authorization_gate.py` (`check_for_bypass_attempts`)
- **Ownership:** Security Engineering
- **Reproduction Steps:** Set `neuroadhd_allow_real_scans=1` (lowercase) or `NEUROADHD_BYPASS_SAFETY=force`. Prior to fix, exact uppercase matching missed lowercase keys, and narrow truthy checks missed alternative truthy values.
- **Observed Result:** Bypass variables with lowercase names or non-standard truthy values were ignored.
- **Expected Result:** Any prohibited variable in any case or truthy form must trigger `AuthorizationBypassAttemptError`.
- **Potential Impact:** Inadvertent or deliberate bypass of safety controls in environments with lowercase environment conventions.
- **Remediation Implemented:** Scanned `os.environ` keys with `.upper()` and rejected any value not in `("", "0", "false", "no", "none", "off")`.
- **Verification Evidence:** `tests/security/test_phase_a7_adversarial_authorization.py::test_case_insensitive_and_alternative_truthy_env_bypasses` (`PASSED`).
- **Current Status:** **REMEDIATED (CLOSED)**

---

### Finding SEC-05: Optional SHA-256 Digest on Approved Evidence Files
- **Finding ID:** `SEC-05`
- **Severity:** `MEDIUM`
- **Affected Component:** `src/validation/authorization_gate.py` (`verify_item`)
- **Ownership:** Governance Engineering
- **Reproduction Steps:** Mark an item `APPROVED` with valid `evidence_reference` but `evidence_sha256 = None`. Prior to fix, the gate verified file existence but skipped cryptographic digest checks.
- **Observed Result:** File was accepted as verified evidence without cryptographic hash verification.
- **Expected Result:** Every `APPROVED` item must require a valid 64-character hex SHA-256 digest.
- **Potential Impact:** A file on disk could be swapped or modified without detection if its hash was not recorded.
- **Remediation Implemented:** Made `evidence_sha256` mandatory for `APPROVED` status, asserting 64-character hex length and matching via constant-time `hmac.compare_digest()`.
- **Verification Evidence:** `tests/security/test_phase_a7_adversarial_authorization.py::test_approved_item_missing_mandatory_sha256_rejected` (`PASSED`).
- **Current Status:** **REMEDIATED (CLOSED)**

---

### Finding SEC-06: Missing File Type and Regular File Assertion on Evidence Paths
- **Finding ID:** `SEC-06`
- **Severity:** `MEDIUM`
- **Affected Component:** `src/validation/authorization_gate.py` (`verify_item`)
- **Ownership:** Security Engineering
- **Reproduction Steps:** Provide `evidence_reference = "docs/evidence"` (a directory) or `"approval.sh"` (executable script).
- **Observed Result:** Prior to fix, directories passed `.exists()`, and executable scripts were not filtered.
- **Expected Result:** Assertion that evidence is a regular file (`.is_file()`) with approved document extensions.
- **Potential Impact:** Non-file paths or arbitrary binary/executable formats accepted as evidence.
- **Remediation Implemented:** Enforced `.is_file()` and restricted extensions strictly to `{".pdf", ".md", ".txt", ".json", ".png", ".jpg", ".jpeg"}`.
- **Verification Evidence:** `tests/security/test_phase_a7_adversarial_authorization.py::test_directory_as_evidence_rejected` and `test_unsupported_evidence_file_extension_rejected` (`PASSED`).
- **Current Status:** **REMEDIATED (CLOSED)**

---

### Finding SEC-07: Audit Trail PHI Keyword Coverage & Log Injection Risk
- **Finding ID:** `SEC-07`
- **Severity:** `LOW`
- **Affected Component:** `src/validation/authorization_gate.py` (`record_audit_event`)
- **Ownership:** Data Privacy Engineering
- **Reproduction Steps:** Call `record_audit_event()` with a reason containing historical subject IDs (`sub-2344857`) or newline characters (`\n`).
- **Observed Result:** Prior to fix, specific clinical cohort IDs were not filtered, and newlines could corrupt audit formatting.
- **Expected Result:** Rejection of historical clinical subject identifiers and strict rejection of control characters.
- **Potential Impact:** Potential inadvertent logging of clinical IDs; log structure corruption.
- **Remediation Implemented:** Expanded prohibited terms to include `2344857`, `peking`, `kki`, `nyu`, `abide`, `adhd200`, and added explicit newline / control character checks.
- **Verification Evidence:** `tests/security/test_phase_a7_adversarial_authorization.py::test_audit_log_newline_injection_rejected` and `test_audit_log_clinical_subject_keywords_rejected` (`PASSED`).
- **Current Status:** **REMEDIATED (CLOSED)**

---

## 3. Summary of Remediated Findings

| Finding ID | Severity | Description | Remediation Applied | Regression Test Added | Current Status |
|---|---|---|---|---|---|
| **SEC-01** | `HIGH` | Symlink bypass in evidence verification | Raw path and parent component symlink inspection | `test_symlink_evidence_file_rejected` | **CLOSED** |
| **SEC-02** | `HIGH` | Path traversal in evidence references | Strict `relative_to(base_dir)` + absolute path rejection | `test_path_traversal_in_evidence_reference_rejected` | **CLOSED** |
| **SEC-03** | `CRITICAL` | Alternate entrypoint `run_single_scan_poc.py` | Embedded fail-closed `ResearchAuthorizationGate` check | `test_alternate_entrypoint_run_single_scan_poc_fails_closed` | **CLOSED** |
| **SEC-04** | `HIGH` | Casing / truthy bypass in env check | Case-insensitive uppercase scanning + comprehensive truthy check | `test_case_insensitive_and_alternative_truthy_env_bypasses` | **CLOSED** |
| **SEC-05** | `MEDIUM` | Optional SHA-256 on approved evidence | Mandatory 64-char hex SHA-256 for all `APPROVED` items | `test_approved_item_missing_mandatory_sha256_rejected` | **CLOSED** |
| **SEC-06** | `MEDIUM` | Missing regular file & extension check | Enforced `.is_file()` and extension whitelist | `test_unsupported_evidence_file_extension_rejected` | **CLOSED** |
| **SEC-07** | `LOW` | Audit log injection & clinical ID filter | Expanded prohibited terms & control character rejection | `test_audit_log_newline_injection_rejected` | **CLOSED** |

---

## 4. Residual Risks & External Institutional Prerequisites

All code-level vulnerabilities have been remediated. The remaining **residual risks are external institutional dependencies**, which cannot be resolved by software engineering alone:

1. **Institutional Ethics Committee (IEC) Clearance Pending:**
   - Real MRI data ingestion remains blocked until formal written IEC approval is granted and authenticated on disk.
2. **Data Use Agreement (DUA) Execution Pending:**
   - Access to retrospective ADHD-200 scans remains blocked until the institutional signing official and NITRC/INDI execute a formal DUA.
3. **Supervisor Decision Sign-Off Pending:**
   - Preprocessing choices (STC, SDC, Defacing, Scanner Protocol) and the 8 QC acceptance thresholds remain `PENDING_SUPERVISOR_APPROVAL`.
4. **Documentary Evidence Authenticity:**
   - Cryptographic hashing proves file integrity but does not prove institutional legal validity; human review by the Principal Investigator remains mandatory.

---

## 5. Master Sign-Off Block

**Security Audit Decision:**  
`[X] ALL CRITICAL AND HIGH FINDINGS REMEDIATED AND VERIFIED`  
`[ ] REMEDIATION DEFICIENT — ACCESS BLOCKED`  

**Lead Security Auditor Signature:**  
`[ INDEPENDENT AUDIT COMPLETED — ALL REMEDIATIONS VERIFIED ]`  
**Date:** `2026-11-02`

