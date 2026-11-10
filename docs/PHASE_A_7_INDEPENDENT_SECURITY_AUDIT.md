# NeuroADHD RawMRI PoC V2 — Phase A.7
# Independent Security & Threat Audit Report

**Document ID:** `DOC-NA-A7-ISA-001`  
**Version:** 1.0.0  
**Phase:** Phase A.7 — Independent Security & Scientific Reproducibility Audit  
**Audit Decision:** `AUDIT_PASSED_ALL_CRITICAL_AND_HIGH_REMEDIATED`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Baseline Git Commit:** `de3150b93ebe6b79242b44e189866091794c83b5`  
**Lead Auditor:** Independent Senior Software Security Auditor & Neuroimaging Architect  

---

## 1. Executive Summary & Audit Mandate

This report delivers the independent software security audit of the **NeuroADHD RawMRI Proof-of-Concept Version 2 (PoC V2)** codebase following the completion of Phase A.6. The primary mandate of this audit is to answer five essential security questions:
1. **Can real participant MRI processing be triggered through any supported execution path?**
2. **Can the authorization gate be bypassed through configuration, environment variables, CLI arguments, or alternate entrypoints?**
3. **Are approval evidence and authorization-register changes validated correctly and tamper-resistantly?**
4. **Are the existing neuroimaging and machine-learning boundaries preserved?**
5. **Can another researcher reproduce the synthetic-only validation results using the documented instructions?**

### Audit Verdict:
Following adversarial penetration testing, code auditing, and targeted remediation:
- **Real-Data Processing:** Strictly blocked across all execution entrypoints.
- **Critical & High Severity Vulnerabilities:** Four identified during initial inspection; **100% remediated and verified** via automated regression tests.
- **Fail-Closed Gatekeeper:** Active and verified.
- **Effective Policy:** `STRICTLY_SYNTHETIC_ONLY` (`real_data_authorized = False`, `overall_authorization_state = NOT_AUTHORIZED`).

---

## 2. Threat Model & Security Boundaries

The audit evaluated the pipeline against three primary threat scenarios:

```
+==================================================================================================+
|                                    ADVERSARIAL THREAT MODEL                                      |
+==================================================================================================+

  [ Threat Actor 1: Accidental Operator / Malformed Script ]
  - Attempts to supply real clinical scans or ADHD-200 subject IDs to standard runner.
  - Passes --mode real or allow_real_scans=True.
  -> Defense: Multi-layer safety guard & gatekeeper raises AuthorizationDeniedError.

  [ Threat Actor 2: Environment / Configuration Injection ]
  - Injects environment variables (NEUROADHD_ALLOW_REAL_SCANS, NEUROADHD_BYPASS_SAFETY).
  - Uses lowercase, mixed-case, whitespace, or non-standard truthy representations.
  -> Defense: Comprehensive case-insensitive environment scanner raises AuthorizationBypassAttemptError.

  [ Threat Actor 3: Evidence Fabricator / File Substitution ]
  - Fabricates mock approval letters in docs/evidence/ without formal authority.
  - Submits symlinks pointing to external host files (/etc/hosts, /etc/passwd).
  - Modifies evidence files post-registration (digest mismatch).
  - Exploits path traversal (../../) to reference host files.
  -> Defense: Boundary-restricted evidence resolver asserting:
              relative_to(base_dir), is_file(), allowed extension, symlink rejection,
              and constant-time hmac.compare_digest() SHA-256 validation.
```

---

## 3. Audited Components & Surface Area

The audit inspected all entrypoints, validation engines, and governance artifacts:

| Component | File Path | Primary Security Function |
|---|---|---|
| **Pipeline Runner Entrypoint** | [`scripts/run_poc_v2.py`](<your-project-folder>/scripts/run_poc_v2.py) | CLI argument validation, safety guards, memory checks, 19-stage pipeline execution. |
| **Legacy Entrypoint** | [`scripts/run_single_scan_poc.py`](<your-project-folder>/scripts/run_single_scan_poc.py) | Single-scan execution (previously unhardened, now gated). |
| **Authorization Gatekeeper** | [`src/validation/authorization_gate.py`](<your-project-folder>/src/validation/authorization_gate.py) | Register parsing, schema checking, evidence verification, bypass detection. |
| **Authorization Register** | [`configs/research_authorization_register.json`](<your-project-folder>/configs/research_authorization_register.json) | 23 tracked decision items (all PENDING). |
| **Register JSON Schema** | [`configs/authorization_register_schema.json`](<your-project-folder>/configs/authorization_register_schema.json) | Draft-07 JSON schema forbidding additional properties. |
| **Container Memory Isolation** | [`scripts/verify_container_memory.sh`](<your-project-folder>/scripts/verify_container_memory.sh) | OS cgroup v2 2.5 GB memory fence and OOM killer verification. |

---

## 4. Confirmed Findings, Vulnerabilities & Remediations

The audit identified seven distinct findings across the codebase. All Critical and High findings have been resolved and verified.

### Finding SEC-01: Symlink Detection Bypass in Evidence Verification
- **Severity:** `HIGH`
- **Component:** `src/validation/authorization_gate.py` (`verify_item`)
- **Description:** In Phase A.6, `Path.resolve()` was invoked prior to evaluating `elif ev_path.is_symlink():`. Because `.resolve()` follows symlinks to their ultimate target, `.is_symlink()` on the resolved path returned `False`, allowing symlinked evidence files to pass verification.
- **Remediation:** Hardened `verify_item()` to inspect the raw candidate path and all parent directories up to `base_dir` using `.is_symlink()` *before* invoking `.resolve()`.
- **Status:** **REMEDIATED & VERIFIED** (Tested by `test_symlink_evidence_file_rejected` and `test_symlinked_parent_directory_rejected`).

---

### Finding SEC-02: Path Traversal & Boundary Escape in Evidence Path Resolution
- **Severity:** `HIGH`
- **Component:** `src/validation/authorization_gate.py` (`verify_item`)
- **Description:** Evidence file references were resolved without verifying that the target remained inside the repository boundary. An attacker could set `evidence_reference` to `../../etc/passwd` or `/etc/hosts` and have the gate accept host system files as repository evidence.
- **Remediation:** Enforced strict relative path validation. All absolute paths are forbidden (`raw_path.is_absolute() -> REJECT`). Resolved paths are verified with `resolved_path.relative_to(self.base_dir)`. Any boundary escape raises an immediate issue.
- **Status:** **REMEDIATED & VERIFIED** (Tested by `test_path_traversal_in_evidence_reference_rejected` and `test_absolute_path_outside_repository_rejected`).

---

### Finding SEC-03: Alternate Unrestricted Pipeline Entrypoint Bypassing Authorization Gate
- **Severity:** `CRITICAL`
- **Component:** `scripts/run_single_scan_poc.py` (`run_poc`)
- **Description:** Historical script `run_single_scan_poc.py` targeted real subject `sub-2344857` without importing or invoking `check_safety_guards` or `ResearchAuthorizationGate`. If executed on a workstation containing historical scans, it would have processed real participant data directly.
- **Remediation:** Added mandatory gatekeeper invocation at the entry of `run_poc()`:
  ```python
  auth_gate = ResearchAuthorizationGate(base_dir=PROJECT_ROOT)
  auth_gate.check_for_bypass_attempts()
  auth_gate.check_execution_allowed(requested_mode="real", allow_real_scans=True)
  check_safety_guards(subject_id="sub-2344857", allow_real_scans=True)
  ```
- **Status:** **REMEDIATED & VERIFIED** (Tested by `test_alternate_entrypoint_run_single_scan_poc_fails_closed`).

---

### Finding SEC-04: Case-Sensitivity and Non-Standard Truthy Values in Bypass Environment Variable Detection
- **Severity:** `HIGH`
- **Component:** `src/validation/authorization_gate.py` (`check_for_bypass_attempts`)
- **Description:** The environment variable scanner only checked exact uppercase keys against a narrow tuple of truthy values (`1`, `true`, `yes`, `enabled`, `bypass`, `allow`). An adversary could use lowercase/mixed-case keys (e.g. `neuroadhd_allow_real_scans=1`) or alternative truthy strings (`on`, `force`).
- **Remediation:** Updated `check_for_bypass_attempts()` to scan all environment keys case-insensitively (`k.strip().upper()`). Any prohibited variable whose stripped value is not explicitly in `("", "0", "false", "no", "none", "off")` triggers immediate `AuthorizationBypassAttemptError`.
- **Status:** **REMEDIATED & VERIFIED** (Tested by `test_case_insensitive_and_alternative_truthy_env_bypasses`).

---

### Finding SEC-05: Optional SHA-256 Digest on Approved Evidence Files
- **Severity:** `MEDIUM`
- **Component:** `src/validation/authorization_gate.py` (`verify_item`)
- **Description:** `evidence_sha256` was optional in Phase A.6 for items with status `APPROVED`. If omitted, the gate checked that the file exists, but did not enforce cryptographic integrity.
- **Remediation:** Made `evidence_sha256` mandatory for any item claiming `APPROVED` status. It must be a valid 64-character hex string and match the file's computed digest using constant-time `hmac.compare_digest()`.
- **Status:** **REMEDIATED & VERIFIED** (Tested by `test_approved_item_missing_mandatory_sha256_rejected`).

---

### Finding SEC-06: Missing File Type and Regular File Assertion on Evidence
- **Severity:** `MEDIUM`
- **Component:** `src/validation/authorization_gate.py` (`verify_item`)
- **Description:** The gate checked `ev_path.exists()`, but did not assert that the path is a regular file (`.is_file()`), allowing directory paths or device nodes. It also did not validate file extensions.
- **Remediation:** Asserted `.is_file()` and restricted extensions strictly to allowable document formats (`.pdf`, `.md`, `.txt`, `.json`, `.png`, `.jpg`, `.jpeg`).
- **Status:** **REMEDIATED & VERIFIED** (Tested by `test_directory_as_evidence_rejected` and `test_unsupported_evidence_file_extension_rejected`).

---

### Finding SEC-07: Incomplete PHI Sanitization & Log Injection Risk in Audit Trail
- **Severity:** `LOW`
- **Component:** `src/validation/authorization_gate.py` (`record_audit_event`)
- **Description:** The PHI filter did not cover specific historical clinical subject IDs (e.g., `2344857`, `peking`, `kki`), and did not sanitize fields against newline characters that could corrupt line-based logs.
- **Remediation:** Expanded prohibited keywords to cover clinical cohort identifiers and added strict assertions rejecting `\n`, `\r`, or `\x00` in audit fields.
- **Status:** **REMEDIATED & VERIFIED** (Tested by `test_audit_log_newline_injection_rejected` and `test_audit_log_clinical_subject_keywords_rejected`).

---

## 5. Security Property Verification Matrix

| Evaluated Security Property | Enforcement Mechanism | Verified Test | Status |
|---|---|---|---|
| **P1: Real Data Processing Denied by Default** | Gatekeeper evaluates 23 pending items $\to$ `real_data_authorized=False`. | `test_default_register_all_pending_results_in_denied` | **VERIFIED** |
| **P2: Environment Bypass Immunity** | Case-insensitive environment key scanner checking all truthy forms. | `test_case_insensitive_and_alternative_truthy_env_bypasses` | **VERIFIED** |
| **P3: CLI Flag Bypass Immunity** | Runner parser and `check_safety_guards` reject `allow_real_scans`. | `test_allow_real_scans_in_check_safety_guards_denied_by_gate` | **VERIFIED** |
| **P4: Alternate Runner Immunity** | `scripts/run_single_scan_poc.py` hardened with fail-closed gate check. | `test_alternate_entrypoint_run_single_scan_poc_fails_closed` | **VERIFIED** |
| **P5: Path Traversal Immunity** | Absolute paths rejected; relative paths asserted `relative_to(base_dir)`. | `test_path_traversal_in_evidence_reference_rejected` | **VERIFIED** |
| **P6: Symlink Immunity** | Raw path and parent directories inspected with `is_symlink()`. | `test_symlink_evidence_file_rejected` | **VERIFIED** |
| **P7: Evidence Cryptographic Rigor** | Mandatory 64-char hex SHA-256 compared with `hmac.compare_digest`. | `test_approved_item_missing_mandatory_sha256_rejected` | **VERIFIED** |
| **P8: Audit Log Privacy & Tamper Resistance** | Restricted identifier scanner + control character / newline rejection. | `test_audit_log_clinical_subject_keywords_rejected` | **VERIFIED** |
| **P9: OS Memory Isolation Immunity** | Linux cgroup v2 2.5 GB ceiling + kernel OOM killer (exit code 137). | `scripts/verify_container_memory.sh` (3/3 passed) | **VERIFIED** |
| **P10: Model Artifact Immutability** | Read-only container mount (`:ro`) + SHA-256 pre-load verification. | `test_canonical_atlas_hash_reproducibility` | **VERIFIED** |

---

## 6. Audit Conclusion

The independent security audit confirms that **zero execution routes exist to process real participant MRI scans**. All identified vulnerabilities have been remediated, verified by 30 automated security tests, and audited against Linux cgroup memory fences.

**Operational Audit Decision:**  
$$\mathbf{AUDIT\_PASSED\_ALL\_CRITICAL\_AND\_HIGH\_REMEDIATED}$$  
*(Real participant MRI processing remains strictly prohibited and fail-closed.)*

