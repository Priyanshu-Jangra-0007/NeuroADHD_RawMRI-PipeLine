# NeuroADHD RawMRI PoC V2 — Phase A.8
# Technical Evidence & Cryptographic Verification Manifest

**Document ID:** `DOC-NA-A8-TEM-001`  
**Version:** 1.0.0  
**Phase:** Phase A.8 — Supervisor Handoff & External Research Authorization Preparation  
**Target Reviewer:** Technical Auditor, Software Verification Engineer, Research Supervisor  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Verified Commit:** [`0547574a6db2eb40668046e065fe8dcf989f09b1`](https://github.com/Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine/commit/0547574a6db2eb40668046e065fe8dcf989f09b1)  
**GitHub Actions CI Run:** [`41461226564`](https://github.com/Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine/actions/runs/41461226564) (Status: `completed`, Conclusion: `success`)  
**Pipeline Execution Mode:** `STRICTLY_SYNTHETIC_ONLY`

---

## 1. Executive Summary of Technical Evidence

This manifest provides the formal record of empirical test execution, cryptographic checksums, security audit findings, and memory-isolation verification for **NeuroADHD RawMRI PoC V2**.

Every technical claim in this document is backed by reproducible automated tests in the repository and verified continuous integration logs on GitHub Actions.

---

## 2. Authorization Security Architecture & Enforcement Controls

The pipeline's fail-closed governance is enforced by the [`ResearchAuthorizationGate`](<your-project-folder>/src/validation/authorization_gate.py#L65) engine:

1. **Schema Validation:** The authorization register ([`configs/research_authorization_register.json`](<your-project-folder>/configs/research_authorization_register.json)) is validated against Draft-07 JSON Schema ([`configs/authorization_register_schema.json`](<your-project-folder>/configs/authorization_register_schema.json)) asserting non-empty strings, valid enum states, and strict object structures.
2. **23 Authorization Items:** Tracks 12 supervisor decisions, 5 institutional legal clearances, and 6 operational safeguards. All 23 items must be in `APPROVED` state to enable real data.
3. **Physical File Existence:** Every `APPROVED` item must link to an existing regular file in `evidence_reference`.
4. **Mandatory SHA-256 Digest Verification:** Approved evidence files must match an expected 64-character hexadecimal SHA-256 digest, evaluated using constant-time comparison (`hmac.compare_digest`).
5. **Path Traversal Protection:** Absolute paths and directory traversal sequences (`..`) in evidence paths are strictly rejected. Paths must resolve within the repository boundaries.
6. **Symlink Traversal Protection:** Candidate evidence paths and all parent directory components are verified via `.is_symlink()` *prior* to path resolution, preventing symlink substitution attacks.
7. **Environment Variable Bypass Protection:** The engine performs a case-insensitive scan across all `os.environ` keys. Any non-falsy value for prohibited override variables (`NEUROADHD_OVERRIDE_SAFETY`, `NEUROADHD_FORCE_REAL_DATA`, `NEUROADHD_DISABLE_AUTH_GATE`, `NEUROADHD_AUTH_OVERRIDE`) triggers an immediate unrecoverable `AuthorizationBypassAttemptError`.
8. **Entrypoint Enforcement:** Both pipeline entrypoints ([`scripts/run_poc_v2.py`](<your-project-folder>/scripts/run_poc_v2.py) and [`scripts/run_single_scan_poc.py`](<your-project-folder>/scripts/run_single_scan_poc.py)) invoke the gate engine and safety guards at function entry before any file operations.
9. **Audit Trail Sanitization:** Audit log entries are sanitized against newline/control character injection (`\n`, `\r`, `\x00`) and prohibited clinical cohort keywords (`sub-2344857`, `peking`, `kki`, `nyu`, `adhd200`).

---

## 3. Phase A.7 Security Audit Findings & Remediations

During adversarial security testing in Phase A.7, seven vulnerabilities were uncovered and remediated:

| Finding ID | Severity | Vulnerability Mechanism | Implemented Remediation | Verification Test |
| :--- | :---: | :--- | :--- | :--- |
| **`SEC-01`** | **HIGH** | Symlink traversal via `.resolve()` prior to `.is_symlink()` check | Recursively inspect candidate path and all parent components with `.is_symlink()` before canonical resolution | `test_symlink_evidence_rejected` |
| **`SEC-02`** | **HIGH** | Filesystem boundary escape via absolute evidence paths (`/etc/hosts`) | Explicitly prohibit absolute paths (`raw_path.is_absolute() -> REJECT`) and assert `resolved_path.relative_to(base_dir)` | `test_absolute_path_outside_repository_rejected`, `test_path_traversal_evidence_rejected` |
| **`SEC-03`** | **CRITICAL** | Alternate legacy entrypoint (`scripts/run_single_scan_poc.py`) lacked gate checks | Embedded fail-closed `ResearchAuthorizationGate` and `check_safety_guards` at top of `run_poc()` | `test_legacy_entrypoint_enforces_gate` |
| **`SEC-04`** | **HIGH** | Case-sensitive scanning and truthy parsing in env var bypass checks | Case-insensitive uppercase inspection across all `os.environ` keys; reject any non-empty value not in `("", "0", "false", "no", "none", "off")` | `test_environment_bypass_case_insensitive`, `test_environment_bypass_truthy_variations` |
| **`SEC-05`** | **MEDIUM** | SHA-256 digest was optional on approved items | Mandated 64-char hex `evidence_sha256` for all `APPROVED` items; evaluated via `hmac.compare_digest` | `test_missing_evidence_sha256_rejected`, `test_tampered_evidence_file_rejected` |
| **`SEC-06`** | **MEDIUM** | Directory paths and arbitrary file extensions accepted as evidence | Enforce `.is_file()` and restrict extensions to `{".pdf", ".md", ".txt", ".json", ".png", ".jpg", ".jpeg"}` | `test_directory_evidence_rejected`, `test_unsupported_evidence_extension_rejected` |
| **`SEC-07`** | **LOW** | Audit log vulnerable to newline injection and clinical subject keywords | Enforced control character rejection (`\n`, `\r`, `\x00`) and comprehensive clinical subject token blocklist | `test_audit_log_phi_rejection`, `test_audit_log_control_character_injection_rejected` |

---

## 4. Empirical Test Suite Results & Coverage

### 4.1 Test Execution Summary

```
========================================================================================
FULL REPOSITORY TEST EXECUTION (Python 3.11.15, Pytest 9.2.0)
========================================================================================
Total Items Collected:                      209 items
- Unit & Security Tests Passed (Local):     202 passed
- Integration / Docker Tests Skipped:         5 skipped (Requires active Docker daemon)
- External Model Tests Skipped (Local):       2 skipped (When NEUROADHD_MODEL_DIR unset)
- Total Passing with Model Benchmark Dir:   204 passed, 5 skipped
- Failure Count:                              0 failures (100% pass rate among executed)
Total Execution Time:                       39.18 seconds
========================================================================================
```

### 4.2 Breakdown by Test Suite

| Test Suite Module | Target Scope | Collected | Passed | Skipped | Failed | Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [`tests/security/test_phase_a7_adversarial_authorization.py`](<your-project-folder>/tests/security/test_phase_a7_adversarial_authorization.py) | Adversarial Security & Gate Bypasses | 14 | 14 | 0 | 0 | 88% |
| [`tests/unit/test_phase_a6_authorization_gate.py`](<your-project-folder>/tests/unit/test_phase_a6_authorization_gate.py) | Gatekeeper Functionality & State Machine | 16 | 16 | 0 | 0 | 88% |
| [`tests/unit/test_phase_a4_model_compatibility.py`](<your-project-folder>/tests/unit/test_phase_a4_model_compatibility.py) | Frozen Classifier & Feature Pipeline | 16 | 16* | 0* | 0 | 100% |
| [`tests/unit/test_phase_a4_safety_gate.py`](<your-project-folder>/tests/unit/test_phase_a4_safety_gate.py) | Phase A.4 Safety Guards & Flag Rejection | 8 | 8 | 0 | 0 | 95% |
| [`tests/unit/test_registration.py`](<your-project-folder>/tests/unit/test_registration.py) | Rigid EPI-T1w & SyN T1w-MNI Registration | 28 | 28 | 0 | 0 | 76% |
| [`tests/unit/test_scientific_registration_validation.py`](<your-project-folder>/tests/unit/test_scientific_registration_validation.py) | CC200 Canonical Grid Alignment | 16 | 16 | 0 | 0 | 87% |
| [`tests/unit/test_segmentation.py`](<your-project-folder>/tests/unit/test_segmentation.py) | Tissue Segmentation & Mask Erosion | 21 | 21 | 0 | 0 | 83% |
| [`tests/unit/test_nuisance.py`](<your-project-folder>/tests/unit/test_nuisance.py) | aCompCor & Confound Matrix Construction | 18 | 18 | 0 | 0 | 89% |
| [`tests/unit/test_qc.py`](<your-project-folder>/tests/unit/test_qc.py) | 8 Quantitative QC Metric Assertions | 10 | 10 | 0 | 0 | 98% |
| [`tests/unit/test_connectivity.py`](<your-project-folder>/tests/unit/test_connectivity.py) | Correlation & Fisher-z Transformation | 13 | 13 | 0 | 0 | 100% |
| [`tests/unit/test_memory_guard.py`](<your-project-folder>/tests/unit/test_memory_guard.py) | In-Memory Allocation & Process Limits | 9 | 9 | 0 | 0 | 81% |
| [`tests/unit/test_input_validation.py`](<your-project-folder>/tests/unit/test_input_validation.py) | BIDS Input Contract & Sidecar Parsing | 6 | 6 | 0 | 0 | 94% |
| [`tests/integration/test_runner_poc_v2_synthetic.py`](<your-project-folder>/tests/integration/test_runner_poc_v2_synthetic.py) | End-to-End 19-Stage Synthetic Execution | 28 | 28 | 0 | 0 | 92% |
| [`tests/integration/test_pipeline_v2_synthetic.py`](<your-project-folder>/tests/integration/test_pipeline_v2_synthetic.py) | Synthetic Pipeline Assembly | 1 | 1 | 0 | 0 | 90% |
| [`tests/integration/test_container_model_integration.py`](<your-project-folder>/tests/integration/test_container_model_integration.py) | Containerized Model Inference | 5 | 0 | 5* | 0 | N/A |

*\*Note: 16/16 model compatibility tests pass when `NEUROADHD_MODEL_DIR` is set to the benchmark artifact directory. Container tests execute and pass in GitHub Actions CI where Docker is active.*

### 4.3 Code Coverage
- **Core Library (`src/`):** **84% total statement coverage** (1084 / 1290 statements).
- **Authorization Engine ([`src/validation/authorization_gate.py`](<your-project-folder>/src/validation/authorization_gate.py)):** **88% coverage** (198 / 224 statements).
- **Quality Control Engine ([`src/preprocessing/qc.py`](<your-project-folder>/src/preprocessing/qc.py)):** **98% coverage** (230 / 234 statements).

---

## 5. Frozen Scientific Research Assets & Cryptographic Hashes

To ensure exact reproducibility and prevent model drift, all reference templates and machine-learning models are cryptographically locked:

| Research Asset Description | Repository Location | Canonical Geometry | Cryptographic SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| **Canonical CC200 Atlas** | `data/atlases/cc200/` | Matrix: `(63, 75, 61)`<br>Voxel: `3.0 mm` iso<br>ROIs: 200 contiguous | `9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650` |
| **Frozen Preprocessing Pipeline** | External Benchmark Artifact | Input: $\mathbb{R}^{19,900}$<br>Output: $\mathbb{R}^{100}$ | `228d3f0180e4855d49fd732327ab1fc4f255beb7bea43dd4fd1db3064aa86249` |
| **Frozen Classifier Binary** | External Benchmark Artifact | Scikit-learn RF<br>Estimators: 100 | `d8eb23eb8c6d25ffc12b2bab515a2a7112daeeb0afa8f4e9eaa499dbe5b5f2ef` |

---

## 6. Cryptographic Signatures vs. Legal Governance Clarifications

### 6.1 Role of SHA-256 Hashes
The pipeline utilizes SHA-256 cryptographic digests extensively. It is vital to recognize what SHA-256 proves:
- **What SHA-256 Proves:** It provides mathematical proof of **bit-level data integrity** relative to a known reference digest. It guarantees that an atlas file, model binary, or evidence PDF has not been altered, truncated, corrupted, or tampered with since the hash was recorded.
- **What SHA-256 Does NOT Prove:** SHA-256 **does not prove the legal validity or institutional authenticity** of an external approval. A fraudulent or fabricated document also produces a valid SHA-256 hash. Institutional authenticity depends on out-of-band verification by the research supervisor and institutional authorities.

### 6.2 Role of Supervisor HMAC Signature
The pipeline includes an HMAC-SHA256 signature verification mechanism for the authorization register:
- **Technical Purpose:** HMAC provides cryptographic attestation that the authorization register was reviewed and approved by an authorized entity possessing the shared secret key (`NEUROADHD_HMAC_KEY`).
- **Legal Limitation:** An HMAC signature **is not a legally recognized electronic signature** under statutory contract or bioethics law (such as India's Information Technology Act 2000).
- **Key Custody Requirements:** The `NEUROADHD_HMAC_KEY` must be retained exclusively by the Primary Research Supervisor. It must never be committed to source code or shared in plain text.
- **Institutional Alignment:** Prior to processing real participant data, the institution's formal physical or e-signature mechanisms (e.g., DSC, signed physical letter on letterhead) must be documented in addition to software HMAC attestation.

---

## 7. OS-Level Memory Isolation Verification

The pipeline is constrained to a **2.5 GB RAM ceiling** to ensure operation on standard compute infrastructure and prevent unmanaged memory expansion during 4D BOLD registration:
- **Enforcement Mechanism:** Linux `cgroup v2` memory controllers enforced via Docker: `--memory=2560m --memory-swap=2560m`.
- **Kernel OOM Killer:** Enabled. If container memory consumption exceeds 2560 MiB, the Linux kernel terminates the process rather than paging to disk.
- **Verification Script:** [`scripts/verify_container_memory.sh`](<your-project-folder>/scripts/verify_container_memory.sh).
- **CI Result:** Successfully verified in GitHub Actions run `41461226564` (Step: "Build Container Image & Verify OS Memory Isolation").
