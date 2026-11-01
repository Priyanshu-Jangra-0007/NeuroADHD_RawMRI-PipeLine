# NeuroADHD RawMRI PoC V2 — Phase A.1.1 Manifest Verification Hardening Report

**Date**: 2026-11-02  
**Baseline Commit**: `14ff974326278aa8ed58a0b389c300532aa3f3b6`  
**Execution Context**: Local Python 3.11.15 Virtual Environment (`./.venv`)  
**Pipeline Mode**: Strictly Synthetic / Fail Closed  

---

## 1. Executive Summary

This remediation report documents the resolution of the two manifest verification weaknesses identified in Phase A.1:
1. **Mandatory HMAC Integrity Verification**: Removed hardcoded fallback keys, enforced mandatory retrieval from `NEUROADHD_HMAC_KEY`, validated token presence and format, applied constant-time `hmac.compare_digest()`, and protected all 5 input entries with deterministic JSON serialization.
2. **Strict Canonical Path Matching**: Replaced filename-only fallback logic with strict canonical path equality (`actual_resolved != manifest_resolved`), verified all five primary inputs, and rejected symlink inputs and symlink manifests.

---

## 2. Detailed Technical Remediations

### 2.1 Mandatory HMAC Verification (`scripts/run_poc_v2.py`)
- **Key Configuration**: Retrieves key via `get_integrity_key()` from `NEUROADHD_HMAC_KEY` (or `NEUROADHD_INTEGRITY_KEY`). If unset or whitespace, execution fails closed with a `SAFETY VIOLATION`. No hardcoded fallback is used. Key values are never logged.
- **Payload Canonical Serialization**: `build_canonical_manifest_payload()` serializes:
  - `manifest_magic`
  - `subject_id`
  - All five input entries: `bold_nii`, `bold_json`, `t1w_nii`, `mni_template`, `cc200_atlas`
  - For each entry: `filename`, canonical `resolved_path`, and `sha256` digest.
  Uses `json.dumps(..., sort_keys=True, separators=(',', ':'))` for deterministic byte serialization across platforms.
- **Token Verification**:
  - Rejects missing `integrity_hmac` in manifest (`SAFETY VIOLATION`).
  - Rejects empty, whitespace, or malformed non-hex tokens (`SAFETY VIOLATION`).
  - Compares token using `hmac.compare_digest(token.lower(), expected_hmac.lower())`.

### 2.2 Strict Canonical Path Equality
- **No Basename Fallback**: Removed `and p.name != entry.get("filename")`. Both canonical path (`actual_resolved == manifest_resolved`) and filename (`actual_resolved.name == entry.get("filename")`) must match exactly.
- **All Five Inputs Required**: `bold_nii`, `bold_json`, `t1w_nii`, `mni_template`, `cc200_atlas` must all be supplied and exist on disk.
- **Symlink Prohibition**: Input paths and manifest paths are checked via `is_symlink()`; any symlink triggers an immediate `SAFETY VIOLATION`.

---

## 3. Test Suite & Coverage Evidence

### Test Suite Execution
- **Total Tests Passing**: 119 / 119 (100%)
- **Unit Tests**: 68 / 68 passed
- **Synthetic Pipeline Integration**: 1 / 1 passed
- **Synthetic Runner Integration**: 25 / 25 passed
  - `test_runner_safety_guard_blocks_missing_hmac_token` [PASSED]
  - `test_runner_safety_guard_blocks_empty_or_malformed_hmac_token` [PASSED]
  - `test_runner_safety_guard_blocks_fabricated_manifest_altered_hmac` [PASSED]
  - `test_runner_safety_guard_blocks_modified_manifest_filename` [PASSED]
  - `test_runner_safety_guard_blocks_modified_resolved_path_with_same_basename` [PASSED]
  - `test_runner_safety_guard_blocks_swapped_input_files` [PASSED]
  - `test_runner_safety_guard_blocks_symlink_inputs` [PASSED]
  - `test_runner_safety_guard_blocks_symlink_manifest` [PASSED]
  - `test_runner_safety_guard_blocks_missing_integrity_key` [PASSED]
  - `test_runner_safety_guard_accepts_valid_synthetic_manifest` [PASSED]

### Coverage Results
- **Core Library (`src/`)**: **87% statement coverage** (673 / 777 statements)
- **Runner Orchestrator (`scripts/run_poc_v2.py`)**: **90% statement coverage** (443 / 491 statements)

---

## 4. Verification & Constraints
- Execution was strictly performed on synthetic phantom sessions and reference templates.
- Real participant MRI data processing remains disabled.
- No downstream models or checkpoints were modified.
