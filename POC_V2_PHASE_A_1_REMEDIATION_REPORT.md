# NeuroADHD RawMRI PoC V2 — Phase A.1 Final Independent Remediation Report

**Date**: 2026-11-02  
**Baseline Commit**: `836868a`  
**Execution Context**: Local Python 3.11.15 Virtual Environment (`./.venv`)  
**Pipeline Mode**: Strictly Synthetic / Fail Closed  

---

## 1. Executive Summary

This remediation report documents the resolution of the five core findings identified during the Phase A review of `NeuroADHD_RawMRI-PipeLine`. All changes preserve the research pipeline's scientific integrity, spatial consistency, and strict prohibition on processing real participant data while achieving high verifiable reproducibility and test coverage.

| Remediation Area | Status | Key Implementation |
|---|---|---|
| **1. Synthetic Manifest Authenticity** | Completed & Verified | HMAC-SHA256 integrity token across all 5 inputs; resolved path validation; strict symlink rejection. Accurately documented as integrity protection (not PKI digital signature). |
| **2. Canonical CC200 Enforcement** | Completed & Verified | Replaced filename string heuristics with explicit `--atlas-mode` policy (`canonical` vs `synthetic_phantom`); cryptographic SHA-256 and geometry verification independent of filename. |
| **3. Mandatory Provenance Validation** | Completed & Verified | Pipeline fails closed (`FileNotFoundError`) if schema missing; jsonschema validation gate; post-write output verification against `output_manifest.json`. |
| **4. Failure & Output Isolation** | Completed & Verified | Staging directory lifecycle (`.staging_run_{run_tag}`); quarantined on failure to `failed_run_{run_tag}` with `run_manifest.json` (`status: "FAILED"`); only verified runs promoted to `run_{run_tag}` (`status: "SUCCESS"`). |
| **5. Reproducibility & CI** | Completed & Verified | Generated `requirements-lock.txt`; created `.github/workflows/ci.yml` (Python 3.11); separate coverage reporting: `src/` (87%) and `scripts/` (90%). |

---

## 2. Remediation Details

### 2.1 Synthetic Manifest Authenticity (`scripts/run_poc_v2.py`)
- **Threat Model Clarification**: Documented that the manifest is an internal integrity mechanism protecting against accidental modification, file swapping, and path misconfiguration. It does not claim public-key digital signatures.
- **HMAC-SHA256 Token**: Added `compute_manifest_hmac()` using a configurable integrity anchor (`NEUROADHD_INTEGRITY_SALT`).
- **All 5 Inputs Registered & Verified**:
  - `bold_nii` (4D BOLD time series)
  - `bold_json` (BIDS sidecar metadata)
  - `t1w_nii` (Anatomical T1w volume)
  - `mni_template` (Standard space template)
  - `cc200_atlas` (CC200 ROI parcellation)
- **Path Resolution & Swapping Defense**: Rejects inputs when resolved canonical path or filename does not match the manifest registration.
- **Symlink Prohibition**: Input files and manifests that are symlinks are rejected unconditionally with a `SAFETY VIOLATION`.

### 2.2 Canonical CC200 Enforcement (`scripts/run_poc_v2.py`, `src/parcellation/extractor.py`)
- Removed filename substring checks (`"cc200_roi_atlas" in str(atlas_path)`).
- Added explicit parameter `--atlas-mode` (`canonical` | `synthetic_phantom`):
  - In `canonical` mode: Invokes `verify_canonical_cc200_atlas()` to assert canonical SHA-256 (`9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650`), spatial shape `(63, 75, 61)`, 3.0 mm isotropic voxel zooms, canonical affine, and 200 contiguous integer labels, regardless of filename.
  - In `synthetic_phantom` mode: Verifies the phantom atlas against the authenticated synthetic manifest.

### 2.3 Mandatory Provenance Validation & Post-Write Verification
- `configs/provenance_schema.json` is strictly required. If missing, execution halts immediately with `FileNotFoundError`.
- Validates the 19-stage provenance dictionary against the JSON Schema using `jsonschema.validate()`.
- Distinguishes the 11 pre-provenance artifact checksums stored in `provenance.json` from the 12 entries recorded in `output_manifest.json` (including `provenance.json`).
- Post-write verification: Re-reads every written artifact from disk, verifies its SHA-256 against `output_manifest.json`, and verifies all 6 validation gates.

### 2.4 Failure and Partial-Output Isolation
- Runs execute within a temporary hidden staging directory: `outputs/<sub_id>/.staging_run_<run_tag>/`.
- On success: Staging directory is finalized to `outputs/<sub_id>/run_<run_tag>/` and includes `run_manifest.json` with `"status": "SUCCESS"`.
- On error/failure: Pipeline catches exception, dumps diagnostic `pipeline_failure_report.json` and `run_manifest.json` with `"status": "FAILED"`, and quarantines the directory as `outputs/<sub_id>/failed_run_<run_tag>/`. No partial `run_<run_tag>` is ever created.

### 2.5 Reproducibility & CI
- Created `requirements-lock.txt` pinning all 53 virtual environment dependencies.
- Added `.github/workflows/ci.yml` running on Ubuntu Latest with Python 3.11, executing unit and integration tests and generating separate XML/terminal coverage reports.
- Local verification note: The test suite and coverage have been executed and verified in the local environment. Remote GitHub Actions execution will occur upon git push.

---

## 3. Test Coverage & Verification Evidence

### Test Suite Execution
- **Total Tests Passing**: 112 / 112
- **Unit Tests**:
  - `tests/unit/test_connectivity.py`: 13 passing (includes arbitrary filename canonical atlas verification and bad-atlas canonical-name rejection)
  - `tests/unit/test_input_validation.py`: 6 passing
  - `tests/unit/test_nuisance.py`: 18 passing
  - `tests/unit/test_qc.py`: 7 passing
  - `tests/unit/test_registration.py`: 28 passing
  - `tests/unit/test_segmentation.py`: 21 passing
- **Integration Tests**:
  - `tests/integration/test_pipeline_v2_synthetic.py`: 1 passing
  - `tests/integration/test_runner_poc_v2_synthetic.py`: 18 passing (includes HMAC tampering, missing manifest entry, swapped inputs, symlink rejection, canonical vs synthetic atlas modes, missing schema fail-closed, post-write output verification, failure staging quarantine)

### Code Coverage Breakdown

#### 1. Core Source Library (`src/`)
```
Name                                Stmts   Miss  Cover   Missing
-----------------------------------------------------------------
src/__init__.py                         1      0   100%
src/features/__init__.py                0      0   100%
src/features/connectivity.py           26      2    92%   34, 44
src/integration/__init__.py             0      0   100%
src/parcellation/__init__.py            0      0   100%
src/parcellation/extractor.py          61      8    87%   78, 84, 89, 97, 101, 163, 165, 171
src/preprocessing/__init__.py           6      0   100%
src/preprocessing/motion.py            63      6    90%   33, 72, 93, 137-139
src/preprocessing/nuisance.py         106     12    89%   91, 95, 104, 151, 193, 195, 199, 226, 228, 232, 289, 323
src/preprocessing/qc.py               121      0   100%
src/preprocessing/registration.py     178     43    76%   33, 107, 172, 174, 299-304, 307-312, 315-320, 323-328, 413, 436-439, 486, 493-500, 544, 550, 562-570, 578
src/preprocessing/segmentation.py     179     31    83%   49, 100, 108-109, 215, 257-285, 389-390, 430, 455
src/validation/__init__.py              2      0   100%
src/validation/bids_validator.py       34      2    94%   38-39
-----------------------------------------------------------------
TOTAL                                 777    104    87%
```

#### 2. End-to-End Orchestrator (`scripts/run_poc_v2.py`)
```
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
scripts/run_poc_v2.py     446     45    90%   53, 100, 194, 201, 203, 210, 227, 242-243, 248, 253, 284, 527, 535-539, 551, 959-960, 997, 1066-1067, 1074-1075, 1082-1176, 1180
-----------------------------------------------------
TOTAL                     446     45    90%
```

---

## 4. Conclusion & Next Steps

Phase A.1 independent remediation is complete. The pipeline enforces zero-trust synthetic safety, strictly verifies canonical CC200 parcellation geometry, validates provenance schema and output integrity, and isolates partial/failed runs while maintaining comprehensive test coverage (87% on `src/`, 90% on `scripts/run_poc_v2.py`).
