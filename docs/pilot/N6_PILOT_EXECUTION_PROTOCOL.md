# N=6 FEASIBILITY PILOT — EXECUTION PROTOCOL & OPERATIONAL PLAN

> **`DRAFT — NOT AUTHORIZED FOR REAL MRI DATA`**
>
> **Document ID:** `DOC-PILOT-EXEC-001`  
> **Project:** NeuroADHD — Raw MRI Processing and ADHD Classification  
> **Pipeline Baseline:** `PoC V2 — 23-Stage Preprocessing & CC200 Feature Extraction Pipeline`  
> **Step 8 Authorization Status:** `NOT_AUTHORIZED` (`STRICTLY_SYNTHETIC_ONLY`, `real_data_authorized = false`)  
> **Pilot Cohort Scope Limit:** Strictly **$N = 6$** placeholder feasibility slots (`sub-pilot-01` through `sub-pilot-06`)  
> **Related Governance Documents:** [`docs/governance/MASTER_RESEARCH_PROTOCOL.md`](../governance/MASTER_RESEARCH_PROTOCOL.md) (`DOC-GOV-PROT-001`), [`docs/governance/N6_PILOT_AUTHORIZATION_MATRIX.md`](../governance/N6_PILOT_AUTHORIZATION_MATRIX.md) (`DOC-GOV-PILOTMAT-001`), [`docs/governance/STEP8_AUTHORIZATION_VALIDATION_REPORT.md`](../governance/STEP8_AUTHORIZATION_VALIDATION_REPORT.md)

---

## 1. Scientific & Engineering Scope

This protocol defines the **operational execution workflow** for the planned $N = 6$ **technical feasibility pilot** of the 23-stage NeuroADHD MRI preprocessing pipeline (`scripts/run_poc_v2.py`) prior to any large-scale retrospective cohort work or clinical validation.

### Explicit Non-Clinical & Scope Boundaries
1. **Feasibility Scope Only:** The future authorized $N = 6$ pilot tests solely:
   - BIDS intake, metadata completeness, and optional DICOM-to-BIDS conversion (`src/intake/service.py`);
   - Fail-closed per-participant authorization verification (`src/validation/authorization_gate.py`, `src/validation/authorization_evidence.py`);
   - End-to-end 23-stage preprocessing within the `<= 2.50 GB` peak RSS memory ceiling;
   - Anatomical defacing, sidecar sanitization, QC overlay generation, and `200` CC200 ROIs / `19,900` Fisher-z connectivity extraction;
   - Historical C-PAC concordance benchmarking ([`CPAC_CONCORDANCE_PLAN.md`](./CPAC_CONCORDANCE_PLAN.md)).
2. **Not a Clinical Study:** $N = 6$ provides **zero** statistical power for ADHD diagnostic sensitivity, specificity, or biomarker discovery.
3. **No Clinical Outputs or Diagnoses:** Pilot tracking tables, cohort manifests, and dashboard views must **never** record clinical diagnoses (`ADHD` / `Control` clinical labels), hospital MRNs, or personal identifiers.
4. **Zero Cohort Generalization:** Authorization for any single pilot participant slot never authorizes other slots, other sessions, or any unapproved cohort subjects.

---

## 2. Placeholder N=6 Slot Definitions (Synthetic Rehearsal vs Future Real Scope)

All six pilot slots below are initialized as **`PENDING` (`NOT_AUTHORIZED`)**. During Step 9 (**Synthetic Dry Run Only**), these slots map 1-to-1 to synthetic phantom participants (`sub-synthetic001` through `sub-synthetic006`) exercising diverse valid acquisition configurations.

| Pilot Slot | Placeholder ID | Synthetic Dry-Run Alias | Required Modalities | Target Session | Synthetic Rehearsal Profile (Step 9) | Current Real-Data Auth Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `01` | `sub-pilot-01` | `sub-synthetic001` | 3D T1w + 4D rs-fMRI BOLD + BIDS JSON (+ optional fmap) | `ses-1` | Standard interleaved `k` SliceTiming + direct `Hz` B0 fieldmap (`PE=j`) + physiological sinusoid correlation | `PENDING` (`NOT_AUTHORIZED`) |
| `02` | `sub-pilot-02` | `sub-synthetic002` | 3D T1w + 4D rs-fMRI BOLD + BIDS JSON (+ optional fmap) | `ses-1` | Descending `k-` SliceTiming + direct `Hz` B0 fieldmap (`PE=j-`) + elevated controlled motion (`6-DOF`) | `PENDING` (`NOT_AUTHORIZED`) |
| `03` | `sub-pilot-03` | `sub-synthetic003` | 3D T1w + 4D rs-fMRI BOLD + BIDS JSON | `ses-1` | Ascending `k` SliceTiming + missing fieldmap (verifies honest `SKIPPED_NO_FIELDMAP` without `--strict-sdc`) | `PENDING` (`NOT_AUTHORIZED`) |
| `04` | `sub-pilot-04` | `sub-synthetic004` | 3D T1w + 4D rs-fMRI BOLD + BIDS JSON | `ses-1` | Missing `SliceTiming` + valid TR (`2.0 s`, verifies honest `SKIPPED_MISSING_METADATA` without guessing) | `PENDING` (`NOT_AUTHORIZED`) |
| `05` | `sub-pilot-05` | `sub-synthetic005` | 3D T1w + 4D rs-fMRI BOLD + BIDS JSON (+ PEPOLAR fmap) | `ses-1` | PEPOLAR spin-echo pair (`j` / `j-`) exercising honest `SKIPPED_TOOL_UNAVAILABLE` when FSL `topup` is absent | `PENDING` (`NOT_AUTHORIZED`) |
| `06` | `sub-pilot-06` | `sub-synthetic006` | 3D T1w + 4D rs-fMRI BOLD + BIDS JSON (+ optional fmap) | `ses-1` | Custom TR (`2.5 s`, `t_vol=60`) + interleaved `i` SliceEncoding + negative/positive ROI correlation benchmark | `PENDING` (`NOT_AUTHORIZED`) |

> **Intake & DICOM Note:** DICOM inspection and `dcm2niix` conversion readiness are validated in isolated unit/integration tests using synthetic DICOM bytes (`tests/integration/test_dicom_conversion_gate.py`) confirming fail-closed authorization denial (`NOT_AUTHORIZED`) whenever `sub-01` or any pilot participant lacks valid signed authorization.

---

## 3. Mandatory Pre-Execution Authorization & Governance Prerequisites

Before **any** real participant scan may enter preprocessing or DICOM conversion, **all** items below must be satisfied and verified by [`AuthorizationEvidenceValidator`](../../src/validation/authorization_evidence.py) and [`ResearchAuthorizationGate`](../../src/validation/authorization_gate.py):

1. **IEC Ethics Approval (`AUTH-IEC-01`):** Signed approval letter from `[IEC_NAME]` naming `[PI_NAME]`, protocol ID `DOC-GOV-PROT-001`, protocol version, approved scope ($N = 6$ feasibility pilot), approval date, and expiry date. -> **`[PENDING]`**
2. **Dataset Data Use Agreement (`AUTH-DUA-02`):** Signed DUA / institutional source clearance covering the exact dataset or site source (`[PENDING_DATASET_OR_SCANNER_REF]`) with explicit prohibition against re-identification. -> **`[PENDING]`**
3. **Institutional & Hospital Governance Sign-off (`AUTH-INST-04` / `[INSTITUTION_NAME]`):** Counter-signed institutional authorization naming approved workspace boundaries and storage path controls. -> **`[PENDING]`**
4. **Privacy & InfoSec Clearance (`AUTH-DSU-03`):** Signed de-identification SOP compliance check (defacing visual review + DICOM tag purge + DPDP Act 2023 / DPDP Rules 2025 `G.S.R. 846(E)` readiness review). -> **`[PENDING]`**
5. **Participant-Scoped Authorization Record (`N6_PILOT_AUTHORIZATION_MATRIX.md`):**
   - Exact `participant_id` matching BIDS subject ID;
   - Exact `session_scope` (e.g., `["ses-01"]`);
   - Exact `permitted_operations` subset (`bids_validation`, `dicom_conversion`, `preprocessing`, `cc200_extraction`, `quality_control`);
   - Matching `protocol_version` and study tier (retrospective DUA cannot unlock prospective scans);
   - Non-expired `effective_date <= current_date <= expiry_date`;
   - Non-revoked `authorization_status == "APPROVED"` and valid `hmac.compare_digest` HMAC-SHA256 signature computed with institutional offline key (`NEUROADHD_HMAC_KEY`). -> **`[PENDING — 0/6 AUTHORIZED]`**

If **any** check fails, execution **must abort immediately (`AuthorizationDeniedError`)** before opening, listing, copying, or converting any real NIfTI/DICOM file.

---

## 4. Per-Participant Execution Workflow (Once Authorized — Currently Locked)

> **Current Guard:** Until Step 8 returns `APPROVED_SCOPED` and valid HMAC signatures exist, every real-data invocation returns `NOT_AUTHORIZED` and leaves real MRI files untouched.

### Step 4.1 — Intake Staging & Read-Only Header Validation (`src/intake/service.py`)
1. Stage participant session (`BIDS ZIP` or `DICOM study`) inside isolated local workspace (`NeuroADHD_Converted/staged_sessions/<session_id>`).
2. Enforce path-traversal, symlink, compression-bomb, and size guards.
3. For DICOM studies: inspect series metadata (`dcm2niix -i y`), require unambiguous T1w + rest BOLD selection, verify `ResearchAuthorizationGate` permits `dicom_conversion` for that `participant_id`, convert only if authorized, and record DICOM checksums + anonymized sidecar audit.
4. Validate 4D BOLD (`t_vol >= 50`, `RepetitionTime > 0`, `TaskName == "rest"`), 3D T1w, matching BIDS subject ID, and optional fieldmap association (`B0FieldIdentifier` / `IntendedFor`).

### Step 4.2 — Strict Execution Policy for Real Scans
Real-participant runs (once authorized in future) **must** enable strict metadata and privacy flags:
- `--strict-slice-timing` (fail closed if `SliceTiming` missing; never invent timings);
- `--strict-sdc` (fail closed if validated B0 fieldmap or opposite-PE spin-echo pair with `topup` is unavailable);
- `--strict-defacing` and `--require-export-ready` (fail closed unless external defacer succeeds and metadata sanitization + manual orthoslice QC are completed).

*(Note: On the current validation host as audited on `2026-11-12`, `/opt/homebrew/bin/dcm2niix` is present, whereas `pydeface`, FSL `flirt`/`topup`/`prelude`/`fugue`, and AFNI `3dTshift` are absent; strict real-data execution requires provisioning verified external binaries first.)*

### Step 4.3 — Sequential 23-Stage Pipeline Execution (`scripts/run_poc_v2.py`)
1. Execute sequentially (1 subject at a time, memory guard `max_memory_gb <= 2.50 GB`) preserving original NIfTI inputs immutable (`chmod`/read-only verification + pre/post SHA-256 comparison).
2. Stages 1–7: BIDS validation, raw BOLD 6-DOF motion estimation & Power FD, DEC-STC-01 slice timing, DEC-SDC-02 B0 estimation, native-ref resampling, functional ref/mask, DEC-DEF-03 defacing + derivative sidecar purge.
3. Stages 8–15: Otsu brain extraction, log-poly bias correction, GMM tissue segmentation, eroded CSF/WM masks, EPI->T1w MI coregistration, T1w->MNI SyN normalization, single-step composed pullback resampling, mask warping.
4. Stages 16–23: 5 aCompCor components, Friston-24 expansion, SVD collinearity check, simultaneous GLM regression + `[0.01, 0.10] Hz` bandpass, Craddock CC200 `200`-ROI timeseries, `19,900` Fisher-z connectivity features, 4 multi-planar QC PNGs, schema-validated `provenance.json` + SHA-256 `output_manifest.json`.

### Step 4.4 — Post-Run QC, Manual Review & Incident Handling
1. Inspect QC metrics against [`N6_PILOT_QC_CRITERIA.md`](./N6_PILOT_QC_CRITERIA.md).
2. Complete mandatory human orthoslice visual review for defacing (`manual_visual_review_required = true`) and registration overlays before any derivative leaves the secure workspace.
3. Update [`N6_PILOT_TRACKER.csv`](./N6_PILOT_TRACKER.csv) for that slot only.
4. **Incident / Failure Isolation & Rollback:**
   - Failed runs quarantine into `failed_run_<timestamp>` with `pipeline_failure_report.json`; healthy subjects in the cohort are unaffected (`continue_on_error=True`).
   - Cohort checkpoint (`cohort_manifest.json`) records partial failure without marking remaining slots complete and supports verified `--resume` after tampering/corruption detection.
   - Any privacy exposure, suspected brain erosion during defacing, or unapproved access triggers immediate pause (`REVOKED` register status) and escalation under [`docs/governance/INCIDENT_RESPONSE_AND_ROLLBACK_PLAN.md`](../governance/INCIDENT_RESPONSE_AND_ROLLBACK_PLAN.md).

