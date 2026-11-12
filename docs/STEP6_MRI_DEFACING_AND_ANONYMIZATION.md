# Step 6: Anatomical MRI Defacing & Metadata De-Identification (`DEC-DEF-03`)

## 1. Overview & Safety Boundaries

Step 6 implements reproducible anatomical MRI defacing, metadata de-identification, quantitative defacing QC, and export privacy gating in [`src/preprocessing/defacing.py`](../src/preprocessing/defacing.py), integrated into the **23-stage** [`scripts/run_poc_v2.py`](../scripts/run_poc_v2.py) preprocessing pipeline and live dashboard telemetry ([`scripts/serve_dashboard.py`](../scripts/serve_dashboard.py), [`src/dashboard/service.py`](../src/dashboard/service.py), [`frontend/index.html`](../frontend/index.html), [`frontend/app.js`](../frontend/app.js)).

**Strict Safety Invariants:**
- `execution_policy = STRICTLY_SYNTHETIC_ONLY`
- `real_data_authorized = false`
- `overall_authorization_state = NOT_AUTHORIZED`
- `sub-01 authorization_status = PENDING` (`cryptographic_signature = null`)
- `configs/research_authorization_register.json`, `src/validation/`, `src/integration/`, and `check_safety_guards()` in `scripts/run_poc_v2.py` remain untouched and fail-closed.
- Clinical directory patterns (`NeuroADHD_Converted`, `sub-01`, `sub-2344857`, `down/outputs/adhd200`, `peking`, `kki`, `nyu`, `abide`, `adhd-200`, `adhd200`), `..` traversals, and symlinks are rejected fail-closed.
- Original raw MRI volumes (`*_T1w.nii.gz`) and BIDS sidecars/TSVs remain immutable (verified via pre/post SHA-256 checksums).

---

## 2. Pipeline Integration — 23-Stage Preprocessing Order

Stage 7 (`Anatomical MRI Defacing, Metadata De-Identification & Privacy Audit`) executes **before** anatomical brain extraction and registration (Stages 8–13) while leaving functional raw-BOLD motion estimation (Stage 2), slice-timing correction (Stage 3), and susceptibility distortion correction (Stage 4) completely independent:

| Stage | Module / Operation | Role in Step 6 |
| :--- | :--- | :--- |
| **Stage 1/23** | **BIDS Input & Sidecar Metadata Validation** | Validates functional BIDS JSON sidecar, synthetic integrity manifest, and CC200 atlas. |
| **Stage 2/23** | **3D Rigid Motion Parameter Estimation** | Estimates 6-DOF rigid transforms and Power FD strictly on **raw uncorrected BOLD** prior to STC/SDC. |
| **Stage 3/23** | **Metadata-Driven Slice-Timing Correction (`DEC-STC-01`)** | Temporal slice alignment (`+TR/2` `StartTime` shift, 4-TR odd reflection) on unresampled native slice geometry. |
| **Stage 4/23** | **B0 Susceptibility Distortion Correction (`DEC-SDC-02`)** | Discovers BIDS `fmap/`, validates PE direction & `TotalReadoutTime`, checks external binaries (`topup`), and estimates pullback field `displacement_vox`. |
| **Stage 5/23** | **Motion Transform (+ Optional SDC) Application** | Aligns STC BOLD into native reference space for Stage 6 functional reference generation. |
| **Stage 6/23** | **Functional Reference Generation** | Computes 3D mean EPI and EPI brain mask. |
| **Stage 7/23** | **Anatomical MRI Defacing, Metadata De-Identification & Privacy Audit (`DEC-DEF-03`)** | Probes `pydeface` availability (`SKIPPED_TOOL_UNAVAILABLE` if absent; fail-closed under `--strict-defacing`), defaces 3D `T1w` into `derivatives/defaced/<sub-id>/anat/<sub-id>_desc-defaced_T1w.nii.gz`, runs quantitative QC, sanitizes permitted BIDS sidecar/TSV/header copies without stripping scientific fields or mutating originals, evaluates the privacy export gate, and routes the accepted defaced `T1w` (when `APPLIED`) into Stage 8–13 anatomical processing. |
| **Stages 8–13/23** | **Anatomical Brain Extraction, Bias Correction, Segmentation & Registration** | `median_otsu` brain extraction, order-2 Legendre bias correction, 3-class GMM segmentation, `0.96` threshold + 1-voxel erosion, EPI-to-T1w rigid coregistration, and T1w-to-MNI SyN normalization. |
| **Stages 14–23/23** | **Single-Step Pullback, Nuisance Regression, CC200, Fisher-$z$, QC & Provenance** | Single-step composed MNI resampling, eroded mask resampling, 5 aCompCor + Friston-24 + detrending (31 columns, GSR OFF), bandpass `[0.01–0.10 Hz]`, 200-ROI CC200 extraction, 19,900 Fisher-$z$ features, 4 orthogonal QC figures, schema-validated `provenance.json` (`pipeline_stages_total = 23`), and additive `defacing_status` in `output_manifest.json`. |

---

## 3. Selected Defacing Tool & Software Discovery Policy

`detect_defacing_software()` audits local binaries without silently substituting skull stripping or inventing homemade facial-removal filters:

- **Preferred Tool:** `pydeface` (BIDS-standard affine registration + face mask removal; invokes `pydeface <input_t1w> --outfile <derivative_t1w> --force`). Requires `fsl`/`flirt` at runtime.
- **Additional Probed Tools (Inventory Only):** `mri_deface`, `@afni_refacer_run`, `mri_reface`, `quickshear`, `flirt` (`FSLDIR`), and installed Python packages (`nibabel`, `numpy`).
- **Host Capability Audit (Current Environment):**
  - `pydeface`: `unavailable`
  - `mri_deface`: `unavailable`
  - `@afni_refacer_run`: `unavailable`
  - `mri_reface`: `unavailable`
  - `quickshear`: `unavailable`
  - `flirt` / `FSLDIR`: `unavailable`
  - Python dependencies present: `nibabel 5.4.2`, `numpy 2.4.6`.
- **Missing-Tool Behavior:** Emits explicit status `SKIPPED_TOOL_UNAVAILABLE`, records `missing_dependencies: ["pydeface", "flirt (FSL)"]`, creates **no** fabricated defaced image or binary face mask, preserves the original `T1w` byte-for-byte, and continues synthetic pipeline execution unless `--strict-defacing` or `--require-export-ready` is set (which fails closed).

---

## 4. Quantitative Defacing QC & Status Matrix

Every executed or attempted defacing operation is classified into explicit, non-misleading states:

| Status | Meaning | Downstream Accepted? |
| :--- | :--- | :--- |
| `APPLIED` | Defaced `T1w` created into BIDS derivatives, input SHA-256 unchanged, geometry & brain retention QC passed. | Yes (`manual_visual_review_required = true`) |
| `QC_PENDING_MANUAL_REVIEW` | Technical QC passed, awaiting mandatory human orthoslice visual review sign-off when `require_manual_qc_signoff=True`. | Conditional on workflow |
| `SKIPPED_TOOL_UNAVAILABLE` | Supported tool (`pydeface`) is missing; original `T1w` passed untouched; never substitutes skull stripping. | Fallback to raw synthetic T1w in non-strict mode only |
| `SKIPPED_UNSUPPORTED_MODALITY` | Non-anatomical volume (4D BOLD, DWI, fieldmap, etc.) or disabled `T2w`/`FLAIR` rejected cleanly. | No |
| `FAILED_INVALID_INPUT` | Missing file, corrupted NIfTI, NaN/Inf voxels, all-zeros volume, or singular affine. | No |
| `FAILED_OUTPUT_VALIDATION` | Missing output, non-finite values, shape/affine/qform/sform mismatch, runner error, or broken input immutability. | No (rejected output removed) |
| `REJECTED_BRAIN_TISSUE_LOSS` | Brain tissue preservation failed (`retention < 0.99` or mask Dice `< 0.95`). | No (rejected output removed) |

**Authentic Face-Mask Policy:** Binary face masks (`derivatives/defaced/<sub-id>/anat/<sub-id>_desc-facemask_mask.nii.gz`) are preserved **only** when actually emitted by the defacing tool; otherwise `defacing_mask_available = false` and `defacing_mask_relative_path = null`.

---

## 5. Metadata Privacy Audit & Sanitization Policy

`audit_metadata_identifiers()` and `sanitize_metadata()` inspect BIDS JSON sidecars, tabular TSVs (`participants.tsv`), NIfTI header text fields (`descrip`, `aux_file`, `db_name`, `intent_name`), `provenance.json`, and `output_manifest.json`:

1. **Direct Identifier Denylist Removal:** Strips fields such as `PatientName`, `PatientID`, `PatientBirthDate`, `BirthDate`, `DOB`, `PatientAddress`, `PatientTelephoneNumbers`, `MedicalRecordLocator`, `MRN`, `AccessionNumber`, `StudyID`, `InstitutionAddress`, `InstitutionName`, `InstitutionalDepartmentName`, `DeviceSerialNumber`, `StationName`, `OperatorsName`, `PerformingPhysicianName`, `ReferringPhysicianName`, `PersonName`, `FullName`, `Email`, `Phone`, and `HomeAddress`.
2. **Scientific Parameter Preservation:** Unconditionally preserves `RepetitionTime`, `SliceTiming`, `SliceEncodingDirection`, `PhaseEncodingDirection`, `TotalReadoutTime`, `EchoTime`, `EffectiveEchoSpacing`, `ReconMatrixPE`, `MagneticFieldStrength`, `FlipAngle`, `InversionTime`, `TaskName`, `Units`, `IntendedFor`, `B0FieldIdentifier`, `B0FieldSource`, `StartTime`, `ManufacturersModelName`, `Manufacturer`, and `Modality`, plus pseudonymous `participant_id` labels.
3. **Zero PHI Leakage:** Audit reports record **only detected field names and counts** (`raw_identifier_values_logged = false`), never raw values.
4. **Derivative-Only Sanitization:** Sanitized copies are written strictly under `derivatives/defaced/<sub-id>/metadata_sanitized/`; original files are never modified.
5. **Upstream DICOM Limitation:** Records `dicom_header_phi_removed = false` and states explicitly that DICOM header PHI de-identification is a separate upstream responsibility.

---

## 6. Privacy & Export-Readiness Gate

`evaluate_privacy_export_gate()` computes an overall privacy status and **never** uses misleading terms such as `ANONYMIZED` or `ANONYMOUS`:

- `PRIVACY_READY_FOR_REVIEW` — Defacing `APPLIED`, metadata audit clean, raw SHA-256s immutable, export paths strictly contained, and manual visual inspection flagged as required.
- `SKIPPED_TOOL_UNAVAILABLE` — Defacing skipped because `pydeface` is absent; rejected when `require_export_ready=True`.
- `DEFACING_PENDING` — Defacing not completed (`SKIPPED_UNSUPPORTED_MODALITY`, pending manual review).
- `METADATA_SANITIZATION_PENDING` — Metadata audit/sanitization incomplete.
- `PRIVACY_AUDIT_FAILED` — Identifier fields remain in export, raw files altered, paths escaped containment, or defacing failed/rejected.

---

## 7. Explicit Five-Way Distinction & Limitations

To maintain strict scientific honesty, documentation, UI panels, and provenance distinguish:

1. **Actual PyDeface Execution:** Requires locally installed `pydeface` + FSL `flirt`; **not executed** on this host because both binaries are unavailable (`SKIPPED_TOOL_UNAVAILABLE`).
2. **Mocked / Injected Runner Validation:** Synthetic `64×64×64` unit and E2E tests inject a `defacing_runner` callable purely to validate CLI construction (`--outfile`, `--force`), QC geometry checks, mask handling, brain-loss rejection, privacy gating, and Stage 7 -> Stage 8–13 routing. Mocked execution does **not** constitute real facial de-identification.
3. **Metadata Privacy Audit & Sanitization:** Executed and verified across synthetic BIDS JSON sidecars, `participants.tsv`, NIfTI header text fields, `provenance.json`, and `output_manifest.json`; does **not** sanitize DICOM binary headers.
4. **Manual Visual QC (Mandatory):** Automated checks (finiteness, affine preservation, brain-mask retention $\ge 0.99$, Dice $\ge 0.95$) cannot verify complete facial feature removal or absence of subtle cortical clipping; human orthoslice visual review remains mandatory (`manual_visual_review_required = true`).
5. **Real Participant Anonymization:** **NOT PERFORMED.** `sub-01` and real clinical MRI directories remain untouched and `PENDING` behind the research authorization gate.

