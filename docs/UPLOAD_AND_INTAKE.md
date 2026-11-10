# Local MRI upload, DICOM-to-BIDS intake, and validation

## 1. Purpose

The dashboard supports an automated local upload, DICOM inspection, DICOM-to-BIDS conversion, and validation workflow.

Inputs supported:
1. Complete BIDS-style MRI session ZIP archives.
2. Raw DICOM studies (ZIP or folders of `.dcm` / `.ima` files) converted automatically to BIDS via `dcm2niix`.
3. Individual BIDS NIfTI (`_bold.nii.gz`, `_T1w.nii.gz`) and JSON sidecars.
4. Separate radiology reports (PDF, JPEG, PNG) processed via local OCR tools.

> [!IMPORTANT]
> **Research Notice:** This feature does **not** enable real-participant preprocessing or CC200 feature extraction. The research authorization register remains authoritative; the real-scan pipeline and model inference remain strictly disabled until all institutional, ethics, and technical validation gates are approved.

---

## 2. External MRI Storage Configuration

All real participant data, uploaded DICOM studies, converted BIDS datasets, validated session ZIPs, and conversion audit logs are strictly stored **outside the project code repository**:

```bash
export NEUROADHD_DATA_ROOT=~/NeuroADHD_Data
```

If `NEUROADHD_DATA_ROOT` is unset, the pipeline defaults to `~/NeuroADHD_Data`.

### Directory Layout

```text
~                             <-- User home directory (outside the project repository)
└── NeuroADHD_Data/           <-- External storage root (default when NEUROADHD_DATA_ROOT is unset)
    ├── dicom_input/          <-- Uploaded raw DICOM studies & archives
    ├── staged_sessions/      <-- Intake staging sessions (replaces outputs/intake_sessions)
    ├── neuroadhd_bids/       <-- Converted BIDS datasets (sub-<id>/ses-<id>/...)
    ├── validated_sessions/   <-- Packaged, downloadable BIDS session ZIPs
    └── conversion_logs/      <-- Anonymized conversion audit logs (no patient identifiers)
```

Directories are created with restrictive permissions (`0o700`) and files with (`0o600`). Path traversal and symlinks are strictly rejected.

---

## 3. Automated DICOM-to-BIDS Ingestion Workflow

### Step A: Upload DICOM Study
Upload a ZIP containing scanner DICOM files or individual DICOM files (`.dcm`, `.ima`) into a scan intake session. Files are verified for cryptographic integrity.

### Step B: Inspect Series (Inspect-Only — Allowed)
The pipeline runs `dcm2niix -n -1` / `-n -2` header listing and temporary metadata extraction:
- Detects all series without converting or writing NIfTI image files (inspect-only header listing remains allowed without conversion authorization).
- Extracts series numbers, descriptions, protocols, image counts, dimensions, modality, and repetition time ($TR$).
- **Strict Anonymization:** Patient name, Patient ID, date of birth, age, sex, institution address, and accession numbers are redacted and never returned or logged.
- Identifies candidate scan types (`T1w`, `BOLD`, `LOCALIZER`, `DWI`, `OTHER`).
- Flags ambiguous series (e.g. BOLD without explicit "rest" designation, or multiple BOLD runs).

### Step C: Select Series & Convert (Gated by Research Authorization)
- **Fail-Closed Authorization Gate:** Before invoking `dcm2niix` or writing any files, `convert_dicom_to_bids` queries `ResearchAuthorizationGate.verify_dataset_authorization(...)` against `configs/research_authorization_register.json` using the exact `dataset_id` (no normalization or whitespace trimming) and `requested_operation="dicom_conversion"`. Conversion is blocked with `IntakeError` unless an unexpired, HMAC-SHA256 signed `APPROVED` record exists for that exact dataset permitting `"dicom_conversion"` and a valid HMAC key is configured (`sub-01` is currently `PENDING` and blocked).
- User selects one Anatomical T1-weighted series (T1w).
- User selects one or more Resting-state BOLD functional series (supports multi-run acquisition: `run-01`, `run-02`, etc.).
- User specifies de-identified BIDS identifiers (e.g. `sub-001`, optional `ses-01`) matching the exact authorized `dataset_id`.
- User explicitly confirms any series with ambiguous classification.
- Pipeline executes `dcm2niix` with gzip compression (`-z y`), BIDS sidecar generation (`-b y`), and anonymization (`-ba y`).
- Generates valid BIDS layout:
  ```text
  neuroadhd_bids/
  ├── dataset_description.json
  └── sub-001/
      └── ses-01/
          ├── anat/
          │   └── sub-001_ses-01_T1w.nii.gz
          └── func/
              ├── sub-001_ses-01_task-rest_run-01_bold.nii.gz
              └── sub-001_ses-01_task-rest_run-01_bold.json
  ```
- Generates an anonymized conversion audit log in `conversion_logs/`.

---

## 4. BIDS Validation Rules

After conversion (or upon direct upload), the session is validated:
1. **BOLD NIfTI:** Valid 4D image with at least 50 time points (volumes).
2. **T1w NIfTI:** Valid 3D image.
3. **BOLD Sidecar:** Valid JSON containing `RepetitionTime` between 0.5 and 5.0 seconds. Missing parameters are never fabricated.
4. **Subject/Session Matching:** Identifiers match across anatomical and functional scans.
5. **BIDS Entities:** Filenames adhere strictly to BIDS specification (`sub-<id>_task-rest_bold.nii.gz`, `_T1w.nii.gz`).

---

## 5. Downloadable Validated BIDS Session ZIP

Upon successful conversion and validation, the dashboard provides a **Download validated BIDS session ZIP** button.
- The archive is saved under `~/NeuroADHD_Data/validated_sessions/sub-<id>_bids.zip` (or `$NEUROADHD_DATA_ROOT/validated_sessions/sub-<id>_bids.zip`).
- Contains `dataset_description.json`, `anat/`, and `func/` NIfTI and sidecar files.
- Original DICOM files and unredacted scanner metadata are strictly excluded.

---

## 6. How to Start the Dashboard

~~~bash
cd "<your-project-folder>"
source .venv/bin/activate
export NEUROADHD_DATA_ROOT=~/NeuroADHD_Data
python scripts/serve_dashboard.py
~~~

Open **http://127.0.0.1:8765** in your browser.

---

## 7. Current Boundaries and Remaining Limitations

* **DICOM Conversion Authorization Gate:** `convert_dicom_to_bids` fails closed unless `configs/research_authorization_register.json` contains a valid, unexpired, HMAC-SHA256 signed `APPROVED` record matching the exact `dataset_id` with `"dicom_conversion"` in `permitted_operations` (and `NEUROADHD_HMAC_KEY` is present). Inspect-only (`inspect_dicom`, `dcm2niix -n -1` / `-n -2` header listing without writing NIfTI images) remains allowed.
* **DICOM-to-BIDS is ingestion only:** Even when a dataset is authorized for `dicom_conversion`, conversion does not authorize real-participant preprocessing or model inference.
* **Research Gate Enforcement:** `processing_enabled` and `real_data_authorized` remain `false` across all intake and converted sessions until Phase 2 research authorization is complete.
* **Synthetic Cohort Runs:** Pipeline runs launched from the dashboard continue to execute against deterministic synthetic phantoms only.
