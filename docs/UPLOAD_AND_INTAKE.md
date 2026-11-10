# Local MRI upload and intake

## Purpose

The dashboard now supports a local upload-and-validation workflow. Its primary input is one complete BIDS-style MRI session ZIP. Individual NIfTI files and DICOM files can also be staged. PDF/JPEG/PNG reports use a separate text-extraction path and are never treated as fMRI inputs.

This feature does **not** enable real-participant preprocessing. The research authorization register remains authoritative; the real-scan pipeline and model inference remain disabled until all required approvals and technical validation are complete.

## Start the dashboard

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python scripts/serve_dashboard.py
~~~

Open http://127.0.0.1:8765. The server must remain bound to loopback. Do not expose it to a LAN, public host, reverse proxy, or cloud deployment.

## Preferred upload: complete session ZIP

Upload one ZIP containing one participant/session. A typical BIDS-style layout is:

~~~text
session.zip
└── sub-001/
    ├── anat/
    │   └── sub-001_T1w.nii.gz
    └── func/
        ├── sub-001_task-rest_bold.nii.gz
        └── sub-001_task-rest_bold.json
~~~

The BOLD image must be 4D with at least 50 volumes. Its JSON sidecar must contain RepetitionTime between 0.5 and 5.0 seconds. The T1w image must be 3D. Use one participant/session per upload. The BOLD and T1w names must retain their BIDS suffixes (_bold and _T1w) so that the intake validator can identify them.

## Individual scan uploads

The individual-file picker accepts BIDS-named .nii, .nii.gz, associated .json sidecars, DICOM .dcm / .ima files, and common BIDS supporting files. Upload the BOLD image, its JSON sidecar, and the T1w image from the same session.

DICOM files can be staged, but this release does not identify DICOM series or execute DICOM-to-NIfTI conversion. Convert and review the series with a trusted DICOM conversion workflow (for example, dcm2niix) and upload the resulting BIDS-compatible NIfTI files only after the research/data-handling requirements for that dataset are satisfied.

## Report uploads

PDF, JPEG and PNG reports are accepted separately. Text extraction uses available local tools:
- pdftotext for text-layer PDFs;
- optional pypdf if installed;
- pdftoppm plus tesseract for OCR of the first five pages of scanned PDFs;
- tesseract for JPEG/PNG images.

Without those tools, the report is staged and the dashboard explains which extraction engine is unavailable. Install Poppler and Tesseract locally if OCR is required (for example, brew install poppler tesseract on macOS). Extracted text files are downloadable from the intake result. Report text is not passed to the BOLD preprocessing or ML model.

## What intake validation checks

- Allowed upload type and file size.
- ZIP member count, expansion size, compression ratio, path traversal and symlinks.
- NIfTI header readability and dimensionality.
- BOLD time-point count and RepetitionTime sidecar value.
- Presence of one matching BOLD and one T1w image.

A successful scan intake status is VALIDATED_WAITING_FOR_RESEARCH_AUTHORIZATION. It means the required files passed the checks above; it does not mean the scan has passed image-registration QC or that real-data processing is authorized.

## Atlas modes and execution performance

The dashboard provides two atlas modes for synthetic cohort runs:
- **Synthetic phantom atlas (`synthetic_phantom`)**: Fast demonstration mode taking ~10–16 seconds per subject.
- **Canonical CC200 atlas (`canonical`)**: Uses the canonical Craddock CC200 reference atlas (`cc200_roi_atlas.nii.gz`). Mutual-information non-linear and affine registration on CPU is computationally intensive and can take many minutes per subject (e.g. ~5–6 minutes per subject on CPU). Note that selecting the canonical atlas changes only the reference atlas; the input data is still a synthetic phantom. Real-participant preprocessing remains strictly locked.

## Storage and limits

- Uploads are saved under outputs/intake_sessions/, which is excluded from version control.
- The app writes staged files with owner-only permissions where supported.
- The dashboard streams upload bodies to disk and does not buffer an entire MRI file in memory.
- One scan file is limited to 2 GB; report files to 50 MB; total session uploads to 8 GB.
- ZIP archives are limited to 5,000 entries and 8 GB uncompressed.
- The local dashboard uses loopback-only binding. It has no user authentication; do not expose it to other machines or users.
- Uploaded MRI and report data may contain sensitive health information. Use only data you are authorized to handle and remove local intake folders when no longer needed.

## Not yet implemented / intentionally locked

- Real-scan preprocessing and generation of roi_CC200.csv.
- Automatic DICOM series detection and conversion.
- Clinical or diagnostic interpretation of radiology reports.
- Real-participant model inference.
- Formal QC pass/fail thresholds.

These are separate milestones and must not be inferred from a successful upload or file validation.
