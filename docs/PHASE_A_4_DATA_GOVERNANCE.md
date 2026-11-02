# PHASE A.4: DATA GOVERNANCE & RESEARCH SAFETY FRAMEWORK

**Document Version:** 1.0.0  
**Phase:** Phase A.4 — Pre-Pilot Readiness Gate  
**Status:** DRAFT — PENDING ETHICS BOARD & SUPERVISOR REVIEW  
**Policy Level:** MANDATORY INSTITUTIONAL RESEARCH POLICY  
**Execution Environment:** Strictly Synthetic-Only

---

## 1. Regulatory Context & Ethical Foundation

The **NeuroADHD RawMRI Preprocessing Pipeline** is an exploratory computational neuroscience tool designed to extract resting-state functional connectivity features for academic research. It is **not** an FDA-cleared, CE-marked, or clinically certified diagnostic medical device.

Processing human brain MRI scans requires adherence to:
1. **The Declaration of Helsinki** (Ethical Principles for Medical Research Involving Human Subjects).
2. **HIPAA Privacy Rule** (45 CFR Part 160 and Part 164, Subparts A and E) & **GDPR** (Regulation EU 2016/679).
3. **Institutional Review Board (IRB) / Research Ethics Board (REB)** protocols governing neuroimaging research.

> [!CAUTION]
> Public availability of a scientific dataset (e.g., ADHD-200, OpenNeuro, ABIDE, UK Biobank) does **NOT** grant unrestricted or automatic permission for every operational or commercial use. Every research pilot requires explicit protocol alignment, institutional authorization, and compliance with data use agreements (DUAs).

---

## 2. Dataset Sourcing & Usage Permissions

### 2.1 Authorized Data Sources
1. **Synthetic Phantoms (Current Operational Baseline)**:
   - Generated entirely in silico via `generate_synthetic_phantom_session()`.
   - Free from human biological material; unencumbered by human subject regulations.
   - Protected by canonical HMAC integrity tokens.
2. **ADHD-200 Consortium Data (Prospective Research Pilot)**:
   - Governed by the ADHD-200 Consortium Data Use Agreement.
   - Restricted to non-commercial academic research into attention disorders.
   - Multi-site data (e.g., Peking University, NYU, KKI, OHSU) retains site-specific institutional oversight.
   - Re-identification of participants, their families, or their healthcare providers is strictly prohibited.
3. **Prospective Clinical Cohorts**:
   - Requires an active, formal IRB/REB approved protocol with written informed consent from participants or legal guardians.

### 2.2 Data Authorization Workflow
Prior to introducing any prospective real MRI dataset:
1. **Data Use Agreement (DUA) Execution**: Signed by institutional authorized official.
2. **IRB Protocol Approval**: Formal determination (exempt, expedited, or full review).
3. **Supervisor Gate Authorization**: Explicit cryptographic signing of the dataset intake manifest.

---

## 3. Participant De-Identification & Anonymization

Under HIPAA §164.514(b) and neuroimaging privacy standards, the pipeline enforces strict de-identification rules:

### 3.1 Prohibited Direct Identifiers (18 HIPAA Safe Harbor Elements)
The following attributes must **never** enter the pipeline, appear in file paths, or be embedded in NIfTI/JSON headers:
1. Participant names, initials, or aliases.
2. Geographic subdivisions smaller than state/country (no hospital clinic, street address, or zip code).
3. Exact calendar dates (birth date, scan date, admission date, discharge date). Age may only be reported in integer years.
4. Telephone, fax, email addresses, or social media handles.
5. Medical Record Numbers (MRNs), health plan beneficiary numbers, or national identification numbers.
6. Certificate/license numbers or device serial numbers.
7. Biometric identifiers (facial photographs, full 3D facial surface meshes).

### 3.2 DICOM-to-BIDS De-Identification Requirements
- Conversion from raw DICOM to BIDS NIfTI must occur in an isolated clinical landing zone outside the analysis environment.
- All DICOM headers containing patient tags (e.g., `(0010,0010)` PatientName, `(0010,0020)` PatientID, `(0010,0030)` PatientBirthDate, `(0008,0080)` InstitutionName) must be permanently stripped during conversion (e.g., using `dcm2niix -ba y`).
- Anatomical T1w images must undergo automated facial defacing (e.g., `pydeface`, `mri_deface`, or `bidsonym`) to prevent 3D facial surface reconstruction attacks, while ensuring cortical boundaries remain intact.

---

## 4. Participant Identifier Management & Pseudonymization

1. **Synthetic Identifiers**:
   - In synthetic development mode, subjects must use the reserved prefix `sub-syntheticXXX`.
2. **Research Subject Identifiers (RSID)**:
   - Real research participants must be assigned an opaque, non-reversible pseudonymous Research Subject ID (e.g., `sub-res001`).
   - The master key-linking table (re-identification code linking RSID to clinical identity) must reside exclusively on an air-gapped, encrypted hospital server managed by the Principal Investigator.
   - The key-linking table must **never** be copied into the computational environment, code repositories, or container images.

---

## 5. Prevention of PHI Exposure in Logs, Reports, & Filenames

To prevent inadvertent leakage of Protected Health Information (PHI):

### 5.1 Pipeline Logging Controls
1. **Console & File Logging**:
   - Only sanitized subject IDs (`sub-<id>`), stage names, numerical metrics (e.g., FD, SNR, execution time), and error codes are emitted.
   - System error handlers must truncate traceback variables to prevent dumping in-memory strings containing patient metadata.
2. **Output Artifact Filenames**:
   - Standardized, anonymous filenames only:
     - `roi_CC200.csv`
     - `features_19900_fisher_z.csv`
     - `motion_parameters.tsv`
     - `provenance.json`
     - `qc_registration_epi_to_t1w.png`
   - No patient demographics, clinical notes, or hospital codes in filenames or folder paths.
3. **Provenance Metadata**:
   - `provenance.json` records software versions, hardware RSS, and validation flags.
   - Hardware hostname, MAC addresses, and specific local user home directory paths must be scrubbed or standardized to container paths (`/opt/neuro_adhd`).

---

## 6. Secure Storage, Access Restrictions, & Encryption

1. **Storage Encryption**:
   - All resting-state fMRI raw data, preprocessed outputs, and scratch files must reside on volumes encrypted at rest with AES-256 (e.g., FileVault on macOS, LUKS on Linux, or BitLocker).
2. **Access Control**:
   - Access restricted to authorized research personnel using role-based access control (RBAC).
   - Write permissions to production atlas and model directories are locked (`chmod 444` / Docker `:ro` mount).
3. **Air-Gapping & Network Isolation**:
   - Pipeline container execution must operate with network access disabled (`--network none`) to prevent exfiltration.
4. **Git Repository Hygiene**:
   - `.gitignore` strictly prohibits committing raw NIfTI files (`*.nii`, `*.nii.gz`), DICOMs (`*.dcm`), tabular participant data (`*.csv`, `*.tsv`), and serialized model binaries (`*.joblib`).
   - Automated pre-commit hooks and repository scanning tools (e.g., `git-secrets`, `trufflehog`) inspect for credentials and inadvertent data commits.

---

## 7. Data Retention & Secure Disposal

1. **Retention Period**:
   - Raw pilot scan data and intermediate artifacts will be retained only for the duration specified by the IRB protocol (typically 5 to 7 years post-publication).
2. **Scratch & Staging Sanitization**:
   - Intermediate scratch files (e.g., uncompressed NIfTI slices, temporary registration warps) must be purged immediately upon pipeline completion or stage failure.
   - Failed staging directories must be wiped using cryptographic shredding (`srm` or overwriting).
3. **Decommissioning**:
   - Upon project completion or protocol revocation, all raw and intermediate MRI volumes must be permanently erased from non-archival storage.

---

## 8. Incident Response & Unauthorized Access Protocol

In the event of an accidental PHI disclosure or unauthorized real data ingestion:
1. **Immediate Containment**:
   - Terminate running pipeline processes immediately (`kill -9`).
   - Isolate the affected host or container instance from the local network.
2. **Incident Notification**:
   - Report the incident within 24 hours to the Principal Investigator, Institutional Data Protection Officer (DPO), and IRB/REB.
3. **Forensic Audit & Remediation**:
   - Inspect git commit history, terminal logs, and disk volumes to identify all leaked data.
   - Purge affected artifacts from version control using repository cleaning tools (e.g., `git filter-repo`).
   - Document root cause analysis and implement preventative code-level safety guards before resuming operations.

