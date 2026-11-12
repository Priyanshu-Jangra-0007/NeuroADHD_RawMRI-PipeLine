# DATA MANAGEMENT AND PRIVACY PLAN (DMPP)

> **`DRAFT — NOT SUBMITTED — NOT APPROVED`**
>
> **Status:** `DRAFT_FOR_INSTITUTIONAL_REVIEW`  
> **Document ID:** `DOC-GOV-DMP-001`  
> **Study Title:** NeuroADHD — ADHD Classification Using Resting-State Functional MRI Connectivity and Machine Learning  
> **Pipeline Version:** NeuroADHD RawMRI PoC V2 (**23 stages**; `STRICTLY_SYNTHETIC_ONLY`)  
> **Execution Policy:** `STRICTLY_SYNTHETIC_ONLY` | `real_data_authorized = false` | `overall_authorization_state = NOT_AUTHORIZED`

> [!IMPORTANT]
> **Regulatory Honesty Note:** Documenting technical controls in this plan does **not** establish legal compliance under the DPDP Act 2023, DPDP Rules 2025, or ICMR Guidelines 2017. Statutory compliance and retention schedules must be independently reviewed and certified by `[DPO_NAME]`, `[INSTITUTION_LEGAL_OFFICE]`, and `[IEC_NAME]`.

---

## 1. Data Collection & Authorized Sources

1. **Authorized Operational Baseline (Current):** Synthetic phantom datasets (`sub-synthetic*`) generated in silico by `scripts/run_poc_v2.py` — zero human subjects.
2. **Future Authorized Acquisition Sources (Currently Locked / `PENDING`):**
   - **Tier 2 (Retrospective Secondary Data):** Publicly distributed ADHD-200 Consortium BIDS/NIfTI scans (`N=153` usable subjects cited in engineering docs: 50 ADHD, 103 Control; `N=6` feasibility pilot proposal) transferred under verified NITRC/INDI DUA (`AUTH-DUA-02`) and IEC authorization (`AUTH-IEC-01`).
   - **Tier 3 (Prospective Pilot, Conditional):** Newly acquired T1w + rs-fMRI BOLD scans (`N=6`) **only** if explicitly approved by `[IEC_NAME]` with informed consent/assent and DPO sign-off.
3. **Data Minimization:** Collect/retain only MRI volumes (`T1w`, `task-rest_bold`, optional `fmap`), essential BIDS acquisition parameters (`RepetitionTime`, `SliceTiming`, `SliceEncodingDirection`, `PhaseEncodingDirection`, `TotalReadoutTime`, `EchoTime`), and minimal group/age/sex covariates required by protocol. Direct clinical notes, full DOBs, addresses, hospital billing IDs, and contact details are excluded (`SAFE-DEID-01`).
4. **Collection & Transfer Method:** Encrypted offline staging or local authenticated institutional transfer (`TLS 1.3` / `SSH ed25519` / encrypted removable media); never public cloud sync, email, or unencrypted HTTP/FTP.

---

## 2. Data Storage, Encryption, Access Control & Backups

| Storage Layer | Requirement & Control | Current Implementation / Institutional Confirmation Needed |
| :--- | :--- | :--- |
| **Approved Storage Location** | Strictly **outside** the Git repository under `NEUROADHD_DATA_ROOT` (default `~/NeuroADHD_Data`: `dicom_input/`, `staged_sessions/`, `neuroadhd_bids/`, `validated_sessions/`, `conversion_logs/`). | Implemented in `src/intake/service.py` with `0o700` dir / `0o600` file modes (`UPLOAD_AND_INTAKE.md`). Institutional dedicated storage path `[APPROVED_INSTITUTIONAL_STORAGE_PATH]` pending sign-off. |
| **Encryption at Rest** | Full-disk / volume **AES-256** encryption (`SAFE-STOR-03`) for all workstation disks, external drives, and backup media holding real data. | `[HOST_ENCRYPTION_VERIFICATION_EVIDENCE]` required from `[IT_SECURITY_OFFICER]` prior to real-data unlock. |
| **Encryption in Transit** | `TLS 1.3` or `SSH (ed25519)` for authorized institutional transfers; dashboard bound to loopback `127.0.0.1:8765` only (no public bind). | Loopback dashboard binding verified (`scripts/serve_dashboard.py`); external network transfer disabled (`--network none` in container mode). |
| **Access Controls & RBAC** | Least-privilege role-based access (`SAFE-ACC-04`), non-root container UID `10001` (`neuro`), read-only input/model mounts (`:ro`), and cryptographically gated `ResearchAuthorizationGate` (`configs/research_authorization_register.json`). | `ResearchAuthorizationGate` active (`NOT_AUTHORIZED`, 23 `PENDING` items). Institutional multi-user OS/RBAC roster `[RBAC_ROSTER_PLACEHOLDER]` pending confirmation. |
| **Backup Strategy** | Encrypted, access-controlled institutional backups separated from public cloud sync (`iCloud`/`Google Drive`/`OneDrive` disabled on `NEUROADHD_DATA_ROOT`). | `[INSTITUTIONAL_BACKUP_POLICY_AND_FREQUENCY]` pending `[DATA_CUSTODIAN]` & `[IT_SECURITY_OFFICER]` confirmation. |
| **Audit Logging** | Immutable, PHI-scrubbed logs recording SHA-256 checksums, gate checks, pipeline stage provenance (`provenance.json`), and intake conversions (`conversion_logs/`) with `raw_identifier_values_logged = false`. | Active in `src/preprocessing/defacing.py` & `src/intake/service.py`. |

---

## 3. Data Processing Safeguards (23-Stage Pipeline Controls)

1. **Pseudonymization & Key Separation:** Real participants must receive opaque coded IDs (`sub-...`). Any clinical linkage key mapping real identity to coded ID **must stay exclusively** on an air-gapped institutional system managed by `[PI_NAME]` and **never** enter the pipeline repo, Docker container, or dashboard (`DATA_MANAGEMENT` linkage isolation rule).
2. **DICOM Header Scrubbing & Gated Conversion:** `inspect_dicom` runs read-only series listing without writing NIfTI files; `convert_dicom` (`dcm2niix -z y -b y -ba y`) is fail-closed blocked unless `ResearchAuthorizationGate.verify_dataset_authorization` verifies an unexpired HMAC-signed `APPROVED` entry for the exact `dataset_id` (`sub-01` currently `PENDING`).
3. **Anatomical MRI Defacing (`DEC-DEF-03`, Stage 7/23):** `src/preprocessing/defacing.py` applies `pydeface` (when available; emits `SKIPPED_TOOL_UNAVAILABLE` otherwise without homemade masking or skull stripping; fail-closed under `--strict-defacing`), verifies brain tissue preservation (`>= 0.99` retention, Dice `>= 0.95`), requires mandatory manual visual orthoslice review, and routes the defaced `T1w` derivative downstream when `APPLIED`.
4. **Metadata Sanitization:** Strips denylisted identifier fields (`35` identifiers) from BIDS JSON sidecars, `participants.tsv`, and NIfTI headers into derivative copies (`derivatives/defaced/<sub-id>/metadata_sanitized/`) while preserving scientific BIDS fields and auditing `provenance.json` + `output_manifest.json` before disk write.
5. **Input Immutability & Staging Isolation:** Pre/post SHA-256 hashes confirm original raw `bold.nii.gz`, `bold.json`, `T1w.nii.gz`, and `TSV` files remain untouched. Temporary processing runs inside `.staging_run_*` and is cleaned up / quarantined cleanly upon failure (`failed_run_*`).
6. **Derivative Tracking & Output Integrity:** Every run validates `provenance.json` against `configs/provenance_schema.json`, writes SHA-256 checksums into `output_manifest.json` (plus additive `defacing_status` and `correction_status`), and enforces tamper detection in `DashboardDataService`.

---

## 4. Data Access Responsibilities by Role

| Role | Permitted Access Scope (Subject to Authorization) | Specific Governance Duties |
| :--- | :--- | :--- |
| **Principal Investigator (`[PI_NAME]`)** | Authorized protocol custodian; holds air-gapped linkage key (if Tier 3). | Accountable for IEC compliance, DUA adherence, team roster maintenance, and withdrawal execution. |
| **Academic Supervisor (`[SUPERVISOR_NAME]`)** | Oversight review of synthetic benchmarks, QC summaries, and governance evidence. | Reviews technical milestones (`DEC-STC-01`, `DEC-SDC-02`, `DEC-DEF-03`, `QC-01..08`) and approves submission readiness. |
| **Institutional Data Custodian (`[DATA_CUSTODIAN]`)** | Infrastructure/storage administration for `[APPROVED_INSTITUTIONAL_STORAGE_PATH]`. | Enforces AES-256 storage, encrypted backups, user provisioning/revocation, and certified destruction. |
| **Research Student / Developer (`[STUDENT_DEVELOPER]`)** | Synthetic development & code testing (`STRICTLY_SYNTHETIC_ONLY`); real coded data **only if** explicitly named on approved IEC & DUA roster. | Maintains pipeline safety guards, tests, and provenance hygiene; never commits NIfTI/PHI/secrets to Git. |
| **Approved Collaborators (`[COLLABORATORS_IF_ANY]`)** | **Zero access** unless named on executed DUA, IEC amendment, and institutional MTA/DSA. | Restricted to explicitly shared de-identified derivatives/summary outputs approved by `[IEC_NAME]`. |

---

## 5. Data Retention & Destruction Placeholders

> [!WARNING]
> **Retention Duration Not Invented:** Do **not** assume a default retention period in final submissions. The exact statutory and institutional duration must be supplied by `[INSTITUTION_NAME]` Legal / `[IEC_NAME]`.

| Lifecycle Item | Required Specification | Institutional Confirmation / Placeholder |
| :--- | :--- | :--- |
| **Active & Post-Publication Retention Duration** | Duration mandated by `[INSTITUTION_NAME]` research-records policy, `[IEC_NAME]`, and DPDP Rules 2025 Second Schedule storage-limitation standards | `[INSTITUTIONAL_RETENTION_YEARS — REQUIRES_AUTHORITY_DECISION]` |
| **Approved Archival Storage Location** | Physical/encrypted institutional repository or server outside GitHub | `[INSTITUTIONAL_ARCHIVE_LOCATION]` |
| **Primary Storage Destruction Method** | Certified secure/cryptographic erasure (e.g., NIST SP 800-88 Rev. 1 or institutional IT decommissioning certificate) | `[CERTIFIED_DESTRUCTION_SOP]` |
| **Backup Deletion Window** | Maximum lag time after expiry/withdrawal for encrypted backup rotation purge | `[BACKUP_EXPIRY_WINDOW_DAYS]` |
| **Institutional Archival Obligations** | Whether anonymized aggregate thesis artifacts or audit logs must be retained permanently after raw scan deletion | `[INSTITUTIONAL_ARCHIVAL_MANDATE]` |

---

## 6. Data Sharing Rules

- **Raw MRI Scans (Un-defaced DICOM / NIfTI):** Strictly **prohibited** from external sharing, public upload, Git commits, or transfer to unauthorized personnel.
- **Defaced Anatomical & Preprocessed BOLD Derivatives:** Restricted strictly to IEC- & DUA-authorized `[INSTITUTION_NAME]` investigators within `NEUROADHD_DATA_ROOT`.
- **Phenotype Metadata (`participants.tsv`):** Strictly coded minimal variables; never shared externally without explicit DUA + IEC approval.
- **Derived Features (`roi_CC200.csv`, `19,900` Fisher-$z$ vectors):** External sharing requires explicit `[IEC_NAME]` + INDI/NITRC DUA confirmation (`REQUIRES_DATA_OWNER_CONFIRMATION`).
- **Collaborations & Publication:** External collaborations require formal DSA/MTA (plus HMSC assessment if international). Scientific papers may publish aggregate statistics and synthetic/code artifacts; publishing individual recognizable brain renderings or unauthorized dataset subsets is strictly prohibited.

---

## 7. Security Incident & Breach Handling SOP (`SAFE-AUD-05` Alignment)

1. **Incident Identification:** Detection of accidental real-data processing without authorization, unredacted PHI in logs/outputs, lost/unencrypted media, unauthorized access, or accidental Git/cloud exposure.
2. **Immediate Containment (Target <= 1 hour):** Halt affected pipeline/dashboard processes immediately; disconnect affected host/mount from network if exfiltration suspected; preserve container/system logs without destructive overwriting that erases forensic evidence.
3. **Access Revocation & Credential Rotation:** Revoke compromised SSH/filesystem/NITRC credentials and rotate `NEUROADHD_HMAC_KEY`.
4. **Evidence Preservation & Breach Assessment (<= 24 hours):** Determine exact files exposed, whether data were coded/defaced vs. raw/identifiable, and whether DPDP Act `Section 8(5)` / `Rule` personal data breach triggers apply.
5. **Internal Reporting Authority:** Escalate immediately to `[PI_NAME]`, `[SUPERVISOR_NAME]`, `[DPO_NAME]`, `[IT_SECURITY_OFFICER]`, and `[IEC_NAME]` Member Secretary.
6. **Statutory Institutional Notification:** Where required under applicable Indian law (including **DPDP Act 2023 / Rules 2025 72-hour breach notification workflow** once enforceable / per institutional legal directive) and `[IEC_NAME]` reporting rules, `[INSTITUTION_ADMINISTRATION]` / `[DPO_NAME]` executes formal notification to the **Data Protection Board of India (DPBI)**, affected participants (if applicable), and dataset providers (INDI/NITRC).
7. **Post-Incident Remediation:** Root-cause analysis, `git filter-repo` cleanup if repository exposure occurred (followed by key revocation), and documented corrective verification before resuming authorized activity.
