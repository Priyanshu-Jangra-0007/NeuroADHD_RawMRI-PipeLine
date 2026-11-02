# NeuroADHD RawMRI PoC V2 — Phase A.5
# Research Ethics & Comprehensive Data Governance Review

**Document ID:** `DOC-NA-A5-EDG-001`  
**Version:** 1.0.0  
**Status:** `READY_FOR_SUPERVISOR_DECISION`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Jurisdiction & Context:** Indian Academic Institution & International Research Collaboration  

---

## 1. Executive Context & Regulatory Scope

The **NeuroADHD RawMRI PoC V2** pipeline is an investigational academic computational neuroimaging platform. This document establishes the ethical, legal, and operational governance framework required to safeguard participant privacy, ensure compliance with statutory data protection mandates, and govern data handling prior to any real magnetic resonance imaging (MRI) data ingestion.

Because this project is conducted under the auspices of an **Indian academic institution** while interacting with international benchmark datasets (e.g., ADHD-200), data governance cannot rely solely on United States frameworks (such as HIPAA). Instead, it must strictly harmonize across:
1. **Indian Council of Medical Research (ICMR) Ethical Guidelines for Biomedical Research Involving Human Participants (2017)**
2. **Digital Personal Data Protection (DPDP) Act 2023 (Government of India)**
3. **International Data Transfer & De-Identification Norms (including HIPAA Safe Harbor & GDPR principles where cross-border data is involved)**

> [!CAUTION]
> **Strict Operational Reality:**
> Currently, **no active Institutional Ethics Committee (IEC) clearance or Data Use Agreement (DUA) is in effect** for real participant data in this pipeline repository. Real participant MRI processing is strictly prohibited and cryptographically blocked.

---

## 2. Fifteen Comprehensive Governance Pillars

### 1. Research Purpose & Non-Diagnostic Scope
- The pipeline is designed exclusively for **academic biomarker discovery, computational reproducibility testing, and methodological benchmarking**.
- It is **not** a clinical diagnostic system or Medical Device (SaMD). Outputs (functional connectivity matrices, class probabilities) are research variables and must never be disclosed to participants, legal guardians, or healthcare providers as diagnostic findings.

### 2. Study Population & Vulnerability Considerations
- ADHD predominantly affects **children and adolescents**. Pediatric research participants constitute a vulnerable population requiring elevated ethical scrutiny under ICMR Guidelines (Section 5).
- Heightened vulnerability mandates:
  - Formal parental/guardian informed consent alongside developmentally appropriate participant assent.
  - Zero-risk secondary data exposure.
  - Strict prohibition against using research data for behavioral profiling, academic tracking, or commercial insurance underwriting.

### 3. Classification of Data Sources
The pipeline recognizes three mutually exclusive data tiers:
1. **Tier 1: Synthetic Phantom Data (Current Default)**
   - Procedurally generated mathematical phantoms containing zero human data.
   - Requires zero ethical clearance or DUA.
2. **Tier 2: Retrospective Open-Science Datasets (e.g., ADHD-200 Consortium)**
   - Pre-existing, de-identified research scans from contributing international sites (NYU, Peking University, OHSU, etc.).
   - Governed by formal Data Use Agreements through NITRC/INDI.
3. **Tier 3: Prospective Real Participant Data (Future Research Pilot)**
   - Scans acquired directly from human subjects under institutional research protocols.
   - Requires full Institutional Ethics Committee (IEC) clearance, verified informed consent/assent, and clinical trial registration if applicable.

### 4. Institutional Ethics Committee (IEC) Clearances
- Before any real participant scan (Tier 2 or Tier 3) can be processed, an application must be formally reviewed and approved by the institution's Institutional Ethics Committee (IEC).
- The IEC submission must include:
  - The comprehensive research protocol.
  - Informed consent and assent forms (in English and relevant regional languages).
  - Software technical specifications and data security architecture.
  - Investigator credentials and supervisor oversight declarations.

### 5. Alignment with ICMR Ethical Guidelines (2017)
The pipeline architecture complies with the **ICMR National Ethical Guidelines for Biomedical and Health Research Involving Human Participants (2017)**:
- **Principle of Essentiality (Section 1.1):** Processing is restricted strictly to features necessary to test the scientific hypothesis ($\binom{200}{2} = 19{,}900$ connectivity edges).
- **Principle of Non-Exploitation (Section 1.3):** Vulnerable pediatric participants are protected against stigmatization; results remain anonymized.
- **Principle of Privacy and Confidentiality (Section 1.4):** Implemented via mandatory defacing, cryptographic tokenization, and air-gapped identity management.
- **Storage and Secondary Use (Section 5.6):** Data will not be shared with unauthorized third parties without specific ethics committee re-clearance.

### 6. Digital Personal Data Protection (DPDP) Act 2023 Compliance
As research conducted within the Indian jurisdiction, the pipeline aligns with the statutory principles of the DPDP Act 2023:
- **Data Fiduciary Obligations:** The academic institution and Principal Investigator act as the Data Fiduciary, responsible for data minimization, purpose limitation, and storage limitation.
- **Special Provisions for Children's Data (Section 9):** Processing of personal data belonging to children requires verifiable parental consent. Tracking, behavioral monitoring, or targeted advertising directed at children is strictly absent from the pipeline design.
- **Security Safeguards (Section 8(5)):** Mandatory technical and organizational measures (encryption, container isolation, access controls) to prevent personal data breaches.

### 7. Data Use Agreement (DUA) Requirements
> [!IMPORTANT]
> **Debunking the "Public Domain" Myth:**
> Public research repositories such as the **ADHD-200 Consortium** are **not** unrestricted public domain. Access is granted under formal Data Use Agreements (DUAs) administered by NITRC and the International Neuroimaging Data-sharing Initiative (INDI).
- Under the ADHD-200 DUA, researchers explicitly agree to:
  1. Not attempt to re-identify any research subject.
  2. Not distribute raw data files outside the authorized research team.
  3. Properly acknowledge data contributors and granting agencies in scientific publications.
  4. Maintain secure local data storage.
- **Current Status:** No DUA has been executed for this pipeline environment. Execution of an institutional DUA is a mandatory blocking prerequisite for Tier 2 research.

### 8. De-Identification Standards
All incoming raw imaging data must undergo rigorous de-identification prior to pipeline ingestion:
- **Removal of 18 HIPAA Safe Harbor Identifiers:** Names, geographic subdivisions smaller than a state, all dates (except year), telephone numbers, email addresses, medical record numbers (MRNs), health plan IDs, device identifiers, and full-face photographic images.
- **Removal of Indian Context Identifiers:** Aadhaar numbers, hospital unique health IDs (ABHA), patient registration IDs, institution-specific radiologist notes, and billing identifiers.
- **Complete DICOM Header Scrubbing:** Conversion from DICOM to BIDS NIfTI using tools like `dcm2niix` configured to strip private DICOM tags (`-ba n` / `-m y`).

### 9. Facial Defacing & 3D Reconstruction Privacy Risks
- Structural T1w MRI scans contain high-resolution surface data of the skull and facial soft tissue. Modern 3D rendering algorithms can reconstruct a participant's recognizable face from raw T1w volumes.
- To prevent biometric facial recognition:
  - All T1w volumes must be defaced prior to persistent storage or processing using an approved defacing algorithm (e.g., `pydeface`, FreeSurfer `mri_deface`).
  - Defaced volumes must undergo automated algorithmic verification to confirm that facial tissue is eliminated while frontal cortical gray matter remains 100% intact.

### 10. Pseudonymization & Research Subject Identifier (RSID) Scheme
- Direct participant identifiers are permanently decoupled from imaging data.
- The pipeline enforces a standardized BIDS-compliant alphanumeric pseudonymization scheme:
  $$\text{RSID} = \texttt{sub-NA}\langle\text{SiteID}\rangle\langle\text{SubjectIndex}\rangle \quad (\text{e.g., } \texttt{sub-NA01001})$$
- All intermediate outputs, logs, connectivity matrices, and QC reports reference only the assigned RSID.

### 11. Cryptographic Protection & Encryption Standards
- **Data at Rest:** All participant data, raw scans, and processed features must reside on filesystems encrypted with **AES-256** (e.g., LUKS volume encryption on Linux or FileVault on macOS).
- **Data in Transit:** Any transmission across local networks or authorized institutional storage must utilize **TLS 1.3** or **SSH with ed25519** authentication. Unencrypted protocols (HTTP, FTP, Telnet) are forbidden.
- **Integrity Verification:** HMAC-SHA256 signatures are enforced on input manifests to detect bitrot or unauthorized data tampering.

### 12. Access Control Architecture & Principle of Least Privilege
- Access to raw MRI data and analysis environments is restricted strictly to authorized investigators named in the IEC approval.
- **Container Isolation:** The Docker execution environment runs strictly under an unprivileged user (`neuro`, UID 10001, GID 10001) without sudo or root privileges.
- **Mount Protections:** Raw data inputs and frozen model artifacts are mounted as read-only (`:ro`), preventing unintentional modification or deletion.

### 13. Data Retention, Archival & Destruction Lifecycles
- Research datasets and derived feature matrices will be retained for a minimum period mandated by institutional policy (typically **5 to 7 years post-publication**) to ensure scientific reproducibility.
- Following the retention period, data will be destroyed using certified cryptographic erasure standards (NIST SP 800-88 Rev. 1).
- Ephemeral temporary files generated during pipeline execution (e.g., intermediate SyN warp iterations) are deleted immediately upon stage completion.

### 14. Security Incident & Breach Notification Procedures
In the event of an unauthorized data access, leakage, or loss:
1. **Immediate Containment:** Revoke affected filesystem access and network connections within 1 hour of detection.
2. **Investigation & Risk Assessment:** Determine the scope of compromised data and re-identification risk within 24 hours.
3. **Regulatory Notification:** If real participant data is affected, formally notify the Institutional Ethics Committee (IEC) and statutory authorities (e.g., Data Protection Board of India under DPDP Act 2023) within the prescribed statutory timeframes.
4. **Remediation:** Conduct post-incident technical forensic audit and update security controls.

### 15. Key & Linkage Code Physical and Logical Separation
- The **Linkage Key** (the master table mapping RSID to real participant hospital ID / identity) must **never** be stored within the codebase, inside the Docker container, or on the analysis workstation.
- The Linkage Key must be maintained in an encrypted, password-protected database stored on an **air-gapped, physically secured storage medium** held exclusively by the Principal Investigator or designated clinical study coordinator.

---

## 3. Current Ethics Compliance State

| Governance Requirement | Applicable Standard | Current Status | Action Required Prior to Real Data |
|---|---|---|---|
| Institutional Ethics Clearance | ICMR 2017 Guidelines | `PENDING` | Prepare and submit formal protocol to Institutional Ethics Committee. |
| Data Use Agreement (ADHD-200) | NITRC / INDI Terms | `PENDING` | Submit institutional DUA application for ADHD-200 benchmark access. |
| Statutory Privacy Compliance | DPDP Act 2023 | `DOCUMENTED` | Ensure data fiduciary policies are instituted at laboratory level. |
| De-identification & Defacing | HIPAA Safe Harbor & Defacing Protocol | `VALIDATED (SYNTHETIC)` | Implement automated frontal-pole preserving defacing pipeline. |
| Encryption at Rest & In Transit | AES-256 / TLS 1.3 | `READY` | Configure encrypted storage partition on deployment host. |
| Linkage Key Separation | ICMR Privacy Mandate | `ARCHITECTED` | Establish offline, air-gapped key management protocol. |

---

## 4. Conclusion & Directive

Under no circumstances will real participant scans be ingested, processed, or staged within this repository until:
1. The Institutional Ethics Committee has granted formal written clearance.
2. A formal Data Use Agreement is fully executed with the dataset repository.
3. The Primary Research Supervisor has signed the authorization block in the master review package.

