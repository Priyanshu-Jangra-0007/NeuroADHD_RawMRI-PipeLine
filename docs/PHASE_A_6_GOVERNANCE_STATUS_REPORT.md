# NeuroADHD RawMRI PoC V2 — Phase A.6
# Institutional Governance & Regulatory Compliance Status Report

**Document ID:** `DOC-NA-A6-GSR-001`  
**Version:** 1.0.0  
**Phase:** Phase A.6 — Supervisor Feedback Intake & Research Authorization Gate  
**Report Date:** 2026-11-02  
**Current Governance State:** `UNAPPROVED_PENDING_EXTERNAL_ACTIONS`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Statutory Frameworks:** ICMR Ethical Guidelines (2017), DPDP Act 2023 (India), NITRC/INDI DUA Terms  

---

## 1. Executive Declaration of Unauthorized State

> [!CAUTION]
> **BINDING OPERATIONAL STATEMENT:**
> Real participant magnetic resonance imaging (MRI) data processing is **STRICTLY UNAUTHORIZED AND PROHIBITED**.
> 
> No Institutional Ethics Committee (IEC) clearance, Institutional Review Board (IRB) approval, Data Use Agreement (DUA), or verified participant informed consent/assent is currently active or on file for this pipeline repository. The pipeline execution policy remains enforced as **`STRICTLY_SYNTHETIC_ONLY`**.

---

## 2. Institutional Ethics & Regulatory Clearances Status

### 2.1 Institutional Ethics Committee (IEC) Approval (`AUTH-IEC-01`)
- **Applicable Standard:** ICMR National Ethical Guidelines for Biomedical and Health Research Involving Human Participants (2017), Section 5 (Vulnerable Populations - Pediatric Research).
- **Current Status:** `PENDING`
- **Current Findings:**
  - A comprehensive study protocol for computational secondary analysis of pediatric neuroimaging data has been drafted in Phase A.5 ([`docs/PHASE_A_5_FIRST_RESEARCH_PILOT_PROTOCOL.md`](<your-project-folder>/docs/PHASE_A_5_FIRST_RESEARCH_PILOT_PROTOCOL.md)).
  - Formal submission to the institutional ethics committee is awaiting Primary Research Supervisor sign-off on the Phase A.5 master review package.
  - Zero ethics clearance numbers, approval certificates, or validity windows currently exist.
- **Prerequisite to Clear:** Formal institutional submission, committee review, and receipt of an authenticated written approval letter with registered protocol ID.

---

### 2.2 Data Use Agreement (DUA) Execution (`AUTH-DUA-02`)
- **Applicable Standard:** International Neuroimaging Data-sharing Initiative (INDI) / Neuroimaging Informatics Tools and Resources Clearinghouse (NITRC) ADHD-200 Consortium Data Access Policy.
- **Current Status:** `PENDING`
- **Current Findings:**
  - Open-science neuroimaging repositories (including ADHD-200) are not unrestricted public domain. Access is conditioned upon signing a formal Data Use Agreement prohibiting re-identification attempts, restricting redistribution, and mandating secure storage.
  - No institutional DUA application has been executed or countersigned by the institutional signing official.
- **Prerequisite to Clear:** Institutional execution of the NITRC/INDI ADHD-200 Data Use Agreement and deposit of the countersigned PDF in `docs/evidence/`.

---

### 2.3 Pediatric Informed Consent & Assent Protocols (`AUTH-DSU-03`)
- **Applicable Standard:** ICMR 2017 Guidelines (Section 5.3) & Declaration of Helsinki.
- **Current Status:** `PENDING`
- **Current Findings:**
  - For prospective data: Parental/guardian written informed consent and participant age-appropriate assent are mandatory.
  - For retrospective open-access data: A formal secondary-use waiver must be confirmed by the Institutional Ethics Committee.
- **Prerequisite to Clear:** Written approval of consent/assent templates or formal waiver documentation from the IEC.

---

## 3. Statutory Data Protection & Indian Legal Alignment

### 3.1 Digital Personal Data Protection (DPDP) Act 2023 Compliance (`AUTH-DPP-05` & `AUTH-SUA-04`)
- **Jurisdiction:** Republic of India (Academic Research Institution).
- **Compliance Status:** `DOCUMENTED_PENDING_INSTITUTIONAL_AUDIT`
- **Key Statutory Protections Implemented:**
  1. **Data Fiduciary Obligations (Section 8):** Data minimization enforced by extracting only CC200 functional connectivity features ($19{,}900$ floating-point scalars), discarding non-essential voxel timeseries.
  2. **Special Provisions for Children's Data (Section 9):** Processing data of minors requires verifiable parental consent. Tracking, behavioral profiling, or advertising targeting children is strictly absent from the pipeline design.
  3. **Security Safeguards (Section 8(5)):** Kernel-enforced memory ceilings (2.5 GB), unprivileged container users (`neuro`, UID 10001), and read-only host mounts prevent data exfiltration.
- **Outstanding Requirement:** Formal review by the Institutional Data Protection Officer / Legal Counsel.

---

## 4. Operational Safeguards Audit

The status of the six operational safeguards is summarized below:

| Safeguard ID | Security / Privacy Domain | Operational Requirement | Current Audit Status | Blocker Summary |
|---|---|---|---|---|
| **SAFE-DEID-01** | De-Identification | Complete scrubbing of 18 HIPAA Safe Harbor identifiers and Indian statutory identifiers (Aadhaar, hospital ID). | `PENDING_INSPECTION` | Verified on synthetic phantoms; requires formal validation against clinical DICOM headers prior to real intake. |
| **SAFE-FACE-02** | Facial Defacing Privacy | Quantitative volumetric audit confirming 0% intersection between defacing mask and brain mask. | `PENDING_SUPERVISOR_SIGN_OFF` | Algorithmic test implemented; awaiting supervisor sign-off on defacing approach (DEC-DEF-03). |
| **SAFE-STOR-03** | Storage Encryption | Physical / logical storage encrypted with AES-256 at rest; 2.5 GB Linux cgroup memory isolation. | `READY (INFRASTRUCTURE)` | Cgroup memory isolation verified (3/3 passed); AES-256 encrypted partition required on production host. |
| **SAFE-ACC-04** | Access Control | Role-based least-privilege access; unprivileged container execution (`UID 10001`). | `VERIFIED (CONTAINER)` | Verified in container integration test suite; host user permissions pending lab assignment. |
| **SAFE-AUD-05** | Incident Response | SOP for breach containment, risk assessment, and 72-hour statutory notification to Data Protection Board. | `PENDING_APPROVAL` | Protocol drafted in Phase A.5; requires formal adoption by laboratory director. |
| **SAFE-PILOT-06** | Pilot Scope Governance | Restriction of first real-data test to $N=6$ retrospective scans for computational feasibility only. | `PENDING_APPROVAL` | Protocol drafted (DOC-NA-A5-FRP-001); awaiting supervisor authorization. |

---

## 5. Master Governance Gating Table

All 11 governance and operational safeguard requirements remain unresolved:

```
====================================================================================================
                        GOVERNANCE PREREQUISITES TRACKING SUMMARY
====================================================================================================

Item ID       Domain                   Responsible Authority       Required Evidence         Status
----------------------------------------------------------------------------------------------------
AUTH-IEC-01   Ethics Clearance         Institutional Ethics Comm.  Formal Approval Letter    PENDING
AUTH-DUA-02   Data Use Agreement       NITRC / INDI / Legal        Countersigned DUA PDF     PENDING
AUTH-DSU-03   Consent / Assent         Clinical Co-Investigator    IEC Waiver / Consent Doc  PENDING
AUTH-SUA-04   ICMR / DPDP Compliance   Institutional Legal Counsel Compliance Certification  PENDING
AUTH-DPP-05   Data Protection Audit    Data Protection Officer     Privacy Impact Assessment PENDING
SAFE-DEID-01  De-Identification        Software Lead               Header Scrubbing Audit    PENDING
SAFE-FACE-02  Facial Defacing          Supervisor / Neuroanatomist Defacing Intersect Report PENDING
SAFE-STOR-03  Storage & Memory Guard   Systems Administrator       Host LUKS/AES Audit Log   PENDING
SAFE-ACC-04  Access Control            Lab Director                Access Authorization List PENDING
SAFE-AUD-05  Incident Response SOP     Data Safety Officer         Adopted SOP Document      PENDING
SAFE-PILOT-06 Pilot Scope Authorization Research Supervisor         Signed Pilot Protocol     PENDING
====================================================================================================
```

---

## 6. Conclusion & Fail-Closed Enforcement

Because 100% of the institutional and governance requirements remain `PENDING`, the Research Authorization Gate enforces:
$$\mathbf{REAL\_DATA\_AUTHORIZED = FALSE}$$

Any attempt to execute the pipeline with `allow_real_scans=True` or `--mode real` will immediately terminate with an `AuthorizationDeniedError`. Synthetic-only execution remains the sole authorized operational mode.

