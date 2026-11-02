# NeuroADHD RawMRI PoC V2 — Phase A.5
# Governance & Research Authorization Checklist

**Document ID:** `DOC-NA-A5-AC-001`  
**Version:** 1.0.0  
**Status:** `READY_FOR_SUPERVISOR_DECISION`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Application Target:** Real-Data Ingestion Gate (Pre-Pilot Authorization)  

---

## 1. Purpose & Enforcement Mechanism

This checklist establishes the mandatory, actionable prerequisites that must be satisfied and verified before any real participant magnetic resonance imaging (MRI) data may be ingested, staged, or processed within the NeuroADHD RawMRI PoC V2 environment.

Under the fail-closed safety policy of this repository:
- Every requirement is initially set to `PENDING`.
- No single individual can unilaterally authorize real data execution.
- Real MRI data processing (`allow_real_scans = True`) remains **cryptographically blocked and programmatically disabled** until all 10 checklist items are formally certified.

> [!CAUTION]
> **Audit Rule:**
> No external approvals, institutional clearances, or supervisor signatures are marked complete in this document. All requirements remain strictly **PENDING**. Fabricating dates, names, or approval statuses is strictly forbidden.

---

## 2. Ten-Point Master Governance & Technical Checklist

| Item ID | Governance / Technical Requirement | Responsible Party | Mandatory Evidence Required | Current Status | Reviewer Role | Notes & Prerequisites |
|---|---|---|---|---|---|---|
| **REQ-01** | **Institutional Ethics Committee (IEC) Clearance** | Principal Investigator / Supervisor | Formal written approval letter from Institutional Ethics Committee with protocol number and validity dates. | `PENDING` | Institutional Ethics Committee | Protocol must explicitly describe secondary computational analysis of pediatric neuroimaging data. |
| **REQ-02** | **Data Use Agreement (DUA) Execution** | Institutional Signing Official / PI | Executed Data Use Agreement with NITRC / INDI for the ADHD-200 Consortium dataset. | `PENDING` | Institutional Legal / DUA Administrator | Debunks "unrestricted public domain" myth; ensures compliance with non-reidentification covenants. |
| **REQ-03** | **Informed Consent & Pediatric Assent Protocols** | Clinical Co-Investigator | Verified parental consent forms and participant assent templates (or retrospective waiver from IEC). | `PENDING` | Ethics Committee & Legal Counsel | Must satisfy ICMR 2017 pediatric vulnerability safeguards and age-appropriate assent guidelines. |
| **REQ-04** | **ICMR 2017 & DPDP Act 2023 Statutory Compliance Review** | Institutional Data Protection Officer / PI | Formal compliance audit documentation verifying data fiduciary obligations under DPDP Act 2023 Section 9. | `PENDING` | Institutional Privacy / Compliance Officer | Verifies purpose limitation, data minimization, and children's data protection measures. |
| **REQ-05** | **Facial Defacing Protocol & Frontal-Pole Preservation Sign-Off** | Lead Software Developer / Supervisor | Visual and quantitative audit demonstrating 0% intersection between defacing mask and cortical gray matter mask. | `PENDING` | Research Supervisor / Neuroanatomist | Prevents biometric facial reconstruction while guaranteeing no frontal cortex tissue is inadvertently removed. |
| **REQ-06** | **MRI Acquisition Protocol Specification** | Neuroimaging Physicist / Supervisor | Documented MRI acquisition parameter specification (TR, TE, FA, voxel size, multiband factor, coil channels). | `PENDING` | Research Supervisor | Minimizes domain shift against the historical CC200 training baseline. |
| **REQ-07** | **8-Point QC Acceptance Threshold Formal Sign-Off** | Research Supervisor | Signed copy of the Supervisor Decision Register (`DOC-NA-A5-SDR-001`) with all 8 metrics approved. | `PENDING` | Research Supervisor | Approves T1w volume, EPI-T1w NMI, SyN Dice, Mean FD, Max Spike, Scrubbing %, tSNR, and DVARS cutoffs. |
| **REQ-08** | **Secure Computing Environment & Memory Isolation Audit** | Lead Systems Engineer | Verified execution of `scripts/verify_container_memory.sh` on the target deployment host confirming 2.5 GB cgroup limits. | `PENDING` | Systems / Security Auditor | Confirms Linux kernel OOM enforcement (code 137), non-root execution (`neuro` UID 10001), and read-only mounts. |
| **REQ-09** | **Data Breach Incident Response Plan Approval** | Data Safety Officer / PI | Approved standard operating procedure (SOP) for data containment, risk assessment, and statutory notification within 72 hours. | `PENDING` | Data Protection Officer | Satisfies Section 8(5) of DPDP Act 2023 and institutional incident response policies. |
| **REQ-10** | **Pilot Cohort Definition & Sample Size Authorization** | Biostatistician / Research Supervisor | Formal approval of the First Research Pilot Protocol (`DOC-NA-A5-FRP-001`) with explicit cohort size ($N=6$). | `PENDING` | Research Supervisor | Confirms pilot scope is strictly computational feasibility and not an underpowered clinical trial. |

---

## 3. Checklist Verification & Certification Log

This section is reserved for formal logging when each requirement reaches verified completion. **No entries are recorded at this time.**

```
====================================================================================================
                        FORMAL VERIFICATION LOG (CURRENTLY UNFILLED)
====================================================================================================

Item ID   Verification Date   Authorizing Official   Document Reference / Hash   Verification Status
----------------------------------------------------------------------------------------------------
REQ-01    [ PENDING ]         [ PENDING ]            [ PENDING ]                 UNVERIFIED
REQ-02    [ PENDING ]         [ PENDING ]            [ PENDING ]                 UNVERIFIED
REQ-03    [ PENDING ]         [ PENDING ]            [ PENDING ]                 UNVERIFIED
REQ-04    [ PENDING ]         [ PENDING ]            [ PENDING ]                 UNVERIFIED
REQ-05    [ PENDING ]         [ PENDING ]            [ PENDING ]                 UNVERIFIED
REQ-06    [ PENDING ]         [ PENDING ]            [ PENDING ]                 UNVERIFIED
REQ-07    [ PENDING ]         [ PENDING ]            [ PENDING ]                 UNVERIFIED
REQ-08    [ PENDING ]         [ PENDING ]            [ PENDING ]                 UNVERIFIED
REQ-09    [ PENDING ]         [ PENDING ]            [ PENDING ]                 UNVERIFIED
REQ-10    [ PENDING ]         [ PENDING ]            [ PENDING ]                 UNVERIFIED
====================================================================================================
```

---

## 4. Final Authorization Gatekeeper Sign-Off Block

> [!IMPORTANT]
> **Real Data Ingestion Gate:**
> Real MRI data ingestion remains **DISABLED**. This gatekeeper block may only be signed when all 10 checklist requirements above have achieved verified `APPROVED` status with attached evidence.

**Current Authorization State:**  
$$\mathbf{BLOCKED\text{ }--\text{ }PENDING\_ALL\_PREREQUISITES}$$

**Primary Research Supervisor:**  
Name: `__________________________________________________`  
Title: `__________________________________________________`  
Signature: `[ PENDING SUPERVISOR REVIEW AND SIGNATURE ]`  
Date: `YYYY-MM-DD`  

**Institutional Signing Authority / Ethics Chair:**  
Name: `__________________________________________________`  
Title: `__________________________________________________`  
Signature: `[ PENDING INSTITUTIONAL ETHICS REVIEW AND SIGNATURE ]`  
Date: `YYYY-MM-DD`  

