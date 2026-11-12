# DATASET PROVENANCE AND PERMISSIONS AUDIT

> **`DRAFT — NOT SUBMITTED — NOT APPROVED`**
>
> **Status:** `DRAFT_FOR_INSTITUTIONAL_REVIEW`  
> **Document ID:** `DOC-GOV-DATA-001`  
> **Dataset Investigated:** ADHD-200 Consortium Resting-State fMRI Sample (plus Synthetic Phantoms & Conditional Future Prospective Pilot)  
> **Execution Policy:** `STRICTLY_SYNTHETIC_ONLY` | `real_data_authorized = false`  
> **Safety Note:** Compiled strictly from repository documentation and official public repository policies without opening participant MRI volumes or individual phenotypic records.

---

## 1. Purpose and Core Governance Principles

This document establishes dataset provenance, licensing/Data Use Agreement (DUA) constraints, and unresolved institutional questions for every data tier referenced by **NeuroADHD — ADHD Classification Using Resting-State Functional MRI Connectivity and Machine Learning**:

1. **Public downloadability is NOT unrestricted permission.** Repositories hosted by INDI / NITRC remain subject to registration, attribution, non-commercial rules, and anti-re-identification undertakings.
2. **Source-institution ethics approval does NOT automatically waive local IEC review.** Contributing sites obtained local IRB/REB clearance to collect and share anonymized scans; the receiving Indian institution's IEC must determine local review or exemption requirements for secondary analysis.
3. **Derivative data sharing requires explicit legal/DUA verification.** Even when derivatives (`roi_CC200.csv`, connectivity matrices) remove voxel geometry, onward redistribution must comply with DUA and institutional policy.

---

## 2. Dataset Tier Overview

| Tier | Dataset Source | Cohort Scope Verified in Docs | Current Pipeline Usage | Overall Permission Posture |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | In-silico Synthetic Phantoms (`generate_synthetic_phantom_session`) | Mathematical phantoms (`sub-synthetic*`) | Active default validation (Stages 1–23) | `VERIFIED` (zero human data) |
| **Tier 2** | **ADHD-200 Consortium Sample** via INDI / NITRC (`fcon_1000.projects.nitrc.org/indi/adhd200/`) | Engineering docs: **153 usable subjects (50 ADHD, 103 Control)**; Model artifacts: `114` train + `20` locked `Peking_2` test (`134`, unreconciled manifest) | Historical `roi_CC200.csv` read-only baseline + **locked** raw-scan target | `REQUIRES_INSTITUTIONAL_REVIEW` (`AUTH-DUA-02` = `PENDING`) |
| **Tier 3** | Prospective Local Pilot ($N=6$, conditional) | **Not collected**; proposed $N=6$ feasibility only if approved | Blocked / unapproved | `REQUIRES_INSTITUTIONAL_REVIEW` |

---

## 3. Detailed ADHD-200 Provenance & Permission Assessment

| Governance Dimension | Documented Finding (Docs & Official Public Terms) | Status | Exact Unresolved Question & Responsible Authority |
| :--- | :--- | :--- | :--- |
| **Original Source & Hosting Organization** | Multi-site **ADHD-200 Consortium**, distributed via the **International Neuroimaging Data-sharing Initiative (INDI)** / **1000 Functional Connectomes Project (FCP)** hosted on **NITRC** (`http://fcon_1000.projects.nitrc.org/indi/adhd200/`), alongside Preprocessed Connectomes Project (PCP) C-PAC derivatives indexed in local S3 manifests. | `VERIFIED` | N/A for source identity; local copy provenance (exact download date, account, and manifest checksums) requires `[DATA_CUSTODIAN]` / `[PI_NAME]` verification (`PENDING_DOCUMENTATION`). |
| **Original Dataset Documentation** | Described in ADHD-200 Consortium (2012, *Frontiers in Systems Neuroscience*) and PCP benchmark notes (Bellec et al., 2017); local technical audit documented in `docs/HISTORICAL_PREPROCESSING_AUDIT.md`. | `VERIFIED` | Exact original split manifest explaining `153` usable subjects (`50` ADHD, `103` Control) vs. `134` (`114+20`) model split requires original manifest inspection -> `[PI_NAME]` / `[SUPERVISOR_NAME]` (`REQUIRES_DATA_OWNER_CONFIRMATION`). |
| **Source Institutional Approvals** | Official INDI/ADHD-200 documentation states each contributing acquisition site (e.g., Peking University, NYU Child Study Center, Kennedy Krieger Institute, OHSU, NeuroImage, Brown, Pittsburgh, Washington University) obtained approval from its local IRB/ethics board and participant/parental consent/assent prior to public de-identified upload. | `VERIFIED` (per provider statements); local verification of specific contributing site letters: `PENDING_DOCUMENTATION` | Does `[IEC_NAME]` require archived copies of contributing-site IRB summary notices for secondary use, and does secondary analysis require local IEC review or formal exemption? -> `[IEC_NAME]`. |
| **Dataset Access & Registration Requirements** | INDI / NITRC requires registered user account sign-in and acceptance of portal terms before downloading raw data or site releases. | `VERIFIED` (portal policy); individual team registration proof on file: `PENDING_DOCUMENTATION` | Which named `[INSTITUTION_NAME]` investigators are registered under the institutional/personal NITRC account holding any local copies? -> `[PI_NAME]` / `[SUPERVISOR_NAME]`. |
| **Data Use Agreement (DUA)** | Governed by FCP / INDI / NITRC non-commercial research use & anti-re-identification undertakings (`AUTH-DUA-02` tracked as `PENDING` in `configs/research_authorization_register.json`). No signed institutional DUA archive is present in this repository. | `PENDING_DOCUMENTATION` | Is an institution-countersigned NITRC/INDI DUA confirmation required for all named team members (`[PI_NAME]`, `[SUPERVISOR_NAME]`, student developer) prior to raw scan staging? -> `[INSTITUTION_LEGAL_OFFICE]` & `[DATA_CUSTODIAN]`. |
| **Licensing & Redistribution Restrictions** | INDI / 1000 FCP policies restrict ADHD-200 data strictly to **non-commercial academic/scientific research** (aligned with **Creative Commons Attribution-NonCommercial / CC-BY-NC** terms referenced on INDI/NITRC pages). Commercial exploitation and profit-driven redistribution are prohibited. | `VERIFIED` (public INDI/CC-BY-NC notice); exact site-by-site license text snapshot not archived in repo (`PENDING_DOCUMENTATION`) | Can trained model weights or derived CC200 summary statistics be released under an academic license without violating CC-BY-NC or site-specific NITRC constraints? -> `[INSTITUTION_LEGAL_OFFICE]` & INDI/NITRC data steward (`REQUIRES_DATA_OWNER_CONFIRMATION`). |
| **Citation Requirements** | Publications must cite: (1) *The ADHD-200 Consortium (2012), Front. Syst. Neurosci. 6:62 (doi:10.3389/fnsys.2012.00062)*; (2) INDI / NITRC ADHD-200 sample URL & contributing sites used; and (3) *Bellec et al. (2017)* if PCP/C-PAC preprocessed derivatives are used, plus relevant NIH/site grant acknowledgments. | `VERIFIED` | Confirm exact contributing sites included in the `153`-subject engineering cohort before manuscript submission -> `[PI_NAME]` (`PENDING_DOCUMENTATION`). |
| **Public Release & Anonymization Conditions** | INDI documentation states volumes were defaced/anonymized under HIPAA-aligned protocols (removal of direct identifiers and facial information prior to release). However, defacing completeness varies historically across multi-site releases and requires re-verification (Step 6 `defacing.py`). | `VERIFIED` (provider claim); independent verification on held raw copies: `PENDING_DOCUMENTATION` | Have all raw anatomical files in the external local store passed modern defacing and DICOM/BIDS header verification without residual facial geometry? -> `[PI_NAME]` / `[DATA_CUSTODIAN]` post-authorization. |
| **Restrictions on Onward Sharing** | Prohibits unauthorized re-uploading, mirroring, or redistributing raw ADHD-200 MRI scans outside authorized research personnel, and strictly prohibits any attempt to re-identify subjects or link phenotypic profiles to real individuals. | `VERIFIED` | Define explicit institutional boundary preventing accidental commit/upload of NIfTI files, staged ZIPs, or subject manifests to GitHub/cloud tools -> `[DATA_CUSTODIAN]` / `[DPO_NAME]`. |
| **Sharing of Derivative Data (`roi_CC200.csv`, 19,900 Fisher-$z$, Code)** | Software code and synthetic validation outputs carry zero human data (`VERIFIED`). Sharing **real-subject** derivatives (`roi_CC200.csv`, Fisher-$z$ matrices, QC PNGs) externally is **not permitted** unless cleared under INDI/ADHD-200 derivative rules and local IEC/DUA terms. | `REQUIRES_INSTITUTIONAL_REVIEW` | Under local IEC rules and NITRC/INDI terms, may de-identified extracted `roi_CC200.csv` or aggregate metrics be included in theses or supplementary archives? -> `[IEC_NAME]` & INDI/NITRC (`REQUIRES_DATA_OWNER_CONFIRMATION`). |
| **Cross-Institutional & International-Use Restrictions** | Data originate from international sites (USA, China, Netherlands) and are processed by an **Indian academic institution** (`[INSTITUTION_NAME]`), triggering cross-border governance review (ICMR guidelines, DHR/HMSC screening assessment if international collaboration/funding exists, DPDP Act 2023). | `REQUIRES_INSTITUTIONAL_REVIEW` | Does secondary computational analysis of an existing international neuroimaging repository—or future collaborative benchmark sharing—require DHR/HMSC screening or institutional cross-border approval? -> `[IEC_NAME]` & `[INSTITUTION_ADMINISTRATION]`. |

---

## 4. Prohibited Assumptions Checklist

- **NO** licence, DUA execution date, or institutional waiver has been invented.
- **NO** raw MRI scans (`sub-01` or ADHD-200 NIfTIs) were opened or inspected during this audit.
- **NO** real-data authorization item in `configs/research_authorization_register.json` was altered; `AUTH-DUA-02` and `sub-01` remain `PENDING`.
