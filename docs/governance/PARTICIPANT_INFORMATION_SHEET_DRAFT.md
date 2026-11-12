# PARTICIPANT INFORMATION SHEET (CONDITIONAL DRAFT)

> **`DRAFT — NOT SUBMITTED — NOT APPROVED`**
>
> **Applicability Notice:** This document is a **conditional template** prepared strictly in case **prospective human-participant MRI acquisition ($N=6$ pilot)** is planned and approved, or `[IEC_NAME]` directs participant notification. It does **not** apply to synthetic phantom tests and does **not** authorize recruitment.
> **Document ID:** `DOC-GOV-PIS-001`  
> **Study Title:** NeuroADHD — ADHD Classification Using Resting-State Functional MRI Connectivity and Machine Learning  
> **Institution:** `[INSTITUTION_NAME]` | **IEC Name & Reg. No.:** `[IEC_NAME]` (`[IEC_REGISTRATION_NUMBER]`)  
> **Protocol Reference:** `[IEC_PROTOCOL_NUMBER]` (Approval Date: `[APPROVAL_DATE]`)

---

## 1. Invitation and Research Purpose
You (or your child/ward) are being invited to consider taking part in an academic research study conducted at `[INSTITUTION_NAME]`. The purpose of this study is to evaluate computational methods that process brain Magnetic Resonance Imaging (MRI) scans and measure functional brain network patterns across research groups. We are studying how software algorithms organize brain connectivity information—**this study is strictly for scientific research and is NOT a medical diagnostic test for Attention-Deficit/Hyperactivity Disorder (ADHD).**

## 2. Why Have You Been Invited? (Conditional Scope)
If `[IEC_NAME]` approves a prospective pilot study (`N = 6` participants), participants may be invited based on approved clinical and MRI safety eligibility criteria `[ELIGIBILITY_CRITERIA_PLACEHOLDER]`. Taking part is completely optional.

## 3. What Participation Involves (If Prospective MRI Collection Is Approved)
If you choose to participate in an approved prospective MRI session:
1. **MRI Safety Screening:** Standard screening to ensure there are no unsafe metal implants, pacemakers, claustrophobia concerns, or pregnancy/safety contraindications.
2. **MRI Scan Session:** Lying still inside a standard MRI scanner (`[SCANNER_FIELD_STRENGTH_AND_DURATION_PLACEHOLDER]`) while structural brain images (`T1w`) and resting-state functional brain images (`BOLD fMRI`, while resting quietly with eyes open/closed per `[SCAN_PROTOCOL_INSTRUCTIONS]`) are recorded. No contrast injections or ionizing radiation (X-rays) are used.
3. **Minimal Research Metadata:** Only approved research-relevant demographic/group labels (e.g., age range, sex, study group code) needed for the protocol will be linked to a coded study ID.

## 4. Potential Risks, Discomforts, and Incidental Findings
- **MRI Noise & Confinement:** The MRI scanner makes loud knocking noises (earplugs/headphones will be provided) and requires lying still inside a narrow tunnel, which can cause mild discomfort or claustrophobia. You may ask to stop the scan at any time using the call squeeze-ball.
- **No Radiation Risk:** MRI uses magnetic fields and radio waves, not X-rays or radiation.
- **Privacy / Re-Identification Risk:** Brain structural images show facial contours. To reduce facial recognition risk, images undergo automated facial removal ("defacing") and direct personal identifiers (names, hospital IDs, birth dates) are stripped into coded study identifiers (`sub-...`).
- **Incidental Findings (If New Scans Are Acquired):** `[INCIDENTAL_FINDINGS_POLICY_PLACEHOLDER — e.g., routine clinical radiology review pathway if an unexpected abnormality is seen]`.

## 5. Research-Only Nature of Machine-Learning Outputs & No Clinical Benefit
- Any computer-generated ADHD-pattern probability score or connectivity chart produced by our research software is **purely experimental** and has **no proven clinical diagnostic accuracy**.
- These computer outputs will **never** be used to diagnose you, change your medical care, or evaluate school/clinical status, and **you should not expect direct personal medical benefit** from participating.

## 6. Confidentiality, Data Storage, Access & Retention
- **Coded Pseudonymization:** Direct personal identifiers are replaced with a coded study number (`sub-...`). The key linking your coded ID to your identity is kept separately in locked institutional custody managed by `[PI_NAME]` and is never placed in public code or online dashboards.
- **Technical Safeguards:** MRI files and derived features are stored externally outside code repositories (`NEUROADHD_DATA_ROOT`) using strong disk encryption (`AES-256` where configured per institutional IT rules) with access restricted strictly to authorized IEC-approved study personnel.
- **Use of Derived Research Features:** Brain connectivity measurements (200 brain region averages / CC200 connectivity numbers, without your name or facial image) may be analyzed computationally within the study and, **only if separately approved by `[IEC_NAME]` and institutional policy**, shared in aggregate scientific publications.
- **Retention:** Data will be retained only for `[INSTITUTIONAL_RETENTION_YEARS]` as determined by `[INSTITUTION_NAME]` policy and `[IEC_NAME]`, after which identifiers/files are securely destroyed per institutional rules.

## 7. Voluntary Participation & Right to Withdraw
- Taking part is completely voluntary. Saying no or stopping participation will **not** affect your medical care, academic standing, or legal rights.
- You may withdraw consent at any time without giving a reason. **Constraint Notice:** Once data have been irreversibly anonymized or incorporated into completed aggregate statistical analyses/publications per `[IEC_NAME]`-approved terms, it may no longer be possible to extract those specific summary statistics, though unanalyzed identifiable files and linkage keys will be removed upon withdrawal request.

## 8. Contacts for Questions and Ethics Concerns (Placeholders)
- **Principal Investigator:** `[PI_NAME]`, `[PI_DEPARTMENT]`, `[INSTITUTION_NAME]` | Phone: `[PI_PHONE]` | Email: `[PI_EMAIL]`
- **Academic Supervisor / Guide:** `[SUPERVISOR_NAME]`, `[SUPERVISOR_DEPARTMENT]` | Email: `[SUPERVISOR_EMAIL]`
- **Institutional Ethics Committee (for rights or complaints):** Member Secretary, `[IEC_NAME]` (DHR Reg. No.: `[IEC_REGISTRATION_NUMBER]`), `[IEC_ADDRESS]` | Phone: `[IEC_PHONE]` | Email: `[IEC_EMAIL]`
