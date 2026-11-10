# NeuroADHD RawMRI PoC V2 — Phase A.8
# Supervisor Decision Request Sheet

**Document ID:** `DOC-NA-A8-SDR-001`  
**Version:** 1.0.0  
**Phase:** Phase A.8 — Supervisor Handoff & External Research Authorization Preparation  
**Target Reviewer:** Primary Research Supervisor & Thesis Advisory Committee  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Baseline Commit:** [`0547574a6db2eb40668046e065fe8dcf989f09b1`](https://github.com/Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine/commit/0547574a6db2eb40668046e065fe8dcf989f09b1)  
**Current Authorization Status:** `PENDING_SUPERVISOR_APPROVAL`

---

## 1. Preamble & Instructions for the Supervisor

This decision sheet presents the **twelve scientific and quality-control determinations** required before the NeuroADHD RawMRI pipeline can be authorized to process real participant MRI scans.

Under the fail-closed governance implemented in Phase A.6 and audited in Phase A.7:
1. Every item below currently resides in the `PENDING` state in the repository's machine-readable authorization register ([`configs/research_authorization_register.json`](<your-project-folder>/configs/research_authorization_register.json)).
2. Real participant MRI processing is strictly blocked by code until these determinations are recorded, signed, and cryptographically verified.
3. The supervisor should review each proposal, indicate their determination, provide comments or modifications where desired, and sign the authorization block at the conclusion of this document.

---

## 2. Section A: Algorithmic & Protocol Determinations

### Determination A: Slice-Timing Correction (STC) Policy
**Register ID:** `DEC-STC-01`

#### Scientific Context
Functional MRI volumes are acquired slice-by-slice across the repetition time ($\text{TR}$). Slices at the top of a volume are acquired up to 2.0 seconds after slices at the bottom in sequential sequences.
- **Prospective Acquisitions:** Modern multi-band sequences acquire volumes with short $\text{TR} < 0.8\text{ s}$, where STC provides negligible benefit and may introduce temporal interpolation artifacts.
- **Retrospective Data (ADHD-200):** Historical scans exhibit $\text{TR} \approx 2.0\text{ s}$, where slice-acquisition phase delays introduce variance into resting-state connectivity matrices. However, slice-order metadata (`SliceTiming` JSON key) is frequently missing or ambiguous across historical repositories.

#### Evaluated Alternatives
- **Option B (Mandatory STC for $\text{TR} \ge 1.5\text{ s}$):** Enforce BIDS-driven STC using exact slice timings prior to rigid motion realignment whenever $\text{TR} \ge 1.5\text{ s}$. Bypass if $\text{TR} < 1.0\text{ s}$.
- **Option C (Dataset-Adaptive STC with Metadata Fallback):** Apply STC if `SliceTiming` is populated in BIDS sidecar. If absent for scans with $\text{TR} \ge 2.0\text{ s}$, fail closed and halt pipeline unless explicit supervisor exemption is logged.

#### Engineering Recommendation
**Option C** is recommended: it enforces scientific rigor for prospective scans while providing fail-closed protection against missing metadata in historical retrospective cohorts.

#### Supervisor Action Field
```
[ ] OPTION B (Mandatory STC for TR >= 1.5s)
[ ] OPTION C (Dataset-Adaptive with Fail-Closed Metadata Fallback - Recommended)
[ ] ALTERNATE DETERMINATION (Specify below)

Supervisor Comments:
____________________________________________________________________________________
```

---

### Determination B: Susceptibility Distortion Correction (SDC) Architecture
**Register ID:** `DEC-SDC-02`

#### Scientific Context
Echo-planar imaging (EPI) experiences severe geometric distortions along the phase-encoding direction due to magnetic susceptibility gradients near tissue-air boundaries (orbitofrontal cortex, inferior temporal lobes). Because the Craddock CC200 atlas includes orbitofrontal and temporal ROIs, uncorrected distortions displace regional BOLD signals across parcel boundaries.
- **Prospective Scans:** Standard spin-echo reverse phase-encoded EPI pairs (AP/PA) allow gold-standard FSL TOPUP distortion unwarping.
- **Retrospective Data (ADHD-200):** Historical datasets lack reverse-phase fieldmaps, precluding TOPUP unwarping.

#### Evaluated Alternatives
- **Dual-Phase TOPUP Only:** Require reverse phase-encoded spin-echo fieldmaps. Reject any scan lacking fieldmaps. (Precludes retrospective ADHD-200 processing).
- **Synthetic B0 Fallback (`syn-b0`):** Implement non-linear deformation estimation directly between distorted EPI and structural T1w volumes as a fallback when fieldmaps are unavailable, gracefully degrading to rigid coregistration if deformation estimation fails.

#### Engineering Recommendation
Adopt **Synthetic B0 Fallback with Prospective TOPUP Mandate**: Mandate TOPUP for all newly acquired prospective scans, while permitting verified synthetic B0 unwarping for retrospective ADHD-200 data.

#### Supervisor Action Field
```
[ ] MANDATE FIELD MAPS ONLY (Strict TOPUP; no retrospective processing)
[ ] SYNTHETIC B0 FALLBACK ALLOWED (Recommended: TOPUP for prospective, Syn-b0 for retrospective)
[ ] RIGID COREGISTRATION ONLY (Omit non-linear unwarping)

Supervisor Comments:
____________________________________________________________________________________
```

---

### Determination C: Facial Defacing Methodology & Anonymization
**Register ID:** `DEC-DEF-03`

#### Scientific Context
High-resolution 3D structural T1w scans permit volumetric facial reconstruction, posing participant re-identification risks under HIPAA and DPDP Act 2023. Automated defacing removes facial tissue. However, overly aggressive defacing tools risk clipping the frontal pole or anterior prefrontal cortex, which would corrupt brain extraction and spatial normalization into MNI space.

#### Evaluated Alternatives
- **Standard `pydeface`:** Affine registration of standard facial mask. Fast and lightweight, but lacks algorithmic cortical preservation verification.
- **`pydeface` with Algorithmic Cortical Preservation Verification:** Execute `pydeface` followed by an automated spatial intersection check:
  $$\text{Intersection}(M_{\text{deface}}, M_{\text{brain}}) = 0\text{ voxels}$$
  If the defacing mask clips even a single brain mask voxel, the pipeline halts immediately.

#### Engineering Recommendation
Adopt **`pydeface` with Algorithmic Cortical Preservation Verification**: Guarantees zero cortical gray matter erosion while achieving complete facial de-identification.

#### Supervisor Action Field
```
[ ] APPROVED: pydeface with Algorithmic Cortical Preservation Verification
[ ] ALTERNATE DEFACING TOOL REQUIRED (e.g., mri_deface, bidsonym)
[ ] REJECTED / DEFER DEFACING TO DATA PROVIDER

Supervisor Comments:
____________________________________________________________________________________
```

---

### Determination D: Prospective MRI Acquisition Protocol Parameters
**Register ID:** `DEC-ACQ-04`

#### Scientific Context
To prevent domain shift between future prospective scans and the frozen CC200 feature extraction pipeline, prospective scanner acquisition protocols must be standardized.

#### Proposed Acquisition Baseline
- **Magnetic Field Strength:** 3.0 Tesla
- **RF Head Coil:** Minimum 32-channel phased-array head coil
- **Structural T1w (MP-RAGE):**
  - Voxel Size: $1.0\text{ mm}$ isotropic ($1.0 \times 1.0 \times 1.0\text{ mm}$)
  - Field of View (FOV): $256 \times 256\text{ mm}$
  - Flip Angle: $7^\circ\text{--}9^\circ$, $\text{TI} \approx 900\text{--}1000\text{ ms}$
- **Functional rs-fMRI (BOLD EPI):**
  - Voxel Resolution: $3.0\text{ mm}$ isotropic ($3.0 \times 3.0 \times 3.0\text{ mm}$)
  - Repetition Time ($\text{TR}$): $2000\text{ ms}$ (to match benchmark) OR $800\text{ ms}$ (multiband factor 8)
  - Echo Time ($\text{TE}$): $30\text{ ms}$ (at 3T)
  - Acquisition Duration: Minimum 6 minutes (180 volumes at $\text{TR}=2.0\text{s}$)
  - Participant State: Eyes open, fixating on central crosshair

#### Supervisor Action Field
```
[ ] APPROVED: Baseline Prospective Acquisition Protocol
[ ] MODIFICATIONS REQUIRED (Specify below)
[ ] REJECTED

Supervisor Comments:
____________________________________________________________________________________
```

---

## 3. Section B: Quality Control (QC) Acceptance Thresholds

The pipeline evaluates **eight quantitative quality control metrics** across structural and functional domains. Every numerical threshold below matches the baseline established in [`docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md`](<your-project-folder>/docs/PHASE_A_5_SUPERVISOR_DECISION_REGISTER.md) and requires supervisor sign-off.

---

### QC-01: T1w Brain Extraction Total Intracranial Volume
- **Stage & Target:** Stage 5: T1w Brain Extraction
- **Metric Formulation:** Total intracranial volume: $V_{\text{brain}} = \sum v \cdot \Delta x \Delta y \Delta z$
- **Proposed Acceptance Threshold:** **$600\text{--}2200\text{ cm}^3$**
- **Scientific Purpose:** Verifies skull-stripping validity across pediatric through adult ranges, rejecting gross brain-extraction clipping or under-stripping.
- **Consequences of Rejection:** Scan is halted; anatomical staging directory quarantined; prevented from corrupting MNI registration.
- **Supervisor Action:** `[ ] APPROVED    [ ] MODIFIED: ______ cm³    [ ] REJECTED`
- **Reviewer Comments:** `__________________________________________________`

---

### QC-02: EPI-to-T1w Coregistration Normalized Mutual Information
- **Stage & Target:** Stage 9: Functional-to-Structural Coregistration
- **Metric Formulation:** $\text{NMI}(X, Y) = \frac{H(X) + H(Y)}{H(X, Y)}$
- **Proposed Acceptance Threshold:** **$\text{NMI} \ge 0.55$**
- **Scientific Purpose:** Ensures robust rigid cross-modal alignment between mean functional BOLD and skull-stripped anatomical T1w volume.
- **Consequences of Rejection:** Scan rejected; prevents spatial misalignment of CC200 parcels relative to functional cortex.
- **Supervisor Action:** `[ ] APPROVED    [ ] MODIFIED: NMI >= ______    [ ] REJECTED`
- **Reviewer Comments:** `__________________________________________________`

---

### QC-03: T1w-to-MNI Spatial Normalization Dice Overlap
- **Stage & Target:** Stage 10: T1w-to-MNI Normalization
- **Metric Formulation:** $\text{Dice} = \frac{2 \|M_{\text{T1w}} \cap M_{\text{MNI}}\|}{\|M_{\text{T1w}}\| + \|M_{\text{MNI}}\|}$
- **Proposed Acceptance Threshold:** **$\text{Dice} \ge 0.85$**
- **Scientific Purpose:** Validates convergence of non-linear symmetric diffeomorphic (SyN) registration into MNI152 stereotaxic space.
- **Consequences of Rejection:** Scan rejected; prevents warped BOLD from entering parcellation on a distorted stereotaxic grid.
- **Supervisor Action:** `[ ] APPROVED    [ ] MODIFIED: Dice >= ______    [ ] REJECTED`
- **Reviewer Comments:** `__________________________________________________`

---

### QC-04: Mean Framewise Displacement (FD) Head Motion Ceiling
- **Stage & Target:** Stage 3: Head Motion Realignment
- **Metric Formulation:** Power et al. (2012): $\overline{\text{FD}} = \frac{1}{N_t - 1} \sum_{t=2}^{N_t} \text{FD}_t$
- **Proposed Acceptance Threshold:** **$\overline{\text{FD}} \le 0.50\text{ mm}$**
- **Scientific Purpose:** Enforces standard head-movement cutoff in pediatric ADHD literature, preventing motion-induced spurious network inflation.
- **Consequences of Rejection:** Scan rejected; excludes excessively restless participants whose connectivity profiles are motion-dominated.
- **Supervisor Action:** `[ ] APPROVED    [ ] MODIFIED: Mean FD <= ______ mm    [ ] REJECTED`
- **Reviewer Comments:** `__________________________________________________`

---

### QC-05: Maximum Instantaneous Motion Spike
- **Stage & Target:** Stage 3: Head Motion Realignment
- **Metric Formulation:** $\text{FD}_{\max} = \max_t (\text{FD}_t)$
- **Proposed Acceptance Threshold:** **Flag if $> 3.0\text{ mm}$; Reject if $> 5.0\text{ mm}$**
- **Scientific Purpose:** Detects sudden catastrophic movements (coughing, shifting) that cause bulk slice disruption across volumes.
- **Consequences of Rejection:** Scans exceeding 5.0 mm bulk shift are discarded; scans with 3.0-5.0 mm spikes require visual inspection.
- **Supervisor Action:** `[ ] APPROVED    [ ] MODIFIED: Flag > ___ mm, Reject > ___ mm    [ ] REJECTED`
- **Reviewer Comments:** `__________________________________________________`

---

### QC-06: Scrubbing / Volume Censoring Proportion Upper Bound
- **Stage & Target:** Stage 3 & 14: Volume Scrubbing
- **Metric Formulation:** $P_{\text{scrub}} = \frac{N_{\text{censored}}}{N_t} \times 100\%$ where volume is censored if $\text{FD}_t > 0.5\text{ mm}$
- **Proposed Acceptance Threshold:** **$\le 30\%$ of total run volumes**
- **Scientific Purpose:** Preserves sufficient temporal degrees of freedom ($N_{\text{eff}} \ge 120$ volumes in a 180-volume run) for stable correlation matrix estimation.
- **Consequences of Rejection:** Scan rejected if $>30\%$ of volumes are censored, avoiding singular or unstable covariance matrices.
- **Supervisor Action:** `[ ] APPROVED    [ ] MODIFIED: Scrub <= ______ %    [ ] REJECTED`
- **Reviewer Comments:** `__________________________________________________`

---

### QC-07: Temporal Signal-to-Noise Ratio (tSNR)
- **Stage & Target:** Stage 4: Raw BOLD Quality
- **Metric Formulation:** Median $\text{tSNR} = \frac{\mu_t(v)}{\sigma_t(v)}$ within intracranial brain mask
- **Proposed Acceptance Threshold:** **Median $\text{tSNR} \ge 40$; Flag if $< 30$**
- **Scientific Purpose:** Verifies raw scanner signal stability and RF head coil performance prior to nuisance regression.
- **Consequences of Rejection:** Identifies scanner RF receiver failure, severe ghosting, or poor baseline signal.
- **Supervisor Action:** `[ ] APPROVED    [ ] MODIFIED: Median tSNR >= ______    [ ] REJECTED`
- **Reviewer Comments:** `__________________________________________________`

---

### QC-08: DVARS Outlier Metric Threshold
- **Stage & Target:** Stage 3: Intensity Transition Realignment
- **Metric Formulation:** $\text{DVARS}_t = \sqrt{\frac{1}{V}\sum_v (\Delta I_{v, t})^2}$
- **Proposed Acceptance Threshold:** **No volumes $> 1.5 \times \text{median}(\text{DVARS})$**
- **Scientific Purpose:** Identifies global BOLD signal intensity step-functions and RF spike artifacts independent of head translation.
- **Consequences of Rejection:** Volumes with anomalous intensity transitions are flagged or censored during scrubbing.
- **Supervisor Action:** `[ ] APPROVED    [ ] MODIFIED: Outlier threshold ______    [ ] REJECTED`
- **Reviewer Comments:** `__________________________________________________`

---

## 4. Formal Supervisor Sign-Off & Determination Block

```
========================================================================================
FORMAL SUPERVISOR RESEARCH DETERMINATION
========================================================================================

I have reviewed the scientific context, architectural alternatives, prospective acquisition
parameters, and eight quantitative QC acceptance thresholds presented in this document.

Overall Determination:
[ ] ALL PROPOSALS APPROVED AS SPECIFIED
[ ] APPROVED SUBJECT TO SPECIFIC MODIFICATIONS NOTED ABOVE
[ ] REJECTED / FURTHER METHODOLOGICAL INVESTIGATION REQUIRED

Primary Research Supervisor Name:      __________________________________________________
Academic Title & Institution:          __________________________________________________
Date:                                  __________________________________________________
Signature:                             __________________________________________________

HMAC Key Custody Confirmation:
[ ] I confirm that the NEUROADHD_HMAC_KEY will be stored securely on isolated hardware
    and will only be provided to authorize verified pipeline runs.
========================================================================================
```
