# NeuroADHD RawMRI PoC V2 — Phase A.5
# Supervisor Decision Register & Preprocessing Specification

**Document ID:** `DOC-NA-A5-SDR-001`  
**Version:** 1.0.0  
**Status:** `READY_FOR_SUPERVISOR_DECISION`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Review Target:** Primary Research Supervisor & Thesis Evaluation Committee  

---

## 1. Purpose & Preamble

This register serves as the formal repository of critical scientific, algorithmic, and methodological decisions requiring direct review and authorized sign-off by the **Primary Research Supervisor** prior to enabling real participant MRI data processing.

Under the fail-closed governance established in Phase A.4, no real MRI data (prospective or retrospective) may be introduced into the pipeline while any decision in this register remains unapproved. All items below are currently marked with status:
$$\mathbf{PENDING\_SUPERVISOR\_APPROVAL}$$

---

## 2. Section A: MRI Acquisition & Preprocessing Technical Decisions

### Decision 1: Slice-Timing Correction (STC) Policy

#### Clinical & Technical Context
Functional MRI volumes are acquired slice-by-slice over the duration of one repetition time ($\text{TR}$). Slices at the top of the volume may be acquired nearly 2 seconds after slices at the bottom in conventional sequential sequences.
- In **historical datasets** (such as ADHD-200), acquisitions typically had long repetition times ($\text{TR} \approx 2.0\text{ s}$), and participating sites exhibited inconsistent preprocessing practices (some applied STC, others omitted it).
- In **modern acquisitions** utilizing simultaneous multi-slice / multiband acceleration, the effective $\text{TR}$ is often short ($\text{TR} < 0.8\text{ s}$), rendering STC of minimal benefit while potentially introducing temporal interpolation artifacts.

#### Options Evaluated
- **Option A (Skip STC Universally):** Omit slice-timing correction across all datasets.
  - *Pros:* Prevents temporal blurring and avoids interacting with head motion realignment order.
  - *Cons:* Residual phase delays between slices remain in long-$\text{TR}$ ($\ge 2.0\text{ s}$) acquisitions.
- **Option B (Mandatory STC for $\text{TR} \ge 1.5\text{ s}$):** Perform BIDS-metadata-driven STC using exact `SliceTiming` arrays prior to motion realignment for acquisitions with $\text{TR} \ge 1.5\text{ s}$; bypass for $\text{TR} < 1.0\text{ s}$.
  - *Pros:* Optimizes temporal alignment according to sequence physics.
  - *Cons:* Requires complete and accurate `SliceTiming` JSON metadata, which is frequently missing or malformed in retrospective public data.
- **Option C (Dataset-Adaptive STC with Metadata Fallback):** Require explicit user flag or infer from BIDS `SliceTiming`; fail closed if $\text{TR} \ge 2.0\text{ s}$ and `SliceTiming` is absent unless supervisor override is logged.

#### Engineering Recommendation
Option B is scientifically optimal for prospective studies; Option C provides robust fail-closed handling for mixed retrospective cohorts.

#### Formal Decision Record
- **Supervisor Action:** `[ ] OPTION A    [ ] OPTION B    [ ] OPTION C    [ ] REJECTED / OTHER`
- **Current Status:** `PENDING_SUPERVISOR_APPROVAL`
- **Supervisor Notes:** `__________________________________________________`

---

### Decision 2: Susceptibility Distortion Correction (SDC) Architecture

#### Clinical & Technical Context
Echo-planar imaging (EPI) sequences suffer from severe geometric and intensity distortions along the phase-encoding direction, caused by local magnetic susceptibility gradients near air-tissue interfaces (e.g., orbitofrontal cortex, inferior temporal lobes). Because the Craddock CC200 atlas includes orbitofrontal and temporal ROIs, uncorrected distortions may displace signal outside regional boundaries.

#### Options Evaluated
- **Option A (Dual Phase-Encoding Fieldmap / TOPUP):** Mandate acquisition of pair of spin-echo EPI images with reversed phase-encoding directions (e.g., AP/PA) and apply FSL TOPUP.
  - *Pros:* Gold standard for geometric unwarping; highest recovery of orbitofrontal geometry.
  - *Cons:* Strictly impossible for retrospective public repositories (e.g., ADHD-200) where reverse-phase fieldmaps were not acquired.
- **Option B (B0 Fieldmap Phase-Difference):** Support dual-echo gradient-echo fieldmaps ($\Delta\text{TE}$).
  - *Pros:* Standard on clinical scanners.
  - *Cons:* Phase unwrapping errors near sinuses; unavailable in historical ADHD-200 data.
- **Option C (Fieldmap-Less Synthetic B0 Normalization / Syn-b0):** Implement a fallback synthetic distortion correction estimating deformation directly against the structural T1w volume, with graceful degradation to rigid coregistration if fieldmaps are unavailable.
  - *Pros:* Allows backward compatibility with historical datasets while supporting high-fidelity fieldmaps when present.
  - *Cons:* Synthetic unwarping has higher computational overhead and slight risk of blurring.

#### Engineering Recommendation
Implement Option C: require formal fieldmaps (Option A or B) for any prospective scanning protocol, while providing a verified fallback for retrospective ADHD-200 benchmark validation.

#### Formal Decision Record
- **Supervisor Action:** `[ ] OPTION A    [ ] OPTION B    [ ] OPTION C    [ ] REJECTED / OTHER`
- **Current Status:** `PENDING_SUPERVISOR_APPROVAL`
- **Supervisor Notes:** `__________________________________________________`

---

### Decision 3: Facial Defacing Methodology & Anonymization

#### Clinical & Technical Context
High-resolution 3D structural T1w MRI scans can be rendered into accurate 3D surface meshes of the participant's face, allowing potential biometric re-identification when cross-referenced against photographic databases or school records. Defacing is mandatory under ethical data governance. However, overly aggressive defacing tools can accidentally clip the frontal pole or anterior prefrontal cortex, degrading brain extraction and spatial normalization.

#### Options Evaluated
- **Option A (`pydeface`):** Python-based automated defacing applying an MNI-derived facial mask via affine registration.
  - *Pros:* Native Python implementation; easily containerized without FreeSurfer overhead.
  - *Cons:* Occasionally clips anterior frontal bone or frontal poles if head tilt is substantial.
- **Option B (FreeSurfer `mri_deface`):** Highly conservative facial stripping based on a multi-subject facial template.
  - *Pros:* Robust preservation of frontal cortex; proven standard in ADNI and ABCD.
  - *Cons:* Requires FreeSurfer licensing and large container dependencies ($>5\text{ GB}$).
- **Option C (`bidsonym` Multi-Tool Defacing with Frontal Preservation Audit):** Apply `pydeface` coupled with an automated post-defacing spatial audit asserting that the intersection between the defacing mask and the subject's brain mask is strictly zero voxels.
  - *Pros:* Guarantees 100% preservation of cortical tissue while eliminating facial features; fully containerizable within 2.5 GB RAM.
  - *Cons:* Requires fail-closed verification step.

#### Engineering Recommendation
Option C: automated defacing with algorithmic verification of zero cortical mask erosion.

#### Formal Decision Record
- **Supervisor Action:** `[ ] OPTION A    [ ] OPTION B    [ ] OPTION C    [ ] REJECTED / OTHER`
- **Current Status:** `PENDING_SUPERVISOR_APPROVAL`
- **Supervisor Notes:** `__________________________________________________`

---

### Decision 4: Prospective MRI Acquisition Protocol Parameters

#### Clinical & Technical Context
Should the supervisor and institutional ethics committee approve prospective participant scanning in future phases, scanning parameters must be formally standardized to minimize domain shift relative to the CC200 training benchmark.

#### Proposed Acquisition Baseline (For Supervisor Determination):
- **Magnetic Field Strength:** 3.0 Tesla (recommended; 1.5T acceptable with protocol adjustment).
- **RF Head Coil:** Minimum 32-channel phased-array head coil.
- **Structural Acquisition (T1w MP-RAGE):**
  - Voxel Size: $1.0\text{ mm}$ isotropic ($1.0 \times 1.0 \times 1.0\text{ mm}$)
  - Field of View (FOV): $256 \times 256\text{ mm}$
  - Flip Angle: $7^\circ\text{--}9^\circ$
  - Inversion Time (TI): $\approx 900\text{--}1000\text{ ms}$
- **Functional Acquisition (Resting-State BOLD EPI):**
  - Repetition Time ($\text{TR}$): $2000\text{ ms}$ (to match historical ADHD-200 dynamics) OR $800\text{ ms}$ (multiband factor 8)
  - Echo Time ($\text{TE}$): $30\text{ ms}$ (at 3T)
  - Flip Angle: Ernst angle ($\approx 70^\circ\text{--}77^\circ$ for $\text{TR}=2\text{s}$)
  - Voxel Resolution: $3.0\text{ mm}$ isotropic ($3.0 \times 3.0 \times 3.0\text{ mm}$)
  - Scan Duration: Minimum 6 minutes (180 volumes at $\text{TR}=2.0\text{s}$) to stabilize functional connectivity matrices
  - Participant State: Eyes open, fixating on central crosshair (to reduce spontaneous sleep transitions).

#### Formal Decision Record
- **Supervisor Action:** `[ ] PROTOCOL ACCEPTED    [ ] MODIFICATIONS REQUIRED    [ ] REJECTED`
- **Current Status:** `PENDING_SUPERVISOR_APPROVAL`
- **Supervisor Notes:** `__________________________________________________`

---

## 3. Section B: Quality Control (QC) Acceptance Criteria Register

The pipeline evaluates 8 quantitative metrics across structural and functional domains. Every threshold must be formally reviewed and provisionally approved by the supervisor.

| QC Metric ID | Pipeline Stage & Target | Evaluated Metric & Formula | Proposed Acceptance Threshold | Scientific Justification | Supervisor Approval Status |
|---|---|---|---|---|---|
| **QC-01** | Stage 5: T1w Brain Extraction | Total intracranial volume: $V_{\text{brain}} = \sum v \cdot \Delta x \Delta y \Delta z$ | **$600\text{--}2200\text{ cm}^3$** | Bounds encompass 99.9% of pediatric through adult populations while rejecting gross skull-stripping failures. | `PENDING_SUPERVISOR_APPROVAL` |
| **QC-02** | Stage 9: EPI-to-T1w Coregistration | Normalized Mutual Information: $\text{NMI}(X, Y) = \frac{H(X) + H(Y)}{H(X, Y)}$ | **$\text{NMI} \ge 0.55$** | Ensures structural-functional spatial alignment; prevents ROI label misassignment. | `PENDING_SUPERVISOR_APPROVAL` |
| **QC-03** | Stage 10: T1w-to-MNI Normalization | Dice Overlap Coefficient: $\text{Dice} = \frac{2 \|M_{\text{T1w}} \cap M_{\text{MNI}}\|}{\|M_{\text{T1w}}\| + \|M_{\text{MNI}}\|}$ | **$\text{Dice} \ge 0.85$** | Validates diffeomorphic SyN convergence to standard CC200 stereotaxic space. | `PENDING_SUPERVISOR_APPROVAL` |
| **QC-04** | Stage 3: Head Motion (Mean) | Mean Framewise Displacement: $\overline{\text{FD}} = \frac{1}{N_t - 1} \sum_{t=2}^{N_t} \text{FD}_t$ | **$\overline{\text{FD}} \le 0.50\text{ mm}$** | Standard cutoff in pediatric ADHD literature (Power et al., 2012) preventing motion-induced spurious network inflation. | `PENDING_SUPERVISOR_APPROVAL` |
| **QC-05** | Stage 3: Head Motion (Spikes) | Maximum Instantaneous FD: $\text{FD}_{\max} = \max_t (\text{FD}_t)$ | **Flag $> 3.0\text{ mm}$<br>Reject $> 5.0\text{ mm}$** | Flags severe bulk movements; rejects scans with physical head repositioning during acquisition. | `PENDING_SUPERVISOR_APPROVAL` |
| **QC-06** | Stage 3 & 14: Scrubbing Proportion | Percentage of volumes with $\text{FD} > 0.5\text{ mm}$: $P_{\text{scrub}} = \frac{N_{\text{censored}}}{N_t} \times 100\%$ | **$\le 30\%$ of total run** | Guarantees sufficient remaining degrees of freedom ($N_{\text{eff}} \ge 120\text{ vols}$) for stable covariance estimation. | `PENDING_SUPERVISOR_APPROVAL` |
| **QC-07** | Stage 4: Functional SNR | Median Temporal SNR: $\text{tSNR}(v) = \frac{\mu_t(v)}{\sigma_t(v)}$ across brain voxels | **Median $\ge 40$<br>Flag $< 30$** | Asserts raw scanner data quality and RF coil integrity before signal processing. | `PENDING_SUPERVISOR_APPROVAL` |
| **QC-08** | Stage 3: DVARS Outliers | Temporal derivative of RMS variance: $\text{DVARS}_t = \sqrt{\frac{1}{V}\sum_v (\Delta I_{v, t})^2}$ | **No volumes $> 1.5 \times \text{median}(\text{DVARS})$** | Identifies BOLD signal intensity step-functions and RF spike artifacts. | `PENDING_SUPERVISOR_APPROVAL` |

---

## 4. Section C: Status Definitions & Formal Approval Block

### Status Definitions
- `PENDING_SUPERVISOR_APPROVAL`: Initial default state. Technical proposal complete; awaits supervisor evaluation.
- `APPROVED`: Formally signed off without modification. Becomes binding parameter.
- `REQUIRES_REVISION`: Supervisor requests modification to thresholds or algorithmic parameters.
- `REJECTED`: Decision option or threshold rejected; engineering team must supply alternative.

---

### Formal Supervisor Decision & Sign-Off Record

> [!IMPORTANT]
> **Declaration of Independence & Oversight:**
> By signing below, the Research Supervisor acknowledges review of the preprocessing specifications and QC acceptance criteria above. This signature certifies review of the methodological framework but **DOES NOT authorize live participant data processing** without concurrent Institutional Ethics Committee approval and signed Data Use Agreements.

**Supervisor Decision:**  
`[ ] ALL DECISIONS & THRESHOLDS APPROVED AS PROPOSED`  
`[ ] APPROVED WITH SPECIFIED MODIFICATIONS (SEE NOTES)`  
`[ ] REVISIONS REQUIRED PRIOR TO APPROVAL`  

**Supervisor Name:** `__________________________________________________`  
**Academic Title / Designation:** `__________________________________________________`  
**Department / Institution:** `__________________________________________________`  
**Date of Decision:** `YYYY-MM-DD`  

**Supervisor Signature:**  
`[ PENDING SUPERVISOR REVIEW AND SIGNATURE ]`

