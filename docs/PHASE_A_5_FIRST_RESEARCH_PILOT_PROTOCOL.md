# NeuroADHD RawMRI PoC V2 — Phase A.5
# First Research Pilot Protocol: Real MRI Technical Feasibility

**Document ID:** `DOC-NA-A5-FRP-001`  
**Version:** 1.0.0  
**Status:** `READY_FOR_SUPERVISOR_DECISION`  
**Repository:** `Priyanshu-Jangra-0007/NeuroADHD_RawMRI-PipeLine`  
**Branch:** `main`  
**Execution Scope:** Controlled Retrospective Pilot ($N=5\text{--}10$) [Subject to Ethics Approval]  

---

## 1. Study Objective & Scientific Rationale

The primary objective of the **First Research Pilot** is to demonstrate the **technical and numerical feasibility** of the containerized NeuroADHD RawMRI PoC V2 pipeline when processing real human magnetic resonance imaging (MRI) scans.

To date, the pipeline has been extensively verified on mathematically generated synthetic phantoms (Phases A.1 through A.4.2.1). However, real human MRI data presents biological complexities not modeled in synthetic phantoms:
- True anatomical morphological variability (cortical folding, ventricular dilation, brain size variance).
- Realistic subject motion (head nodding, swallowing artifacts, sudden drift).
- Complex scanner physics (B0 susceptibility distortions near sinuses, B1 radiofrequency transmit/receive coil profile inhomogeneities).
- Physiological noise (respiratory and cardiac cycles).

> [!IMPORTANT]
> **Strict Feasibility Scope (Non-Clinical):**
> This pilot is **strictly an engineering and computational feasibility assessment**. It is **not** a clinical diagnostic trial, nor is it statistically powered to determine diagnostic sensitivity or specificity. Downstream machine learning predictions will be evaluated purely for mathematical stability and pipeline continuity.

---

## 2. Pilot Cohort Definition & Sample Size Determination

### Proposed Pilot Cohort Structure
- **Dataset Source:** Retrospective, de-identified scans from the **ADHD-200 Consortium** public repository (e.g., New York University Child Study Center or Peking University site).
- **Proposed Sample Size:** $N = 6$ participants
  - 3 Typically Developing Controls (TDC)
  - 3 ADHD diagnosed participants (ADHD-Combined or Inattentive)
  - Balanced for sex and age range ($8\text{--}14\text{ years}$).
- **Inclusion Criteria:**
  - High-resolution 3D T1w MP-RAGE ($1\text{ mm}$ isotropic nominal resolution).
  - Resting-state BOLD fMRI ($TR \approx 2.0\text{ s}$, duration $\ge 5\text{ minutes}$, $\ge 150\text{ volumes}$).
  - Complete demographic and scanner sidecar metadata.
- **Exclusion Criteria:**
  - Severe gross motion artifacts identifiable on raw visual inspection ($\text{FD}_{\max} > 5.0\text{ mm}$).
  - Corrupted or truncated NIfTI volumes.

### Formal Sample Size & Power Calculation Status
Because this is a technical feasibility pilot designed to evaluate computational stability and memory limits:
- **Statistical Power Calculation:** Not applicable for diagnostic efficacy at $N=6$.
- **Sample Size Final Approval:**  
  $$\mathbf{PENDING\_SUPERVISOR\_APPROVAL}$$
  *(The supervisor may expand or contract the pilot cohort size between $N=3$ and $N=15$ based on available computing resources and institutional approvals).*

---

## 3. Ten-Stage Pilot Execution Protocol

```
+-----------------------------------------------------------------------------------+
|               FIRST RESEARCH PILOT: 10-STAGE EXECUTION WORKFLOW                   |
+-----------------------------------------------------------------------------------+
  [Stage 1]  Institutional Clearances Verified (IEC Approval + DUA Active)
       │
       ▼
  [Stage 2]  Intake Authorization & Cryptographic Manifest Generation
       │
       ▼
  [Stage 3]  Secure Execution Environment Deployment (Docker 2.5 GB Limit, Offline)
       │
       ▼
  [Stage 4]  Facial Defacing Execution & Frontal-Pole Preservation Audit
       │
       ▼
  [Stage 5]  BIDS Schema & Structural Integrity Validation
       │
       ▼
  [Stage 6]  Controlled Neuroimaging Preprocessing (Stages 1-14 Execution)
       │
       ▼
  [Stage 7]  Comprehensive QC Evaluation Against Approved 8-Point Thresholds
       │
       ▼
  [Stage 8]  Parcellation & CC200 Functional Connectivity Feature Extraction
       │
       ▼
  [Stage 9]  Downstream Frozen ML Model Inference (Read-Only joblib Execution)
       │
       ▼
  [Stage 10] Technical Post-Pilot Audit & Formal Supervisor Reporting
+-----------------------------------------------------------------------------------+
```

### Detailed Stage Breakdown:

#### Stage 1: Authorization & Governance Verification
- Verify that Institutional Ethics Committee (IEC) clearance is formally documented.
- Verify that the Data Use Agreement (DUA) with NITRC/INDI is active and covers all investigators.
- Verify that supervisor sign-off on the pilot protocol is executed.
- *Gate Check:* If any document is missing, execution is aborted immediately.

#### Stage 2: Intake Approval & Manifest Checksum Verification
- Stage raw NIfTI files into a designated, secure intake volume (`/data/intake/`).
- Generate `manifest.json` containing SHA-256 hashes of all anatomical and functional files.
- Sign manifest using `NEUROADHD_HMAC_KEY` via `scripts/run_poc_v2.py` / `scripts/run_synthetic_cohort.py` (`scripts/generate_manifest.py` is not present in this repository; manifest creation and HMAC signing are built inline).

#### Stage 3: Secure Environment Deployment
- Spin up the verified Docker container (`neuro_adhd_pipeline:v2`).
- Enforce strict kernel limits: `--memory=2560m`, `--network=none` (strictly offline execution).
- Run as unprivileged user `neuro` (`UID 10001`).
- Mount intake and model directories as read-only (`:ro`).

#### Stage 4: Facial Defacing & Privacy Verification
- Execute automated defacing on all structural T1w volumes using approved defacing tooling.
- Execute automated spatial audit: assert zero overlap between defacing mask and brain mask.
- Perform visual inspection of 3D orthogonal slices to confirm elimination of facial features and complete preservation of frontal pole tissue.

#### Stage 5: BIDS Validation & Structural Integrity
- Run BIDS validation check confirming subject folder hierarchy (`sub-<ID>/anat/`, `sub-<ID>/func/`).
- Validate JSON sidecars for repetition time, echo time, and phase encoding direction.
- Verify image matrix dimensions and voxel zooms against scanner acquisition protocols.

#### Stage 6: Controlled Preprocessing Execution
- Execute Stages 1 through 14 of the pipeline sequentially:
  - BOLD motion realignment and reference volume generation.
  - T1w skull stripping, N4 bias field correction, and 3-class tissue segmentation.
  - Rigid EPI-to-T1w coregistration.
  - Non-linear SyN T1w-to-MNI spatial normalization.
  - Pullback resampling of 4D BOLD into canonical CC200 space ($63 \times 75 \times 61$, 3.0 mm).
  - aCompCor (5 CSF + 5 WM components) and Friston-24 motion regression.
  - Simultaneous bandpass filtering ($0.01\text{--}0.10\text{ Hz}$).

#### Stage 7: Comprehensive Quality Control Evaluation
- Quantify all 8 QC metrics for each pilot participant:
  1. T1w brain volume ($600\text{--}2200\text{ cm}^3$)
  2. EPI-to-T1w coregistration NMI ($\ge 0.55$)
  3. T1w-to-MNI SyN Dice ($\ge 0.85$)
  4. Mean Framewise Displacement ($\le 0.50\text{ mm}$)
  5. Maximum Motion Spike ($\le 3.0\text{ mm}$; flag $>3.0\text{ mm}$, reject $>5.0\text{ mm}$)
  6. Scrubbing proportion ($\le 30\%$)
  7. Temporal SNR (median $\ge 40$)
  8. DVARS outlier count ($>1.5 \times \text{median}$)
- Generate automated PDF/HTML visual summary sheets with orthogonal registration overlays.

#### Stage 8: CC200 Parcellation & Feature Extraction
- Verify canonical CC200 atlas SHA-256 checksum (`9467afce...`).
- Extract mean time-series across all 200 ROIs.
- Compute $200 \times 200$ Pearson correlation matrix.
- Apply Fisher-$z$ transformation.
- Extract upper-triangular feature vector ($1 \times 19{,}900$ float64).

#### Stage 9: Downstream Frozen ML Model Inference
- Mount authentic frozen model artifacts:
  - `final_preprocessing_pipeline.joblib` (SHA-256: `228d3f01...`)
  - `final_model.joblib` (SHA-256: `d8eb23eb...`)
- Verify checksums before deserialization.
- Transform features through frozen `SimpleImputer`, `VarianceThreshold`, `StandardScaler`, and `SelectKBest(k=100)`.
- Execute `RandomForestClassifier.predict_proba()` to compute exploratory class probabilities.
- Store results in structured JSON summary.

#### Stage 10: Technical Post-Pilot Audit & Reporting
- Audit container logs, peak memory usage, and execution timing.
- Compile comprehensive Pilot Execution Report.
- Submit report to Research Supervisor for evaluation.

---

## 4. Feasibility Success Criteria

The pilot will be deemed successful if the following technical benchmarks are met across the cohort:

| Evaluation Domain | Feasibility Metric | Success Threshold |
|---|---|---|
| **System Stability** | Pipeline crash rate | **0% unhandled exceptions** across all scans |
| **Memory Isolation** | Peak container RAM usage | **$< 2,300\text{ MiB}$** (within 2,560 MiB limit; zero OOM kills) |
| **Registration Fidelity** | T1w-to-MNI SyN Dice overlap | **$\ge 0.85$** for all subjects |
| **Registration Fidelity** | EPI-to-T1w Coregistration NMI | **$\ge 0.55$** for all subjects |
| **Feature Extraction** | Connectivity matrix validity | **100% valid** (19,900 finite values, no NaN/Inf, symmetric) |
| **Model Compatibility** | Frozen model execution | **100% valid execution** without deserialization errors or retraining |

> [!NOTE]
> **Diagnostic Accuracy Disclaimer:**
> Classification accuracy on the pilot cohort is explicitly **excluded** from the feasibility success criteria. A cohort of $N=6$ is statistically incapable of establishing diagnostic sensitivity or specificity.

---

## 5. Risk Mitigation & Stop-Work Triggers

Execution of the pilot will be halted immediately under any of the following stop-work conditions:

1. **Kernel OOM Termination (Exit code 137):** Any container termination by the Linux OOM killer indicates memory leakage or inadequate chunking; processing stops until code optimization is reviewed.
2. **Registration Divergence ($\text{Dice} < 0.70$):** Gross spatial normalization failure indicates severe anatomical abnormality or corrupted input geometry.
3. **Facial Defacing Failure:** Detection of any residual facial recognizable geometry or accidental clipping of frontal cortical tissue halts the pipeline.
4. **Data Integrity Violation:** Any hash mismatch on inputs, atlases, or frozen model binaries triggers immediate fail-closed termination.
5. **Governance Incident:** Any indication that un-anonymized metadata is present halts all processing immediately.

