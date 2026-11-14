# Step 6: MRI Defacing and Anonymization Design (`DEC-DEF-01`)

**Author:** NeuroADHD Development & Safety Engineering Team  
**Status:** Approved Design Specification  
**Milestone:** Step 6 — MRI Defacing and Anonymization  
**Repository:** `NeuroADHD_RawMRI-PipeLine`  
**Branch:** `feat/step6-defacing`  
**Execution Policy:** `STRICTLY_SYNTHETIC_ONLY` (`real_data_authorized = false`)

---

## 1. Executive Summary & Safety Invariants

Neuroimaging datasets containing high-resolution 3D anatomical magnetic resonance imaging (MRI) scans pose a recognized patient re-identification risk. Modern surface-rendering algorithms can reconstruct recognizable facial geometry (nose, lips, eyes, chin, and jawline) from high-resolution T1-weighted (T1w) scans, enabling automated or manual visual matching against reference photographs.

This document establishes the formal design for **Step 6: MRI Defacing and Metadata De-Identification (`DEC-DEF-01`)** within the NeuroADHD processing pipeline.

### Core Architectural Guarantees:
1. **Zero External Binary Dependencies:** Implements a pure-Python, deterministic Quickshear-style defacing algorithm utilizing NumPy and SciPy, eliminating external binary dependencies on unavailable tools (FSL `flirt`/`pydeface` and FreeSurfer `mri_deface`).
2. **Brain Anatomy Bit-Identical Preservation:** Brain tissue voxels identified by the anatomical brain mask are guaranteed bit-identical between the input and defaced T1w volumes.
3. **Fail-Closed Privacy Boundary:** If defacing fails or encounters invalid input, execution halts immediately with a quarantine report. Zero outputs containing facial structures are persisted.
4. **Leak-Proof QC Pipeline:** All downstream quality control images (PNGs) and derivative visual figures are rendered exclusively from defaced T1w volumes.
5. **Regulatory Invariant:** Step 6 does not authorize real human data processing; real data processing remains strictly locked (`overall_authorization_state = NOT_AUTHORIZED`).

---

## 2. Threat Model: Facial Re-Identification in Neuroimaging

### 2.1 Identifiable Features in T1-Weighted Anatomical Scans
High-resolution 3D T1w scans (typically acquired at $1.0\text{ mm}$ isotropic voxel resolution) capture soft tissue and external skin surface anatomy with high fidelity. The identifiable features include:
- **Nose Surface & Bridge:** The nasal cartilage, septum, and skin contour form a distinct profile highly characteristic of individual facial identity.
- **Orbital & Periorbital Regions:** Eye sockets, globes, and eyebrow ridges.
- **Mouth, Lips, and Chin:** Mandibular contours and soft tissue thickness.
- **Dentition and Jawbone:** High-contrast dental enamel and mandibular geometry.
- **Ears and Facial Cheeks:** External pinna structure and cheek contours.

Three-dimensional isosurface extraction algorithms (e.g. marching cubes) applied to un-defaced T1w scans can generate textured 3D polygon meshes that match photographic identification records with $> 80\%$ automated recognition accuracy.

### 2.2 Functional BOLD Scans Are Non-Identifiable
In contrast to T1w anatomical scans, 4D functional blood-oxygen-level-dependent (BOLD) scans do **not** pose a facial re-identification threat:
- **Low Spatial Resolution:** BOLD acquisitions typically use $3.0\text{–}4.0\text{ mm}$ isotropic voxel grids, smoothing out fine facial contours.
- **Susceptibility Artifacts & Field Distortions:** The air-tissue interfaces in the nasal cavity, frontal sinuses, and oral cavity create severe magnetic susceptibility field inhomogeneities, causing signal loss (dropout) and severe spatial distortion in anterior regions.
- **Restricted Functional Field of View (FOV):** Functional acquisitions frequently truncate the inferior facial soft tissue (lips, nose tip, chin) to maximize brain coverage within the required repetition time ($T_R \approx 2.0\text{ s}$).

Therefore, facial defacing applies strictly to 3D T1w anatomical scans, while BOLD scans require temporal slice-timing correction (`DEC-STC-01`), motion correction, and susceptibility distortion correction (`DEC-SDC-02`).

### 2.3 Quality Control PNGs as a High-Risk Leak Vector
A critical vulnerability in neuroimaging pipelines is the inadvertent leak of anatomical facial features via 2D quality control (QC) summary images. Pipelines that deface the 3D NIfTI volume but render QC PNGs from the raw input scan, or render sagittal slices extending beyond the brain into the facial profile, leak the patient's face in unencrypted web dashboards and static HTML reports.

**Design Requirement:**
- Every QC image displaying anatomical anatomy (orthogonal slice views, coregistration contour overlays, and tissue segmentation maps) must be generated strictly from the **defaced** T1w volume.
- Under no circumstance may raw anatomical facial features be plotted, rendered, or cached.

---

## 3. Tool Evaluation: Quickshear vs. pydeface / mri_deface

| Criterion | Quickshear (Pure Python) | pydeface (FSL Wrapper) | mri_deface (FreeSurfer) |
| :--- | :--- | :--- | :--- |
| **Language & Dependencies** | Pure Python (`numpy`, `scipy`, `nibabel`) | Python script calling FSL binary `flirt` | C binary requiring FreeSurfer environment |
| **Availability on Host/CI** | **100% Available** (Standard Scientific Python) | **UNAVAILABLE** (FSL not installed; `FSLDIR` unset) | **UNAVAILABLE** (FreeSurfer not installed) |
| **External Model / Template** | None (Uses subject's own brain mask) | Requires `defacing_mask.nii.gz` template | Requires `face.gca` / `talairach.gca` (~10 GB) |
| **Computational Footprint** | $< 1.5\text{ seconds}$, $< 200\text{ MB}$ RAM | $\approx 45\text{ seconds}$, requires affine registration | $\approx 60\text{ seconds}$, requires FreeSurfer setup |
| **Brain Tissue Preservation** | Mathematical guarantee: cutting plane is exterior to brain mask | Warped template mask can clip frontal cortex on atypical heads | Conservative template, occasional frontal clipping |
| **Robustness on Phantoms** | High; adapts to any geometry with valid brain mask | Fails on synthetic phantoms lacking standard MNI geometry | Fails on synthetic phantoms without Talairach alignment |

### Recommendation
**Quickshear-style geometric shearing is the chosen algorithm for NeuroADHD.** It is mathematically robust, deterministic, has zero external binary dependencies, runs instantaneously in containerized CI environments, and guarantees that brain tissue within the extracted brain mask remains completely untouched.

### Quickshear Algorithmic Specification:
1. **Inputs:** 3D T1w anatomical volume $V_{\text{T1w}} \in \mathbb{R}^{X \times Y \times Z}$ and binary brain mask $M_{\text{brain}} \in \{0, 1\}^{X \times Y \times Z}$.
2. **Mid-Sagittal Brain Profile:** Determine the mid-sagittal slice $x_0$ of the brain mask (or average across central sagittal slices).
3. **Convex Boundary & Tangent Plane:**
   - Locate the most anterior voxel of the brain mask: $(y_{\text{front}}, z_{\text{front}})$.
   - Locate the most ventral-anterior voxel of the frontal lobe: $(y_{\text{ventral}}, z_{\text{ventral}})$.
   - Compute the 2D cutting line in the sagittal plane $(y, z)$ passing through a standoff point exterior to the brain mask (with protective safety margin $\delta = 2\text{ voxels}$):
     $$a \cdot y + b \cdot z + c = 0$$
     where the normal vector $(a, b)$ points toward the anterior-inferior facial quadrant.
4. **3D Cutting Half-Space:** Extend the 2D cutting line across all sagittal slices $x \in [0, X-1]$ to define the 3D half-space equation:
   $$\mathcal{H}_{\text{face}} = \{(x, y, z) \mid a \cdot y + b \cdot z + c > 0\}$$
5. **Facial Masking:** Set all voxels in $\mathcal{H}_{\text{face}}$ to zero:
   $$V_{\text{defaced}}(x, y, z) = \begin{cases} 0 & \text{if } (x, y, z) \in \mathcal{H}_{\text{face}} \\ V_{\text{T1w}}(x, y, z) & \text{otherwise} \end{cases}$$
6. **Bit-Identical Invariant Check:** Verify that:
   $$\forall (x, y, z) \text{ where } M_{\text{brain}}(x, y, z) == 1, \quad V_{\text{defaced}}(x, y, z) \equiv V_{\text{T1w}}(x, y, z)$$

---

## 4. Pipeline Stage Position & Integration Architecture

### 4.1 Stage Ordering (From 22 to 23 Stages)
To maintain complete scientific and physical integrity, defacing is integrated as **Stage 8** of the 23-stage pipeline:

```
 [Stage 1]  BIDS Input & Sidecar Metadata Validation
    ↓
 [Stage 2]  3D Rigid Motion Estimation (Raw BOLD; FD calculation)
    ↓
 [Stage 3]  Metadata-Driven Slice-Timing Correction (DEC-STC-01; unresampled BOLD)
    ↓
 [Stage 4]  B0 Susceptibility Distortion Correction (DEC-SDC-02; fieldmap estimation)
    ↓
 [Stage 5]  Motion (+ Optional SDC) Transform Application (Native reference alignment)
    ↓
 [Stage 6]  3D Functional Reference Generation (Mean EPI & EPI brain mask)
    ↓
 [Stage 7]  T1w Anatomical Brain Extraction (median_otsu -> t1w_brain_mask)
    ↓
 [Stage 8]  NEW: Anatomical MRI Defacing & Metadata Sanitization (DEC-DEF-01)
    ↓       - Quickshear facial removal using Stage 7 brain mask
    ↓       - Header & sidecar privacy sanitization
    ↓       - Emits derivatives/defaced/*_desc-defaced_T1w.nii.gz
    ↓
 [Stage 9]  T1w Bias-Field Correction (Order-2 Legendre polynomial on defaced T1w)
    ↓
 [Stage 10] 3-Class GMM Tissue Segmentation (GM/WM/CSF segmentation on defaced T1w)
    ↓
 [Stage 11] Eroded WM/CSF Mask Generation (WM & CSF nuisance masks)
    ↓
 [Stage 12] EPI-to-T1w Rigid Coregistration (Aligns EPI to defaced T1w)
    ↓
 [Stage 13] T1w-to-MNI SyN Nonlinear Normalization (Normalizes defaced T1w to MNI152)
    ↓
 [Stage 14] Single-Step Composed Pullback Resampling (Motion + SDC + Coreg + SyN)
    ↓
 [Stage 15] Standard-Space Anatomical Mask Resampling
    ↓
 [Stage 16] Anatomical aCompCor Confounds Extraction (5 WM + 5 CSF components)
    ↓
 [Stage 17] Friston-24 Motion & Polynomial Detrending Confounds Assembly (31 regressors)
    ↓
 [Stage 18] Simultaneous OLS Nuisance Regression & Butterworth Bandpass (GSR OFF)
    ↓
 [Stage 19] CC200 Atlas Parcellation (roi_CC200.csv: 200 ROIs x N timepoints)
    ↓
 [Stage 20] Fisher-z Functional Connectivity Matrix Extraction (19,900 features)
    ↓
 [Stage 21] Orthogonal Quality Control Overlays (Using defaced T1w only)
    ↓
 [Stage 22] Output Manifest & Checksum Generation (output_manifest.json with correction_status & defacing_status)
    ↓
 [Stage 23] Schema-Compliant Provenance Generation (provenance.json with 23-stage audit)
```

### 4.2 Justification for Stage 8 Positioning:
1. **Dependency on Brain Extraction (Stage 7):** Quickshear relies upon an accurate binary brain mask to guarantee that cortical tissue is never clipped. Placing defacing immediately after Stage 7 satisfies this prerequisite directly.
2. **Precedence over Downstream Processing (Stages 9–21):** By defacing before bias correction, segmentation, coregistration, and normalization, the entire downstream workflow operates on privacy-compliant imagery. Crucially, Stage 21 (QC image generation) renders exclusively from the defaced T1w image.
3. **Decoupled from BOLD Processing (Stages 2–6):** The functional BOLD stages operate independently on BOLD data and are unaffected by anatomical defacing.

---

## 5. Header Sanitization & Metadata De-Identification

### 5.1 NIfTI-1 Header Sanitization
NIfTI-1 binary headers contain legacy 80-character and 24-character string fields often populated by DICOM-to-NIfTI converters with sensitive metadata:
- `descrip`: Often contains scanner serial numbers, institutional protocols, operator initials, or study dates. Cleared to `"NeuroADHD_Deidentified"`.
- `aux_file`: Cleared to empty string `""`.
- `intent_name`: Cleared to `""` unless an explicit scientific intent is registered.
- **Affine and Geometry Invariance:** Voxel dimensions (`pixdim`), spatial transform matrices (`qform`, `sform`), and spatial codes (`qform_code`, `sform_code`) must be preserved identically.

### 5.2 BIDS JSON Sidecar Sanitization
- **Permitted Scientific Fields (Whitelisted):**
  `RepetitionTime`, `EchoTime`, `FlipAngle`, `MagneticFieldStrength`, `Manufacturer`, `ManufacturersModelName`, `PhaseEncodingDirection`, `TotalReadoutTime`, `SliceTiming`, `EffectiveEchoSpacing`.
- **Prohibited Direct Identifiers (Stripped):**
  `PatientName`, `PatientID`, `PatientBirthDate`, `PatientSex`, `PatientAge`, `PatientWeight`, `InstitutionName`, `InstitutionAddress`, `InstitutionalDepartmentName`, `ReferringPhysicianName`, `PerformingPhysicianName`, `OperatorsName`, `DeviceSerialNumber`, `AccessionNumber`.
- **Free-Text Fields:** Any free-text or unstructured comment fields (`PatientComments`, `StudyComments`, `ImageComments`) are stripped fail-closed.

---

## 6. Fail-Closed Policy & Quarantine Protocol

1. **Defacing Verification Checks:**
   - **Brain Mask Safety Check:** The brain mask must contain $> 0$ voxels and the cutting plane must not intersect any voxel where $M_{\text{brain}}(x, y, z) == 1$.
   - **Facial Removal Check:** At least one voxel in the facial region must be removed; if the input volume already had zero facial voxels, the operation logs an idempotency notice.
   - **Bit-Identical Brain Check:** $\max |V_{\text{defaced}} \cdot M_{\text{brain}} - V_{\text{T1w}} \cdot M_{\text{brain}}| == 0$.
2. **Failure Handling:**
   - If any verification check fails, or if an unexpected exception occurs, execution aborts immediately.
   - A structured quarantine record (`defacing_quarantine_report.json`) is emitted describing the exact failure cause.
   - No un-defaced T1w file is copied or linked into derivatives or outputs.
3. **Strict Mode Enforcement (`--strict-defacing`):**
   - In synthetic exploratory mode, missing defacing dependencies can emit a skip flag if configured.
   - In real-data mode, `--strict-defacing` is **mandatory**. Any skip or defacing anomaly raises `DefacingError` and terminates the run.

---

## 7. Explicit Non-Guarantees & Scientific Boundaries

1. **Non-Anatomical Records:** Defacing does not de-identify genomic, demographic, retinal, or dental records.
2. **Skull Morphometry:** Defacing removes facial soft tissue (eyes, nose, mouth, chin) but does not alter cranial vault shape; re-identification through specialized skull-geometry matching remains theoretically possible.
3. **Authorization Boundary:** Successfully defacing a synthetic or real scan does **not** grant authorization for real human data processing. Real data processing remains governed strictly by institutional ethics approvals, data use agreements, and cryptographic signatures in `configs/research_authorization_register.json`.

---

## 8. Idempotency Guarantee

Defacing an already-defaced T1w scan is strictly **idempotent**:
- The cutting plane calculation on a previously defaced image with the same brain mask yields the identical cutting plane.
- Voxels anterior to the plane are already zero, so applying defacing a second time results in bitwise-identical output:
  $$\text{Deface}(\text{Deface}(V, M), M) \equiv \text{Deface}(V, M)$$
- Metadata records the defacing provenance tag (`desc-defaced`) and prevents redundant re-computation.

