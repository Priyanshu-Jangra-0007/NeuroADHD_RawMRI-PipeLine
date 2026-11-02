# PHASE A.4: REAL MRI INPUT CONTRACT & BIDS SPECIFICATION

**Document Version:** 1.0.0  
**Phase:** Phase A.4 — Pre-Pilot Readiness Gate  
**Status:** DRAFT — PENDING SUPERVISOR REVIEW & AUTHORIZATION  
**Execution Environment:** Strictly Synthetic-Only (Real scans disabled until formal gate approval)

---

## 1. Scope & Objective

This document establishes the authoritative input specification and acceptance boundary for resting-state functional MRI (rs-fMRI) and structural T1-weighted (T1w) anatomical MRI scans intended for downstream ingestion by the **NeuroADHD RawMRI Preprocessing Pipeline**.

The pipeline transforms raw DICOM-converted NIfTI images into Craddock 200 (CC200) functional connectivity matrices and 19,900 Fisher-$z$ transformed edge features compatible with the frozen downstream machine-learning classifier (`final_model.joblib`). To guarantee scientific validity, numerical reproducibility, and algorithmic stability, all candidate real MRI scans must strictly adhere to the standards outlined herein.

> [!IMPORTANT]
> Real MRI execution remains strictly prohibited during Phase A.4. This document serves as the regulatory and technical specification to evaluate candidate datasets prior to requesting institutional and supervisory authorization.

---

## 2. Dataset Architecture & BIDS Standard Compliance

All prospective input scans must be formatted according to the **Brain Imaging Data Structure (BIDS)** specification (v1.8.0 or higher).

### 2.1 Directory Structure & File Hierarchy
The input directory must follow standard subject/session BIDS hierarchy:

```text
bids_root/
├── sub-<subject_id>/
│   ├── [ses-<session_id>/]
│   │   ├── anat/
│   │   │   ├── sub-<subject_id>[_ses-<session_id>]_T1w.nii.gz
│   │   │   └── sub-<subject_id>[_ses-<session_id>]_T1w.json
│   │   └── func/
│   │       ├── sub-<subject_id>[_ses-<session_id>]_task-rest_bold.nii.gz
│   │       └── sub-<subject_id>[_ses-<session_id>]_task-rest_bold.json
```

---

## 3. Mandatory vs Optional Input Specifications

| Field / Asset | Modality | Status | Required Value / Type | Validation Rule |
| :--- | :--- | :--- | :--- | :--- |
| **BOLD NIfTI** | Functional | **Mandatory** | 4D NIfTI (`.nii` or `.nii.gz`) | Finite 4D tensor, $T \ge 50$ volumes |
| **BOLD JSON** | Functional | **Mandatory** | JSON Sidecar | Valid JSON containing `RepetitionTime` |
| **T1w NIfTI** | Anatomical | **Mandatory** | 3D NIfTI (`.nii` or `.nii.gz`) | Finite 3D tensor, whole-brain coverage |
| **T1w JSON** | Anatomical | **Optional** | JSON Sidecar | Scanner metadata (e.g., TI, TE, FlipAngle) |
| **Phase Encoding** | Functional | **Optional** | Metadata (`PhaseEncodingDirection`) | Distortion correction parameter |
| **Fieldmap (FMAP)**| Fieldmap | **Optional** | 3D/4D B0 map or dual-echo PE | Reserved for future susceptibility unwarping |
| **Physio Data** | Physiological | **Optional** | BIDS `.tsv.gz` cardiac/respiratory | Currently not ingested by PoC V2 |

---

## 4. Functional BOLD Input Contract (`task-rest_bold`)

### 4.1 Spatial & Grid Requirements
1. **Dimensionality**: Strictly 4D tensor ($X \times Y \times Z \times T$).
2. **Voxel Size**: Isotropic or near-isotropic acquisition:
   - In-plane resolution ($\Delta x, \Delta y$): Between $2.0\text{ mm}$ and $4.0\text{ mm}$.
   - Slice thickness ($\Delta z$): Between $2.0\text{ mm}$ and $4.5\text{ mm}$.
   - Slices must span whole brain (minimum $Z \ge 25$ slices).
3. **Orientation**: Canonical neurological (RAS) or radiological (LPI/RPI) affine representation.
   - Coordinate transform $S_{\text{form}}$ or $Q_{\text{form}}$ must be valid (code 1, 2, or 4).
   - Oblique acquisitions $> 15^\circ$ require header de-obliquing or strict validation during coregistration.

### 4.2 Temporal Requirements
1. **Repetition Time ($T_R$)**:
   - $T_R$ must be explicitly defined in seconds in `sub-*_bold.json` under key `"RepetitionTime"`.
   - Permissible range: $0.5\text{ s} \le T_R \le 3.5\text{ s}$.
   - Header `pixdim[4]` must match JSON `RepetitionTime` within $\pm 0.001\text{ s}$.
2. **Scan Length ($T$)**:
   - Minimum volume count: $T \ge 50$ timepoints (equivalent to $\ge 100\text{ s}$ at $T_R = 2.0\text{ s}$).
   - Recommended resting-state duration: $T \ge 120$ timepoints ($4\text{ to }10\text{ minutes}$).
   - Scans with $T < 50$ are immediately rejected due to unstable sample covariance estimation.

### 4.3 Signal Integrity
1. **Finite Values**: Zero NaN, Inf, or null values within the spatial bounding box.
2. **Zero-Variance / Dropout**: No constant or dead voxel columns within intracranial volume.
3. **Artifact / Motion Ceilings**:
   - Mean Framewise Displacement (FD): $\text{Mean FD} < 0.50\text{ mm}$.
   - Scans exceeding $30\%$ scrubbed volumes ($\text{FD} > 0.5\text{ mm}$) are flagged for rejection.

---

## 5. Anatomical T1w Input Contract (`_T1w`)

### 5.1 Spatial & Structural Requirements
1. **Dimensionality**: Strictly 3D tensor ($X \times Y \times Z$).
2. **Resolution**: High-resolution structural acquisition:
   - Voxel spacing: $\le 1.2\text{ mm}$ isotropic (nominal $1.0\text{ mm}$ isotropic preferred).
   - Field of view (FOV) must encompass full cerebrum, cerebellum, and brainstem.
3. **Contrast & Tissue Differentiation**:
   - High gray-matter / white-matter / cerebrospinal fluid contrast-to-noise ratio (CNR).
   - Defaced or anonymized facial features must not clip the frontal pole or anterior skull.

---

## 6. Geometric Alignment & Affine Verification

Every candidate NIfTI image must pass rigorous spatial geometry verification:

```python
# Formal Geometry Verification Rules
assert img.header.get_sform(coded=True)[1] in (1, 2, 4) or img.header.get_qform(coded=True)[1] in (1, 2, 4), "Uncoded affine geometry"
assert np.linalg.det(img.affine[:3, :3]) != 0.0, "Singular / non-invertible affine"
assert not np.isnan(img.affine).any(), "NaN values detected in coordinate affine matrix"
```

1. **Affine Invertibility**: $\det(A) \ne 0$; matrix condition number $\kappa(A) < 10^5$.
2. **Voxel Coordinate Frame**: Orthogonal or mildly sheared grid compatible with DIPY/NiBabel coordinate pullback.
3. **Target Standard Space Alignment**:
   - Resampled canonical BOLD output must match CC200 template:
     - Dimensions: $(63, 75, 61)$
     - Voxel Spacing: $(3.0, 3.0, 3.0)\text{ mm}$
     - Affine Diagonal: $[-3.0, 3.0, 3.0]$
     - Offsets: $[93.0, -129.0, -72.0]$

---

## 7. Rejection Criteria Matrix (Fail-Closed Gates)

A scan session must be rejected with an explicit error code under any of the following conditions:

| Category | Condition for Rejection | Error Code | Pipeline Action |
| :--- | :--- | :--- | :--- |
| **Missing Asset** | Missing functional BOLD NIfTI | `ERR_INPUT_BOLD_MISSING` | Fail closed immediately |
| **Missing Asset** | Missing anatomical T1w NIfTI | `ERR_INPUT_T1W_MISSING` | Fail closed immediately |
| **Missing Metadata** | Missing or non-positive `RepetitionTime` in BOLD JSON | `ERR_METADATA_TR_INVALID` | Fail closed immediately |
| **Temporal Duration**| BOLD scan duration $T < 50$ volumes | `ERR_BOLD_DURATION_TOO_SHORT`| Fail closed immediately |
| **Spatial Anomaly** | 4D BOLD spatial grid mismatch across volumes | `ERR_BOLD_SHAPE_INCONSISTENT` | Fail closed immediately |
| **Geometry Failure** | Singular, corrupted, or non-finite NIfTI affine matrix | `ERR_AFFINE_SINGULAR` | Fail closed immediately |
| **Corrupted Signal** | In-brain NaNs, Infs, or global intensity scale $= 0$ | `ERR_SIGNAL_NON_FINITE` | Fail closed immediately |
| **Brain Extraction** | Skull-stripping failure ($\text{Brain Volume} < 600\text{ cm}^3$ or $> 2200\text{ cm}^3$)| `ERR_BET_VOLUME_ANOMALY` | Abort; quarantine staging |
| **Registration** | Coregistration mutual information below threshold | `ERR_REGISTRATION_COREG_POOR` | Abort; quarantine staging |
| **Parcellation** | $\ge 1$ CC200 ROI has zero voxel overlap or constant signal | `ERR_PARCELLATION_ZERO_VARIANCE` | Reject; fail Gate 5 |
| **Motion Excess** | Gross head movement ($> 5.0\text{ mm}$ translation or $> 5.0^\circ$ rotation) | `ERR_EXCESSIVE_HEAD_MOTION` | Reject; fail Gate 2 |
| **Governance** | Unsigned research consent or unauthorized subject identifier | `ERR_GOVERNANCE_UNAUTHORIZED` | Abort execution; security audit |

---

## 8. Supervisor Review & Open Technical Questions

> [!NOTE]
> The following technical questions remain pending supervisor approval before the real-data gate can be opened:

1. **Slice-Timing Correction**:
   - *Status*: `PENDING_SUPERVISOR_APPROVAL`
   - *Question*: For multi-band vs interleaved axial acquisitions, should slice-timing correction be enabled if `SliceTiming` is populated in BIDS JSON, or should it remain bypassed to preserve exact C-PAC benchmark equivalence?
2. **Susceptibility Distortion Correction (SDC)**:
   - *Status*: `PENDING_SUPERVISOR_APPROVAL`
   - *Question*: Candidate pilot sites may lack reverse phase-encoded fieldmaps. Is non-SDC preprocessing approved for the initial exploratory pilot?
3. **Defacing Impact on Skull Stripping**:
   - *Status*: `PENDING_SUPERVISOR_APPROVAL`
   - *Question*: Are facial-defaced T1w scans permitted, provided skull-stripping masks are independently verified not to crop the orbitofrontal cortex?

