# DATA INPUT SPECIFICATION: BIDS & RESTING-STATE fMRI

## 1. Overview & Input Scope

To ensure reproducibility and adherence to neuroimaging standards, the raw preprocessing pipeline accepts resting-state functional MRI data organized according to the **Brain Imaging Data Structure (BIDS)** specification.

Structural MRI (T1w) alone is **NOT** interchangeable with resting-state fMRI. The downstream classifier requires temporal BOLD signal fluctuations; a T1w scan may only be used as an anatomical reference for spatial normalization, never as a substitute for functional data.

---

## 2. Directory Layout & File Hierarchy

A valid subject input folder must follow BIDS naming conventions:

```
sub-<subject_id>/
├── [ses-<session_id>/]
│   ├── anat/
│   │   ├── sub-<subject_id>[_ses-<session_id>]_T1w.nii[.gz]         # Highly recommended for BBR
│   │   └── sub-<subject_id>[_ses-<session_id>]_T1w.json             # Optional metadata
│   └── func/
│       ├── sub-<subject_id>[_ses-<session_id>]_task-rest_bold.nii[.gz] # MANDATORY 4D BOLD
│       └── sub-<subject_id>[_ses-<session_id>]_task-rest_bold.json     # MANDATORY Sidecar
```

### 2.1 File Role & Requirement Matrix

| File Type | Pattern | Required? | Purpose |
|---|---|---|---|
| **Resting-State BOLD** | `*task-rest*_bold.nii[.gz]` | **MANDATORY** | 4D timeseries containing resting-state BOLD signal |
| **BOLD JSON Sidecar** | `*task-rest*_bold.json` | **MANDATORY** | Contains acquisition Repetition Time (`RepetitionTime`), phase encoding, etc. |
| **Anatomical T1w** | `*_T1w.nii[.gz]` | **RECOMMENDED** | High-resolution anatomical image for coregistration and tissue segmentation |
| **Anatomical Sidecar** | `*_T1w.json` | **OPTIONAL** | Anatomical acquisition parameters |
| **Field Maps** | `fmap/*` | **OPTIONAL** | B0 distortion correction (if acquired) |
| **Task Events** | `*events.tsv` | **NOT APPLICABLE** | Resting-state scans have no event onsets; ignored if present |

---

## 3. Data Integrity & Validation Constraints

Before raw data is passed into any preprocessing step, it must satisfy strict programmatic checks:

### 3.1 4D BOLD Volume Checks
1. **Dimensionality**: Must be exactly 4D (`len(img.shape) == 4`). 3D images are rejected immediately with an explanatory error.
2. **Temporal Length ($T$)**: Must contain at least **50 valid timepoints** ($T \ge 50$). Scans shorter than 50 volumes provide inadequate degrees of freedom for 0.01–0.10 Hz bandpass filtering and stable correlation estimation.
3. **Spatial Dimensions**: Minimum spatial matrix size $[32, 32, 15]$ voxels.
4. **Data Types & Values**: Voxels must contain finite numeric values (no `NaN` or `Inf` within the brain mask).

### 3.2 BOLD Metadata (`.json`) Checks
1. **RepetitionTime ($T_R$)**: Must be explicitly defined in seconds. Range: $0.5 \le T_R \le 5.0$ seconds.
2. **TaskName**: Must confirm resting state (e.g. `"rest"`, `"rest_acq-rest"`, etc.). Task-based fMRI (e.g. n-back, flanker) will be rejected to prevent out-of-distribution feature generation.

### 3.3 Explicit Failure Conditions
The validation module will abort processing and produce a structured error report if:
- Input path does not exist or is empty.
- Functional scan is missing or cannot be parsed by `nibabel`.
- Functional scan header indicates a 3D volume instead of 4D.
- Functional scan length $T < 50$.
- `RepetitionTime` is missing from the JSON sidecar.
- Voxel matrix affine contains singular or invalid transformation matrices.
