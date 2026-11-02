# NeuroADHD RawMRI PoC V2 — Phase A.3 Deployment Readiness & Memory Isolation Report

## Executive Summary

Phase A.3 prepares the **NeuroADHD RawMRI PoC V2** pipeline for reproducible containerized deployment with strict operating-system-level memory isolation.

All objectives of Phase A.3 were accomplished under strict synthetic-only constraints:
1. **Canonical CC200 Atlas Deployment**: The authentic CC200 atlas (`data/atlases/cc200_roi_atlas.nii.gz`) was deployed, mounted read-only (`:ro`), and verified against all canonical specifications: SHA-256 `9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650`, spatial dimensions `(63, 75, 61)`, 3.0 mm isotropic spacing, canonical affine matrix, and exactly 200 contiguous ROI labels ($1..200$).
2. **Operating-System Memory Isolation**: Container isolation was configured and enforced at a hard ceiling of **2560 MiB (2.5 GiB)** (`--memory=2560m --memory-swap=2560m`). Kernel-enforced cgroup limits were verified at `2,684,354,560` bytes. The Linux kernel Out-Of-Memory (OOM) killer was demonstrated via a test container that triggered exit code 137 (`SIGKILL`) upon memory violation.
3. **Dual-Layer Memory Safety**: Passive Python RSS monitoring was reinforced with proactive kernel cgroup enforcement (`src/validation/memory_guard.py`). The pipeline fails closed if memory isolation is unconfigured or misconfigured when `--enforce-container-memory` is active.
4. **Reproducible Container Build**: Created a non-root `python:3.11-slim` `Dockerfile` utilizing pre-compiled `manylinux_2_17_aarch64` wheels from `requirements-lock.txt` with zero build-time C compilation.
5. **End-to-End Containerized Validation**: Executed the complete 19-stage pipeline inside the hardened container with OS memory isolation and canonical atlas verification, producing all 200 CC200 ROI time series, 19,900 Fisher-z connectivity features, quality control overlays, and validated provenance metadata.
6. **Zero Real MRI / Participant Data**: 100% of execution was performed using synthetic phantom data. Real scan processing remained blocked and failed closed.

---

## 1. Canonical CC200 Atlas Deployment & Verification

The authentic canonical CC200 atlas was deployed to `data/atlases/cc200_roi_atlas.nii.gz`.

### Atlas Contract Verification Results:
| Attribute | Contract Requirement | Observed Value | Status |
| :--- | :--- | :--- | :--- |
| **File Path** | `data/atlases/cc200_roi_atlas.nii.gz` | `data/atlases/cc200_roi_atlas.nii.gz` | **MATCH** |
| **SHA-256** | `9467afce...89c650` | `9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650` | **MATCH** |
| **Spatial Shape** | `(63, 75, 61)` | `(63, 75, 61)` | **MATCH** |
| **Voxel Zooms** | `(3.0, 3.0, 3.0)` mm | `(3.0, 3.0, 3.0)` mm | **MATCH** |
| **Parcel Labels** | Exactly 200 contiguous ($1..200$) | Exactly 200 contiguous labels ($1$ to $200$, min=1, max=200) | **MATCH** |
| **Canonical Affine** | $\begin{bmatrix} -3 & 0 & 0 & 93 \\ 0 & 3 & 0 & -129 \\ 0 & 0 & 3 & -72 \\ 0 & 0 & 0 & 1 \end{bmatrix}$ | $\begin{bmatrix} -3.0 & 0.0 & 0.0 & 93.0 \\ 0.0 & 3.0 & 0.0 & -129.0 \\ 0.0 & 0.0 & 3.0 & -72.0 \\ 0.0 & 0.0 & 0.0 & 1.0 \end{bmatrix}$ | **MATCH** |
| **Total Brain Voxels**| $> 40,000$ | $43,413$ voxels | **MATCH** |
| **Container Mount** | Read-Only (`:ro`) | Mounted as `/opt/neuro_adhd/data/atlases:ro` | **ENFORCED** |

The canonical atlas file was preserved unaltered without compression, replacement, or re-generation.

---

## 2. Operating-System Memory Isolation Architecture

### The Limitations of Python RSS Monitoring Alone
Process-level tracking via `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` or `psutil` has critical shortcomings in scientific production pipelines:
1. **Passive and Retrospective**: `ru_maxrss` only records the high-water mark after memory allocations have already succeeded. It cannot prevent a process from attempting to allocate beyond memory limits.
2. **Native Extension Bypass**: NumPy, SciPy, and DIPY invoke C/Cython extensions where large multi-gigabyte matrix buffers are allocated directly via libc `malloc` or kernel `mmap`. Python's garbage collector has no visibility into or immediate control over native heap growth.
3. **Host Destabilization**: If memory is unconstrained at the OS level, an unexpected burst allocation will cause kernel page swapping, disk thrashing, or uncontrollable host-wide Out-Of-Memory termination.

### Dual-Layer Defense Architecture
To guarantee memory ceiling compliance, the pipeline implements two complementary layers:

```
+---------------------------------------------------------------------------------------+
| LAYER 1: HOST OS KERNEL CGROUP ENFORCEMENT                                             |
|                                                                                       |
| - Docker Flags:     --memory=2560m --memory-swap=2560m                                |
| - Linux Cgroup v2:  /sys/fs/cgroup/memory.max == 2684354560 (2560 MiB / 2.5 GiB)      |
| - Kernel Action:    If container memory reaches 2560 MiB, Linux OOM Killer             |
|                     immediately issues SIGKILL (exit code 137). Host RAM is protected.|
+---------------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------------+
| LAYER 2: APPLICATION-LEVEL MEMORY GUARD & AUDITING (src/validation/memory_guard.py)   |
|                                                                                       |
| - Startup Check:    verify_os_memory_isolation() inspects cgroup filesystem.          |
|                     Fails closed if cgroup is unlimited or ceiling exceeds 2560 MiB.   |
| - Runtime Tracking: Polls /sys/fs/cgroup/memory.current and memory.peak.               |
| - Gate 1 Audit:     Records peak RSS and cgroup peak in run_manifest & provenance.    |
+---------------------------------------------------------------------------------------+
```

### Empirical Kernel OOM-Killer Verification
A dedicated verification test was executed using a constrained container (`--memory=50m --memory-swap=50m`) attempting to allocate 120 MiB of memory in Python:
- **Result**: The Linux kernel OOM killer terminated the container process immediately.
- **Exit Code**: `137` (`SIGKILL`).
- **Confirmation**: Verified that the operating-system memory ceiling is physical and unconditional.

---

## 3. Container Deployment Specification

### Docker Configuration
- **Base Image**: `python:3.11-slim`
- **User Security**: Runs as unprivileged service user `neuro:neuro` (`UID 999:GID 999`).
- **Dependencies**: Pinned in `requirements-lock.txt`, downloaded as pre-compiled wheels:
  - `numpy==2.4.6` (`manylinux_2_28_aarch64`)
  - `scipy==1.17.1` (`manylinux_2_28_aarch64`)
  - `dipy==1.12.1` (`manylinux_2_28_aarch64`)
  - `nilearn==0.14.1`
  - `nibabel==5.4.2`
  - `scikit-learn==1.9.1` (`manylinux_2_28_aarch64`)
- **Zero Native Compilation**: All dependencies install from pre-built binary wheels, eliminating build-time GCC/gfortran toolchain dependencies.

### Container Mount Configuration
```bash
docker run --rm \
    --memory=2560m \
    --memory-swap=2560m \
    -e NEUROADHD_HMAC_KEY="${NEUROADHD_HMAC_KEY}" \
    -v "$(pwd)/data/atlases:/opt/neuro_adhd/data/atlases:ro" \
    -v "$(pwd)/outputs/docker_deployment:/opt/neuro_adhd/outputs:rw" \
    neuro_adhd_pipeline:v2 \
    --mode synthetic \
    --atlas-mode canonical \
    --enforce-container-memory \
    --output-dir /opt/neuro_adhd/outputs
```

---

## 4. End-to-End Synthetic Pipeline Validation

The synthetic runner was executed end-to-end inside the container under full OS memory isolation.

### Processing Stages Execution Summary:
| Stage | Component | Output / Validation Result | Status |
| :--- | :--- | :--- | :--- |
| **Stage 1** | BIDS Validation & Atlas Check | BOLD JSON validated (TR=2.0s); Canonical CC200 atlas verified | **PASSED** |
| **Stage 2** | Rigid Motion Realignment | 60 volumes aligned; Mean FD = 0.098 mm; 0 scrubbed | **PASSED** |
| **Stage 3** | Functional Reference Image | Generated 3D reference EPI on canonical grid `(63, 75, 61)` | **PASSED** |
| **Stage 4** | Anatomical Brain Extraction | Extracted brain voxels: 48,395 | **PASSED** |
| **Stage 5** | N4 Bias Field Correction | Bias field computed; mean intensity = 0.747 | **PASSED** |
| **Stage 6** | Tissue Segmentation | GMM/FAST tissue segmentation into CSF, GM, WM | **PASSED** |
| **Stage 7** | Tissue Mask Erosion | Threshold 0.96; 1-voxel 3D erosion: 1,687 WM, 159 CSF voxels | **PASSED** |
| **Stage 8** | EPI-to-T1w Coregistration | DIPY Rigid registration; coregistration pullback derived | **PASSED** |
| **Stage 9** | T1w-to-MNI SyN Registration | Affine + SyN non-linear diffeomorphic mapping computed | **PASSED** |
| **Stage 10** | Transformation Composition | Composed SyN + Affine + Rigid pullback mapping | **PASSED** |
| **Stage 11** | Tissue Mask Resampling | WM/CSF masks resampled into canonical grid via SyN | **PASSED** |
| **Stage 12** | aCompCor Extraction | Extracted 5 physiological PCA components (shape: $60 \times 5$) | **PASSED** |
| **Stage 13** | Friston-24 Expansion | 6 motion parameters expanded to Friston-24 ($60 \times 24$) | **PASSED** |
| **Stage 14** | Confound Matrix Assembly | 31-column nuisance matrix; full rank (31/31); cond: $3.1 \times 10^5$ | **PASSED** |
| **Stage 15** | Denoising & Butterworth Filter | Cleaned BOLD shape: `(63, 75, 61, 60)`; bandpass [0.01–0.10 Hz] | **PASSED** |
| **Stage 16** | CC200 ROI Extraction | Extracted 200 ROIs; zero constant parcels ($200 \times 60$) | **PASSED** |
| **Stage 17** | Fisher-z Connectivity Features | 19,900 Fisher-z transformed correlation features; all finite | **PASSED** |
| **Stage 18** | Orthogonal QC Overlays | Coregistration, normalization, tissue masks, parcellation PNGs | **PASSED** |
| **Stage 19** | Provenance & Output Manifest | Validated against `configs/provenance_schema.json` | **PASSED** |

### Resource Metrics (Gate 1):
- **Memory Ceiling**: 2560 MiB (2.5 GiB / 2,684,354,560 bytes)
- **Container Cgroup Status**: `ENFORCED (2560.0 MiB max, cgroup v2)`
- **Peak Process RSS**: 0.77 GB ($788$ MiB)
- **Total Runtime**: $\approx 159.3$ seconds
- **Memory Gate 1 Verdict**: **PASSED** (0.77 GB $\ll$ 2.50 GB ceiling)

---

## 5. Exported Output Artifacts

| Artifact | Filename | Dimensionality / Contract |
| :--- | :--- | :--- |
| **CC200 ROI Timeseries** | `roi_CC200.csv` | 200 rows $\times$ 61 columns (`node/volume` + 60 timepoints) |
| **19,900 Fisher-z Features (CSV)** | `features_19900_fisher_z.csv` | 19,900 rows $\times$ 2 columns (`feature_name`, `fisher_z`) |
| **19,900 Fisher-z Features (NPY)** | `features_19900_fisher_z.npy` | 1D NumPy float64 array, shape: `(19900,)` |
| **Motion Parameters** | `motion_parameters.tsv` | 6 rigid motion parameters across 60 timepoints |
| **Framewise Displacement** | `framewise_displacement.tsv` | Power FD series with threshold indicator |
| **Nuisance Confounds** | `confounds.tsv` | 31 columns (Friston-24 + 5 aCompCor + Cosines) |
| **Quality Control Overlays** | `qc_*.png` | 4 high-resolution diagnostic overlays |
| **Provenance Record** | `provenance.json` | Validated JSON with hardware & cgroup telemetry |
| **Run Manifest** | `run_manifest.json` | Pipeline status `SUCCESS` with SHA-256 output hashes |

---

## 6. Test Suite & Verification Results

```
=================================== Summary ====================================
Total Test Modules:      13
Total Passing Tests:     148 (138 baseline + 9 memory guard + 1 canonical runner)
Test Pass Rate:          100%
Local Execution Time:    ~45 seconds
Memory Guard Suite:      9 passed in 0.03s
Container Memory Test:   3/3 passed in scripts/verify_container_memory.sh
================================================================================
```

---

## 7. Safety & Compliance Sign-Off

1. **Synthetic-Only Data**: No participant data, ADHD-200 scans, or Peking_2 files were accessed, staged, or processed.
2. **Immutable Canonical Atlas**: The authentic CC200 atlas was verified via cryptographic SHA-256 and mounted strictly read-only.
3. **Fail-Closed Memory Boundary**: Container memory is bounded at 2560 MiB by the Linux kernel; attempting to run outside container isolation with `--enforce-container-memory` immediately aborts execution.
4. **No ML Model Drift**: Downstream ML classifiers, weights, and scoring matrices remain untouched.

