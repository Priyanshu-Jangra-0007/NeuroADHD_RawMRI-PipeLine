# NeuroADHD RawMRI PoC V2 — Deployment & Container Isolation Guide

## 1. Overview and Operational Purpose

This document specifies the deployment architecture, OS-level memory isolation controls, and reproducible runtime configuration for the **NeuroADHD RawMRI PoC V2** pipeline.

The pipeline processes synthetic resting-state functional MRI (rs-fMRI) phantoms through a 19-stage preprocessing orchestrator, culminating in 200 CC200 ROI time series and 19,900 Fisher-z transformed functional connectivity features.

### Core Deployment Contracts
1. **Operating-System Memory Ceiling**: A hard container limit of **2560 MiB (2.5 GiB)** enforced by the Linux kernel via Control Groups (cgroup v2 / v1).
2. **Canonical Atlas Integrity**: Authentic CC200 atlas (`data/atlases/cc200_roi_atlas.nii.gz`) mounted strictly read-only (`:ro`) and validated via SHA-256 (`9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650`), spatial shape `(63, 75, 61)`, 3.0 mm isotropic voxel zooms, and 200 contiguous ROI labels (1–200).
3. **Fail-Closed Execution Policy**: Strict input authenticity verification via HMAC-SHA256 (`NEUROADHD_HMAC_KEY`). Any execution missing OS memory constraints or attempting to process unverified or real MRI data fails closed.
4. **Synthetic-Only Guarantee**: Execution is restricted exclusively to cryptographically authenticated synthetic phantom data; real scan processing is strictly blocked.

---

## 2. Architecture & Runtime Environment

### Container Specification
- **Base Image**: `python:3.11-slim` (Debian 12 Bookworm slim)
- **Architecture**: Tested and verified on Apple Silicon ARM64 (`aarch64`) and x86_64 (`amd64`).
- **Dependencies**: 100% pinned via `requirements-lock.txt` using pre-compiled `manylinux_2_17_aarch64` wheels with zero native compilation steps during build.
- **Process Privilege**: Unprivileged service user `neuro` (`UID 999:GID 999`). Root execution is explicitly prohibited.

### Process Environment Variables
| Variable | Value | Purpose |
| :--- | :--- | :--- |
| `PYTHONUNBUFFERED` | `1` | Real-time terminal log flushing |
| `PYTHONDONTWRITEBYTECODE` | `1` | Prevents `.pyc` creation on mounted host paths |
| `MPLCONFIGDIR` | `/tmp/matplotlib` | Writable directory for Matplotlib cache |
| `OMP_NUM_THREADS` | `2` | OpenMP thread limit for deterministic memory usage |
| `OPENBLAS_NUM_THREADS` | `2` | OpenBLAS thread limit |
| `MKL_NUM_THREADS` | `2` | MKL thread limit |
| `NEUROADHD_HMAC_KEY` | *(Secret)* | Integrity key protecting synthetic session manifests |

---

## 3. Dual-Layer Memory Enforcement: OS Cgroups vs Python RSS

A central vulnerability of naive Python memory guards is relying exclusively on Python-level monitoring (such as `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` or `psutil`).

### Why Python RSS Monitoring is Insufficient Alone:
- `ru_maxrss` is **strictly passive and historical**: it reports the maximum memory high-water mark after allocation has already occurred.
- Native C/Cython extensions (e.g., NumPy, SciPy, DIPY C-extensions) allocate memory through `malloc`/`mmap` that Python's garbage collector cannot immediately reclaim.
- A sudden memory spike in C extensions can exceed available physical RAM, causing host system thrashing or uncontrollable host OOM-killing before Python can intercept it.

### Dual-Layer Architecture:
```
+-------------------------------------------------------------------------------+
| HOST OS KERNEL (Linux / macOS Docker Desktop VM)                              |
|                                                                               |
|  [ LAYER 1: HARD KERNEL CGROUP ENFORCEMENT ]                                  |
|   Container Flag: --memory=2560m --memory-swap=2560m                          |
|   Kernel File:    /sys/fs/cgroup/memory.max == 2684354560 (2560 MiB)          |
|   Enforcement:    Linux OOM Killer sends SIGKILL (exit 137) immediately if    |
|                   total physical memory exceeds 2560 MiB.                     |
|                                                                               |
|  +-------------------------------------------------------------------------+  |
|  | DOCKER CONTAINER RUNTIME                                                |  |
|  |                                                                         |  |
|  |   [ LAYER 2: APPLICATION RUNTIME GUARD & PROVENANCE AUDITING ]          |  |
|  |    Module:       src.validation.memory_guard                            |  |
|  |    Active Check: verify_os_memory_isolation() fails closed if limit is  |  |
|  |                  unconfigured or exceeds 2560 MiB.                      |  |
|  |    Telemetry:    Polls memory.current and memory.peak from cgroup.      |  |
|  |    Gate 1:       validation_gates.gate1_resources records both cgroup   |  |
|  |                  peak and process RSS in provenance.json.               |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
```

---

## 4. Volume Mounts & Storage Isolation

To prevent data corruption and ensure zero unintended state modification:

1. **Canonical Atlas Volume (`:ro`)**:
   `-v "${PROJECT_ROOT}/data/atlases:/opt/neuro_adhd/data/atlases:ro"`
   Mounted strictly read-only. Any attempt by the container to rewrite or overwrite the canonical CC200 atlas raises an OS-level `Read-only file system` error.
2. **Output Volume (`:rw`)**:
   `-v "${PROJECT_ROOT}/outputs/docker_deployment:/opt/neuro_adhd/outputs:rw"`
   Dedicated host directory mounted for receiving validated outputs: `roi_CC200.csv`, `features_19900_fisher_z.csv`, `run_manifest.json`, and `provenance.json`.
3. **Scratch Isolation**:
   Container temporary inputs are generated in a transient scratch directory inside the container and verified via HMAC before execution.

---

## 5. Deployment Commands & Verification

### Step 1: Build the Hardened Container Image
```bash
docker build -t neuro_adhd_pipeline:v2 .
```

### Step 2: Verify OS Memory Isolation & Kernel OOM-Killer
Execute the automated test script to confirm cgroup limits and kernel enforcement:
```bash
./scripts/verify_container_memory.sh
```
*Expected Output:*
- Test 1/3: Cgroup memory limit confirmed: 2,684,354,560 bytes (2560 MiB) `[PASSED]`
- Test 2/3: Kernel OOM-killer termination confirmed with exit code 137 `[PASSED]`
- Test 3/3: Fail-closed policy confirmed when run on unconstrained host `[PASSED]`

### Step 3: Execute End-to-End Synthetic Deployment Runner
```bash
export NEUROADHD_HMAC_KEY="deployment_hmac_secret_key_arm64"
./scripts/docker_run_synthetic.sh
```

### Step 4: Verify Output Artifacts
Inspect the generated deployment output directory:
```bash
ls -lh outputs/docker_deployment/sub-synthetic001/ses-1/
```
Verify:
1. `roi_CC200.csv` has exactly 200 rows and $(T+1)$ columns.
2. `features_19900_fisher_z.csv` has exactly 19,900 rows and matching feature names.
3. `run_manifest.json` reports `"pipeline_status": "SUCCESS"`.
4. `provenance.json` validates against `configs/provenance_schema.json` and records cgroup memory telemetry.
