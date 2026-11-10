# HARDWARE AND RESOURCE ESTIMATES

## 1. Measured System Baseline (Local Environment Audit)

Hardware metrics were directly queried from the local operating system:
- **Operating System**: macOS Darwin 27.0.0
- **Host Architecture**: Apple Silicon ARM64 (Apple M2 chip, 8 cores)
- **Physical Memory (RAM)**: **8.00 GB** (`hw.memsize: 8,589,934,592` bytes)
  *(Note: Differs from the assumed 16 GB; memory constraints are stricter than anticipated)*
- **Primary Storage (SSD)**:
  - Total Filesystem: `460 GiB`
  - Used Space: `157 GiB` (37% capacity)
  - **Available Disk Space**: **`274 GiB`**
- **Virtualization Runtime**: Docker CLI `29.7.2` installed; **Docker Daemon is STOPPED / INACTIVE**.

---

## 2. Resource Consumption Analysis (One Scan vs Full Dataset)

### 2.1 Single Scan Processing Budget

| Stage | Memory (RAM) Peak | Disk Usage (Transient) | Disk Usage (Permanent) | Estimated Duration |
|---|---|---|---|---|
| **Raw Ingestion & QC** | ~400 MB | 0 MB (reads in place) | ~100 KB (QC report) | ~5 seconds |
| **Motion Correction** | ~1.2 GB | ~250 MB (realigned 4D) | ~20 KB (`FD.1D`, `motion_params.txt`) | ~45 seconds |
| **Spatial Normalization (MNI)** | ~2.5 GB | ~600 MB (warped 4D) | ~5 MB (affine/warp matrix) | ~2 to 4 minutes |
| **Nuisance Regress & Filter** | ~2.0 GB | ~600 MB (cleaned 4D) | ~50 KB (nuisance matrix) | ~45 seconds |
| **CC200 Parcellation** | ~1.5 GB | 0 MB | **~350 KB (`roi_CC200.csv`)** | ~10 seconds |
| **Connectivity & Inference** | ~300 MB | 0 MB | ~150 KB (19,900 feature CSV) | ~2 seconds |
| **TOTAL (Single Scan)** | **Peak ~2.5 GB** | **Transient: ~1.5 - 2.0 GB** | **Permanent: < 6 MB** | **~4 to 7 minutes** |

> [!IMPORTANT]
> **RAM Constraint Warning**: With 8 GB total system RAM, standard macOS system processes consume ~3.5–4.5 GB. The fMRI pipeline must not exceed **3.0 GB peak RAM** to prevent memory pressure, swapping, and kernel OOM terminations. High-memory containers (e.g. standard fMRIPrep requiring 12–16 GB) cannot run safely without severe swapping.

---

### 2.2 Full Cohort Scaling (Historical 153 Subjects)

> [!NOTE]
> **Audit Note on Subject Counts and Test Suite Metrics (November 2026)**:
> - **Subject Count Discrepancy**: Historical documentation cites "153 subjects" for full cohort scaling, whereas model training documentation references 114 training + 20 test subjects (134 total subjects). This subject count discrepancy is **unreconciled - needs the original split manifest** rather than guessing.
> - **Current Test Suite**: 343 tests collected: 336 passed, 7 skipped (5 Docker tests, 2 requiring `NEUROADHD_MODEL_DIR`), with 80% statement coverage across `src/` as of November 2026.

- **Raw Data Requirements**: 153 subjects $\times \approx 100\text{ MB} \approx 15.3\text{ GB}$.
- **Transient Intermediates**: $153 \times 1.5\text{ GB} \approx 230\text{ GB}$ if stored concurrently.
- **Pruning Strategy**: **MUST** operate serially (process subject $i \to$ extract `roi_CC200.csv` $\to$ prune 4D intermediate files $\to$ proceed to subject $i+1$). Under this serial pruning model, disk consumption never exceeds $\approx 20\text{ GB}$ total.
- **Accidental Full Processing Prevention**: Batch processing must be protected by an explicit `--max-subjects 1` safety flag during early phases.

---

## 3. Data Pruning & Provenance Retention Policy

### 3.1 Transient Files Safe to Delete Post-Extraction
Once `roi_CC200.csv` is extracted and passes dimensionality and variance tests, the following heavy intermediate files can be safely deleted:
- Realigned uncompressed 4D functional volume (`*_desc-realigned_bold.nii`).
- Resampled MNI 4D functional volume (`*_space-MNI152_bold.nii`).
- Filtered residual 4D functional volume (`*_desc-cleaned_bold.nii`).

### 3.2 Permanent Provenance Artifacts (Must Be Kept)
- Raw BIDS input files (`*_bold.nii.gz` and `*_bold.json`).
- Framewise displacement (`FD.1D`) and 6-parameter motion realignment traces (`motion_parameters.txt`).
- Confound regressor matrix (`confounds.tsv`).
- CC200 regional time series (`roi_CC200.csv`).
- Execution log and checksum manifest (`provenance.json`).
