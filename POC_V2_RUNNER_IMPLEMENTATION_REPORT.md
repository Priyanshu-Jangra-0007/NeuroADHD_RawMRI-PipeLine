# NeuroADHD PoC V2 — Functional End-to-End Runner Implementation Report

**Author**: Senior Neuroimaging Software Engineer & QA/Validation Engineer  
**Date**: November 2, 2026  
**Status**: COMPLETE — FUNCTIONAL SYNTHETIC ORCHESTRATOR OPERATIONAL  
**Final Verdict**: **`PASS (SYNTHETIC RUNNER VALIDATED) — CONTROLLED SINGLE-SCAN REVIEW READY`**

---

## 1. Executive Summary

Previously, `scripts/run_poc_v2.py` was a safety stub that immediately raised `NotImplementedError`. While individual processing modules had unit and component-level test coverage, an end-to-end executable orchestrator had not been demonstrated.

In this release, `scripts/run_poc_v2.py` has been fully implemented as an executable pipeline orchestrator. It executes all 19 documented preprocessing stages, incorporates strict spatial boundary validation, exports the complete artifact suite required by downstream modeling, captures schema-compliant provenance, provides isolated run handling with diagnostic failure reports, and enforces strict safety restrictions blocking real participant scans.

An automated integration test suite (`tests/integration/test_runner_poc_v2_synthetic.py`) invokes the actual orchestrator rather than calling individual component functions. All 87 project tests pass with 100% success rate and 87% statement coverage.

---

## 2. Orchestration Architecture & The 19 Pipeline Stages

The orchestrator in `scripts/run_poc_v2.py` coordinates all 19 stages sequentially:

| Stage | Process | Module & Method | Key Validation & Output |
| :---: | :--- | :--- | :--- |
| **1** | BIDS Input Validation | `BIDSInputValidator.validate_sidecar_json()` | TR $\in [0.5, 5.0]\text{s}$, resting task verified |
| **2** | 3D Motion Realignment | `motion.realign_4d_bold()` | 6 motion parameters, Power FD vector, $T \ge 50$ |
| **3** | Functional Reference | `registration.generate_functional_reference()` | Temporal mean volume + functional brain mask |
| **4** | Brain Extraction | `segmentation.extract_anatomical_brain()` | Median Otsu skull stripping with hole-filling |
| **5** | Bias-Field Correction | `segmentation.correct_bias_field()` | Order-2 polynomial log-space correction |
| **6** | Tissue Segmentation | `segmentation.segment_t1w_tissue()` | GMM probability maps: CSF, GM, WM |
| **7** | Mask Erosion | `segmentation.create_eroded_tissue_masks()` | $P \ge 0.96$, 1-voxel erosion, mandatory T1w reference |
| **8** | Coregistration | `registration.register_epi_to_t1w()` | DIPY 6-DOF rigid pullback matrix ($M_{\text{T1w}\to\text{EPI}}$) |
| **9** | Normalization | `registration.register_t1w_to_mni()` | Affine ($M_{\text{MNI}\to\text{T1w}}$) + SyN diffeomorphic map |
| **10** | Pullback Resampling | `registration.compose_transformations()` + `resample_4d_bold()` | Preserves `is_inverse=True`, output matches CC200 grid |
| **11** | Mask Registration | `registration.resample_mask_to_target()` | Nearest-neighbor tissue mask resampling into CC200 grid |
| **12** | aCompCor Extraction | `nuisance.extract_acompcor_components()` | Top 5 physiological noise PCA components |
| **13** | Friston Expansion | `motion.build_friston_24_regressors()` | 24 motion regressors ($R, R^2, R_{t-1}, R_{t-1}^2$) |
| **14** | Confound Assembly | `nuisance.build_confound_matrix()` | 31 columns (24 Friston + 5 aCompCor + 2 poly trends) |
| **15** | Nuisance Regression | `nuisance.clean_functional_bold()` | Butterworth [0.01 - 0.10 Hz], GSR OFF, MeanTissue OFF |
| **16** | CC200 Extraction | `parcellation.extract_cc200_timeseries()` | `resampling_target=None`, $200 \times (T+1)$ DataFrame |
| **17** | Connectivity Vector | `connectivity.compute_connectivity_vector()` | 19,900 Fisher-z edges, row-major lexicographical order |
| **18** | Visual QC Overlays | `qc.generate_*_qc_overlay()` | 4 orthogonal multi-planar PNGs with strict grid checks |
| **19** | Provenance Report | `scripts.run_poc_v2` | Schema-compliant JSON with hardware and validation gates |

---

## 3. Exported Output Artifacts

For each run, outputs are saved into an isolated subject directory (`<output_dir>/<subject_id>/run_<timestamp>/`):

1. **`roi_CC200.csv`**:
   - Matches historical C-PAC format: 200 rows by $(T+1)$ columns.
   - Column 0: `'node/volume'` with contiguous integer labels $1 \dots 200$.
   - Columns $1 \dots T$: mean BOLD intensity per parcel across time.
2. **`features_19900_fisher_z.csv` & `features_19900_fisher_z.npy`**:
   - 19,900 Fisher-z transformed correlation edges in upper-triangular lexicographical order matching the downstream ML classifier schema.
   - Values clipped to $[-8.41, +8.41]$ with zero NaNs or infinities.
3. **`motion_parameters.tsv` & `framewise_displacement.tsv`**:
   - 6-parameter rigid motion series ($x, y, z$ translation in mm, roll, pitch, yaw in degrees).
   - Power Framewise Displacement vector in mm, including mean FD, max FD, and scrubbing counts.
4. **`confounds.tsv` & `confound_report.json`**:
   - 31-column nuisance matrix.
   - Matrix rank ($31/31$) and condition number verification.
5. **Quality Control Multi-Planar Overlays**:
   - `qc_registration_epi_to_t1w.png`: Functional EPI contour on anatomical T1w brain.
   - `qc_registration_t1w_to_mni.png`: Subject T1w contour on standard MNI152 template.
   - `qc_tissue_masks.png`: Eroded WM (blue) and CSF (red) contours on anatomical T1w.
   - `qc_parcellation_cc200.png`: CC200 parcel boundaries overlaid on mean functional image.
6. **`provenance.json`**:
   - Validated against `configs/provenance_schema.json`.
   - Captures UTC timestamps, software versions (`dipy 1.12.1`, `nilearn 0.14.1`, `nibabel 5.4.2`, `numpy 2.4.6`, `scipy 1.17.1`), execution hardware, memory profile, all 6 validation gates (`"PASSED"`), and output paths.

---

## 4. Safety Guard & Error Handling Verification

1. **Cohort & Real Scan Blocking**:
   - `check_safety_guards()` strictly intercepts and blocks any execution targeting:
     - Real subject ID `sub-2344857` -> raises `RuntimeError("SAFETY VIOLATION...")`.
     - Locked `Peking_2` cohort -> raises `RuntimeError("SAFETY VIOLATION...")`.
     - Raw data paths pointing into `../down/` or `ADHD200_RawDataBIDS/` -> raises `RuntimeError`.
   - Verified by unit/integration tests in `tests/integration/test_runner_poc_v2_synthetic.py`.
2. **Fault Isolation & Rollback**:
   - The runner creates an isolated run folder and a sub-scratch directory.
   - If an error occurs during any stage, the runner logs `pipeline_failure_report.json` containing the stage name, error type, message, and diagnostic traceback.
   - Temporary scratch files are automatically removed (`shutil.rmtree`), preventing corrupted partial files from polluting the output space.

---

## 5. Automated Test Suite Results

```text
Platform: macOS Darwin 24.6.0 (Apple Silicon ARM64) | Python: 3.11.15
============================= 87 passed in 20.06s ==============================
Coverage: 87% across 711 statements (0 failures, 0 errors)

Test Breakdown:
- tests/integration/test_runner_poc_v2_synthetic.py:  5 passed (Orchestrator, Safety, Failure)
- tests/integration/test_pipeline_v2_synthetic.py:    1 passed (Synthetic Pipeline Integration)
- tests/unit/test_connectivity.py:                   8 passed (CC200 Extraction & Features)
- tests/unit/test_segmentation.py:                  21 passed (Tissue Segmentation & Erosion)
- tests/unit/test_registration.py:                  21 passed (SyN & Composed Resampling)
- tests/unit/test_nuisance.py:                      18 passed (aCompCor & Friston-24)
- tests/unit/test_qc.py:                             7 passed (Multi-Planar Visual QC)
- tests/unit/test_input_validation.py:               6 passed (BIDS Sidecar Validation)
```

---

## 6. How to Run the Synthetic Pipeline

### Option 1: Via the Executable Runner CLI
```bash
# Run complete 19-stage synthetic preprocessing pipeline
./.venv/bin/python scripts/run_poc_v2.py --mode synthetic --output-dir outputs/synthetic_poc_v2

# Custom volume length and random seed
./.venv/bin/python scripts/run_poc_v2.py --mode synthetic --output-dir outputs/synthetic_poc_v2 --t-vol 65 --seed 123
```

### Option 2: Programmatically in Python
```python
from pathlib import Path
from scripts.run_poc_v2 import generate_synthetic_phantom_session, run_poc_v2_pipeline

# 1. Generate synthetic BIDS inputs
synthetic_inputs = generate_synthetic_phantom_session(
    Path("/tmp/synthetic_session"),
    subject_id="sub-synthetic001",
    t_vol=60,
    seed=42,
)

# 2. Execute orchestrator
results = run_poc_v2_pipeline(
    bold_path=synthetic_inputs["bold_nii"],
    bold_json_path=synthetic_inputs["bold_json"],
    t1w_path=synthetic_inputs["t1w_nii"],
    atlas_path=synthetic_inputs["cc200_atlas"],
    template_path=synthetic_inputs["mni_template"],
    output_dir=Path("outputs/synthetic_poc_v2"),
    subject_id="sub-synthetic001",
)

print(f"Status: {results['status']}")
print(f"ROI CSV: {results['artifacts']['roi_csv']}")
```

---

## 7. Final Verdict

### **`PASS (SYNTHETIC RUNNER VALIDATED) — CONTROLLED SINGLE-SCAN REVIEW READY`**

- The executable runner is **fully operational** and rigorously verified across all 19 processing stages.
- Output artifacts strictly match the contracts required by the frozen NeuroADHD downstream model.
- Real participant data remains guarded and disabled by default.
- The pipeline is prepared for an authorized single-subject dry run on developmental training data.
