# HISTORICAL C-PAC CONCORDANCE & EQUIVALENCY PLAN ($N = 6$ PILOT)

> **`PRE-AUTHORIZATION PLAN — COMPARISON NOT YET EXECUTED ON REAL DATA`**
>
> **Document ID:** `DOC-PILOT-CPAC-001`  
> **Scope:** Future authorized $N = 6$ ADHD-200 / matched feasibility participants (`STRICTLY_SYNTHETIC_ONLY` until Step 8 passes `APPROVED_SCOPED`)  
> **Important Notice:** No real ADHD-200 raw MRI scans or historical C-PAC real-subject outputs were accessed, compared, or reported in Step 9. All quantitative targets below (including `r >= 0.85`) are **proposed engineering targets only**, not measured claims.

---

## 1. Objective

The historical NeuroADHD baseline classifier (`trained_models/`) was trained on archival **C-PAC** (Configurable Pipeline for the Analysis of Connectomes) Craddock **CC200** Fisher-z connectivity features derived from ADHD-200 scans. Before using raw-MRI features produced by `PoC V2` (23-stage pipeline) for cross-pipeline feature evaluation on an authorized future pilot, we must quantify **numerical and anatomical equivalency** between `PoC V2` outputs and matched historical C-PAC derivatives across the same $N = 6$ authorized participant sessions.

---

## 2. Stages & Output Modalities to Compare

| Comparison Domain | `PoC V2` Artifact / Metric | Historical C-PAC Derivative | Primary Comparison Metrics | Proposed Engineering Target *(Not Validated Yet)* |
| :--- | :--- | :--- | :--- | :--- |
| **1. Motion Parameterization & Power FD** | `motion_parameters.tsv` (`6-DOF`), `framewise_displacement.tsv` | C-PAC `coordinate_transformation` / FD TSV (FSL MCFLIRT / AFNI `3dvolreg`) | Pearson `r`, MAE (mm/rad), RMSE of per-volume Power FD | Report exact `r`, MAE, RMSE; inspect systematic scale differences |
| **2. Reference Geometry & Brain Masks** | `functional_mask`, `anat_brain_mask`, `epi_ref` | C-PAC `functional_brain_mask` / `anat_brain_mask` | Dice similarity coefficient (`DSC`), Jaccard overlap, Hausdorff/surface delta (mm) | High spatial mask overlap (`DSC >= 0.85` proposed benchmark) |
| **3. MNI Spatial Normalization** | `bold_mni` + `t1w_to_mni` transform (`qc_registration_t1w_to_mni.png`) | C-PAC MNI-normalized functional mean / anatomical MNI image | Normalized Cross-Correlation (`NCC`), Normalized Mutual Information (`NMI`), anatomical landmark displacement error (mm) | Landmark displacement `< 3.0 mm` (1 CC200 voxel) & visual inspection |
| **4. CC200 Regional Time Series** | `roi_CC200.csv` (`200 x T`, post-regression + bandpass) | C-PAC `roi_timeseries` (Craddock 200 atlas) | Per-ROI mean Pearson `r`, normalized RMSE (`nRMSE`), non-empty parcel count (`200/200`) | Consistent zero-lag correlation polarity & parcel coverage (`200/200`) |
| **5. Connectivity & Fisher-z Vector** | `features_19900_fisher_z.csv` (`19,900` unique edges) | Historical C-PAC CC200 upper-triangular correlation / Fisher-z vector | Vector-wide **Pearson `r`**, **MAE**, **RMSE**, Bland-Altman bias/LOA, top-10% strongest edge Jaccard overlap, frozen classifier score delta | **Proposed Target Only: Pearson `r >= 0.85`** across matched Fisher-z edge vectors; report MAE/RMSE and outlier parcels transparently |

---

## 3. Protocol Execution Sequence (Post-Authorization Only)

1. **Verify Participant Authorization:** Confirm `AuthorizationEvidenceValidator.verify_participant_authorization()` returns `APPROVED_SCOPED` with valid HMAC signature for both raw input access and matched archival derivative access for each of the $N = 6$ IDs.
2. **Verify Exact Scan & Atlas Alignment:** Confirm identical `subject_id`, `session_id`, `task-rest` run index, TR, number of volumes `T`, and matching canonical Craddock CC200 atlas checksum (`9467afce...`, `200` labels, `(63, 75, 61)` or documented resampling grid).
3. **Run PoC V2 in Strict Mode:** Execute `run_poc_v2.py` with `--atlas-mode canonical`, `--strict-slice-timing` (or matched C-PAC STC config recorded in comparison metadata), `--strict-sdc` (matched fieldmap policy), and `--strict-defacing`.
4. **Compute Concordance Metrics (Out-of-Band Comparison Script):**
   - Align edge indices `(i, j)` (`1 <= i < j <= 200`, excluding diagonal/self-connections, verifying identical parcel label mapping).
   - Compute Pearson `r`, MAE, RMSE, edge-wise sign agreement, per-parcel mean concordance (identifying edge/parcel outliers such as orbitofrontal/temporal susceptibility zones), and Dice/NMI for spatial normalization.
5. **Document & Escalate Divergences:** Never adjust classifier weights or tune thresholds on $N = 6$ to force agreement. Report actual discrepancies transparently to supervisor and governance reviewers.

---

## 4. Documented Confounding & Divergence Factors

Any deviation below `r = 0.85` (or higher/lower observed concordance) **must** be interpreted alongside these known architectural differences between `PoC V2` and historical ADHD-200 C-PAC runs:

1. **Software Toolchain Differences:** `PoC V2` uses `DIPY` + `SciPy` + `nilearn` (+ optional `topup`/`pydeface`), whereas historical ADHD-200 C-PAC (`c-pac` v0.3.x/v1.x) relied on `AFNI` (`3dvolreg`, `3dTshift`, `3dBandpass`), `FSL` (`mcflirt`, `fnirt`/`flirt`, `fast`), and `ANTs` SyN C++ binaries.
2. **Slice-Timing & Reference Slice Conventions:** `PoC V2` shifts to `+TR/2` using cubic splines when BIDS `SliceTiming` exists (or skips honestly if missing); archival ADHD-200 scans vary by acquisition site (Peking, NYU, KKI, NeuroIMAGE, OHSU) in slice order and missing slice-timing metadata.
3. **Susceptibility Distortion Correction Availability:** Legacy ADHD-200 datasets frequently lack BIDS fieldmaps or reverse-PE spin-echo acquisitions, meaning historical C-PAC runs typically ran **without SDC**, whereas `PoC V2` applies direct/validated B0 unwarping only when physical fieldmap evidence exists.
4. **Interpolation & Transform Composition:** `PoC V2` composes Motion + SDC + Coreg + SyN into a **single-step pullback resampling** directly on slice-time-corrected BOLD to minimize blurring, whereas older C-PAC branches applied sequential motion reslicing followed by separate MNI resampling.
5. **Nuisance Regression Ordering & Bandpass Implementation:** `PoC V2` performs **simultaneous** GLM nuisance removal (`aCompCor 5` + `Friston-24` + drift/mean) with `[0.01, 0.10] Hz` Butterworth filtering (avoiding reintroducing removed noise), vs sequential `Nuisance -> Bandpass` pipelines in legacy C-PAC.
6. **Global Signal Regression (GSR) Strategy:** Archival C-PAC releases were distributed in `no-GSR` vs `GSR` variants; GSR strongly shifts correlation means toward zero and introduces negative correlations, so concordance **must** match the exact historical C-PAC strategy (non-GSR baseline first).
7. **Defacing Impact Order:** Running anatomical defacing (`Stage 7`) strictly downstream of EPI/T1w brain mask boundaries avoids facial tissue registration bias, which may differ slightly from C-PAC `3dSkullStrip`/`BET` on un-defaced T1w images.
8. **Atlas Grid Resampling:** MNI template voxel size (`3mm`/`4mm` vs `1mm`/`2mm`) and nearest-neighbor label interpolation can alter boundary voxel membership of small CC200 parcels.

