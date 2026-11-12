# Step 4 / 4.1 / 4.2: Metadata-Driven Slice-Timing Correction (`DEC-STC-01`)

## 1. Overview & Safety Boundaries

Step 4, Step 4.1, and Step 4.2 implement and scientifically audit metadata-driven BIDS slice-timing correction (STC) inside [`src/preprocessing/slice_timing.py`](../src/preprocessing/slice_timing.py), time-axis-preserving CC200 extraction in [`src/parcellation/extractor.py`](../src/parcellation/extractor.py), composed spatial resampling in [`src/preprocessing/registration.py`](../src/preprocessing/registration.py), and the **21-stage** [`scripts/run_poc_v2.py`](../scripts/run_poc_v2.py) preprocessing pipeline.

**Strict Safety Invariants:**
- `execution_policy = STRICTLY_SYNTHETIC_ONLY`
- `real_data_authorized = false`
- `overall_authorization_state = NOT_AUTHORIZED`
- `sub-01 authorization_status = PENDING` (`cryptographic_signature = null`)
- All clinical directories (`NeuroADHD_Converted`, `sub-01`, `sub-2344857`, `down/outputs/adhd200`, `peking`) and symlink inputs are rejected fail-closed.
- Validation fixtures live strictly outside the Git repository (`../NeuroADHD_Step4_Synthetic/`, overrideable via `NEUROADHD_STEP4_ROOT`) and are never tracked in Git.

---

## 2. Phase A Audit & Corrected 21-Stage Preprocessing Order

### Problem Identified in Step 4 Baseline
In the initial Step 4 integration (`d9f4c95`), STC was placed immediately before `realign_4d_bold()`. Because `realign_4d_bold()` coupled 6-DOF rigid motion parameter estimation with per-volume spatial resampling, running STC before `realign_4d_bold()` caused motion parameters to be estimated from **temporally interpolated BOLD volumes** (`stc_bold_img`) rather than the original uncorrected BOLD signal.

### Refactored Motion Architecture & Stage Ordering (`Step 4.1`)
`src/preprocessing/motion.py` and `src/preprocessing/registration.py` were refactored so rigid motion fitting (`estimate_motion_parameters`), rigid transform application (`apply_motion_transforms`), and MNI pullback composition (`compose_volume_transform`, `resample_4d_bold(..., motion_transforms=...)`) are independent, composable primitives (while keeping legacy `realign_4d_bold()` intact for existing callers):

| Stage | Module / Operation | Input Signal & Spatial Transform Semantics |
| :--- | :--- | :--- |
| **Stage 1/21** | **BIDS Input & Sidecar Metadata Validation** | Validates BIDS JSON metadata, synthetic integrity manifest, and canonical CC200 atlas before touching MRI voxel arrays. |
| **Stage 2/21** | **3D Rigid Motion Parameter Estimation** (`estimate_motion_parameters`) | Estimates 6-DOF rigid realignment transforms ($\text{FD}$ & `motion_parameters.tsv`) directly on **raw, uncorrected BOLD volumes** against the native mean EPI. Bit-identical regardless of STC method or status. |
| **Stage 3/21** | **Metadata-Driven Slice-Timing Correction** (`write_slice_timing_derivative`) | Corrects temporal slice offsets on **original, unresampled slice geometry** (`bold_img` loaded from disk), writing immutable BIDS derivatives under `derivatives/*_desc-stc_bold.{nii.gz,json}`. |
| **Stage 4/21** | **Motion Transform Application** (`apply_motion_transforms`) | Applies the Stage 2 rigid transforms to `stc_bold_img` solely to construct a motion-aligned 4D native series for **Stage 5** functional reference generation. |
| **Stage 5/21** | **Functional Reference Generation** | Temporal mean EPI & brain mask computed from the motion-aligned native STC series. |
| **Stages 6–9/21** | **Anatomical T1w Processing** | Skull stripping (`median_otsu`), Legendre polynomial bias correction, CSF/GM/WM segmentation, $P \ge 0.96$ thresholding + 1-voxel 3D erosion. |
| **Stages 10–11/21** | **Coregistration & MNI SyN Normalization** | EPI-to-T1w rigid coregistration + T1w-to-MNI affine & diffeomorphic SyN mapping. |
| **Stage 12/21** | **Single-Step Composed 4D Resampling** (`resample_4d_bold(..., motion_transforms=motion_transforms)`) | Evaluates composed per-volume pullback $\text{motion\_affine}_t^{-1} \circ \text{prealign}^{-1} \circ \Phi_{\text{SyN}}^{-1}$ directly on **unresampled `stc_bold_img`** in **one single spatial interpolation** (eliminating double spatial smoothing). |
| **Stages 13–21/21** | **Masks, Nuisance/Bandpass, CC200, QC & Provenance** | Tissue mask SyN resampling, 5 aCompCor + Friston-24 + linear/quadratic trends (31 columns, GSR OFF), simultaneous GLM + `[0.01–0.10 Hz]` Butterworth bandpass, 200-ROI CC200 extraction, 19,900 Fisher-$z$ edges, 4 QC overlays, and schema-validated provenance (`pipeline_stages_total = 21`). |

---

## 3. BIDS Metadata Validation & Slice Axis Defaulting (`validate_slice_timing_metadata`)

- **`RepetitionTime`:** Required finite positive scalar (rejects booleans/strings/`<= 0`). Verified against `expected_tr`, NIfTI header `pixdim[4]` (`sec`, `msec`, `usec`), and `VolumeTiming` step differences when present.
- **`SliceTiming`:** Required 1D finite numeric array with `0 <= timing < TR` matching the slice count along the resolved NIfTI slice axis. Missing `SliceTiming` never invents timings from `TR`: strict mode raises `MissingSliceTimingError`; runner default returns `SKIPPED_MISSING_METADATA` with original BOLD preserved.
- **`SliceEncodingDirection` & NIfTI Header `slice_dim` Resolution (Step 4.1):**
  - Explicit BIDS `i`, `i-`, `j`, `j-`, `k`, `k-` values map to axes `0`, `1`, `2` (`-` reverses slice index order: `timing[::-1]`).
  - When explicit `SliceEncodingDirection` and NIfTI header `header.get_dim_info()[2]` (`slice_dim`) are both present, they **must agree**; conflicts raise `InvalidSliceTimingError` fail-closed.
  - When `SliceEncodingDirection` is **absent or `null`**, `slice_timing.py` consults `header.get_dim_info()[2]`:
    - `slice_dim = 0` $\rightarrow$ `'i'` (increasing index), `slice_axis_source = "nifti_header_slice_dim"`
    - `slice_dim = 1` $\rightarrow$ `'j'` (increasing index), `slice_axis_source = "nifti_header_slice_dim"`
    - `slice_dim = 2` $\rightarrow$ `'k'` (increasing index), `slice_axis_source = "nifti_header_slice_dim"`
    - `slice_dim is None` (unknown) $\rightarrow$ **Never assumes `'k'` universally.** Skips STC cleanly with `status = "SKIPPED_UNKNOWN_SLICE_AXIS"`, `corrected = False`, explicit reason/warning, and bit-identical input preservation.
- **Simultaneous Acquisition & Double-STC Guard:** Equal slice timings (`ptp <= 1e-9 s`) skip with `SKIPPED_SIMULTANEOUS_ACQUISITION`; inputs carrying `SliceTimingCorrected: true` raise `DoubleSliceTimingCorrectionError`.

---

## 4. Software Backend & Temporal Boundary Validation

Host/container inspection (`detect_stc_software()`) confirms AFNI `3dTshift` and FSL `slicetimer` are not installed. Default STC uses pure-Python **SciPy `CubicSpline`** (`bc_type='not-a-knot'`), with selectable `quintic_bspline`, `fourier`, and `linear` alternatives.

### Audit of Unpadded `CubicSpline(..., bc_type='not-a-knot', extrapolate=True)`
Evaluating a single-volume linear-operator perturbation reveals why unpadded `not-a-knot` extrapolation becomes unstable near scan boundaries:
- At shift $\delta = -0.45\text{ TR}$ (`vol 0`) or $\delta = +0.45\text{ TR}$ (`vol T-1`), unpadded `not-a-knot` extrapolation carries filter weight norm $\|w\|_1 = 6.8969$ and $\|w\|_2 = 3.5389$ ($\max |w_j| = 2.2065$), magnifying white-noise variance by $\|w\|_2^2 = 12.52\times$ and causing massive polynomial overshoot (`22.99`–`23.07` RMSE) on boundary transients (`amplitude = 15.0`).

### Selected Boundary Strategy: 4-TR C¹ Odd Point-Reflection + `extrapolate=False`
Implemented `_odd_reflect_pad_series(slice_y, t_vol, k_pad=min(4, T-1))`:
$$y(-k) = 2 y(0) - y(k), \qquad y(T - 1 + k) = 2 y(T - 1) - y(T - 1 - k), \quad k \in \{1,\dots,k_{\text{pad}}\}$$
followed by `CubicSpline(t_ext, y_ext, axis=1, bc_type='not-a-knot', extrapolate=False)` (and `extrapolate=False` for `quintic_bspline`).

**Mathematical Properties & Honest Tradeoffs:**
1. **Linear Operator Norm Reduction:** At `vol 0` ($\delta = -0.45\text{ TR}$) and `vol T-1` ($\delta = +0.45\text{ TR}$), $\|w\|_1$ drops **$2.87\times$** (`6.8969 -> 2.4016`), $\|w\|_2$ drops **$2.09\times$** (`3.5389 -> 1.6901`), noise variance amplification $\|w\|_2^2$ drops **$4.38\times$** ($12.52\times \to 2.86\times$), and unit DC/affine reproduction $\sum w_j = 1.0000$ holds identically.
2. **Interior Invariance:** Interior volumes (`[5, T-6]`) keep exact unperturbed $C^2$ cubic spline weights ($\|w\|_1 = 1.5435$, $\|w\|_2 = 0.8727$; RMSE $= 0.0030$, bit-identical to unpadded `not-a-knot`).
3. **Boundary Stability under Noise & Transients:** Cuts noisy edge RMSE up to **50%** and cuts transient boundary overshoot **$4.28\times$–$4.65\times$** (`22.99 -> 5.37` at `vol 0`, `23.07 -> 4.96` at `vol T-1`; up to $28\times$ at $\delta = -0.95\text{ TR}$).
4. **Measured Tradeoff (Zero-Curvature Bias on Noiseless Extrema):** Anti-symmetric point reflection enforces $y''(0)=y''(T-1)=0$ (natural-spline equivalence). For noiseless oscillatory signals evaluated beyond `[0, T-1]`, ignoring nonzero boundary curvature $y''(0)$ increases single-edge noiseless RMSE from `0.1873` to `0.2403` at `vol 0` ($\delta=-0.45\text{ TR}$) and `0.2670` to `0.6258` at phase-aligned curvature extrema (`vol T-1`, $\delta=+0.45\text{ TR}$) while still halving raw acquisition error on start frames (`1.4586 -> 0.2403`).
5. **No Silent Cropping + Five-Region Boundary Risk Metadata (`Step 4.2`):** All $T$ volumes and `+reference_time` `StartTime` are preserved (`AutomaticBoundaryVolumeRemovalApplied = false`); derivative JSON sidecars and provenance record `BoundaryPaddingStrategy = "odd_point_reflection_4tr"`, `BoundaryExtrapolationDisabled = true`, `BoundaryEdgeVolumesAtRisk = [0, T-1]`, `BoundaryRiskWindowSamplesPerSide = 5`, `BoundaryRiskWindowVolumes = [0..4, T-5..T-1]`, `BoundaryRegions` (distinguishing `first_volume`, `first_five_volumes`, `interior_volumes`, `last_five_volumes`, `final_volume`), and clarified `VolumeCensoringGuidance`. Tested explicitly on both negative (`reference_time = 0.0 s`) and positive (`reference_time = 1.81818 s`) STC shifts (`N = 120`, zero silent truncation).

### Five-Region Boundary Benchmark (`N = 120`, `250 voxels`, `TR = 2.0s`, $\sigma = 0.25$ Gaussian Noise)

| Shift ($\delta$) | Signal Regime | Region (`idx`) | Raw RMSE | Old (`extrap=True`) RMSE | Selected (`odd_reflect_4TR`) RMSE | Old vs New MaxAbsErr |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`-0.45 TR`** | Smooth oscillation | `first_volume` (`0`) | `1.4586` | `0.1873` | `0.2403` | `0.1928` vs `0.3750` |
| **`-0.45 TR`** | Smooth oscillation | `first_five` (`0..4`) | `0.8149` | `0.0847` | `0.1096` | `0.1928` vs `0.3750` |
| **`-0.45 TR`** | Smooth oscillation | `interior` (`5..114`) | `0.8938` | `0.00303` | `0.00303` | `0.00526` vs `0.00526` |
| **`-0.45 TR`** | Smooth oscillation | `last_five` (`115..119`) | `1.3270` | `0.0199` | `0.0599` | `0.0467` vs `0.1372` |
| **`-0.45 TR`** | Smooth oscillation | `final_volume` (`119`) | `0.4070` | `0.0409` | `0.1300` | `0.0467` vs `0.1372` |
| **`-0.45 TR`** | Noisy ($\sigma=0.25$) | **`first_volume` (`0`)** | `1.4645` | **`0.9156`** | **`0.4593`** (`-49.8%`) | **`2.7454` vs `1.4989`** |
| **`-0.45 TR`** | Noisy ($\sigma=0.25$) | `first_five` (`0..4`) | `0.8520` | `0.4647` | `0.2857` | `2.7454` vs `1.4989` |
| **`-0.45 TR`** | Noisy ($\sigma=0.25$) | `interior` (`5..114`) | `0.9300` | `0.2179` | `0.2179` | `1.0393` vs `1.0393` |
| **`-0.45 TR`** | Noisy ($\sigma=0.25$) | `last_five` (`115..119`) | `1.3592` | `0.2317` | `0.2249` | `0.8331` vs `0.7474` |
| **`-0.45 TR`** | Noisy ($\sigma=0.25$) | `final_volume` (`119`) | `0.4770` | `0.2887` | `0.2560` | `0.8331` vs `0.6726` |
| **`-0.45 TR`** | Non-periodic linear trend | `first_volume` / `interior` | `1.5305` / `0.8925` | `0.1873` / `0.00303` | `0.2403` / `0.00303` | trend offset `0.0000` |
| **`-0.45 TR`** | Controlled transient | **`first_volume` (`0`)** | `3.8661` | **`22.9926`** | **`5.3677`** (`4.28x` lower) | **`22.9981` vs `5.5621`** |
| **`-0.45 TR`** | Controlled transient | `first_five` (`0..4`) | `3.9258` | `10.4811` | `2.4978` | `22.9981` vs `5.5621` |
| **`-0.45 TR`** | Controlled transient | `interior` (`5..114`) | `1.0269` | `0.0684` | `0.0684` | `0.3939` vs `0.3939` |
| **`-0.45 TR`** | Controlled transient | `last_five` / `final` | `4.2298` / `6.0169` | `2.3209` / `5.1160` | `0.7575` / `1.6494` | `5.1221` vs `1.6682` |
| **`+0.45 TR`** | Smooth oscillation | `first_volume` / `final_volume` | `1.2926` / `0.4205` | `0.0292` / `0.2670` | `0.0524` / `0.6258` | `0.2907` vs `0.6610` (final) |
| **`+0.45 TR`** | Noisy ($\sigma=0.25$) | **`final_volume` (`119`)** | `0.4705` | **`0.9323`** | **`0.7541`** (`-19.1%`) | **`2.6473` vs `1.9203`** |
| **`+0.45 TR`** | Controlled transient | **`final_volume` (`119`)** | `2.7034` | **`23.0713`** | **`4.9612`** (`4.65x` lower) | **`23.0960` vs `5.0523`** |

---

## 5. Temporal Reference Point & Provenance Immutability

- **Default Reference Time:** `TR / 2.0` (`+1.0 s` for `TR = 2.0 s`), configurable within `[0, TR)`.
- **Temporal-Origin Shift:** BIDS derivative sidecars record `SliceTimingCorrected: true`, `SliceTimingReferenceTime`, `StartTime = reference_time`, `TemporalOriginShiftSeconds = reference_time`, `OriginalSliceTiming`, `Sources`, plus boundary provenance (`BoundaryPaddingStrategy`, `BoundaryExtrapolationDisabled`, `BoundaryEdgeVolumesAtRisk`, `BoundaryRiskWindowSamplesPerSide`, `BoundaryRiskWindowVolumes`, `BoundaryRegions`, `AutomaticBoundaryVolumeRemovalApplied: false`, `VolumeCensoringGuidance`, `HeaderSliceDim`, `SliceAxisSource`).
- **Immutability:** Raw BIDS BOLD NIfTI and JSON sidecars remain byte-identical (verified via pre/post SHA-256 checksums); derivatives are emitted solely under `derivatives/*_desc-stc_bold.{nii.gz,json,_provenance.json}`.

---

## 6. CC200 Time-Axis Audit & Configured Censoring (`Step 4.2`)

### 6.1 Root Cause of `200 × 241` vs `200 × 240`
Tracing the complete data path from Stage 12 resampled BOLD `(63, 75, 61, 240)` through Stage 17 (`clean_functional_bold`), Stage 18 (`extract_cc200_timeseries`), Stage 19 (`compute_connectivity_vector`), and the dashboard (`DashboardDataService.get_timeseries`) confirms:
- **Intentional C-PAC CSV Identifier Column:** `extract_cc200_timeseries()` uses `NiftiLabelsMasker` (no padding, no reference volume prepending, no cropping) to extract `signals` of shape **`(240, 200)`** (`T × 200`). It transposes to `(200, 240)` and inserts column `0` (`df.insert(0, "node/volume", np.arange(1, 201))`), producing the exported CSV `roi_CC200.csv` with shape **`(200, 241)`** (`"node/volume"` + `"0".."239"`).
- **Zero Time-Axis Corruption Downstream:** Stage 19 consumes `signals` `(240, 200)` directly (no label column), and `DashboardDataService.get_timeseries()` asserts `fieldnames[0] == "node/volume"` and slices `list(row.keys())[1:]`, yielding `n_timepoints = 240`. Every ROI has uniform, aligned length `240` with zero silent duplication or padding.
- **Disambiguated Provenance Reporting & Opt-In Censoring:** Earlier Step 4.1 reporting ambiguously printed `roi_df.shape` (`[200, 241]`) as the time-series shape. `validation_gates.gate5_roi_coverage` in `scripts/run_poc_v2.py` now cleanly separates `n_timepoints_input`, `n_timepoints_retained`, `roi_timeseries_shape` (`[200, T_retained]`), `roi_csv_shape` (`[200, T_retained + 1]`), `roi_label_column` (`"node/volume"`), `volume_censoring_applied`, and `censored_volumes`. Explicit `censor_volumes` is supported via `resolve_censor_volumes()`, `extract_cc200_timeseries()`, and `run_poc_v2_pipeline(..., censor_volumes=...)` (applied in Stage 18 after continuous-TR Stage 17 temporal filtering), failing closed on booleans, out-of-range indices, or `< 50` retained volumes.

---

## 7. Quantitative Synthetic Validation (`scripts/validate_step4_slice_timing.py`)

### 7.1 Analytical Multi-Tone Validation (`16x16x37x240`, Hall–Meyer $C^4$ Interior Bound = `0.023130`)

| Case ID | Regime & Axis | Whole-Series RMSE Before -> After (% Drop) | Interior RMSE (`[2:-2]`) | Phase Error (rad) Before -> After | Mean Pearson $r$ | Max Determinism Diff |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `case_a_sequential_k` | Sequential (`k`) | `0.3712 -> 0.01070` (`97.12%`) | `0.00151` | `0.0751 -> 0.00006` | `0.999996` | `0.0e+00` |
| `case_b_interleaved_k` | Interleaved (`k`) | `0.5270 -> 0.01570` (`97.02%`) | `0.00216` | `0.1192 -> 0.00008` | `0.999991` | `0.0e+00` |
| `case_c_multiband_k` | Multiband MB=3 (`k`) | `0.5793 -> 0.01855` (`96.80%`) | `0.00228` | `0.1214 -> 0.00006` | `0.999988` | `0.0e+00` |
| `case_d_negative_k_minus` | Reversed (`k-`) | `0.3750 -> 0.01114` (`97.03%`) | `0.00153` | `0.0758 -> 0.00006` | `0.999996` | `0.0e+00` |
| `case_e_axis_i` | Axis `i` (`37x16x16`) | `0.5270 -> 0.01566` (`97.03%`) | `0.00216` | `0.1192 -> 0.00008` | `0.999991` | `0.0e+00` |
| `case_e_axis_j` | Axis `j` (`16x37x16`) | `0.3712 -> 0.01070` (`97.12%`) | `0.00151` | `0.0751 -> 0.00006` | `0.999996` | `0.0e+00` |

### 7.2 Header Axis Controls & Negative Direction Discrimination
- Missing `SliceEncodingDirection` with header `slice_dim = 0, 1, 2` resolved cleanly to `i, j, k` (`interior_rmse_after = 0.00150` across all 3 axes).
- Missing `SliceEncodingDirection` with unset `slice_dim = None` skipped cleanly (`SKIPPED_UNKNOWN_SLICE_AXIS`, `corrected = false`, bit-identical BOLD preserved).
- Explicit direction vs conflicting `slice_dim` raised `InvalidSliceTimingError` fail-closed.
- `k-` outperformed unreversed `k` (`0.01114` vs `0.73972` RMSE, $66.4\times$ accuracy improvement).

### 7.3 Independent Composed Spatial Transform Landmark Validation (`Step 4.2`)
Audited [`compose_transformations`](../src/preprocessing/registration.py), [`compose_volume_transform`](../src/preprocessing/registration.py), and [`resample_4d_bold`](../src/preprocessing/registration.py) across 5 asymmetric 3D landmarks on a synthetic `32×32×24` (`3×3×4 mm` RAS) EPI $\to$ `36×36×30` (`2.5 mm` LAS) MNI grid, checking both forward (`.transform` + `.transform_points_inverse`) and inverse (`.transform_inverse` + `.transform_points`) mappings:

| Regime | Fwd COM Mean / Max Err (`mm`) | Fwd Max (`vox`) | Inv COM Mean / Max Err (`mm`) | Point Fwd / Inv Max (`mm`) |
| :--- | :--- | :--- | :--- | :--- |
| `identity` | `0.0998 / 0.1377` | `0.0551` | `0.0814 / 0.1427` | `0.0 / 0.0` |
| `pure_translation` | `0.1245 / 0.1983` | `0.0793` | `0.0800 / 0.1455` | `0.0 / 0.0` |
| `pure_rotation` | `0.0900 / 0.1370` | `0.0548` | `0.0831 / 0.1669` | `7.3e-15 / 6.6e-15` |
| `combined_rigid` | `0.0861 / 0.1470` | `0.0588` | `0.0803 / 0.1718` | `4.4e-15 / 7.5e-15` |
| `non_commuting_composed_with_syn` | `0.1162 / 0.1502` | `0.0601` | `0.1690 / 0.3327` | `0.0174 / 0.0089` |

- **Sequential vs Single-Step Composed Resampling:** Max landmark difference `0.0618 mm`.
- **Negative Multiplication-Order Controls:** Swapping motion order (`prealign_inv @ motion` instead of `motion @ prealign_inv`) produces `2.6650 mm` landmark error; swapping coreg/normalization order (`affine_t1w_to_mni @ rigid_epi_to_t1w`) produces `2.8925 mm` landmark error. Singular `4×4` motion matrices fail closed (`ValueError`).

### 7.4 Realistic 37-Slice × 240-Volume End-to-End 21-Stage Integration
Executed full 21-stage preprocessing (`run_poc_v2_pipeline`) on a source-acquisition-like synthetic BIDS scan (`64x64x37x240`, `37` interleaved slice timings, $3.0 \times 3.5 \times 4.0\text{ mm}^3$, `TR = 2.0 s`, all 200 canonical CC200 parcels in FOV) mapped onto canonical MNI/CC200 `(63, 75, 61)` for two independent runs (`Run 1` + `Run 2`):
- **Stage Ordering Verified:** Stage 2 estimated motion directly on raw uncorrected BOLD; Stage 3 applied STC across `37/37` slices (`desc-stc_bold.nii.gz` shape `(64, 64, 37, 240)`); Stage 12 composed `motion_transforms` + coregistration + SyN in one spatial pullback into `(63, 75, 61, 240)`.
- **CC200 Time Axis & Correlation Recovery:** `n_timepoints_input = 240`, `n_timepoints_retained = 240`, `roi_timeseries_shape = (200, 240)`, `roi_csv_shape = (200, 241)` (`roi_label_column = "node/volume"`). Pure-GM positive correlation (`Parcel_066`, `Parcel_128`) recovered **`r = +0.9791`** (`> 0.70`); pure-GM negative correlation (`Parcel_093`, `Parcel_069`) recovered **`r = -0.9800`** (`< -0.65`); `19,900` finite Fisher-$z$ features.
- **Determinism & Resource Budget:** `Run 1` vs `Run 2` `max_abs_diff_timeseries = 0.0`, `max_abs_diff_fisher_z = 0.0`; runtime `81.68 s` (Run 1) / `78.55 s` (Run 2); Peak RSS `1.02 GB` ($\le 2.50\text{ GB}$ ceiling).

### 7.5 Canonical 61-Slice Step 3 Regression Verification (`scripts/validate_step3_pipeline.py`)
Verified end-to-end compatibility against the canonical Step 3 synthetic benchmark (`63×75×61×240`, `61/61` interleaved `k` slices, `TR = 2.0 s`, Stage 2 raw-BOLD motion estimation pre-STC):
- **Status & Time Axis:** `VERIFIED` (`21/21` stages); `timeseries_shape = [200, 240]`, `boundary_risk_window_volumes = [0..4, 235..239]`, `automatic_boundary_volume_removal_applied = false`.
- **Correlation Recovery:** Positive pair (`Parcel_012`, `Parcel_045`) **`r = +0.9818`** (`> 0.70`); Negative pair (`Parcel_070`, `Parcel_085`) **`r = -0.9864`** (`< -0.65`); `19,900` finite Fisher-$z$ edges (`[-2.491, 2.346]`).
- **Determinism & Budget:** `Run 1` vs `Run 2` `max_abs_diff_timeseries = 0.0`, `max_abs_diff_fisher_z = 0.0`; `Run 1` runtime `102.81 s`; Peak RSS `0.95 GB` ($\le 2.50\text{ GB}$ ceiling).

---

## 8. Known Limitations

1. **SciPy Backend vs AFNI `3dTshift`:** AFNI `3dTshift` and FSL `slicetimer` remain absent in this host/container environment; temporal interpolation uses SciPy `CubicSpline` with odd-point boundary reflection (`4 TR`) and `extrapolate=False` rather than AFNI binary output.
2. **Optional Boundary Censoring:** Edge volumes `[0, T-1]` (and spline reflection knot window `BoundaryRiskWindowVolumes = [0..4, T-5..T-1]`) exhibit natural-boundary zero-curvature bias on noiseless oscillatory extrema (while cutting noisy/transient instability); all $T$ volumes are retained cleanly unless callers explicitly pass `censor_volumes`.
3. **Real-Data Concordance Still Locked:** Because `real_data_authorized = false` (`overall_authorization_state = NOT_AUTHORIZED`), real participant BOLD execution (`sub-01`) and historical C-PAC concordance remain disabled (`NOT_ASSESSED`).
