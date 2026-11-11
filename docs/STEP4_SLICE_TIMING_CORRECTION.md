# Step 4 / 4.1: Metadata-Driven Slice-Timing Correction (`DEC-STC-01`)

## 1. Overview & Safety Boundaries

Step 4 and Step 4.1 implement scientifically validated, metadata-driven BIDS slice-timing correction (STC) inside [`src/preprocessing/slice_timing.py`](../src/preprocessing/slice_timing.py) and integrate it cleanly into the **21-stage** [`scripts/run_poc_v2.py`](../scripts/run_poc_v2.py) preprocessing pipeline.

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
5. **No Silent Cropping + Provenance Risk Tagging:** All $T$ volumes and `+reference_time` `StartTime` are preserved; derivative JSON sidecars record `BoundaryPaddingStrategy = "odd_point_reflection_4tr"`, `BoundaryExtrapolationDisabled = true`, `BoundaryEdgeVolumesAtRisk = [0, T-1]`, and explicit `VolumeCensoringGuidance`.

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
- **Temporal-Origin Shift:** BIDS derivative sidecars record `SliceTimingCorrected: true`, `SliceTimingReferenceTime`, `StartTime = reference_time`, `TemporalOriginShiftSeconds = reference_time`, `OriginalSliceTiming`, `Sources`, plus the Step 4.1 boundary provenance fields (`BoundaryPaddingStrategy`, `BoundaryExtrapolationDisabled`, `BoundaryEdgeVolumesAtRisk`, `VolumeCensoringGuidance`, `HeaderSliceDim`, `SliceAxisSource`).
- **Immutability:** Raw BIDS BOLD NIfTI and JSON sidecars remain byte-identical (verified via pre/post SHA-256 checksums); derivatives are emitted solely under `derivatives/*_desc-stc_bold.{nii.gz,json,_provenance.json}`.

---

## 6. Quantitative Synthetic Validation (`scripts/validate_step4_slice_timing.py`)

### 6.1 Analytical Multi-Tone Validation (`16x16x37x240`, Hall–Meyer $C^4$ Interior Bound = `0.023130`)

| Case ID | Regime & Axis | Whole-Series RMSE Before -> After (% Drop) | Interior RMSE (`[2:-2]`) | Phase Error (rad) Before -> After | Mean Pearson $r$ | Max Determinism Diff |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `case_a_sequential_k` | Sequential (`k`) | `0.3712 -> 0.01070` (`97.12%`) | `0.00151` | `0.0751 -> 0.00006` | `0.999996` | `0.0e+00` |
| `case_b_interleaved_k` | Interleaved (`k`) | `0.5270 -> 0.01570` (`97.02%`) | `0.00216` | `0.1192 -> 0.00008` | `0.999991` | `0.0e+00` |
| `case_c_multiband_k` | Multiband MB=3 (`k`) | `0.5793 -> 0.01855` (`96.80%`) | `0.00228` | `0.1214 -> 0.00006` | `0.999988` | `0.0e+00` |
| `case_d_negative_k_minus` | Reversed (`k-`) | `0.3750 -> 0.01114` (`97.03%`) | `0.00153` | `0.0758 -> 0.00006` | `0.999996` | `0.0e+00` |
| `case_e_axis_i` | Axis `i` (`37x16x16`) | `0.5270 -> 0.01566` (`97.03%`) | `0.00216` | `0.1192 -> 0.00008` | `0.999991` | `0.0e+00` |
| `case_e_axis_j` | Axis `j` (`16x37x16`) | `0.3712 -> 0.01070` (`97.12%`) | `0.00151` | `0.0751 -> 0.00006` | `0.999996` | `0.0e+00` |

### 6.2 Header Axis Controls & Negative Direction Discrimination
- Missing `SliceEncodingDirection` with header `slice_dim = 0, 1, 2` resolved cleanly to `i, j, k` (`interior_rmse_after = 0.00150` across all 3 axes).
- Missing `SliceEncodingDirection` with unset `slice_dim = None` skipped cleanly (`SKIPPED_UNKNOWN_SLICE_AXIS`, `corrected = false`, bit-identical BOLD preserved).
- Explicit direction vs conflicting `slice_dim` raised `InvalidSliceTimingError` fail-closed.
- `k-` outperformed unreversed `k` (`0.01114` vs `0.73972` RMSE, $66.4\times$ accuracy improvement).

### 6.3 Realistic 37-Slice × 240-Volume End-to-End 21-Stage Integration
Executed full 21-stage preprocessing (`run_poc_v2_pipeline`) on a source-acquisition-like synthetic BIDS scan (`64x64x37x240`, `37` interleaved slice timings, $3.0 \times 3.5 \times 4.0\text{ mm}^3$, `TR = 2.0 s`, all 200 canonical CC200 parcels in FOV) mapped onto canonical MNI/CC200 `(63, 75, 61)`:
- **Stage Ordering Verified:** Stage 2 estimated motion directly on raw uncorrected BOLD (`Mean FD = 0.2791 mm`); Stage 3 applied STC across `37/37` slices (`desc-stc_bold.nii.gz` shape `(64, 64, 37, 240)`); Stage 12 composed `motion_transforms` + coregistration + SyN in one spatial pullback into `(63, 75, 61, 240)`.
- **Downstream CC200 Outputs:** `roi_CC200.csv` shape `(200, 241)` (`240` timepoints preserved), `19,900` finite Fisher-$z$ features (`[-2.950, 2.698]`), Peak RSS `0.86 GB` ($\le 2.50\text{ GB}$ ceiling), total runtime `396.96s` (`< 600s` budget).

---

## 7. Known Limitations

1. **SciPy Backend vs AFNI `3dTshift`:** AFNI `3dTshift` and FSL `slicetimer` remain absent in this host/container environment; temporal interpolation uses SciPy `CubicSpline` with odd-point boundary reflection (`4 TR`) and `extrapolate=False` rather than AFNI binary output.
2. **Optional Boundary Censoring:** Edge volumes `[0, T-1]` exhibit natural-boundary zero-curvature bias on noiseless oscillatory extrema (while cutting noisy/transient instability); all $T$ volumes are retained cleanly and `[0, T-1]` are flagged via `BoundaryEdgeVolumesAtRisk` if downstream modeling wishes to censor edge frames.
3. **Real-Data Concordance Still Locked:** Because `real_data_authorized = false` (`overall_authorization_state = NOT_AUTHORIZED`), real participant BOLD execution (`sub-01`) and historical C-PAC concordance remain disabled (`NOT_ASSESSED`).
