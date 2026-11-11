# Step 4: Metadata-Driven Slice-Timing Correction (`DEC-STC-01`)

## 1. Overview & Safety Boundaries

Step 4 implements metadata-driven BIDS slice-timing correction (STC) inside [`src/preprocessing/slice_timing.py`](../src/preprocessing/slice_timing.py) and integrates it as **Stage 2** of the 20-stage [`scripts/run_poc_v2.py`](../scripts/run_poc_v2.py) preprocessing architecture.

**Strict Safety Invariants:**
- `execution_policy = STRICTLY_SYNTHETIC_ONLY`
- `real_data_authorized = false`
- `overall_authorization_state = NOT_AUTHORIZED`
- `sub-01 authorization_status = PENDING` (`cryptographic_signature = null`)
- All clinical directories (`NeuroADHD_Converted`, `sub-01`, `sub-2344857`, `down/outputs/adhd200`, `peking`) and symlink inputs are rejected fail-closed.
- Validation fixtures live externally in `../NeuroADHD_Step4_Synthetic/` (overrideable via `NEUROADHD_STEP4_ROOT`) and are never committed to Git.

---

## 2. Phase A Audit & 20-Stage Pipeline Ordering

### Existing Preprocessing Flow Audit
1. **BOLD entry:** BIDS sidecar validation (`BIDSInputValidator.validate_sidecar_json`) followed by `nib.load(bold_path)`.
2. **Motion estimation & realignment:** `realign_4d_bold(img)` in `src/preprocessing/motion.py` couples 6-DOF rigid estimation against the temporal mean EPI with per-volume rigid resampling (`rigid_mat.transform`).
3. **Functional reference & anatomical processing:** `generate_functional_reference`, `extract_anatomical_brain`, `correct_bias_field`, `segment_t1w_tissue`, `create_eroded_tissue_masks`.
4. **Registration & composed resampling:** `register_epi_to_t1w`, `register_t1w_to_mni`, `compose_transformations`, `resample_4d_bold`.
5. **Nuisance & bandpass:** `extract_acompcor_components`, `build_friston_24_regressors`, `build_confound_matrix`, simultaneous GLM + Butterworth `[0.01–0.10 Hz]` in `clean_functional_bold`.
6. **Parcellation & connectivity:** `extract_cc200_timeseries`, `compute_connectivity_vector` (`19,900` Fisher-$z$ edges).

### Correct Architectural Position & Scientific Rationale
Slice-timing correction is placed at **Stage 2**, immediately after **Stage 1: BIDS Input & Sidecar Metadata Validation** and strictly before **Stage 3: 3D Rigid Motion Realignment**:
1. Interpolating 1D voxel time series slice-by-slice **prior to any 3D motion realignment or spatial MNI/CC200 resampling** guarantees each slice's time series remains confined to its true acquisition plane before 3D spatial interpolation mixes intensity values across slices acquired at different times within `TR`.
2. Motion parameters (`Stage 3`) continue to be estimated prior to temporal bandpass filtering (`Stage 16`) and nuisance regression.
3. Keeping `realign_4d_bold` intact preserves deterministic bitwise compatibility for existing motion realignment unit tests while ensuring STC happens prior to spatial mixing.

---

## 3. BIDS Metadata Validation Rules (`validate_slice_timing_metadata`)

- **`RepetitionTime`:** Required numeric (non-boolean), finite, strictly `> 0`. Must agree with external `expected_tr` (when provided), explicit NIfTI temporal header `pixdim[4]` (supporting `sec`, `msec`, and `usec` units), and `VolumeTiming` step differences when present.
- **`SliceTiming`:** Required 1D numeric array matching exact slice count along the NIfTI axis designated by `SliceEncodingDirection` (never guesses axis from array length). All entries must be finite and satisfy `0 <= timing < TR`. Supports sequential (ascending/descending), interleaved, arbitrary custom orders, and repeated acquisition times from multiband SMS sequences.
- **`SliceEncodingDirection`:** Supports `i`, `i-`, `j`, `j-`, `k`, and `k-` mapping to NIfTI axes `0`, `1`, and `2`. Negative directions (`i-`, `j-`, `k-`) reverse the slice-index-to-timing correspondence (`voxel_slice_times = timing[::-1]`). When absent or `null`, applies the BIDS MRI specification default `'k'` (`axis 2`) and logs a documented assumption warning.
- **Missing Metadata Policy:** Never invents or infers `SliceTiming` from `TR` alone. Strict validation (`allow_missing_timing=False`) raises `MissingSliceTimingError`; pipeline runner default skips STC cleanly (`SKIPPED_MISSING_METADATA`) with explicit warnings in logs, derivatives report, and `provenance.json`, leaving original BOLD data unaltered. Present but invalid metadata always fails closed (`InvalidSliceTimingError`).
- **Simultaneous Acquisition:** When all slice timings agree within `1e-9 s` (or `num_slices == 1`), temporal interpolation is skipped (`SKIPPED_SIMULTANEOUS_ACQUISITION`) and voxel arrays remain bit-identical.
- **Double-Correction Prevention:** Rejects inputs carrying `SliceTimingCorrected: true` with `DoubleSliceTimingCorrectionError`.

---

## 4. Software Backend & Boundary Handling

Host inspection (`detect_stc_software()`) shows neither AFNI `3dTshift` nor FSL `slicetimer` is installed on the macOS workspace or in the minimal CI container. Rather than faking unavailability checks, Stage 2 selects pure-Python **SciPy** (`scipy.interpolate.CubicSpline`, `bc_type='not-a-knot'`, $C^2$ continuous) as the default interpolator, alongside selectable `quintic_bspline`, chord-detrended $C^1$ odd-mirror `fourier`, and `linear` backends:
- Non-periodic polynomial spline continuation evaluates sub-TR shifts $|\Delta t| < \mathrm{TR}$ at endpoints ($t=0$, $t=T-1$) without FFT circular wrap-around leakage across scan boundaries.
- Complex voxel dtypes, non-finite voxels, singular affines, and non-4D inputs fail closed; scaled integer NIfTIs (`scl_slope`, `scl_inter`) are decoded via `get_fdata(dtype=np.float64)` and exported with unit scaling `(1.0, 0.0)`, exact input 4D shape, exact spatial 4x4 affine, `pixdim[4] = TR`, and temporal unit `'sec'`.

---

## 5. Temporal Reference Point & Provenance Immutability

- **Default Reference Time:** `TR / 2.0` (e.g., `+1.0 s` for `TR = 2.0 s`), configurable within `[0, TR)`.
- **Temporal-Origin Shift:** BIDS derivative sidecars record `SliceTimingCorrected: true`, `SliceTimingReferenceTime`, `StartTime = reference_time`, `TemporalOriginShiftSeconds = reference_time`, explicit `TemporalOriginNote`, `OriginalSliceTiming`, and `Sources`.
- **Immutability:** Raw BIDS BOLD NIfTI and JSON sidecars are never modified or overwritten; Stage 2 writes separate BIDS derivatives under `derivatives/*_desc-stc_bold.{nii.gz,json,_provenance.json}` and verifies input SHA-256 checksums before and after execution.

---

## 6. Quantitative Synthetic Validation (`scripts/validate_step4_slice_timing.py`)

Evaluated against analytical continuous-time resting-state ground truth (`0.025, 0.048, 0.072 Hz`, `TR = 2.0 s`, `N = 240` volumes, Hall–Meyer $C^4$ interior cubic-spline error bound **`0.023130`**):

| Case ID | Regime & Axis | RMSE Before -> After (% Drop) | Phase Error (rad) Before -> After | Mean Pearson $r$ | Max Determinism Diff |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `case_a_sequential_k` | Sequential (`k`) | `0.3712 -> 0.00277` (`99.25%`) | `0.0751 -> 0.00006` | `1.000000` | `0.0e+00` |
| `case_b_interleaved_k` | Interleaved (`k`) | `0.5270 -> 0.00390` (`99.26%`) | `0.1192 -> 0.00008` | `1.000000` | `0.0e+00` |
| `case_c_multiband_k` | Multiband MB=3 (`k`) | `0.5793 -> 0.00409` (`99.29%`) | `0.1214 -> 0.00006` | `0.999999` | `0.0e+00` |
| `case_d_negative_k_minus` | Reversed (`k-`) | `0.3750 -> 0.00271` (`99.28%`) | `0.0758 -> 0.00006` | `1.000000` | `0.0e+00` |
| `case_e_axis_i` | Axis `i` (`37x16x16`) | `0.5270 -> 0.00393` (`99.25%`) | `0.1192 -> 0.00008` | `0.999999` | `0.0e+00` |
| `case_e_axis_j` | Axis `j` (`16x37x16`) | `0.3712 -> 0.00276` (`99.26%`) | `0.0751 -> 0.00006` | `1.000000` | `0.0e+00` |

Boundary/safety controls verified simultaneously: simultaneous skip (`SKIPPED_SIMULTANEOUS_ACQUISITION`), missing-timing strict error vs safe skip (`SKIPPED_MISSING_METADATA`), invalid timing rejection, input byte-identical SHA-256 preservation, and negative direction (`k-`) outperforming unreversed (`k`) by `> 10x`.

---

## 7. Known Limitations

1. **SciPy Backend vs AFNI `3dTshift`:** AFNI `3dTshift` and FSL `slicetimer` are absent in this host/container environment; the pure-Python SciPy `CubicSpline` implementation is mathematically documented and benchmarked on synthetic fixtures, not runtime-benchmarked against AFNI binary output.
2. **Real-Data Concordance Still Locked:** Because `real_data_authorized = false` (`overall_authorization_state = NOT_AUTHORIZED`), real participant BOLD execution and historical C-PAC concordance remain disabled and `NOT_ASSESSED`.

