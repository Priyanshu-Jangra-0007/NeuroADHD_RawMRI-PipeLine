# Step 5: B0 Susceptibility Distortion Correction (`DEC-SDC-02`)

## 1. Overview & Safety Boundaries

Step 5 implements metadata-driven B0 susceptibility-induced EPI distortion estimation, validation, and single-step composed spatial correction inside [`src/preprocessing/susceptibility_distortion.py`](../src/preprocessing/susceptibility_distortion.py), single-step pullback composition in [`src/preprocessing/registration.py`](../src/preprocessing/registration.py) and [`src/preprocessing/motion.py`](../src/preprocessing/motion.py), and the **22-stage** [`scripts/run_poc_v2.py`](../scripts/run_poc_v2.py) preprocessing pipeline (plus live dashboard telemetry in [`scripts/serve_dashboard.py`](../scripts/serve_dashboard.py), [`src/dashboard/service.py`](../src/dashboard/service.py), [`frontend/index.html`](../frontend/index.html), and [`frontend/app.js`](../frontend/app.js)).

**Strict Safety Invariants:**
- `execution_policy = STRICTLY_SYNTHETIC_ONLY`
- `real_data_authorized = false`
- `overall_authorization_state = NOT_AUTHORIZED`
- `sub-01 authorization_status = PENDING` (`cryptographic_signature = null`)
- `configs/research_authorization_register.json` remains unmodified and fail-closed.
- Clinical directory patterns (`NeuroADHD_Converted`, `sub-01`, `sub-2344857`, `down/outputs/adhd200`, `peking`, `kki`, `nyu`, `abide`, `adhd200`) and symlink substitutions are rejected fail-closed across all SDC discovery and validation paths.
- Generated synthetic NIfTI, PNG, and JSON validation fixtures reside strictly outside the Git repository under `../NeuroADHD_Step5_Synthetic/` (overridable via `NEUROADHD_STEP5_ROOT`) and are never tracked in Git.

---

## 2. Refactored 22-Stage Preprocessing Pipeline Ordering

Inserting Stage 4 (`B0 Susceptibility Distortion Correction`) preserves Step 4.1–4.2's strict physical invariance guarantees:
1. Motion parameters and framewise displacement (`estimate_motion_parameters`) are computed strictly on **raw, uncorrected BOLD** in Stage 2 before either STC or SDC.
2. Slice-timing correction (Stage 3) operates strictly on **original unresampled slice geometry**.
3. SDC field discovery, metadata validation, and field estimation run in Stage 4, producing an undistorted-to-distorted EPI pullback field `displacement_vox` $(X, Y, Z, 3)$ without mutating raw BIDS inputs.
4. Spatial transforms are composed in Stage 13 (`compose_transformations` + `resample_4d_bold(..., motion_transforms=..., sdc_displacement=...)`) via `compose_sdc_transform()` and evaluated on unresampled `stc_bold_img` in **one single spatial interpolation**.

| Stage | Module / Operation | Transform & Signal Role |
| :--- | :--- | :--- |
| **Stage 1/22** | **BIDS Input & Sidecar Metadata Validation** | Validates BIDS JSON sidecar, synthetic integrity manifest, and CC200 atlas before voxel manipulation. |
| **Stage 2/22** | **3D Rigid Motion Parameter Estimation** (`estimate_motion_parameters`) | Estimates 6-DOF rigid transforms (`FD` & `motion_parameters.txt`) directly on **raw uncorrected BOLD**; proven bit-identical with vs. without Stage 4 SDC (`motion_identical_with_vs_without_sdc = true`). |
| **Stage 3/22** | **Metadata-Driven Slice-Timing Correction** (`write_slice_timing_derivative`) | Corrects temporal slice offsets (`+TR/2` `StartTime` shift, 4-TR odd reflection, `extrapolate=False`) on unresampled native slice geometry (`derivatives/*_desc-stc_bold.{nii.gz,json}`). |
| **Stage 4/22** | **B0 Susceptibility Distortion Correction** (`run_susceptibility_distortion_correction`) | Discovers BIDS `fmap/` inputs, validates PE direction and `TotalReadoutTime`, probes external binaries (`topup`, `prelude`, `fugue`), estimates physical $B_0$ pullback `displacement_vox`, checks Jacobian invertibility ($J > 0$), and writes SDC derivatives (`*_desc-sdc_fieldmap.nii.gz`, `*_desc-sdc_displacement.nii.gz`, `*_desc-sdc_qc.png`, `*_desc-sdc_report.json`). |
| **Stage 5/22** | **Motion Transform (+ Optional SDC) Application** (`apply_motion_transforms`) | Applies Stage 2 rigid transforms (and Stage 4 `sdc_displacement` when active) to construct an aligned native reference series solely for Stage 6 functional reference estimation. |
| **Stage 6/22** | **Functional Reference Generation** (`generate_functional_reference`) | Computes 3D mean EPI and EPI brain mask on the motion-aligned (and unwarped when SDC applied) reference series. |
| **Stages 7–10/22** | **Anatomical T1w Processing** | `median_otsu` brain extraction, order-2 log-polynomial bias correction, 3-class GMM segmentation, $P \ge 0.96$ thresholding + 1-voxel 3D erosion. |
| **Stage 11/22** | **EPI-to-T1w Rigid Coregistration** (`register_epi_to_t1w`) | Rigid EPI-to-T1w alignment; when SDC is active, evaluates distorted vs. SDC-unwarped EPI against coregistered T1w (`correction_qc`: `ncc`, `nmi`, `mask_dice`) and regenerates contoured `*_desc-sdc_qc.png`. |
| **Stage 12/22** | **T1w-to-MNI SyN Nonlinear Normalization** (`register_t1w_to_mni`) | Computes affine + diffeomorphic SyN transform onto canonical MNI/CC200 target space. |
| **Stage 13/22** | **Single-Step Composed Pullback Resampling** (`resample_4d_bold(..., motion_transforms=..., sdc_displacement=...)`) | Evaluates single-step pullback $v_{\text{vol}}(t) = M_{\text{ref}\to\text{vol}}(t)\big(v_{\text{undist}} + u_{\text{SDC}}(v_{\text{undist}})\big)$ directly on unresampled `stc_bold_img`. |
| **Stages 14–22/22** | **Mask Resampling, Confounds, Parcellation, QC & Provenance** | Eroded CSF/WM resampling, 5 aCompCor + Friston-24 + linear/quadratic trends (31 columns, GSR OFF), simultaneous GLM + `[0.01–0.10 Hz]` Butterworth bandpass, 200-ROI CC200 extraction, 19,900 Fisher-$z$ features, 4 orthogonal QC figures, and JSON schema validation (`pipeline_stages_total = 22`). |

---

## 3. BIDS Fieldmap Discovery, Physics & Validation Rules

### 3.1 Strict BIDS Association Priority (`discover_sdc_inputs`)
1. **`B0FieldSource` / `B0FieldIdentifier` (Primary BIDS $\ge 1.6$ mechanism):** Matches exact field identifier tokens between the target BOLD sidecar's `B0FieldSource` and candidate `fmap/*.json` sidecars' `B0FieldIdentifier`.
2. **`IntendedFor` (Secondary BIDS mechanism):** Resolves relative/BIDS-URI targets canonically against `subject_dir`, `session_dir`, and `bids_root` (rejecting path traversals `..`, symlinks, and ambiguity across bare basenames).
3. **Fail-Closed Ambiguity & Isolation:**
   - Unassociated files in `fmap/` are never silently picked up and are recorded transparently in `unassociated_fmap_files`.
   - Conflicting or multiple ambiguous matching fieldmap sets raise `AmbiguousFieldmapAssociationError` (`FAILED_INVALID_INPUT`).

### 3.2 Phase-Encoding & Readout Timing Validation
- **`parse_phase_encoding_direction()`:** Accepts strictly BIDS voxel-index tokens `i`, `i-`, `j`, `j-`, `k`, `k-` mapping to axes `0, 1, 2` and polarities `+1, -1`. Ambiguous scanner/patient orientation strings (`AP`, `PA`, `RL`, `LR`, `IS`, `SI`) and null values are rejected fail-closed (`InvalidSDCInputError` / `MissingSDCMetadataError`).
- **`compute_total_readout_time()`:**
  - Uses explicit finite `TotalReadoutTime` $T_{\text{ro}} \in (0, 1.0)\text{ s}$ when present, rejecting negative/zero values, non-physical durations $\ge 1.0\text{ s}$, or values matching `RepetitionTime` (`TR`).
  - Otherwise computes:
    $$T_{\text{ro}} = \text{EffectiveEchoSpacing} \times (\text{ReconMatrixPE} - 1)$$
    requiring finite $\text{EffectiveEchoSpacing} \in (0, 0.1)\text{ s}$ and integer $\text{ReconMatrixPE} \ge 2$. Never substitutes `TR` or arbitrary defaults.
- **Fieldmap Modalities:**
  - `pepolar_epi` (`*_epi.nii.gz`): Validates identical voxel dimensions and affine alignment against BOLD (`atol = 1e-3`), same PE axis (`fmap_axis == bold_pe_axis`), and identifiability (either opposite polarity or distinct $T_{\text{ro}}$; rejects identical polarity + identical readout time fail-closed).
  - `direct_fieldmap` (`*_fieldmap.nii.gz`): Requires `Units` $\in \{\text{"Hz"}, \text{"rad/s"}\}$ (converts $\text{rad/s} \to \text{Hz}$ via $\Delta B_0 / (2\pi)$) and $\ge 50\%$ BOLD FOV overlap before linear resampling to the BOLD grid.
  - `phasediff` / `phase1` + `phase2`: Validates $0 < \text{EchoTime}_1 < \text{EchoTime}_2$ and `Units`; requires external `prelude` + `fugue` for phase unwrapping (never reinterprets wrapped phase radians as Hz).

### 3.3 Physical $B_0$-to-Displacement & Single-Step Pullback Equations
For off-resonance field $\Delta f(v)$ in $\text{Hz}$, PE axis $a \in \{0, 1, 2\}$, polarity $p \in \{-1, +1\}$, and total readout time $T_{\text{ro}}$ (`s`), `field_hz_to_displacement()` defines the undistorted-to-distorted pullback shift along unit PE vector $\mathbf{e}_a$:
$$\mathbf{u}_{\text{SDC}}(v) = p \cdot T_{\text{ro}} \cdot \Delta f(v)\, \mathbf{e}_a \quad [\text{voxels}], \qquad \mathbf{u}_{\text{mm}}(v) = A_{\text{EPI}, 3\times 3}\,\mathbf{u}_{\text{SDC}}(v)$$

`validate_displacement_field()` verifies `100%` finite values, zero orthogonal leakage along non-PE axes (`orthogonal_residual_max_vox = 0.0`), physical bound (`max_abs_mm <= 35.0 mm`), and diffeomorphic invertibility via the 1D directional Jacobian determinant along PE axis $a$:
$$J(v) = 1 + \frac{\partial u_a(v)}{\partial v_a}$$
Any voxel with $J(v) \le 0$ raises `SDCFoldingError` fail-closed.

In Stage 13 (`compose_sdc_transform()`), MNI/CC200 target voxel $i_{\text{MNI}}$ maps through target affine, SyN backward displacement $\Phi_{\text{SyN}}^{-1}$, EPI-to-T1w coregistration $\text{prealign}^{-1}$, EPI SDC pullback, and per-volume Stage 2 rigid motion $M_{\text{ref}\to\text{vol}}(t)$ in **one spatial resampling step**:
1. $x_{\text{target\_def}} = A_{\text{target}}[i_{\text{MNI}}, 1] + \Phi_{\text{SyN}}^{-1}(x_{\text{target}})$
2. $v_{\text{undist}} = A_{\text{EPI}}^{-1}\,\text{prealign}^{-1}\,x_{\text{target\_def}}$
3. $v_{\text{dist\_ref}} = v_{\text{undist}} + \mathbf{u}_{\text{SDC}}(v_{\text{undist}})$
4. $v_{\text{vol}}(t) = A_{\text{EPI}}^{-1}\,M_{\text{ref}\to\text{vol}}(t)\,A_{\text{EPI}}\,v_{\text{dist\_ref}}$

### 3.4 Odd-Slice TOPUP Configuration (`select_topup_config` / `format_topup_cnf`)
Standard FSL `b02b0.cnf` uses subsampling schedule `[4, 2, 1, 1]` or `[2, 2, 1, 1]`, which aborts on odd dimensions such as **37 slices**. When any dimension in `(X, Y, Z)` is odd, `select_topup_config()` selects preset `odd_dimension_safe_subsamp1` (`subsamp = [1, 1, 1, 1]`, `slice_modification_applied = false`) without padding, cropping, or duplicating anatomical slices.

---

## 4. External Tooling Audit & Honest Fallback Status Matrix

Host/container capability probing (`detect_sdc_software()`) on this machine confirms:
- External neuroimaging binaries: `topup = unavailable`, `applytopup = unavailable`, `prelude = unavailable`, `fugue = unavailable`, `fslmaths = unavailable`, `antsRegistration = unavailable`, `3dQwarp = unavailable`, `FSLDIR = unset`.
- Python packages: `numpy == 2.4.6`, `scipy == 1.17.1`, `nibabel == 5.4.2`, `dipy == 1.12.1`, `nilearn == 0.14.1`, `sdcflows = null`.

When external binaries are missing, NeuroADHD **never** fabricates fieldmaps or substitutes generic nonlinear/affine SyN warps:

| Condition / Request | Emitted SDC Status | `corrected` | Geometry Behavior & Provenance |
| :--- | :--- | :--- | :--- |
| Valid PEPOLAR pair + FSL `topup` installed (or verified `topup_runner`) | `APPLIED_PEPOLAR_TOPUP` | `true` | Runs `topup --fout=...`, validates $J > 0$, applies Stage 4/5/13 unwarping |
| Valid calibrated `*_fieldmap` (`Hz` or `rad/s`) | `APPLIED_DIRECT_FIELDMAP` | `true` | Resamples fieldmap onto BOLD grid, validates $J > 0$, applies single-step Stage 13 unwarping |
| Valid PEPOLAR pair + `topup` missing | `SKIPPED_TOOL_UNAVAILABLE` | `false` | Records validated `acqparams_lines`, `odd_dimension_safe_subsamp1`, `planned_command`, `missing_tools=["topup"]`; original BOLD preserved |
| Valid `phasediff` / `phase12` + `prelude`/`fugue` missing | `SKIPPED_TOOL_UNAVAILABLE` | `false` | Preserves original geometry, records missing unwrapping binaries (`["prelude", "fugue"]`); never treats wrapped radians as Hz |
| No BIDS-associated fieldmap (`B0FieldSource`/`IntendedFor` absent) or `sdc_method="none"` | `SKIPPED_NO_ASSOCIATED_FIELDMAP` | `false` | Preserves original geometry and Stage 2–3 outputs without distortion warps |
| Explicit `sdc_method="fieldmap_less_syn"` | `SKIPPED_UNSUPPORTED_METHOD` | `false` | Refuses generic unvalidated SyN-b0 fallback |
| Missing mandatory PE/TRT/Units/EchoTimes (`allow_incomplete_metadata=true`) | `SKIPPED_INSUFFICIENT_METADATA` | `false` | Preserves original geometry with exact missing metadata reason (`MissingSDCMetadataError` in strict mode) |
| Conflicting ambiguity, non-invertible Jacobian ($J \le 0$), FOV $< 50\%$, or safety violation | Raises (`AmbiguousFieldmapAssociationError`, `SDCFoldingError`, `InvalidSDCInputError`) | — | Fail-closed rejection before any spatial corruption |

---

## 5. Quantitative Synthetic Validation (`scripts/validate_step5_sdc.py`)

### 5.1 12-Case Phase-Encoding & Units Matrix (`26×26×21`, `TRT = 0.055 s`, `38 Hz` Peak B0)
Every direction `(i, i-, j, j-, k, k-)` was evaluated under both `Hz` and `rad/s` ($2\pi \cdot \Delta f$) inputs using physics-based forward distortion (`generate_synthetic_distorted_epi`, damped fixed-point residual $\le 9.6 \times 10^{-9}$ vox):

| PE & Units | Max Disp (`vox` / `mm`) | Min Jacobian | NCC Before -> After (Opp Pol) | NMI Before -> After | Dice Before -> After |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `i` (`Hz`, `rad/s`) | `2.0641` / `6.1924` | `0.636135` | `0.962718 -> 0.998198` (`0.918679`) | `1.6209 -> 1.8077` | `0.9920 -> 0.9970` |
| `i-` (`Hz`, `rad/s`) | `2.0641` / `6.1924` | `0.638591` | `0.965775 -> 0.999175` (`0.890771`) | `1.6355 -> 1.8349` | `0.9921 -> 0.9976` |
| `j` (`Hz`, `rad/s`) | `2.0641` / `6.1924` | `0.636644` | `0.975729 -> 0.998921` (`0.937838`) | `1.6447 -> 1.8382` | `0.9902 -> 0.9966` |
| `j-` (`Hz`, `rad/s`) | `2.0641` / `6.1924` | `0.632306` | `0.977241 -> 0.999558` (`0.914764`) | `1.6323 -> 1.8635` | `0.9915 -> 0.9977` |
| `k` (`Hz`, `rad/s`) | `2.0641` / `7.2245` | `0.626714` | `0.981340 -> 0.999379` (`0.933205`) | `1.6242 -> 1.8466` | `0.9899 -> 0.9978` |
| `k-` (`Hz`, `rad/s`) | `2.0641` / `7.2245` | `0.625516` | `0.980527 -> 0.999043` (`0.944949`) | `1.6174 -> 1.8395` | `0.9901 -> 0.9978` |

*Note:* `rad/s` outputs match `Hz` identically to 6+ decimals across every direction (`2*pi` conversion invariance verified).

### 5.2 Negative Physics, Odd-Slice & Tool Controls
- **Jacobian Folding Reject (`SDCFoldingError`):** Extreme $B_0$ gradient rejected fail-closed (`1728` voxels with $J \le 0$, `min J = -1.8139`).
- **Odd-37-Slice TOPUP Config `(64, 64, 37)`:** Selected `odd_dimension_safe_subsamp1` with `subsamp_schedule = [1, 1, 1, 1]` and `slice_modification_applied = false`.
- **Unequal Readout Times:** Accepted distinct `["0 1 0 0.04500000", "0 1 0 0.06800000"]`; rejected identical polarity + readout time fail-closed.
- **Ambiguity & Missing Tools:** Multiple ambiguous matching fieldmaps raised `AmbiguousFieldmapAssociationError`; `fieldmap_less_syn` returned `SKIPPED_UNSUPPORTED_METHOD`; `phasediff` missing `prelude`/`fugue` returned `SKIPPED_TOOL_UNAVAILABLE`; mocked PEPOLAR `topup_runner` returned `APPLIED_PEPOLAR_TOPUP` (`ncc_after = 0.999323`).

### 5.3 Single-Step Composed SDC Transform Landmark Validation
Validated `compose_sdc_transform()`, `apply_composed_sdc_volume()`, and `resample_4d_bold(..., sdc_displacement=...)` across 5 separated 3D landmarks (`sigma_mm = 3.6`, non-commuting rigid motion + PE=`j` SDC + EPI-to-T1w coreg + MNI SyN):
- **Single-Step Landmark Errors (`mm`):** `[0.048606, 0.050789, 0.023290, 0.027936, 0.023291]`
- **Mean / Max TRE:** `0.034782 mm` mean, `0.050789 mm` max (sub-voxel)
- **Sharpness Preservation:** Single-step composed mean landmark peak `82.0691` vs. sequential two-step resampling `70.5877` (**`+11.4814` intensity preservation gain**).

### 5.4 Canonical `64×64×37×240` 22-Stage End-to-End Integration (`PE=j`, `TRT=0.055 s`, `TR=2.0 s`, `32 Hz` Orbitofrontal Field)
1. **Run A — PEPOLAR Pair without FSL `topup`:**
   - Emitted `status = "SKIPPED_TOOL_UNAVAILABLE"`, `corrected = false`, `missing_tools = ["topup"]`, `subsamp_schedule = [1, 1, 1, 1]`, 2 valid `acqparams_lines`, `single_step_composed_resampling_with_sdc = false`.
   - Raw BIDS inputs byte-identical (pre/post SHA-256 verified); runtime `299.83 s`, Peak RSS `1.0454 GB` ($\le 2.50\text{ GB}$).
2. **Run B — Direct Fieldmap (`Hz`) via `IntendedFor` (Run 1, Run 2 & `sdc_method="none"` Motion Control):**
   - **Stage 4–13 Execution:** `status = "APPLIED_DIRECT_FIELDMAP"`, `corrected = true`, `max_abs_vox = 1.76`, `max_abs_mm = 6.16`, `min_jacobian = 0.8138`, `folding_voxel_count = 0`, `single_step_composed_resampling_with_sdc = true`.
   - **Stage 11 Coregistered T1w QC:** `ncc_before = 0.244804 -> ncc_after = 0.245178` (`ncc_improved = true`); cross-modality comparisons against `median_otsu`-stripped T1w have lower absolute NCC/NMI baseline than single-modality phantoms.
   - **Motion Invariance:** Stage 2 raw-BOLD motion parameters are bit-identical between `sdc_method="auto"` and `sdc_method="none"` (`motion_identical_with_vs_without_sdc = true`).
   - **CC200 Time Axis & Correlation Recovery:** `roi_timeseries_shape = [200, 240]`, `roi_csv_shape = [200, 241]`, `19,900` finite Fisher-$z$ features; `Parcel_066` & `Parcel_128` recovered **`r = +0.982149`**, `Parcel_093` & `Parcel_069` recovered **`r = -0.980832`**.
   - **Determinism & Resource Budget:** `Run 1` vs. `Run 2` `max_abs_diff_timeseries = 0.0`, `max_abs_diff_fisher_z = 0.0`; runtimes `305.03 s` / `305.81 s`; Peak RSS `1.1176 GB` ($\le 2.50\text{ GB}$ ceiling).

### 5.5 Regression Verification Across Steps 3, 4 & Unit Tests
- **Unit & Dashboard Integration Suite ([`tests/unit/test_susceptibility_distortion.py`](../tests/unit/test_susceptibility_distortion.py)):** `53/53 passed` (`0 failed`), covering PE/TRT rules, `B0FieldSource`/`IntendedFor` discovery, Jacobian folding, mocked TOPUP execution, odd-37 `subsamp=1`, single-step composed TRE, E2E provenance schema validation, and `DashboardDataService` ingestion.
- **Step 3 22-Stage Canonical Regression (`scripts/validate_step3_pipeline.py`):** `VERIFIED` (`22/22` stages, `sdc = SKIPPED_NO_ASSOCIATED_FIELDMAP`), `r_pos = +0.981839`, `r_neg = -0.986359`, `det_diff = 0.0`, runtime `101.99 s`, Peak RSS `0.9274 GB`.
- **Step 4.2 22-Stage Regression (`scripts/validate_step4_slice_timing.py`):** `VERIFIED` (`22/22` stages), `37×240` E2E `r_pos = +0.979133`, `r_neg = -0.979995`, `det_diff = 0.0`, runtimes `83.21 s` / `97.42 s`, Peak RSS `1.0594 GB`.

---

## 6. Known Limitations

1. **External FSL Binaries Unavailable on Host/Container:** `topup`, `applytopup`, `prelude`, `fugue`, `fslmaths`, `antsRegistration`, and AFNI `3dQwarp` are not installed on the current host; PEPOLAR (`*_epi`) and phase-difference (`*_phasediff`) fieldmap estimation honestly report `SKIPPED_TOOL_UNAVAILABLE` until FSL is installed (validated with an explicit mock runner in tests).
2. **Stage 4–5 vs. Stage 13 Interpolation Count:** When SDC is active, Stage 5 applies `sdc_displacement` together with Stage 2 rigid transforms *solely* to compute the unwarped Stage 6 functional reference for Stage 11 rigid EPI-to-T1w coregistration; the actual science timeseries consumed by Stages 14–20 comes exclusively from unresampled `stc_bold_img` via Stage 13's **single-step** composed pullback (`compose_sdc_transform`).
3. **Real-Data Concordance Locked:** Because `real_data_authorized = false` (`overall_authorization_state = NOT_AUTHORIZED`), real `sub-01` execution and historical C-PAC concordance remain disabled (`NOT_ASSESSED`).

