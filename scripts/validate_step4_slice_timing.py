#!/usr/bin/env python3
"""Step 4 Synthetic Ground-Truth Generator and Scientific Validation Script (DEC-STC-01).

Generates synthetic 4D BOLD datasets sampled from known continuous-time resting-state
signals across multiple acquisition regimes inside an external directory outside Git:
    <parent_dir>/NeuroADHD_Step4_Synthetic/  (or NEUROADHD_STEP4_ROOT env var)

Covers:
A. Sequential ascending acquisition along axis k (16x16x37x240, TR=2.0s)
B. Interleaved acquisition along axis k (16x16x37x240, TR=2.0s)
C. Multiband repeated acquisition timings along axis k (16x16x37x240, TR=2.0s)
D. Negative SliceEncodingDirection ('k-') reversing slice-to-timing index mapping
E. Alternative slice axes ('i' on 37x16x16x240 and 'j' on 16x37x16x240)
Plus simultaneous, missing-metadata, and invalid-metadata boundary controls.

SAFETY:
- Enforces ResearchAuthorizationGate (execution_policy=STRICTLY_SYNTHETIC_ONLY).
- Never accesses clinical MRI data or modifies authorization/register files.
- Keeps all generated NIfTI/JSON test fixtures outside Gate-tracked repository files.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.slice_timing import (  # noqa: E402
    SLICE_ENCODING_DIRECTION_MAP,
    InvalidSliceTimingError,
    MissingSliceTimingError,
    compute_file_sha256,
    correct_slice_timing_with_report,
    detect_stc_software,
    write_slice_timing_derivative,
)
from src.validation.authorization_gate import ResearchAuthorizationGate  # noqa: E402

# Store synthetic Step 4 MRI fixtures outside the Git repository by default
DEFAULT_STEP4_ROOT = PROJECT_ROOT.parent / "NeuroADHD_Step4_Synthetic"
STEP4_ROOT = Path(os.environ.get("NEUROADHD_STEP4_ROOT") or DEFAULT_STEP4_ROOT).resolve()

# Multi-frequency continuous-time resting-state signal model inside passband [0.01, 0.10 Hz]
SIGNAL_COMPONENTS: Tuple[Tuple[float, float, float], ...] = (
    # (amplitude, frequency_hz, base_phase_rad)
    (4.0, 0.025, 0.35),
    (3.0, 0.048, -0.80),
    (2.0, 0.072, 1.15),
)


def compute_theoretical_cubic_spline_bound(tr: float) -> float:
    """Derive analytical interior cubic spline error bound from Hall & Meyer (1976).

    For f in C^4 smoothed by cubic spline on uniform grid spacing h = TR:
        ||f - S_3||_inf <= (5 / 384) * TR^4 * ||f^(4)||_inf
    where ||f^(4)||_inf <= sum_k |A_k| * (2*pi*f_k)^4.
    """
    fourth_deriv_sup = sum(
        abs(amp) * ((2.0 * math.pi * freq) ** 4) for amp, freq, _ in SIGNAL_COMPONENTS
    )
    return float((5.0 / 384.0) * (tr ** 4) * fourth_deriv_sup)


def _eval_continuous_signal(
    t_seconds: np.ndarray,
    baseline: float,
    voxel_phase_offset: float,
) -> np.ndarray:
    """Evaluate exact analytical multi-tone resting-state BOLD time series at t_seconds."""
    sig = np.full(t_seconds.shape, float(baseline), dtype=np.float64)
    for amp, freq, base_phi in SIGNAL_COMPONENTS:
        sig = sig + amp * np.sin(2.0 * np.pi * freq * t_seconds + base_phi + voxel_phase_offset)
    return sig


def build_synthetic_stc_dataset(
    case_dir: Path,
    case_id: str,
    *,
    spatial_shape: Tuple[int, int, int] = (16, 16, 37),
    n_vols: int = 240,
    tr: float = 2.0,
    reference_time: Optional[float] = None,
    slice_timing: Sequence[float],
    slice_encoding_direction: str = "k",
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate observed 4D BOLD NIfTI + BIDS JSON + analytical reference-time ground truth."""
    case_dir.mkdir(parents=True, exist_ok=True)
    if reference_time is None:
        reference_time = tr / 2.0

    slice_axis, is_reversed = SLICE_ENCODING_DIRECTION_MAP[slice_encoding_direction]
    n_slices = spatial_shape[slice_axis]
    if len(slice_timing) != n_slices:
        raise ValueError(
            f"SliceTiming length {len(slice_timing)} != axis-{slice_axis} slice count {n_slices}."
        )

    timing_arr = np.asarray(slice_timing, dtype=np.float64)
    voxel_slice_times = timing_arr[::-1] if is_reversed else timing_arr

    rng = np.random.default_rng(seed)
    obs_data = np.zeros((*spatial_shape, n_vols), dtype=np.float32)
    gt_data = np.zeros((*spatial_shape, n_vols), dtype=np.float64)

    # Create interior spherical ROI mask matching physical voxel coordinates so edge artifacts
    # are cleanly represented alongside core brain signals
    gx, gy, gz = np.ogrid[: spatial_shape[0], : spatial_shape[1], : spatial_shape[2]]
    cx, cy, cz = (d / 2.0 for d in spatial_shape)
    rx, ry, rz = (max(2.0, d * 0.42) for d in spatial_shape)
    brain_mask = (((gx - cx) / rx) ** 2 + ((gy - cy) / ry) ** 2 + ((gz - cz) / rz) ** 2) <= 1.0

    vol_onsets = np.arange(n_vols, dtype=np.float64) * float(tr)
    gt_times = vol_onsets + float(reference_time)

    # Spatial variation in baseline intensity and smooth spatial phase gradient
    baselines = (100.0 + rng.uniform(-8.0, 8.0, size=spatial_shape)) * brain_mask
    phase_offsets = rng.uniform(-0.4, 0.4, size=spatial_shape)

    for s_idx in range(n_slices):
        st = float(voxel_slice_times[s_idx])
        obs_times = vol_onsets + st

        sel: List[Any] = [slice(None), slice(None), slice(None)]
        sel[slice_axis] = s_idx
        sl = tuple(sel)

        b_slice = baselines[sl]
        p_slice = phase_offsets[sl]
        m_slice = brain_mask[sl]

        d1, d2 = b_slice.shape
        slice_obs = np.zeros((d1, d2, n_vols), dtype=np.float32)
        slice_gt = np.zeros((d1, d2, n_vols), dtype=np.float64)

        for i1 in range(d1):
            for i2 in range(d2):
                if m_slice[i1, i2]:
                    ts_obs = _eval_continuous_signal(obs_times, float(b_slice[i1, i2]), float(p_slice[i1, i2]))
                    ts_gt = _eval_continuous_signal(gt_times, float(b_slice[i1, i2]), float(p_slice[i1, i2]))
                    slice_obs[i1, i2, :] = ts_obs.astype(np.float32)
                    slice_gt[i1, i2, :] = ts_gt

        out_sel = (sl[0], sl[1], sl[2], slice(None))
        obs_data[out_sel] = slice_obs
        gt_data[out_sel] = slice_gt

    affine = np.array(
        [
            [3.0, 0.0, 0.0, -24.0],
            [0.0, 3.0, 0.0, -24.0],
            [0.0, 0.0, 3.0, -54.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    img = nib.Nifti1Image(obs_data, affine)
    img.header.set_zooms((3.0, 3.0, 3.0, float(tr)))
    img.header.set_xyzt_units("mm", "sec")

    func_dir = case_dir / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    bold_path = func_dir / f"sub-synthetic_{case_id}_task-rest_bold.nii.gz"
    json_path = func_dir / f"sub-synthetic_{case_id}_task-rest_bold.json"
    gt_path = case_dir / f"{case_id}_analytical_ground_truth.nii.gz"
    mask_path = case_dir / f"{case_id}_brain_mask.npy"

    nib.save(img, str(bold_path))
    gt_img = nib.Nifti1Image(gt_data, affine)
    gt_img.header.set_zooms((3.0, 3.0, 3.0, float(tr)))
    gt_img.header.set_xyzt_units("mm", "sec")
    nib.save(gt_img, str(gt_path))
    np.save(mask_path, brain_mask)

    sidecar = {
        "RepetitionTime": float(tr),
        "TaskName": "rest",
        "SliceTiming": [round(float(x), 8) for x in slice_timing],
        "SliceEncodingDirection": slice_encoding_direction,
        "MagneticFieldStrength": 3.0,
        "Manufacturer": "SyntheticStep4Generator",
    }
    json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    return {
        "case_id": case_id,
        "bold_nii": bold_path,
        "bold_json": json_path,
        "gt_nii": gt_path,
        "gt_data": gt_data,
        "brain_mask": brain_mask,
        "affine": affine,
        "spatial_shape": spatial_shape,
        "n_vols": n_vols,
        "tr": float(tr),
        "reference_time": float(reference_time),
        "slice_axis": slice_axis,
        "voxel_slice_times": voxel_slice_times,
        "sidecar": sidecar,
    }


def _wrap_angle_diff(angle_diff: np.ndarray) -> np.ndarray:
    """Wrap phase differences strictly into [-pi, pi]."""
    return (angle_diff + np.pi) % (2.0 * np.pi) - np.pi


def _compute_dft_phase_error(
    series: np.ndarray,
    gt_series: np.ndarray,
    tr: float,
) -> Dict[str, float]:
    """Compute energy-weighted Fourier phase error (radians & equivalent seconds) at resting-state tones."""
    # Demean to isolate AC oscillatory phase cleanly across (N_vox, T)
    s_dm = series - np.mean(series, axis=-1, keepdims=True)
    g_dm = gt_series - np.mean(gt_series, axis=-1, keepdims=True)

    # Interior window avoids edge transient contamination in DFT phase comparison
    if s_dm.shape[-1] > 16:
        s_dm = s_dm[..., 4:-4]
        g_dm = g_dm[..., 4:-4]

    n_pts = s_dm.shape[-1]
    freqs = np.fft.rfftfreq(n_pts, d=tr)
    spec_s = np.fft.rfft(s_dm, axis=-1)
    spec_g = np.fft.rfft(g_dm, axis=-1)

    phase_errors_rad: List[float] = []
    timing_errors_sec: List[float] = []
    weights: List[float] = []

    for amp, target_f, _ in SIGNAL_COMPONENTS:
        bin_idx = int(np.argmin(np.abs(freqs - target_f)))
        true_coeff = spec_g[:, bin_idx]
        est_coeff = spec_s[:, bin_idx]
        valid = np.abs(true_coeff) > 1e-6
        if not np.any(valid):
            continue
        ang_diff = np.abs(_wrap_angle_diff(np.angle(est_coeff[valid]) - np.angle(true_coeff[valid])))
        mean_rad = float(np.mean(ang_diff))
        equiv_sec = float(mean_rad / (2.0 * np.pi * target_f))
        phase_errors_rad.append(mean_rad)
        timing_errors_sec.append(equiv_sec)
        weights.append(float(amp ** 2))

    if not phase_errors_rad:
        return {"mean_abs_phase_error_rad": 0.0, "equivalent_timing_error_sec": 0.0}

    w_arr = np.asarray(weights, dtype=np.float64)
    w_arr /= np.sum(w_arr)
    return {
        "mean_abs_phase_error_rad": float(np.sum(np.asarray(phase_errors_rad) * w_arr)),
        "equivalent_timing_error_sec": float(np.sum(np.asarray(timing_errors_sec) * w_arr)),
    }


def evaluate_case_accuracy(
    dataset: Dict[str, Any],
    corrected_img: nib.Nifti1Image,
) -> Dict[str, Any]:
    """Compute all 9 Phase F scientific validation metrics on a synthetic dataset."""
    tr = dataset["tr"]
    ref_time = dataset["reference_time"]
    slice_axis = dataset["slice_axis"]
    voxel_times = dataset["voxel_slice_times"]
    brain_mask = dataset["brain_mask"]
    gt_data = dataset["gt_data"]

    raw_img = nib.load(str(dataset["bold_nii"]))
    raw_data = np.asarray(raw_img.get_fdata(dtype=np.float64), dtype=np.float64)
    corr_data = np.asarray(corrected_img.get_fdata(dtype=np.float64), dtype=np.float64)

    # 1-4. Whole-brain RMSE and MAE before & after correction
    raw_vox = raw_data[brain_mask, :]
    corr_vox = corr_data[brain_mask, :]
    gt_vox = gt_data[brain_mask, :]

    rmse_before = float(np.sqrt(np.mean((raw_vox - gt_vox) ** 2)))
    rmse_after = float(np.sqrt(np.mean((corr_vox - gt_vox) ** 2)))
    rmse_interior_after = float(np.sqrt(np.mean((corr_vox[:, 2:-2] - gt_vox[:, 2:-2]) ** 2)))

    mae_before = float(np.mean(np.abs(raw_vox - gt_vox)))
    mae_after = float(np.mean(np.abs(corr_vox - gt_vox)))

    # 5. Phase error before and after correction
    phase_before = _compute_dft_phase_error(raw_vox, gt_vox, tr=tr)
    phase_after = _compute_dft_phase_error(corr_vox, gt_vox, tr=tr)

    # 6. Temporal signal correlation with analytical ground truth
    def _voxel_corrs(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
        est_c = est - np.mean(est, axis=1, keepdims=True)
        ref_c = ref - np.mean(ref, axis=1, keepdims=True)
        denom = np.sqrt(np.sum(est_c ** 2, axis=1) * np.sum(ref_c ** 2, axis=1))
        valid = denom > 1e-12
        return np.sum(est_c[valid] * ref_c[valid], axis=1) / denom[valid]

    corr_r_before = _voxel_corrs(raw_vox, gt_vox)
    corr_r_after = _voxel_corrs(corr_vox, gt_vox)

    # 7. Slice-wise accuracy
    n_slices = dataset["spatial_shape"][slice_axis]
    slice_wise: List[Dict[str, Any]] = []
    non_ref_improved = 0
    non_ref_total = 0

    for s_idx in range(n_slices):
        sel: List[Any] = [slice(None), slice(None), slice(None)]
        sel[slice_axis] = s_idx
        sl = tuple(sel)
        m_sl = brain_mask[sl]
        if not np.any(m_sl):
            continue
        r_sl = raw_data[sl[0], sl[1], sl[2], :][m_sl, :]
        c_sl = corr_data[sl[0], sl[1], sl[2], :][m_sl, :]
        g_sl = gt_data[sl[0], sl[1], sl[2], :][m_sl, :]

        sl_rmse_b = float(np.sqrt(np.mean((r_sl - g_sl) ** 2)))
        sl_rmse_a = float(np.sqrt(np.mean((c_sl - g_sl) ** 2)))
        sl_mae_b = float(np.mean(np.abs(r_sl - g_sl)))
        sl_mae_a = float(np.mean(np.abs(c_sl - g_sl)))
        shift_s = abs(float(ref_time) - float(voxel_times[s_idx]))
        if shift_s > 1e-6:
            non_ref_total += 1
            if sl_rmse_a < sl_rmse_b:
                non_ref_improved += 1

        slice_wise.append(
            {
                "slice_index": int(s_idx),
                "acquisition_time_sec": float(voxel_times[s_idx]),
                "abs_shift_to_ref_sec": shift_s,
                "rmse_before": sl_rmse_b,
                "rmse_after": sl_rmse_a,
                "mae_before": sl_mae_b,
                "mae_after": sl_mae_a,
            }
        )

    # 8-9. Geometry & finite checks
    total_voxels = int(corr_data.size)
    finite_count = int(np.count_nonzero(np.isfinite(corr_data)))
    rmse_reduction_pct = (
        float(100.0 * (rmse_before - rmse_after) / rmse_before) if rmse_before > 1e-12 else 0.0
    )

    return {
        "temporal_rmse_before": rmse_before,
        "temporal_rmse_after": rmse_after,
        "temporal_rmse_interior_after": rmse_interior_after,
        "rmse_reduction_percent": rmse_reduction_pct,
        "mae_before": mae_before,
        "mae_after": mae_after,
        "phase_error_before": phase_before,
        "phase_error_after": phase_after,
        "pearson_r_before_mean": float(np.mean(corr_r_before)),
        "pearson_r_before_min": float(np.min(corr_r_before)),
        "pearson_r_after_mean": float(np.mean(corr_r_after)),
        "pearson_r_after_min": float(np.min(corr_r_after)),
        "slice_wise_count": len(slice_wise),
        "non_reference_slices_total": non_ref_total,
        "non_reference_slices_improved": non_ref_improved,
        "slice_wise_summary": slice_wise[:5] + slice_wise[-3:],  # Compact sample for JSON report
        "whole_brain_rmse_before": rmse_before,
        "whole_brain_rmse_after": rmse_after,
        "finite_output_count": finite_count,
        "total_output_count": total_voxels,
        "all_finite": finite_count == total_voxels,
        "output_shape": list(corrected_img.shape),
        "output_header_tr": float(corrected_img.header.get_zooms()[3]),
        "affine_preserved": bool(np.allclose(corrected_img.affine, dataset["affine"], atol=1e-12)),
    }


def run_step4_validation(root_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Generate external Step 4 synthetic benchmarks and run full scientific validation."""
    out_root = Path(root_dir or STEP4_ROOT).resolve()

    # Enforce Phase A.6 Research Authorization Gate before generating or running anything
    auth_gate = ResearchAuthorizationGate(base_dir=PROJECT_ROOT)
    auth_state = auth_gate.evaluate_register()
    auth_gate.check_execution_allowed(requested_mode="synthetic", allow_real_scans=False)

    print("=" * 78)
    print("NEUROADHD STEP 4: SLICE-TIMING CORRECTION SCIENTIFIC VALIDATION")
    print(f"Policy        : {auth_state['execution_policy']}")
    print(f"Real Data Auth: {auth_state['real_data_authorized']} ({auth_state['overall_authorization_state']})")
    print(f"Fixture Root  : {out_root}")
    print("=" * 78)

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tr = 2.0
    n_vols = 240
    n_slices_std = 37
    ref_time = tr / 2.0

    # Define Phase E synthetic regimes
    seq_timings = np.linspace(0.0, tr, n_slices_std, endpoint=False).tolist()

    # Interleaved: even slice indices first [0, 2, ...], then odd [1, 3, ...]
    interleaved_order = list(range(0, n_slices_std, 2)) + list(range(1, n_slices_std, 2))
    interleaved_timings = [0.0] * n_slices_std
    for acq_step, sl_idx in enumerate(interleaved_order):
        interleaved_timings[sl_idx] = float(acq_step * tr / n_slices_std)

    # Multiband MB=3 with 13 temporal groups (repeated slice timings)
    n_mb_groups = 13
    multiband_timings = [float((idx % n_mb_groups) * tr / n_mb_groups) for idx in range(n_slices_std)]

    case_specs = [
        {
            "case_id": "case_a_sequential_k",
            "label": "Sequential Ascending (k)",
            "spatial_shape": (16, 16, 37),
            "slice_timing": seq_timings,
            "direction": "k",
            "seed": 42,
        },
        {
            "case_id": "case_b_interleaved_k",
            "label": "Interleaved Even-First (k)",
            "spatial_shape": (16, 16, 37),
            "slice_timing": interleaved_timings,
            "direction": "k",
            "seed": 43,
        },
        {
            "case_id": "case_c_multiband_k",
            "label": "Multiband MB=3 Repeated Groups (k)",
            "spatial_shape": (16, 16, 37),
            "slice_timing": multiband_timings,
            "direction": "k",
            "seed": 44,
        },
        {
            "case_id": "case_d_negative_k_minus",
            "label": "Negative SliceEncodingDirection (k-)",
            "spatial_shape": (16, 16, 37),
            "slice_timing": seq_timings,
            "direction": "k-",
            "seed": 45,
        },
        {
            "case_id": "case_e_axis_i",
            "label": "Alternative Slice Axis (i)",
            "spatial_shape": (37, 16, 16),
            "slice_timing": interleaved_timings,
            "direction": "i",
            "seed": 46,
        },
        {
            "case_id": "case_e_axis_j",
            "label": "Alternative Slice Axis (j)",
            "spatial_shape": (16, 37, 16),
            "slice_timing": seq_timings,
            "direction": "j",
            "seed": 47,
        },
    ]

    # Theoretical error bound from signal bandwidth & Hall-Meyer C4 cubic spline bound
    theoretical_interior_bound = compute_theoretical_cubic_spline_bound(tr)
    print(f"Analytical Hall-Meyer Cubic Spline Interior Bound: {theoretical_interior_bound:.6f}")

    tracemalloc.start()
    software_info = detect_stc_software()
    case_results: Dict[str, Any] = {}

    for spec in case_specs:
        cid = spec["case_id"]
        cdir = out_root / cid
        dataset = build_synthetic_stc_dataset(
            case_dir=cdir,
            case_id=cid,
            spatial_shape=spec["spatial_shape"],
            n_vols=n_vols,
            tr=tr,
            reference_time=ref_time,
            slice_timing=spec["slice_timing"],
            slice_encoding_direction=spec["direction"],
            seed=spec["seed"],
        )

        # Record SHA-256 checksums before correction
        in_bold_sha = compute_file_sha256(dataset["bold_nii"])
        in_json_sha = compute_file_sha256(dataset["bold_json"])

        deriv_dir = cdir / "derivatives"
        corr_img_run1, prov_rec = write_slice_timing_derivative(
            bold_path=dataset["bold_nii"],
            bold_json_path=dataset["bold_json"],
            derivative_dir=deriv_dir,
            reference_time=ref_time,
            interpolation_method="cubic_spline",
        )
        # Re-run to verify bitwise determinism
        corr_img_run2, _ = correct_slice_timing_with_report(
            bold_image=dataset["bold_nii"],
            bold_metadata=dataset["bold_json"],
            reference_time=ref_time,
            interpolation_method="cubic_spline",
        )
        determinism_max_diff = float(
            np.max(
                np.abs(
                    np.asarray(corr_img_run1.dataobj, dtype=np.float64)
                    - np.asarray(corr_img_run2.dataobj, dtype=np.float64)
                )
            )
        )

        # Verify byte-identical preservation of inputs
        post_bold_sha = compute_file_sha256(dataset["bold_nii"])
        post_json_sha = compute_file_sha256(dataset["bold_json"])
        inputs_immutable = (post_bold_sha == in_bold_sha) and (post_json_sha == in_json_sha)

        metrics = evaluate_case_accuracy(dataset, corr_img_run1)

        # For negative direction (k-), confirm unreversed 'k' interpretation degrades alignment
        extra_checks: Dict[str, Any] = {}
        if spec["direction"] == "k-":
            wrong_meta = dict(dataset["sidecar"])
            wrong_meta["SliceEncodingDirection"] = "k"
            wrong_img, _ = correct_slice_timing_with_report(
                bold_image=dataset["bold_nii"],
                bold_metadata=wrong_meta,
                reference_time=ref_time,
            )
            wrong_metrics = evaluate_case_accuracy(dataset, wrong_img)
            extra_checks["unreversed_direction_rmse"] = wrong_metrics["temporal_rmse_after"]
            assert metrics["temporal_rmse_after"] < wrong_metrics["temporal_rmse_after"] * 0.1, (
                "Negative SliceEncodingDirection reversal failed to outperform unreversed mapping."
            )

        # Scientific acceptance criteria derived strictly from signal theory & geometry
        assert metrics["temporal_rmse_interior_after"] <= theoretical_interior_bound, (
            f"{cid}: interior RMSE {metrics['temporal_rmse_interior_after']:.6f} exceeded "
            f"theoretical spline bound {theoretical_interior_bound:.6f}"
        )
        assert metrics["temporal_rmse_after"] < metrics["temporal_rmse_before"] * 0.2, (
            f"{cid}: insufficient RMSE improvement"
        )
        assert metrics["mae_after"] < metrics["mae_before"] * 0.2, (
            f"{cid}: insufficient MAE improvement"
        )
        assert (
            metrics["phase_error_after"]["mean_abs_phase_error_rad"]
            < metrics["phase_error_before"]["mean_abs_phase_error_rad"] * 0.1
        ), f"{cid}: insufficient phase realignment"
        assert metrics["pearson_r_after_mean"] > 0.999, f"{cid}: correlation below 0.999"
        assert metrics["all_finite"], f"{cid}: non-finite values in output"
        assert metrics["output_shape"] == [*spec["spatial_shape"], n_vols]
        assert math.isclose(metrics["output_header_tr"], tr, abs_tol=1e-5)
        assert metrics["affine_preserved"]
        assert determinism_max_diff == 0.0
        assert inputs_immutable

        print(
            f"  [{cid:24s}] RMSE: {metrics['temporal_rmse_before']:.4f} -> {metrics['temporal_rmse_after']:.5f} "
            f"({metrics['rmse_reduction_percent']:.2f}% drop) | "
            f"Phase(rad): {metrics['phase_error_before']['mean_abs_phase_error_rad']:.4f} -> "
            f"{metrics['phase_error_after']['mean_abs_phase_error_rad']:.5f} | "
            f"r={metrics['pearson_r_after_mean']:.6f} | det={determinism_max_diff:.1e}"
        )

        case_results[cid] = {
            "label": spec["label"],
            "spatial_shape": list(spec["spatial_shape"]),
            "n_volumes": n_vols,
            "tr": tr,
            "reference_time": ref_time,
            "slice_encoding_direction": spec["direction"],
            "slice_order_interpretation": prov_rec["slice_order_interpretation"],
            "status": prov_rec["status"],
            "num_corrected_slices": int(prov_rec["num_corrected_slices"]),
            "duration_seconds": prov_rec["duration_seconds"],
            "determinism_max_abs_diff": determinism_max_diff,
            "inputs_immutable_sha256": inputs_immutable,
            "theoretical_interior_spline_bound": theoretical_interior_bound,
            **metrics,
            **extra_checks,
        }

    # Additional boundary/control validations (Simultaneous, Missing, Invalid)
    sim_dir = out_root / "control_simultaneous"
    sim_ds = build_synthetic_stc_dataset(
        case_dir=sim_dir,
        case_id="control_simultaneous",
        spatial_shape=(16, 16, 37),
        n_vols=n_vols,
        tr=tr,
        reference_time=ref_time,
        slice_timing=[1.0] * 37,
        slice_encoding_direction="k",
        seed=99,
    )
    sim_img, sim_rep = correct_slice_timing_with_report(sim_ds["bold_nii"], sim_ds["bold_json"])
    assert sim_rep["status"] == "SKIPPED_SIMULTANEOUS_ACQUISITION"
    assert sim_rep["num_corrected_slices"] == 0
    np.testing.assert_array_equal(
        np.asarray(sim_img.dataobj),
        np.asarray(nib.load(str(sim_ds["bold_nii"])).dataobj),
    )

    # Missing SliceTiming check
    miss_meta = {"RepetitionTime": tr, "TaskName": "rest"}
    miss_img, miss_rep = correct_slice_timing_with_report(
        sim_ds["bold_nii"], miss_meta, allow_missing_timing=True
    )
    assert miss_rep["status"] == "SKIPPED_MISSING_METADATA"
    np.testing.assert_array_equal(
        np.asarray(miss_img.dataobj),
        np.asarray(nib.load(str(sim_ds["bold_nii"])).dataobj),
    )
    try:
        correct_slice_timing_with_report(sim_ds["bold_nii"], miss_meta, allow_missing_timing=False)
        raise AssertionError("Expected MissingSliceTimingError")
    except MissingSliceTimingError:
        pass

    # Invalid SliceTiming check
    bad_meta = {"RepetitionTime": tr, "SliceTiming": [tr + 0.1] * 37}
    try:
        correct_slice_timing_with_report(sim_ds["bold_nii"], bad_meta, allow_missing_timing=True)
        raise AssertionError("Expected InvalidSliceTimingError")
    except InvalidSliceTimingError:
        pass

    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = {
        "step": "Step 4: Slice-Timing Correction Scientific Validation (DEC-STC-01)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VERIFIED",
        "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
        "real_data_authorized": False,
        "overall_authorization_state": auth_state["overall_authorization_state"],
        "software": software_info,
        "interpolation_method": "cubic_spline",
        "reference_time_seconds": ref_time,
        "tr_seconds": tr,
        "n_volumes": n_vols,
        "spatial_dimensions_default": [16, 16, 37],
        "signal_model_frequencies_hz": [f for _, f, _ in SIGNAL_COMPONENTS],
        "acceptance_criterion": {
            "basis": "Hall & Meyer (1976) C4 cubic spline interior error bound + monotonic improvement over uncorrected sampling",
            "theoretical_interior_rmse_bound": theoretical_interior_bound,
            "required_min_mean_pearson_r": 0.999,
            "required_finite_fraction": 1.0,
        },
        "peak_traced_memory_mib": round(peak_mem_bytes / (1024.0 * 1024.0), 3),
        "cases": case_results,
        "controls": {
            "simultaneous_acquisition_status": sim_rep["status"],
            "missing_slice_timing_skip_status": miss_rep["status"],
            "missing_slice_timing_strict_rejected": True,
            "invalid_slice_timing_rejected": True,
        },
    }

    report_path = out_root / "step4_slice_timing_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Also mirror ONLY the small JSON summary (no NIfTI!) if desired under outputs/step4_validation
    # Note: do NOT track outputs/ in Git; keep synthetic NIfTIs strictly outside the repo!
    print("\n" + "=" * 78)
    print("STEP 4 VALIDATION PASSED ACROSS ALL SYNTHETIC ACQUISITION REGIMES")
    print(f"Report Written: {report_path}")
    print("=" * 78)
    return report


if __name__ == "__main__":
    run_step4_validation()
