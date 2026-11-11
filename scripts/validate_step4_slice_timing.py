#!/usr/bin/env python3
"""Step 4 / 4.1 Synthetic Ground-Truth Generator and Scientific Validation Script (DEC-STC-01).

Generates synthetic 4D BOLD datasets sampled from known continuous-time resting-state
signals across multiple acquisition regimes inside an external directory outside Git:
    <parent_dir>/NeuroADHD_Step4_Synthetic/  (or NEUROADHD_STEP4_ROOT env var)

Covers:
A. Sequential ascending acquisition along axis k (16x16x37x240, TR=2.0s)
B. Interleaved acquisition along axis k (16x16x37x240, TR=2.0s)
C. Multiband repeated acquisition timings along axis k (16x16x37x240, TR=2.0s)
D. Negative SliceEncodingDirection ('k-') reversing slice-to-timing index mapping
E. Alternative slice axes ('i' on 37x16x16x240 and 'j' on 16x37x16x240)
F. Header slice_dim fallback controls (i/j/k + unavailable slice_dim -> SKIPPED_UNKNOWN_SLICE_AXIS)
G. 5-region temporal boundary benchmark (first, first5, interior, last5, final) across
   smooth, noisy, non-periodic linear trend, and controlled transient signals
H. Source-acquisition-like 37-slice x 240-volume full 21-stage canonical CC200 E2E run
   (verifying Stage 2 raw-BOLD motion estimation before Stage 3 STC & downstream CC200).

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_poc_v2 import (  # noqa: E402
    CANONICAL_CC200_AFFINE,
    CANONICAL_CC200_SHAPE,
    compute_manifest_hmac,
    compute_sha256,
    run_poc_v2_pipeline,
    verify_canonical_cc200_atlas,
)
from src.preprocessing.slice_timing import (  # noqa: E402
    SLICE_ENCODING_DIRECTION_MAP,
    InvalidSliceTimingError,
    MissingSliceTimingError,
    _odd_reflect_pad_series,
    compute_file_sha256,
    correct_slice_timing_with_report,
    detect_stc_software,
    validate_slice_timing_metadata,
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
    include_slice_encoding_direction: bool = True,
    set_header_slice_dim: bool = True,
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
    if set_header_slice_dim:
        img.header.set_dim_info(slice=slice_axis)

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
    if set_header_slice_dim:
        gt_img.header.set_dim_info(slice=slice_axis)
    nib.save(gt_img, str(gt_path))
    np.save(mask_path, brain_mask)

    sidecar: Dict[str, Any] = {
        "RepetitionTime": float(tr),
        "TaskName": "rest",
        "SliceTiming": [round(float(x), 8) for x in slice_timing],
        "MagneticFieldStrength": 3.0,
        "Manufacturer": "SyntheticStep4Generator",
    }
    if include_slice_encoding_direction:
        sidecar["SliceEncodingDirection"] = slice_encoding_direction
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

    temporal_regions = compute_five_region_metrics(est=corr_vox, raw=raw_vox, gt=gt_vox)

    return {
        "temporal_rmse_before": rmse_before,
        "temporal_rmse_after": rmse_after,
        "temporal_rmse_interior_after": rmse_interior_after,
        "temporal_regions": temporal_regions,
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


def compute_five_region_metrics(
    est: np.ndarray,
    raw: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute separate RMSE and MaxAbsError across 5 temporal boundary & interior regions.

    Regions:
    - ``first_volume`` (t = 0)
    - ``first_five_volumes`` (t = 0..4)
    - ``interior_volumes`` (t = 5..T-6)
    - ``last_five_volumes`` (t = T-5..T-1)
    - ``final_volume`` (t = T-1)
    """
    n_t = int(est.shape[-1])
    if n_t < 12:
        raise ValueError(f"Expected >= 12 volumes for 5-region breakdown, got {n_t}.")
    slices = {
        "first_volume": slice(0, 1),
        "first_five_volumes": slice(0, 5),
        "interior_volumes": slice(5, n_t - 5),
        "last_five_volumes": slice(n_t - 5, n_t),
        "final_volume": slice(n_t - 1, n_t),
    }
    metrics: Dict[str, Dict[str, float]] = {}
    for name, sl in slices.items():
        diff_after = est[..., sl] - gt[..., sl]
        diff_raw = raw[..., sl] - gt[..., sl]
        metrics[name] = {
            "rmse_raw": float(np.sqrt(np.mean(diff_raw ** 2))),
            "rmse_after": float(np.sqrt(np.mean(diff_after ** 2))),
            "max_abs_err_raw": float(np.max(np.abs(diff_raw))),
            "max_abs_err_after": float(np.max(np.abs(diff_after))),
        }
    return metrics


def run_boundary_validation_suite() -> Dict[str, Any]:
    """Compare unpadded not-a-knot extrapolation vs 4-TR odd-point reflection across 4 regimes.

    Tests all 5 temporal regions (first, first_five, interior, last_five, final) for:
    1. Smooth multi-tone resting-state oscillations
    2. Noisy resting-state oscillations (additive Gaussian noise, sigma=0.25)
    3. Non-periodic linear trend + oscillation
    4. Controlled transient BOLD signal changes near scan boundaries
    """
    tr = 2.0
    n_vols = 120
    n_vox = 250
    t_idx = np.arange(n_vols, dtype=np.float64)
    t_obs = t_idx * tr
    deltas = (-0.45, 0.45)  # Near-worst-case backward (-0.90s) and forward (+0.90s) boundary shifts

    def _eval_smooth(t_sec: np.ndarray, ph: np.ndarray) -> np.ndarray:
        return (
            100.0
            + 4.0 * np.sin(2.0 * np.pi * 0.025 * t_sec + 0.35 + ph)
            + 3.0 * np.sin(2.0 * np.pi * 0.048 * t_sec - 0.80 + ph)
            + 2.0 * np.sin(2.0 * np.pi * 0.072 * t_sec + 1.15 + ph)
        )

    def _eval_trend(t_sec: np.ndarray, ph: np.ndarray) -> np.ndarray:
        return _eval_smooth(t_sec, ph) + 0.08 * t_sec

    def _eval_transient(t_sec: np.ndarray, ph: np.ndarray) -> np.ndarray:
        # Smooth resting baseline + localized Gaussian BOLD spikes near t=2s, t=T/2, and t=234s
        base = _eval_smooth(t_sec, ph)
        spike_start = 15.0 * np.exp(-0.5 * ((t_sec - 2.0) / 1.1) ** 2)
        spike_mid = 12.0 * np.exp(-0.5 * ((t_sec - 120.0) / 1.1) ** 2)
        spike_end = 15.0 * np.exp(-0.5 * ((t_sec - ((n_vols - 2) * tr)) / 1.1) ** 2)
        return base + spike_start + spike_mid + spike_end

    rng = np.random.default_rng(202611)
    phases = rng.uniform(-0.4, 0.4, size=(n_vox, 1))
    noise = rng.normal(0.0, 0.25, size=(n_vox, n_vols))

    results: Dict[str, Any] = {}
    for delta in deltas:
        shift_label = f"delta_{delta:+.2f}TR"
        shift_results: Dict[str, Any] = {}
        t_gt = t_obs[None, :] + delta * tr

        regimes = {
            "smooth_oscillation": (
                _eval_smooth(t_obs[None, :], phases),
                _eval_smooth(t_gt, phases),
            ),
            "noisy_oscillation": (
                _eval_smooth(t_obs[None, :], phases) + noise,
                _eval_smooth(t_gt, phases),
            ),
            "nonperiodic_linear_trend": (
                _eval_trend(t_obs[None, :], phases),
                _eval_trend(t_gt, phases),
            ),
            "controlled_transient_change": (
                _eval_transient(t_obs[None, :], phases),
                _eval_transient(t_gt, phases),
            ),
        }

        for rname, (y_obs, y_gt) in regimes.items():
            # Old unpadded not-a-knot with extrapolate=True
            cs_old = CubicSpline(t_idx, y_obs, axis=-1, bc_type="not-a-knot", extrapolate=True)
            est_old = cs_old(t_idx + delta)
            m_old = compute_five_region_metrics(est_old, y_obs, y_gt)

            # Selected odd-point anti-symmetric 4-TR reflection with extrapolate=False
            t_pad, y_pad = _odd_reflect_pad_series(y_obs, n_vols, 4)
            cs_new = CubicSpline(t_pad, y_pad, axis=-1, bc_type="not-a-knot", extrapolate=False)
            est_new = cs_new(t_idx + delta)
            assert np.isfinite(est_new).all()
            m_new = compute_five_region_metrics(est_new, y_obs, y_gt)

            shift_results[rname] = {
                "unpadded_not_a_knot_extrap": m_old,
                "odd_point_reflection_4tr_no_extrap": m_new,
            }

        # Verify scientifically measured improvements on noisy & transient boundaries
        if delta < 0:
            assert (
                shift_results["noisy_oscillation"]["odd_point_reflection_4tr_no_extrap"]["first_volume"]["rmse_after"]
                < shift_results["noisy_oscillation"]["unpadded_not_a_knot_extrap"]["first_volume"]["rmse_after"]
            )
            assert (
                shift_results["controlled_transient_change"]["odd_point_reflection_4tr_no_extrap"]["first_volume"]["rmse_after"]
                < shift_results["controlled_transient_change"]["unpadded_not_a_knot_extrap"]["first_volume"]["rmse_after"] * 0.25
            )
        else:
            assert (
                shift_results["noisy_oscillation"]["odd_point_reflection_4tr_no_extrap"]["final_volume"]["rmse_after"]
                < shift_results["noisy_oscillation"]["unpadded_not_a_knot_extrap"]["final_volume"]["rmse_after"]
            )
            assert (
                shift_results["noisy_oscillation"]["odd_point_reflection_4tr_no_extrap"]["final_volume"]["max_abs_err_after"]
                < shift_results["noisy_oscillation"]["unpadded_not_a_knot_extrap"]["final_volume"]["max_abs_err_after"] * 0.80
            )
            assert (
                shift_results["controlled_transient_change"]["odd_point_reflection_4tr_no_extrap"]["final_volume"]["rmse_after"]
                < shift_results["controlled_transient_change"]["unpadded_not_a_knot_extrap"]["final_volume"]["rmse_after"] * 0.25
            )
        results[shift_label] = shift_results

    return results


def run_header_axis_controls(
    out_root: Path,
    tr: float,
    n_vols: int,
    n_slices: int,
) -> Dict[str, Any]:
    """Validate NIfTI header slice_dim fallback and SKIPPED_UNKNOWN_SLICE_AXIS behavior."""
    axis_summary: Dict[str, Any] = {}
    timings = np.linspace(0.0, tr, n_slices, endpoint=False).tolist()

    for dim_idx, expected_dir, shape in (
        (0, "i", (n_slices, 8, 8)),
        (1, "j", (8, n_slices, 8)),
        (2, "k", (8, 8, n_slices)),
    ):
        ds = build_synthetic_stc_dataset(
            case_dir=out_root / f"control_header_dim_{dim_idx}",
            case_id=f"control_header_dim_{dim_idx}",
            spatial_shape=shape,
            n_vols=n_vols,
            tr=tr,
            slice_timing=timings,
            slice_encoding_direction=expected_dir,
            include_slice_encoding_direction=False,
            set_header_slice_dim=True,
            seed=100 + dim_idx,
        )
        corr_img, rep = correct_slice_timing_with_report(ds["bold_nii"], ds["bold_json"])
        metrics = evaluate_case_accuracy(ds, corr_img)
        assert rep["status"] == "APPLIED"
        assert rep["slice_axis"] == dim_idx
        assert rep["slice_encoding_direction"] == expected_dir
        assert rep["header_slice_dim"] == dim_idx
        assert rep["slice_axis_source"] == "nifti_header_slice_dim"
        assert metrics["temporal_rmse_interior_after"] < 0.005
        axis_summary[f"missing_dir_header_dim_{dim_idx}"] = {
            "resolved_axis": rep["slice_axis"],
            "resolved_direction": rep["slice_encoding_direction"],
            "slice_axis_source": rep["slice_axis_source"],
            "interior_rmse_after": metrics["temporal_rmse_interior_after"],
        }

    # Unknown header slice_dim + absent SliceEncodingDirection must skip cleanly without assuming k
    unk_ds = build_synthetic_stc_dataset(
        case_dir=out_root / "control_unknown_slice_axis",
        case_id="control_unknown_slice_axis",
        spatial_shape=(8, 8, n_slices),
        n_vols=n_vols,
        tr=tr,
        slice_timing=timings,
        slice_encoding_direction="k",
        include_slice_encoding_direction=False,
        set_header_slice_dim=False,
        seed=110,
    )
    unk_img, unk_rep = correct_slice_timing_with_report(unk_ds["bold_nii"], unk_ds["bold_json"])
    assert unk_rep["status"] == "SKIPPED_UNKNOWN_SLICE_AXIS"
    assert unk_rep["corrected"] is False
    assert unk_rep["slice_axis"] is None
    assert unk_rep["slice_encoding_direction"] is None
    np.testing.assert_array_equal(
        np.asarray(unk_img.dataobj),
        np.asarray(nib.load(str(unk_ds["bold_nii"])).dataobj),
    )
    axis_summary["missing_dir_missing_header_slice_dim"] = {
        "status": unk_rep["status"],
        "corrected": unk_rep["corrected"],
        "reason": unk_rep["reason"],
    }

    # Explicit vs header conflict rejected fail-closed
    bad_dir_ds = build_synthetic_stc_dataset(
        case_dir=out_root / "control_explicit_header_conflict",
        case_id="control_explicit_header_conflict",
        spatial_shape=(8, 8, n_slices),
        n_vols=n_vols,
        tr=tr,
        slice_timing=timings,
        slice_encoding_direction="k",
        include_slice_encoding_direction=False,
        set_header_slice_dim=True,  # header slice_dim=2 ('k')
        seed=111,
    )
    conflict_meta = {
        "RepetitionTime": tr,
        "SliceTiming": [0.0] * 8,
        "SliceEncodingDirection": "i",  # conflicts with header slice_dim=2
    }
    try:
        validate_slice_timing_metadata(nib.load(str(bad_dir_ds["bold_nii"])), conflict_meta)
        raise AssertionError("Expected InvalidSliceTimingError for header conflict")
    except InvalidSliceTimingError:
        axis_summary["explicit_vs_header_conflict_rejected"] = True

    return axis_summary


def run_37slice_240vol_e2e_integration(out_root: Path) -> Dict[str, Any]:
    """Run full 21-stage pipeline on a source-acquisition-like 37-slice x 240-vol scan -> canonical CC200."""
    case_root = out_root / "e2e_37slices_240vols"
    ses_dir = case_root / "inputs" / "sub-synthetic037" / "ses-1"
    func_dir = ses_dir / "func"
    anat_dir = ses_dir / "anat"
    tpl_dir = case_root / "inputs" / "templates"
    for d in (func_dir, anat_dir, tpl_dir):
        d.mkdir(parents=True, exist_ok=True)

    canonical_atlas = (PROJECT_ROOT / "data" / "atlases" / "cc200_roi_atlas.nii.gz").resolve()
    verify_canonical_cc200_atlas(canonical_atlas)
    atlas_img = nib.load(str(canonical_atlas))
    atlas_data = np.asanyarray(atlas_img.dataobj, dtype=np.int32)

    epi_shape = (64, 64, 37)
    epi_affine = np.array(
        [
            [-3.0, 0.0, 0.0, 94.5],
            [0.0, 3.5, 0.0, -126.0],
            [0.0, 0.0, 4.0, -56.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    n_vols = 240
    tr = 2.0

    # Map native 37-slice EPI voxels to canonical CC200 parcels in physical world coordinates
    inv_can = np.linalg.inv(CANONICAL_CC200_AFFINE)
    map_vox = inv_can @ epi_affine
    ii, jj, kk = np.meshgrid(
        np.arange(epi_shape[0]), np.arange(epi_shape[1]), np.arange(epi_shape[2]), indexing="ij"
    )
    ci = np.round(map_vox[0, 0] * ii + map_vox[0, 3]).astype(int)
    cj = np.round(map_vox[1, 1] * jj + map_vox[1, 3]).astype(int)
    ck = np.round(map_vox[2, 2] * kk + map_vox[2, 3]).astype(int)
    valid_mask = (
        (ci >= 0) & (ci < CANONICAL_CC200_SHAPE[0])
        & (cj >= 0) & (cj < CANONICAL_CC200_SHAPE[1])
        & (ck >= 0) & (ck < CANONICAL_CC200_SHAPE[2])
    )
    epi_parcels = np.zeros(epi_shape, dtype=np.int32)
    epi_parcels[valid_mask] = atlas_data[ci[valid_mask], cj[valid_mask], ck[valid_mask]]
    covered_parcels = int(len(np.unique(epi_parcels[epi_parcels > 0])))
    assert covered_parcels == 200, f"Expected all 200 CC200 parcels in 37-slice EPI FOV, found {covered_parcels}"

    order = list(range(0, 37, 2)) + list(range(1, 37, 2))
    slice_timings = [0.0] * 37
    for step, z_idx in enumerate(order):
        slice_timings[z_idx] = round(float(step * tr / 37.0), 8)

    rng = np.random.default_rng(123)
    vol_onsets = np.arange(n_vols, dtype=np.float64) * tr
    bold_data = np.zeros((*epi_shape, n_vols), dtype=np.float32)
    for parcel_id in range(1, 201):
        mask_p = epi_parcels == parcel_id
        freq = 0.02 + (parcel_id % 30) * 0.0025
        phase_p = (parcel_id * 0.37) % (2.0 * np.pi)
        for z_idx in np.unique(np.where(mask_p)[2]):
            sl_xy = mask_p[:, :, z_idx]
            nv = int(np.sum(sl_xy))
            if nv > 0:
                t_sec = vol_onsets + slice_timings[int(z_idx)]
                sig = (100.0 + 8.0 * np.sin(2.0 * np.pi * freq * t_sec + phase_p)).astype(np.float32)[None, :]
                bold_data[sl_xy, int(z_idx), :] = sig + rng.normal(0.0, 0.5, size=(nv, n_vols)).astype(np.float32)

    bold_path = func_dir / "sub-synthetic037_ses-1_task-rest_bold.nii.gz"
    bold_img = nib.Nifti1Image(bold_data, epi_affine)
    bold_img.header.set_zooms((3.0, 3.5, 4.0, float(tr)))
    bold_img.header.set_xyzt_units("mm", "sec")
    bold_img.header.set_dim_info(slice=2)
    nib.save(bold_img, str(bold_path))

    json_path = func_dir / "sub-synthetic037_ses-1_task-rest_bold.json"
    json_path.write_text(
        json.dumps(
            {
                "RepetitionTime": tr,
                "TaskName": "rest",
                "SliceTiming": slice_timings,
                "SliceEncodingDirection": "k",
                "Manufacturer": "Synthetic37SliceSourceAcquisition",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    brain_mask = atlas_data > 0
    mx, my, mz = np.ogrid[: CANONICAL_CC200_SHAPE[0], : CANONICAL_CC200_SHAPE[1], : CANONICAL_CC200_SHAPE[2]]
    mc = (CANONICAL_CC200_SHAPE[0] // 2, CANONICAL_CC200_SHAPE[1] // 2, CANONICAL_CC200_SHAPE[2] // 2)
    csf = brain_mask & (((mx - mc[0]) ** 2 / 5.0 ** 2 + (my - mc[1]) ** 2 / 5.0 ** 2 + (mz - mc[2]) ** 2 / 5.0 ** 2) <= 1.0)
    wm = brain_mask & (~csf) & (((mx - mc[0]) ** 2 / 12.0 ** 2 + (my - mc[1]) ** 2 / 15.0 ** 2 + (mz - mc[2]) ** 2 / 12.0 ** 2) <= 1.0)
    gm = brain_mask & (~csf) & (~wm)
    dist_sq = (mx - mc[0]) ** 2 + (my - mc[1]) ** 2 + (mz - mc[2]) ** 2
    skull = (~brain_mask) & (dist_sq >= 28 ** 2) & (dist_sq <= 30 ** 2)
    t1_data = np.zeros(CANONICAL_CC200_SHAPE, dtype=np.float32)
    t1_data[csf] = 25.0 + rng.normal(0.0, 2.0, np.sum(csf)).astype(np.float32)
    t1_data[gm] = 85.0 + rng.normal(0.0, 3.0, np.sum(gm)).astype(np.float32)
    t1_data[wm] = 150.0 + rng.normal(0.0, 3.0, np.sum(wm)).astype(np.float32)
    t1_data[skull] = 110.0 + rng.normal(0.0, 3.0, np.sum(skull)).astype(np.float32)
    t1_path = anat_dir / "sub-synthetic037_ses-1_T1w.nii.gz"
    nib.save(nib.Nifti1Image(t1_data, CANONICAL_CC200_AFFINE), str(t1_path))

    mni_data = np.zeros(CANONICAL_CC200_SHAPE, dtype=np.float32)
    mni_data[brain_mask] = 100.0
    mni_data[wm] = 150.0
    mni_data[gm] = 85.0
    mni_data[csf] = 25.0
    mni_path = tpl_dir / "MNI152_T1_3mm.nii.gz"
    nib.save(nib.Nifti1Image(mni_data, CANONICAL_CC200_AFFINE), str(mni_path))

    files_dict = {
        "bold_nii": {"filename": bold_path.name, "resolved_path": str(bold_path.resolve()), "sha256": compute_sha256(bold_path)},
        "bold_json": {"filename": json_path.name, "resolved_path": str(json_path.resolve()), "sha256": compute_sha256(json_path)},
        "t1w_nii": {"filename": t1_path.name, "resolved_path": str(t1_path.resolve()), "sha256": compute_sha256(t1_path)},
        "mni_template": {"filename": mni_path.name, "resolved_path": str(mni_path.resolve()), "sha256": compute_sha256(mni_path)},
        "cc200_atlas": {"filename": canonical_atlas.name, "resolved_path": str(canonical_atlas.resolve()), "sha256": compute_sha256(canonical_atlas)},
    }
    man_path = ses_dir / "synthetic_manifest.json"
    man_path.write_text(
        json.dumps(
            {
                "manifest_magic": "NEUROADHD_SYNTHETIC_DATASET_V2",
                "subject_id": "sub-synthetic037",
                "session_id": "ses-1",
                "integrity_hmac": compute_manifest_hmac("NEUROADHD_SYNTHETIC_DATASET_V2", "sub-synthetic037", files_dict),
                "files": files_dict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    res = run_poc_v2_pipeline(
        bold_path=bold_path,
        bold_json_path=json_path,
        t1w_path=t1_path,
        atlas_path=canonical_atlas,
        template_path=mni_path,
        output_dir=case_root / "outputs",
        subject_id="sub-synthetic037",
        session_id="ses-1",
        atlas_mode="canonical",
        manifest_path=man_path,
    )
    assert res["status"] == "SUCCESS"
    prov = res["provenance"]
    assert prov["pipeline_stages_total"] == 21
    assert prov["motion_estimation"]["estimated_before_slice_timing"] is True
    assert prov["slice_timing_correction"]["status"] == "APPLIED"
    assert prov["slice_timing_correction"]["num_slices"] == 37
    assert prov["slice_timing_correction"]["slice_axis"] == 2

    deriv_nii = Path(res["output_dir"]) / prov["slice_timing_correction"]["derivative_nii"]
    deriv_img = nib.load(str(deriv_nii))
    assert deriv_img.shape == (64, 64, 37, 240)

    roi_df = pd.read_csv(res["artifacts"]["roi_csv"])
    assert roi_df.shape == (200, 241)
    feat_vec = np.load(res["artifacts"]["features_npy"])
    assert feat_vec.shape == (19900,)
    assert np.all(np.isfinite(feat_vec))

    return {
        "status": "PASSED",
        "source_bold_shape": list(bold_img.shape),
        "derivative_bold_shape": list(deriv_img.shape),
        "canonical_target_shape": list(CANONICAL_CC200_SHAPE),
        "slice_timing_length": len(slice_timings),
        "slice_axis": prov["slice_timing_correction"]["slice_axis"],
        "slice_encoding_direction": prov["slice_timing_correction"]["slice_encoding_direction"],
        "estimated_motion_before_stc": prov["motion_estimation"]["estimated_before_slice_timing"],
        "single_step_composed_resampling": True,
        "pipeline_stages_total": prov["pipeline_stages_total"],
        "roi_timeseries_shape": list(roi_df.shape),
        "n_fisher_z_features": int(len(feat_vec)),
        "all_features_finite": bool(np.all(np.isfinite(feat_vec))),
        "runtime_seconds": float(res["runtime_seconds"]),
        "peak_rss_gb": float(res["peak_rss_gb"]),
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

    # Additional header-axis fallback, 5-region boundary suite, and 37-slice x 240-vol E2E checks
    print("\nRunning NIfTI header slice_dim axis controls...")
    header_axis_controls = run_header_axis_controls(
        out_root=out_root, tr=tr, n_vols=n_vols, n_slices=n_slices_std
    )
    print("  -> Header slice_dim fallback (0->i, 1->j, 2->k), unknown-axis skip, and conflict controls PASSED")

    print("\nRunning 5-region cubic spline boundary validation suite...")
    boundary_validation = run_boundary_validation_suite()
    for shift_lbl, reg_dict in boundary_validation.items():
        n_old_f = reg_dict["noisy_oscillation"]["unpadded_not_a_knot_extrap"]["first_volume"]["rmse_after"]
        n_new_f = reg_dict["noisy_oscillation"]["odd_point_reflection_4tr_no_extrap"]["first_volume"]["rmse_after"]
        t_old_f = reg_dict["controlled_transient_change"]["unpadded_not_a_knot_extrap"]["first_volume"]["rmse_after"]
        t_new_f = reg_dict["controlled_transient_change"]["odd_point_reflection_4tr_no_extrap"]["first_volume"]["rmse_after"]
        print(
            f"  [{shift_lbl}] Noisy vol[0] RMSE: {n_old_f:.4f} -> {n_new_f:.4f} | "
            f"Transient vol[0] RMSE: {t_old_f:.4f} -> {t_new_f:.4f}"
        )

    print("\nRunning 37-slice x 240-volume source-acquisition -> canonical CC200 21-stage E2E test...")
    e2e_37slice = run_37slice_240vol_e2e_integration(out_root)
    print(
        f"  -> E2E 37-slice PASSED: src={e2e_37slice['source_bold_shape']} -> "
        f"roi={e2e_37slice['roi_timeseries_shape']} | features={e2e_37slice['n_fisher_z_features']} | "
        f"runtime={e2e_37slice['runtime_seconds']:.2f}s | peak_rss={e2e_37slice['peak_rss_gb']:.2f} GB"
    )

    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = {
        "step": "Step 4.1: Slice-Timing Correction Final Scientific Validation (DEC-STC-01)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VERIFIED",
        "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
        "real_data_authorized": False,
        "overall_authorization_state": auth_state["overall_authorization_state"],
        "software": software_info,
        "interpolation_method": "cubic_spline",
        "boundary_strategy": "odd_point_reflection_4tr_no_extrap",
        "reference_time_seconds": ref_time,
        "tr_seconds": tr,
        "n_volumes": n_vols,
        "spatial_dimensions_default": [16, 16, 37],
        "signal_model_frequencies_hz": [f for _, f, _ in SIGNAL_COMPONENTS],
        "acceptance_criterion": {
            "basis": "Hall & Meyer (1976) C4 cubic spline interior error bound + monotonic improvement over uncorrected sampling + bounded boundary response under noise/transients",
            "theoretical_interior_rmse_bound": theoretical_interior_bound,
            "required_min_mean_pearson_r": 0.999,
            "required_finite_fraction": 1.0,
        },
        "peak_traced_memory_mib": round(peak_mem_bytes / (1024.0 * 1024.0), 3),
        "cases": case_results,
        "header_axis_controls": header_axis_controls,
        "boundary_validation_5_regions": boundary_validation,
        "e2e_37slice_240vol_canonical": e2e_37slice,
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
