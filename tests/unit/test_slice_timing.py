"""Comprehensive unit, negative, and integration tests for Step 4 / 4.1 Slice-Timing Correction (DEC-STC-01).

Covers all Phase B-I & Step 4.1 requirements using strictly synthetic data generated in-memory
or under pytest's temporary workspace:
- BIDS metadata validation (RepetitionTime, SliceTiming, SliceEncodingDirection)
- All 6 slice encoding directions (i, i-, j, j-, k, k-) & header slice_dim validation/fallback
- Skip with SKIPPED_UNKNOWN_SLICE_AXIS when both SliceEncodingDirection and slice_dim are absent
- Sequential, interleaved, multiband repeated timings, and simultaneous skip
- Missing metadata safety (strict rejection vs explicit skip without timing invention)
- Analytical continuous-time ground-truth accuracy across interpolation backends
- 5-region temporal boundary evaluation (first, first5, interior, last5, final) across smooth,
  noisy, non-periodic linear trend, and controlled transient signals
- Geometric invariance (shape, spatial 4x4 affine, NIfTI header TR & units='sec')
- Scaled NIfTI (scl_slope/scl_inter) handling and bit-exact determinism
- BIDS derivative immutability (input SHA-256 unchanged; StartTime/+TR/2 origin shift)
- Double slice-timing correction prevention (directly and via run_poc_v2_pipeline)
- 22-stage preprocessing order (Stage 2 raw motion estimation before Stage 3 STC & Stage 4 SDC, single-step resampling)
- Realistic 37-slice x 240-volume source-to-61-slice end-to-end integration
- Strict synthetic-only safety enforcement (real-data paths & symlinks rejected)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from unittest import mock

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import CubicSpline

from scripts.run_poc_v2 import (
    compute_manifest_hmac,
    compute_sha256,
    generate_synthetic_phantom_session,
    run_poc_v2_pipeline,
)
from src.preprocessing.motion import (
    apply_motion_transforms as real_apply_motion_transforms,
    estimate_motion_parameters as real_estimate_motion_parameters,
)
from src.preprocessing.registration import resample_4d_bold as real_resample_4d_bold
from src.preprocessing.slice_timing import (
    DEFAULT_STC_METHOD,
    DoubleSliceTimingCorrectionError,
    InvalidSliceTimingError,
    MissingSliceTimingError,
    SliceTimingValidationError,
    _odd_reflect_pad_series,
    compute_file_sha256,
    correct_slice_timing,
    correct_slice_timing_with_report,
    detect_stc_software,
    validate_slice_timing_metadata,
    write_slice_timing_derivative,
)
from src.preprocessing.slice_timing import (
    write_slice_timing_derivative as real_write_slice_timing_derivative,
)
from src.validation.authorization_gate import ResearchAuthorizationGate

_TONES: Tuple[Tuple[float, float, float], ...] = (
    (4.0, 0.025, 0.35),
    (3.0, 0.048, -0.80),
    (2.0, 0.072, 1.15),
)


def _eval_multitone(
    t_sec: np.ndarray,
    baseline: float = 100.0,
    phase: float = 0.0,
    linear_drift: float = 0.0,
) -> np.ndarray:
    y = np.full(t_sec.shape, float(baseline), dtype=np.float64) + linear_drift * t_sec
    for amp, freq, phi0 in _TONES:
        y = y + amp * np.sin(2.0 * np.pi * freq * t_sec + phi0 + phase)
    return y


def _make_analytical_bold(
    spatial_shape: Tuple[int, int, int] = (8, 8, 15),
    n_vols: int = 120,
    tr: float = 2.0,
    slice_timing: Sequence[float] | None = None,
    direction: str = "k",
    reference_time: float | None = None,
    linear_drift: float = 0.0,
    dtype: type = np.float32,
    set_header_slice_dim: bool = True,
) -> Tuple[nib.Nifti1Image, np.ndarray, Dict[str, object], Tuple[float, ...]]:
    """Build synthetic 4D NIfTI BOLD sampled at slice times plus analytical GT at reference_time."""
    axis_map = {"i": (0, False), "i-": (0, True), "j": (1, False), "j-": (1, True), "k": (2, False), "k-": (2, True)}
    axis, is_rev = axis_map[direction]
    n_slices = spatial_shape[axis]
    if slice_timing is None:
        slice_timing = np.linspace(0.0, tr, n_slices, endpoint=False).tolist()
    if reference_time is None:
        reference_time = tr / 2.0

    t_arr = np.asarray(slice_timing, dtype=np.float64)
    voxel_times = t_arr[::-1] if is_rev else t_arr
    vol_onsets = np.arange(n_vols, dtype=np.float64) * float(tr)

    rng = np.random.default_rng(12345)
    baselines = 100.0 + rng.uniform(-5.0, 5.0, size=spatial_shape)
    phases = rng.uniform(-0.5, 0.5, size=spatial_shape)

    obs = np.zeros((*spatial_shape, n_vols), dtype=dtype)
    gt = np.zeros((*spatial_shape, n_vols), dtype=np.float64)

    for s in range(n_slices):
        sl_time = float(voxel_times[s])
        t_obs = vol_onsets + sl_time
        t_gt = vol_onsets + float(reference_time)

        sel = [slice(None), slice(None), slice(None)]
        sel[axis] = s
        b_sl = baselines[tuple(sel)]
        p_sl = phases[tuple(sel)]
        d1, d2 = b_sl.shape
        sl_obs = np.zeros((d1, d2, n_vols), dtype=dtype)
        sl_gt = np.zeros((d1, d2, n_vols), dtype=np.float64)
        for i1 in range(d1):
            for i2 in range(d2):
                sl_obs[i1, i2, :] = _eval_multitone(t_obs, b_sl[i1, i2], p_sl[i1, i2], linear_drift).astype(dtype)
                sl_gt[i1, i2, :] = _eval_multitone(t_gt, b_sl[i1, i2], p_sl[i1, i2], linear_drift)
        out_sel = (sel[0], sel[1], sel[2], slice(None))
        obs[out_sel] = sl_obs
        gt[out_sel] = sl_gt

    affine = np.array(
        [
            [3.0, 0.0, 0.0, -12.0],
            [0.0, 3.0, 0.0, -12.0],
            [0.0, 0.0, 3.5, -21.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    img = nib.Nifti1Image(obs, affine)
    img.header.set_zooms((3.0, 3.0, 3.5, float(tr)))
    img.header.set_xyzt_units("mm", "sec")
    if set_header_slice_dim:
        img.header.set_dim_info(slice=axis)
    meta = {
        "RepetitionTime": float(tr),
        "TaskName": "rest",
        "SliceTiming": [float(v) for v in slice_timing],
        "SliceEncodingDirection": direction,
    }
    return img, gt, meta, tuple(float(v) for v in voxel_times)


def _metrics(est: np.ndarray, raw: np.ndarray, gt: np.ndarray, tr: float) -> Dict[str, float]:
    n_t = int(est.shape[-1])
    rmse_raw = float(np.sqrt(np.mean((raw - gt) ** 2)))
    rmse_est = float(np.sqrt(np.mean((est - gt) ** 2)))
    rmse_interior = float(np.sqrt(np.mean((est[..., 2:-2] - gt[..., 2:-2]) ** 2)))
    rmse_first = float(np.sqrt(np.mean((est[..., 0:1] - gt[..., 0:1]) ** 2)))
    rmse_first5 = float(np.sqrt(np.mean((est[..., 0:5] - gt[..., 0:5]) ** 2)))
    rmse_interior5 = (
        float(np.sqrt(np.mean((est[..., 5:-5] - gt[..., 5:-5]) ** 2)))
        if n_t > 12
        else rmse_interior
    )
    rmse_last5 = float(np.sqrt(np.mean((est[..., -5:] - gt[..., -5:]) ** 2)))
    rmse_final = float(np.sqrt(np.mean((est[..., -1:] - gt[..., -1:]) ** 2)))
    mae_raw = float(np.mean(np.abs(raw - gt)))
    mae_est = float(np.mean(np.abs(est - gt)))

    est_flat = est.reshape(-1, est.shape[-1])
    gt_flat = gt.reshape(-1, gt.shape[-1])
    est_c = est_flat - np.mean(est_flat, axis=1, keepdims=True)
    gt_c = gt_flat - np.mean(gt_flat, axis=1, keepdims=True)
    r_vals = np.sum(est_c * gt_c, axis=1) / np.sqrt(np.sum(est_c ** 2, axis=1) * np.sum(gt_c ** 2, axis=1))

    # Dominant frequency DFT phase error
    s_win = est_c[:, 4:-4]
    g_win = gt_c[:, 4:-4]
    freqs = np.fft.rfftfreq(s_win.shape[1], d=tr)
    f_bin = int(np.argmin(np.abs(freqs - 0.048)))
    dphi = (np.angle(np.fft.rfft(s_win, axis=1)[:, f_bin]) - np.angle(np.fft.rfft(g_win, axis=1)[:, f_bin]) + np.pi) % (
        2.0 * np.pi
    ) - np.pi
    raw_c = raw.reshape(-1, raw.shape[-1]) - np.mean(raw.reshape(-1, raw.shape[-1]), axis=1, keepdims=True)
    raw_win = raw_c[:, 4:-4]
    dphi_raw = (
        np.angle(np.fft.rfft(raw_win, axis=1)[:, f_bin]) - np.angle(np.fft.rfft(g_win, axis=1)[:, f_bin]) + np.pi
    ) % (2.0 * np.pi) - np.pi

    return {
        "rmse_before": rmse_raw,
        "rmse_after": rmse_est,
        "rmse_interior_after": rmse_interior,
        "rmse_first": rmse_first,
        "rmse_first5": rmse_first5,
        "rmse_interior5": rmse_interior5,
        "rmse_last5": rmse_last5,
        "rmse_final": rmse_final,
        "mae_before": mae_raw,
        "mae_after": mae_est,
        "pearson_r_mean": float(np.mean(r_vals)),
        "phase_err_before_rad": float(np.mean(np.abs(dphi_raw))),
        "phase_err_after_rad": float(np.mean(np.abs(dphi))),
    }


def _interleaved_timings(n_slices: int, tr: float) -> List[float]:
    order = list(range(0, n_slices, 2)) + list(range(1, n_slices, 2))
    out = [0.0] * n_slices
    for step, s_idx in enumerate(order):
        out[s_idx] = float(step * tr / n_slices)
    return out


# ==============================================================================
# 1–7. Acquisition Regimes & Slice Encoding Directions
# ==============================================================================


def test_1_valid_sequential_slice_timing_accuracy():
    """Sequential ascending & descending acquisition realign accurately to TR/2."""
    n_slices, tr = 19, 2.0
    for seq_timings, expected_label in (
        (np.linspace(0.0, tr, n_slices, endpoint=False).tolist(), "ascending_sequential"),
        (np.linspace(0.0, tr, n_slices, endpoint=False)[::-1].tolist(), "descending_sequential"),
    ):
        img, gt, meta, _ = _make_analytical_bold((6, 6, n_slices), n_vols=120, tr=tr, slice_timing=seq_timings)
        corr_img, report = correct_slice_timing(img, meta, return_report=True)
        m = _metrics(np.asarray(corr_img.dataobj, dtype=np.float64), np.asarray(img.dataobj, dtype=np.float64), gt, tr)

        assert report["status"] == "APPLIED"
        assert report["slice_order_interpretation"] == expected_label
        assert report["reference_time"] == pytest.approx(1.0)
        assert report["temporal_origin_shift_seconds"] == pytest.approx(1.0)
        assert m["rmse_interior_after"] < 0.005
        assert m["rmse_after"] < m["rmse_before"] * 0.08
        assert m["mae_after"] < m["mae_before"] * 0.05
        assert m["phase_err_after_rad"] < m["phase_err_before_rad"] * 0.05
        assert m["pearson_r_mean"] > 0.9995


def test_2_valid_interleaved_slice_timing_accuracy():
    """Interleaved acquisition recovers true temporal phase across odd/even slices."""
    n_slices, tr = 21, 2.0
    timings = _interleaved_timings(n_slices, tr)
    img, gt, meta, _ = _make_analytical_bold((6, 6, n_slices), n_vols=120, tr=tr, slice_timing=timings)
    corr_img, report = correct_slice_timing_with_report(img, meta)
    m = _metrics(np.asarray(corr_img.dataobj, dtype=np.float64), np.asarray(img.dataobj, dtype=np.float64), gt, tr)

    assert report["status"] == "APPLIED"
    assert "interleaved" in report["slice_order_interpretation"]
    assert m["rmse_interior_after"] < 0.005
    assert m["rmse_after"] < 0.04
    assert m["rmse_after"] < m["rmse_before"] * 0.08
    assert m["pearson_r_mean"] > 0.9995


def test_3_valid_multiband_repeated_acquisition_times():
    """Multiband SMS sequences with repeated timings across groups are supported accurately."""
    n_slices, tr, n_groups = 24, 1.8, 8  # MB=3
    mb_timings = [float((idx % n_groups) * tr / n_groups) for idx in range(n_slices)]
    img, gt, meta, _ = _make_analytical_bold((6, 6, n_slices), n_vols=120, tr=tr, slice_timing=mb_timings)
    corr_img, report = correct_slice_timing_with_report(img, meta)
    m = _metrics(np.asarray(corr_img.dataobj, dtype=np.float64), np.asarray(img.dataobj, dtype=np.float64), gt, tr)

    assert report["status"] == "APPLIED"
    assert report["slice_order_interpretation"] == "multiband_mb3"
    assert m["rmse_interior_after"] < 0.005
    assert m["rmse_after"] < m["rmse_before"] * 0.08
    assert m["pearson_r_mean"] > 0.9995


@pytest.mark.parametrize("direction,shape", [("k-", (6, 6, 17)), ("i-", (17, 6, 6)), ("j-", (6, 17, 6))])
def test_4_negative_slice_encoding_direction_reverses_mapping(direction: str, shape: Tuple[int, int, int]):
    """Negative SliceEncodingDirection (i-, j-, k-) must reverse slice timing mapping."""
    tr = 2.0
    timings = np.linspace(0.0, tr, 17, endpoint=False).tolist()
    img, gt, meta, _ = _make_analytical_bold(shape, n_vols=100, tr=tr, slice_timing=timings, direction=direction)
    corr_img, report = correct_slice_timing_with_report(img, meta)
    m = _metrics(np.asarray(corr_img.dataobj, dtype=np.float64), np.asarray(img.dataobj, dtype=np.float64), gt, tr)

    assert report["status"] == "APPLIED"
    assert report["slice_encoding_direction"] == direction
    assert report["slice_encoding_direction_reversed"] is True
    assert list(report["voxel_slice_times"]) == pytest.approx(list(reversed(timings)))
    assert m["rmse_interior_after"] < 0.005
    assert m["rmse_after"] < m["rmse_before"] * 0.08
    assert m["rmse_after"] < 0.04
    assert m["pearson_r_mean"] > 0.9995

    # Sanity check: using unreversed positive direction on the same data must degrade accuracy dramatically
    wrong_meta = dict(meta)
    wrong_meta["SliceEncodingDirection"] = direction[0]
    wrong_img, _ = correct_slice_timing_with_report(img, wrong_meta)
    m_wrong = _metrics(np.asarray(wrong_img.dataobj, dtype=np.float64), np.asarray(img.dataobj, dtype=np.float64), gt, tr)
    assert m_wrong["rmse_after"] > m["rmse_after"] * 15.0


def test_5_slice_axis_i():
    """SliceEncodingDirection='i' operates along NIfTI spatial axis 0."""
    tr = 2.0
    timings = _interleaved_timings(19, tr)
    img, gt, meta, _ = _make_analytical_bold((19, 6, 6), n_vols=100, tr=tr, slice_timing=timings, direction="i")
    corr_img, report = correct_slice_timing_with_report(img, meta)
    m = _metrics(np.asarray(corr_img.dataobj, dtype=np.float64), np.asarray(img.dataobj, dtype=np.float64), gt, tr)

    assert report["slice_axis"] == 0
    assert report["num_slices"] == 19
    assert m["rmse_interior_after"] < 0.005
    assert m["rmse_after"] < m["rmse_before"] * 0.08
    assert m["rmse_after"] < 0.04
    assert m["pearson_r_mean"] > 0.9995


def test_6_slice_axis_j():
    """SliceEncodingDirection='j' operates along NIfTI spatial axis 1."""
    tr = 2.0
    timings = _interleaved_timings(19, tr)
    img, gt, meta, _ = _make_analytical_bold((6, 19, 6), n_vols=100, tr=tr, slice_timing=timings, direction="j")
    corr_img, report = correct_slice_timing_with_report(img, meta)
    m = _metrics(np.asarray(corr_img.dataobj, dtype=np.float64), np.asarray(img.dataobj, dtype=np.float64), gt, tr)

    assert report["slice_axis"] == 1
    assert report["num_slices"] == 19
    assert m["rmse_interior_after"] < 0.005
    assert m["rmse_after"] < m["rmse_before"] * 0.08
    assert m["rmse_after"] < 0.04
    assert m["pearson_r_mean"] > 0.9995


def test_7_slice_axis_header_dim_fallback_and_unknown_slice_axis_skip():
    """Absent SliceEncodingDirection uses NIfTI header slice_dim (0/1/2) or skips when unavailable."""
    tr = 2.0
    n_slices = 15
    timings = np.linspace(0.0, tr, n_slices, endpoint=False).tolist()

    # 1. Header slice_dim = 0, 1, 2 with absent SliceEncodingDirection -> increasing index convention
    for slice_dim, expected_dir, shape in (
        (0, "i", (n_slices, 6, 6)),
        (1, "j", (6, n_slices, 6)),
        (2, "k", (6, 6, n_slices)),
    ):
        img, gt, meta_exp, _ = _make_analytical_bold(
            shape, n_vols=100, tr=tr, slice_timing=timings, direction=expected_dir, set_header_slice_dim=True
        )
        assert img.header.get_dim_info()[2] == slice_dim
        corr_exp, rep_exp = correct_slice_timing_with_report(img, meta_exp)
        assert rep_exp["slice_axis"] == slice_dim
        assert rep_exp["slice_axis_source"] == "bids_slice_encoding_direction"
        assert rep_exp["slice_encoding_direction_assumed"] is False

        meta_absent = {k: v for k, v in meta_exp.items() if k != "SliceEncodingDirection"}
        corr_hdr, rep_hdr = correct_slice_timing_with_report(img, meta_absent)
        m_hdr = _metrics(
            np.asarray(corr_hdr.dataobj, dtype=np.float64), np.asarray(img.dataobj, dtype=np.float64), gt, tr
        )

        assert rep_hdr["status"] == "APPLIED"
        assert rep_hdr["slice_axis"] == slice_dim
        assert rep_hdr["header_slice_dim"] == slice_dim
        assert rep_hdr["slice_axis_source"] == "nifti_header_slice_dim"
        assert rep_hdr["slice_encoding_direction"] == expected_dir
        assert rep_hdr["slice_encoding_direction_reversed"] is False
        assert rep_hdr["slice_encoding_direction_assumed"] is True
        assert any("NIfTI header dim_info slice_dim" in w for w in rep_hdr["warnings"])
        np.testing.assert_array_equal(np.asarray(corr_hdr.dataobj), np.asarray(corr_exp.dataobj))
        assert m_hdr["rmse_interior_after"] < 0.005
        assert m_hdr["rmse_after"] < 0.04

    # 2. Unavailable header slice_dim (None) + absent SliceEncodingDirection:
    # must SKIP_UNKNOWN_SLICE_AXIS and NEVER assume 'k' universally!
    img_nodim, _, meta_k, _ = _make_analytical_bold(
        (6, 6, n_slices), n_vols=60, tr=tr, slice_timing=timings, direction="k", set_header_slice_dim=False
    )
    assert img_nodim.header.get_dim_info()[2] is None
    meta_no_dir = {k: v for k, v in meta_k.items() if k != "SliceEncodingDirection"}
    out_unk, rep_unk = correct_slice_timing_with_report(img_nodim, meta_no_dir)
    assert rep_unk["status"] == "SKIPPED_UNKNOWN_SLICE_AXIS"
    assert rep_unk["corrected"] is False
    assert rep_unk["slice_axis"] is None
    assert rep_unk["header_slice_dim"] is None
    assert rep_unk["slice_axis_source"] == "undetermined"
    assert rep_unk["slice_encoding_direction"] is None
    assert rep_unk["num_corrected_slices"] == 0
    assert "skip" in rep_unk["reason"].lower()
    np.testing.assert_array_equal(np.asarray(out_unk.dataobj), np.asarray(img_nodim.dataobj))

    # 3. Explicit direction vs conflicting NIfTI header slice_dim must fail closed
    img_z, _, _, _ = _make_analytical_bold(
        (6, 6, n_slices), n_vols=40, tr=tr, slice_timing=timings, direction="k", set_header_slice_dim=True
    )
    with pytest.raises(InvalidSliceTimingError, match="conflicts with NIfTI header dim_info slice_dim"):
        validate_slice_timing_metadata(
            img_z,
            {"RepetitionTime": tr, "SliceTiming": [0.0] * 6, "SliceEncodingDirection": "i"},
        )

    # 4. Explicit direction with unset header slice_dim (None) works and warns about unverified slice_dim
    corr_nodim_exp, rep_nodim_exp = correct_slice_timing_with_report(img_nodim, meta_k)
    assert rep_nodim_exp["status"] == "APPLIED"
    assert rep_nodim_exp["slice_axis"] == 2
    assert rep_nodim_exp["slice_axis_source"] == "bids_slice_encoding_direction"
    assert any("slice_dim is unset" in w for w in rep_nodim_exp["warnings"])
    assert np.isfinite(np.asarray(corr_nodim_exp.dataobj)).all()


# ==============================================================================
# 8–17. Metadata & Geometry Rejection (Fail Closed, Never Invent Timings)
# ==============================================================================


def test_8_missing_slice_timing_never_invented():
    """Missing SliceTiming raises MissingSliceTimingError strictly and skips unchanged when allowed."""
    img, _, meta, _ = _make_analytical_bold((4, 4, 8), n_vols=60, tr=2.0)
    del meta["SliceTiming"]

    with pytest.raises(MissingSliceTimingError, match="refusing to invent"):
        correct_slice_timing_with_report(img, meta, allow_missing_timing=False)

    out_img, rep = correct_slice_timing_with_report(img, meta, allow_missing_timing=True)
    assert rep["status"] == "SKIPPED_MISSING_METADATA"
    assert rep["corrected"] is False
    assert rep["num_corrected_slices"] == 0
    assert rep["original_slice_timing"] is None
    assert "refusing to invent" in rep["reason"].lower()
    np.testing.assert_array_equal(np.asarray(out_img.dataobj), np.asarray(img.dataobj))


def test_9_missing_repetition_time_rejected():
    """Missing RepetitionTime raises InvalidSliceTimingError."""
    img, _, meta, _ = _make_analytical_bold((4, 4, 8), n_vols=40, tr=2.0)
    del meta["RepetitionTime"]
    with pytest.raises(InvalidSliceTimingError, match="RepetitionTime"):
        validate_slice_timing_metadata(img, meta)


@pytest.mark.parametrize("bad_tr", [0.0, -1.5, float("nan"), float("inf"), float("-inf"), True, False, "2.0"])
def test_10_invalid_tr_rejected(bad_tr):
    """Non-positive, non-finite, boolean, and non-numeric RepetitionTime values fail closed."""
    img, _, meta, _ = _make_analytical_bold((4, 4, 8), n_vols=40, tr=2.0)
    meta["RepetitionTime"] = bad_tr
    with pytest.raises(InvalidSliceTimingError):
        validate_slice_timing_metadata(img, meta)


def test_10b_tr_agreement_with_header_expected_and_volume_timing():
    """TR must agree with expected_tr, explicit NIfTI temporal header, and VolumeTiming; msec units convert."""
    img, _, meta, _ = _make_analytical_bold((4, 4, 8), n_vols=40, tr=2.0)

    with pytest.raises(InvalidSliceTimingError, match="expected acquisition TR"):
        validate_slice_timing_metadata(img, meta, expected_tr=2.5)

    # Disagreeing NIfTI header TR with seconds unit
    bad_hdr_img = nib.Nifti1Image(np.asarray(img.dataobj), img.affine)
    bad_hdr_img.header.set_zooms((3.0, 3.0, 3.5, 1.5))
    bad_hdr_img.header.set_xyzt_units("mm", "sec")
    with pytest.raises(InvalidSliceTimingError, match="NIfTI header temporal zoom"):
        validate_slice_timing_metadata(bad_hdr_img, meta)

    # Millisecond header unit (2000 msec == 2.0 s) must succeed
    msec_img = nib.Nifti1Image(np.asarray(img.dataobj), img.affine)
    msec_img.header.set_zooms((3.0, 3.0, 3.5, 2000.0))
    msec_img.header.set_xyzt_units("mm", "msec")
    rep_ms = validate_slice_timing_metadata(msec_img, meta)
    assert rep_ms.header_tr_seconds == pytest.approx(2.0)

    # Disagreeing VolumeTiming
    bad_vt_meta = dict(meta, VolumeTiming=[0.0, 1.5, 3.0])
    with pytest.raises(InvalidSliceTimingError, match="VolumeTiming"):
        validate_slice_timing_metadata(img, bad_vt_meta)


def test_11_nan_slice_timing_rejected():
    """NaN in SliceTiming raises InvalidSliceTimingError even if allow_missing_timing=True."""
    img, _, meta, _ = _make_analytical_bold((4, 4, 8), n_vols=40, tr=2.0)
    meta["SliceTiming"][3] = float("nan")
    with pytest.raises(InvalidSliceTimingError, match="non-finite"):
        correct_slice_timing_with_report(img, meta, allow_missing_timing=True)


def test_12_inf_slice_timing_rejected():
    """Inf and -Inf in SliceTiming raise InvalidSliceTimingError."""
    img, _, meta, _ = _make_analytical_bold((4, 4, 8), n_vols=40, tr=2.0)
    for val in (float("inf"), float("-inf")):
        meta["SliceTiming"][2] = val
        with pytest.raises(InvalidSliceTimingError):
            validate_slice_timing_metadata(img, meta)


def test_13_negative_slice_timing_rejected():
    """Negative timing values violate 0 <= timing < TR."""
    img, _, meta, _ = _make_analytical_bold((4, 4, 8), n_vols=40, tr=2.0)
    meta["SliceTiming"][0] = -1e-4
    with pytest.raises(InvalidSliceTimingError, match="0 <= timing < TR"):
        validate_slice_timing_metadata(img, meta)


@pytest.mark.parametrize("bad_val", [2.0, 2.0001, 4.5])
def test_14_timing_greater_equal_tr_rejected(bad_val: float):
    """Any timing >= TR must be rejected strictly."""
    img, _, meta, _ = _make_analytical_bold((4, 4, 8), n_vols=40, tr=2.0)
    meta["SliceTiming"][-1] = bad_val
    with pytest.raises(InvalidSliceTimingError, match="0 <= timing < TR"):
        validate_slice_timing_metadata(img, meta)


def test_15_slice_timing_length_mismatch_and_never_guess_axis():
    """SliceTiming length mismatch fails closed and NEVER guesses slice axis from other axes' sizes."""
    # X=15, Y=6, Z=8 with SliceEncodingDirection='k' -> 15 matches X, NOT Z=8! Must reject!
    img = nib.Nifti1Image(np.ones((15, 6, 8, 40), dtype=np.float32), np.eye(4))
    meta = {
        "RepetitionTime": 2.0,
        "SliceEncodingDirection": "k",
        "SliceTiming": np.linspace(0.0, 2.0, 15, endpoint=False).tolist(),
    }
    with pytest.raises(InvalidSliceTimingError, match="does not match number of slices \\(8\\)"):
        validate_slice_timing_metadata(img, meta)

    # Empty, boolean entry, and 2D arrays also rejected
    for bad_st in ([], "0,1", [[0.0, 0.5]], [0.0, True, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        with pytest.raises(InvalidSliceTimingError):
            validate_slice_timing_metadata(img, {"RepetitionTime": 2.0, "SliceEncodingDirection": "k", "SliceTiming": bad_st})


@pytest.mark.parametrize("bad_dir", ["z", "k+", "i+", "K", "", 123, None])
def test_16_invalid_slice_encoding_direction_rejected(bad_dir):
    """Unsupported SliceEncodingDirection values fail closed (None follows header slice_dim or skips)."""
    img, _, meta, _ = _make_analytical_bold((4, 4, 8), n_vols=40, tr=2.0, set_header_slice_dim=True)
    if bad_dir is None:
        meta["SliceEncodingDirection"] = None
        rep = validate_slice_timing_metadata(img, meta)
        assert rep.slice_encoding_direction == "k"
        assert rep.slice_encoding_direction_assumed is True
        assert rep.slice_axis_source == "nifti_header_slice_dim"

        # Without header slice_dim, None skips cleanly instead of guessing 'k'
        img_nodim, _, meta_nodim, _ = _make_analytical_bold(
            (4, 4, 8), n_vols=40, tr=2.0, set_header_slice_dim=False
        )
        meta_nodim["SliceEncodingDirection"] = None
        rep_nodim = validate_slice_timing_metadata(img_nodim, meta_nodim)
        assert rep_nodim.status == "SKIPPED_UNKNOWN_SLICE_AXIS"
        assert rep_nodim.slice_axis is None
        return
    meta["SliceEncodingDirection"] = bad_dir
    with pytest.raises(InvalidSliceTimingError, match="SliceEncodingDirection"):
        validate_slice_timing_metadata(img, meta)


def test_17_incorrect_nifti_dimensionality_and_corrupted_values_rejected(tmp_path: Path):
    """Non-4D BOLD, non-finite voxels, singular affine, complex dtype, and too-short time series fail closed."""
    meta = {"RepetitionTime": 2.0, "SliceTiming": [0.0, 0.25, 0.5, 0.75], "SliceEncodingDirection": "k"}

    # 3D image rejected
    img_3d = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4))
    with pytest.raises(InvalidSliceTimingError, match="requires a 4D BOLD NIfTI image"):
        correct_slice_timing_with_report(img_3d, meta)

    # Non-finite voxel rejected
    arr_nan = np.ones((4, 4, 4, 30), dtype=np.float32)
    arr_nan[1, 1, 1, 5] = np.nan
    with pytest.raises(InvalidSliceTimingError, match="non-finite values"):
        correct_slice_timing_with_report(nib.Nifti1Image(arr_nan, np.eye(4)), meta)

    # Singular affine rejected (collinear columns give det=0 without zero-zoom divide-by-zero warning)
    sing_aff = np.eye(4)
    sing_aff[:, 1] = sing_aff[:, 0]
    with pytest.raises(InvalidSliceTimingError, match="singular"):
        correct_slice_timing_with_report(nib.Nifti1Image(np.ones((4, 4, 4, 30), dtype=np.float32), sing_aff), meta)

    # Complex dtype rejected
    c_img = nib.Nifti1Image(np.ones((4, 4, 4, 30), dtype=np.complex64), np.eye(4))
    with pytest.raises(InvalidSliceTimingError, match="not a real numeric type"):
        correct_slice_timing_with_report(c_img, meta)

    # Too few volumes for cubic spline rejected
    short_img = nib.Nifti1Image(np.ones((4, 4, 4, 3), dtype=np.float32), np.eye(4))
    with pytest.raises(InvalidSliceTimingError, match="Insufficient BOLD timepoints"):
        correct_slice_timing_with_report(short_img, meta)

    # Scaled int16 NIfTI on disk with scl_slope/scl_inter must read true continuous values cleanly
    img_raw, gt, meta_ok, _ = _make_analytical_bold((4, 4, 9), n_vols=80, tr=2.0)
    raw_f = np.asarray(img_raw.dataobj, dtype=np.float64)
    slope, inter = 0.01, 50.0
    int_data = np.round((raw_f - inter) / slope).astype(np.int16)
    scaled_img = nib.Nifti1Image(int_data, img_raw.affine)
    scaled_img.header.set_data_dtype(np.int16)
    scaled_img.header.set_slope_inter(slope, inter)
    scaled_img.header.set_zooms((3.0, 3.0, 3.5, 2.0))
    scaled_img.header.set_xyzt_units("mm", "sec")
    scaled_path = tmp_path / "scaled_int16_bold.nii.gz"
    nib.save(scaled_img, str(scaled_path))

    corr_scaled, _ = correct_slice_timing_with_report(scaled_path, meta_ok)
    scaled_arr = np.asarray(corr_scaled.dataobj, dtype=np.float64)
    assert math.sqrt(float(np.mean((scaled_arr[..., 2:-2] - gt[..., 2:-2]) ** 2))) < 0.01
    assert math.sqrt(float(np.mean((scaled_arr - gt) ** 2))) < 0.025
    assert corr_scaled.header.get_slope_inter() in ((None, None), (1.0, 0.0))


# ==============================================================================
# 18–25. Simultaneous Skip, Interpolation Accuracy, Boundaries, Geometry & Provenance
# ==============================================================================


def test_18_simultaneous_slices_skip_without_resampling():
    """Simultaneous acquisition (all slice timings identical) skips temporal interpolation bit-for-bit."""
    tr = 2.0
    sim_timings = [0.75] * 12
    img, _, meta, _ = _make_analytical_bold((5, 5, 12), n_vols=80, tr=tr, slice_timing=sim_timings)
    out_img, rep = correct_slice_timing_with_report(img, meta)

    assert rep["status"] == "SKIPPED_SIMULTANEOUS_ACQUISITION"
    assert rep["corrected"] is False
    assert rep["relative_correction_necessary"] is False
    assert rep["num_corrected_slices"] == 0
    assert rep["slice_order_interpretation"] == "simultaneous"
    np.testing.assert_array_equal(np.asarray(out_img.dataobj), np.asarray(img.dataobj))


@pytest.mark.parametrize(
    "method,max_interior_rmse,max_whole_rmse",
    [
        ("cubic_spline", 0.005, 0.04),
        ("quintic_bspline", 0.003, 0.04),
        ("fourier", 0.005, 0.05),
        ("linear", 0.14, 0.15),
    ],
)
def test_19_interpolation_accuracy_across_backends(method: str, max_interior_rmse: float, max_whole_rmse: float):
    """Compare interpolation backends against exact analytical ground truth."""
    tr = 2.0
    timings = _interleaved_timings(17, tr)
    img, gt, meta, _ = _make_analytical_bold((5, 5, 17), n_vols=120, tr=tr, slice_timing=timings)
    corr_img, rep = correct_slice_timing_with_report(img, meta, interpolation_method=method)
    m = _metrics(np.asarray(corr_img.dataobj, dtype=np.float64), np.asarray(img.dataobj, dtype=np.float64), gt, tr)

    assert rep["interpolation_method"] == method
    assert m["rmse_interior_after"] < max_interior_rmse
    if method == "quintic_bspline":
        assert m["rmse_interior5"] < 0.0005
    assert m["rmse_after"] < max_whole_rmse
    assert m["rmse_after"] < m["rmse_before"]
    assert m["pearson_r_mean"] > 0.999


def test_19b_custom_reference_time_validation():
    """Verify non-half-TR custom reference_time and invalid reference_time rejection."""
    tr = 2.0
    custom_ref = 0.4
    img_c, gt_c, meta_c, _ = _make_analytical_bold((5, 5, 11), n_vols=80, tr=tr, reference_time=custom_ref)
    corr_c, rep_c = correct_slice_timing_with_report(img_c, meta_c, reference_time=custom_ref)
    m_c = _metrics(np.asarray(corr_c.dataobj, dtype=np.float64), np.asarray(img_c.dataobj, dtype=np.float64), gt_c, tr)
    assert rep_c["reference_time"] == pytest.approx(custom_ref)
    assert rep_c["half_tr_shift_applied"] is False
    assert m_c["rmse_interior_after"] < 0.005
    assert m_c["rmse_after"] < 0.045
    assert m_c["rmse_after"] < m_c["rmse_before"] * 0.08

    with pytest.raises(InvalidSliceTimingError, match="reference_time"):
        correct_slice_timing_with_report(img_c, meta_c, reference_time=tr)


def test_20_boundary_handling_no_circular_wraparound_leakage():
    """Start (t=0) and end (t=T-1) preserve linear trends and remain finite without FFT wrap-around."""
    tr = 2.0
    timings = np.linspace(0.0, tr, 13, endpoint=False).tolist()
    img, gt, meta, _ = _make_analytical_bold(
        (5, 5, 13), n_vols=90, tr=tr, slice_timing=timings, linear_drift=0.12
    )
    corr_img, rep = correct_slice_timing_with_report(img, meta, interpolation_method="cubic_spline")
    raw = np.asarray(img.dataobj, dtype=np.float64)
    est = np.asarray(corr_img.dataobj, dtype=np.float64)
    m = _metrics(est, raw, gt, tr)

    assert np.isfinite(est).all()
    assert corr_img.shape[3] == 90  # No silent cropping
    assert rep["boundary_pad_samples"] == 4
    assert m["rmse_interior5"] < 0.005
    err_start_est = m["rmse_first"]
    err_end_est = m["rmse_final"]
    err_start_raw = float(np.sqrt(np.mean((raw[..., 0] - gt[..., 0]) ** 2)))
    err_end_raw = float(np.sqrt(np.mean((raw[..., -1] - gt[..., -1]) ** 2)))
    assert err_start_est < err_start_raw * 0.25
    assert err_end_est < err_end_raw * 0.25


def test_20b_five_region_boundary_stability_noisy_and_transient(tmp_path: Path):
    """Evaluate 5 temporal regions across smooth, noisy, trend, and transient signals + short T=4..6."""
    tr = 2.0
    n_vols = 120
    n_vox = 200
    t_idx = np.arange(n_vols, dtype=np.float64)
    t_obs = t_idx * tr
    rng = np.random.default_rng(777)
    phases = rng.uniform(-np.pi, np.pi, size=(n_vox, 1))
    noise = rng.normal(0.0, 0.25, size=(n_vox, n_vols))

    def _smooth(t_s: np.ndarray) -> np.ndarray:
        return (
            100.0
            + 4.0 * np.sin(2.0 * np.pi * 0.025 * t_s + 0.35 + phases)
            + 3.0 * np.sin(2.0 * np.pi * 0.048 * t_s - 0.80 + phases)
            + 2.0 * np.sin(2.0 * np.pi * 0.072 * t_s + 1.15 + phases)
        )

    for delta in (-0.45, 0.45):
        t_gt = t_obs[None, :] + delta * tr
        y_gt_smooth = _smooth(t_gt)
        y_noisy = _smooth(t_obs[None, :]) + noise
        y_trend = _smooth(t_obs[None, :]) + 0.15 * t_obs[None, :]
        y_trend_gt = y_gt_smooth + 0.15 * t_gt
        y_trans = _smooth(t_obs[None, :]) + 15.0 * np.exp(-0.5 * ((t_obs[None, :] - 2.0) / 1.1) ** 2) + 15.0 * np.exp(
            -0.5 * ((t_obs[None, :] - (n_vols - 2) * tr) / 1.1) ** 2
        )
        y_trans_gt = y_gt_smooth + 15.0 * np.exp(-0.5 * ((t_gt - 2.0) / 1.1) ** 2) + 15.0 * np.exp(
            -0.5 * ((t_gt - (n_vols - 2) * tr) / 1.1) ** 2
        )

        edge_slice = slice(0, 1) if delta < 0 else slice(-1, None)
        for obs, gt_arr, check_strong in (
            (y_noisy, y_gt_smooth, False),
            (y_trans, y_trans_gt, True),
        ):
            old_est = CubicSpline(t_idx, obs, axis=-1, bc_type="not-a-knot", extrapolate=True)(t_idx + delta)
            t_pad, y_pad = _odd_reflect_pad_series(obs, n_vols, 4)
            new_est = CubicSpline(t_pad, y_pad, axis=-1, bc_type="not-a-knot", extrapolate=False)(t_idx + delta)

            assert np.isfinite(new_est).all()
            rmse_old_edge = float(np.sqrt(np.mean((old_est[..., edge_slice] - gt_arr[..., edge_slice]) ** 2)))
            rmse_new_edge = float(np.sqrt(np.mean((new_est[..., edge_slice] - gt_arr[..., edge_slice]) ** 2)))
            max_old_edge = float(np.max(np.abs(old_est[..., edge_slice] - gt_arr[..., edge_slice])))
            max_new_edge = float(np.max(np.abs(new_est[..., edge_slice] - gt_arr[..., edge_slice])))
            rmse_new_int = float(np.sqrt(np.mean((new_est[..., 5:-5] - gt_arr[..., 5:-5]) ** 2)))
            if check_strong:
                assert rmse_new_edge < rmse_old_edge * 0.25
            else:
                assert rmse_new_edge < rmse_old_edge
                assert max_new_edge < max_old_edge * 0.80
                assert rmse_new_int < 0.25

        # Linear trend component preserves exact zero interior RMSE and keeps first5/last5 bounded
        t_pad_tr, y_pad_tr = _odd_reflect_pad_series(y_trend, n_vols, 4)
        est_tr = CubicSpline(t_pad_tr, y_pad_tr, axis=-1, bc_type="not-a-knot", extrapolate=False)(t_idx + delta)
        assert float(np.sqrt(np.mean((est_tr[..., 5:-5] - y_trend_gt[..., 5:-5]) ** 2))) < 0.005

    # Short time series (T=4..6) must pad cleanly without negative-slice bugs and preserve StartTime
    for short_t in (4, 5, 6):
        img_s, _, meta_s, _ = _make_analytical_bold((3, 3, 5), n_vols=short_t, tr=tr)
        b_p = tmp_path / f"sub-s{short_t}_bold.nii.gz"
        j_p = tmp_path / f"sub-s{short_t}_bold.json"
        nib.save(img_s, str(b_p))
        j_p.write_text(json.dumps(meta_s), encoding="utf-8")
        out_s, prov_s = write_slice_timing_derivative(b_p, j_p, tmp_path / f"deriv_{short_t}")
        assert out_s.shape == (3, 3, 5, short_t)
        assert np.isfinite(np.asarray(out_s.dataobj)).all()
        d_json = json.loads(Path(prov_s["derivative_json_path"]).read_text(encoding="utf-8"))
        assert d_json["StartTime"] == pytest.approx(1.0)
        assert d_json["BoundaryPaddingStrategy"] == "odd_point_reflection_4tr"
        assert d_json["BoundaryExtrapolationDisabled"] is True
        assert d_json["BoundaryEdgeVolumesAtRisk"] == [0, short_t - 1]
        assert d_json["BoundaryRiskWindowVolumes"] == list(range(short_t))
        assert d_json["AutomaticBoundaryVolumeRemovalApplied"] is False


def test_20c_boundary_risk_window_and_positive_negative_shift_regimes(tmp_path: Path):
    """Verify 5-region boundary metadata and accuracy under explicit positive, negative, and centered STC shifts."""
    tr = 2.0
    n_vols = 120
    n_slices = 11
    timings = np.linspace(0.0, tr, n_slices, endpoint=False).tolist()

    # Test negative-only shift regime (ref=0.05 -> delta <= 0 across late slices),
    # positive-only shift regime (ref=1.80 -> delta >= 0 across early slices),
    # and centered TR/2 (both positive and negative shifts in one scan).
    for case_label, ref_t, s_dir in (
        ("negative_shift_ref_start_k", float(timings[0]), "k"),
        ("positive_shift_ref_end_k", float(timings[-1]), "k"),
        ("mixed_shift_ref_mid_k_minus", tr / 2.0, "k-"),
    ):
        img, gt, meta, _ = _make_analytical_bold(
            (4, 4, n_slices),
            n_vols=n_vols,
            tr=tr,
            slice_timing=timings,
            direction=s_dir,
            reference_time=ref_t,
        )
        cdir = tmp_path / case_label
        cdir.mkdir(parents=True, exist_ok=True)
        b_p = cdir / "sub-synth_task-rest_bold.nii.gz"
        j_p = cdir / "sub-synth_task-rest_bold.json"
        nib.save(img, str(b_p))
        j_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        out_img, prov = write_slice_timing_derivative(b_p, j_p, cdir / "derivatives", reference_time=ref_t)
        assert out_img.shape == (4, 4, n_slices, n_vols)  # All T=120 volumes preserved

        d_json = json.loads(Path(prov["derivative_json_path"]).read_text(encoding="utf-8"))
        assert d_json["StartTime"] == pytest.approx(ref_t)
        assert d_json["TemporalOriginShiftSeconds"] == pytest.approx(ref_t)
        assert d_json["BoundaryPaddingStrategy"] == "odd_point_reflection_4tr"
        assert d_json["BoundaryExtrapolationDisabled"] is True
        assert d_json["BoundaryEdgeVolumesAtRisk"] == [0, n_vols - 1]
        assert d_json["BoundaryRiskWindowSamplesPerSide"] == 5
        assert d_json["BoundaryRiskWindowVolumes"] == [0, 1, 2, 3, 4, 115, 116, 117, 118, 119]
        assert d_json["AutomaticBoundaryVolumeRemovalApplied"] is False

        regions = d_json["BoundaryRegions"]
        assert regions["first_volume"] == [0]
        assert regions["first_five_volumes"] == [0, 1, 2, 3, 4]
        assert regions["interior_volumes"] == {"start_index": 5, "end_index_inclusive": 114, "count": 110}
        assert regions["last_five_volumes"] == [115, 116, 117, 118, 119]
        assert regions["final_volume"] == [119]

        guidance = d_json["VolumeCensoringGuidance"]
        assert "[0, T-1]" in guidance
        assert "BoundaryRiskWindowVolumes" in guidance
        assert "opt-in" in guidance

        # Verify regional RMSE properties for both shift polarities
        raw_arr = np.asarray(img.dataobj, dtype=np.float64)
        out_arr = np.asarray(out_img.dataobj, dtype=np.float64)
        assert np.isfinite(out_arr).all()
        rmse_interior = float(np.sqrt(np.mean((out_arr[..., 5:-5] - gt[..., 5:-5]) ** 2)))
        rmse_raw_int = float(np.sqrt(np.mean((raw_arr[..., 5:-5] - gt[..., 5:-5]) ** 2)))
        assert rmse_interior < 0.015
        assert rmse_interior < rmse_raw_int * 0.05

        # Negative shifts hit left boundary (0..4); positive shifts hit right boundary (T-5..T-1);
        # both remain far more accurate than uncorrected raw sampling.
        rmse_first5 = float(np.sqrt(np.mean((out_arr[..., :5] - gt[..., :5]) ** 2)))
        rmse_raw_first5 = float(np.sqrt(np.mean((raw_arr[..., :5] - gt[..., :5]) ** 2)))
        rmse_last5 = float(np.sqrt(np.mean((out_arr[..., -5:] - gt[..., -5:]) ** 2)))
        rmse_raw_last5 = float(np.sqrt(np.mean((raw_arr[..., -5:] - gt[..., -5:]) ** 2)))
        assert rmse_first5 < rmse_raw_first5 * 0.35
        assert rmse_last5 < rmse_raw_last5 * 0.35


def test_21_22_23_shape_affine_header_tr_starttime_and_determinism(tmp_path: Path):
    """Verify exact shape, affine, NIfTI TR/sec unit, StartTime derivative metadata, and zero determinism drift."""
    img, _, meta, _ = _make_analytical_bold((6, 7, 13), n_vols=70, tr=2.5)
    bold_p = tmp_path / "sub-synthetic01_ses-1_task-rest_bold.nii.gz"
    json_p = tmp_path / "sub-synthetic01_ses-1_task-rest_bold.json"
    nib.save(img, str(bold_p))
    json_p.write_text(json.dumps(meta), encoding="utf-8")

    deriv1 = tmp_path / "deriv1"
    deriv2 = tmp_path / "deriv2"
    out1, _ = write_slice_timing_derivative(bold_p, json_p, deriv1)
    out2, _ = write_slice_timing_derivative(bold_p, json_p, deriv2)

    assert out1.shape == img.shape == (6, 7, 13, 70)
    np.testing.assert_allclose(out1.affine, img.affine, atol=1e-12)
    assert out1.header.get_zooms()[3] == pytest.approx(2.5)
    assert out1.header.get_xyzt_units()[1] == "sec"
    assert float(np.max(np.abs(np.asarray(out1.dataobj) - np.asarray(out2.dataobj)))) == 0.0

    deriv_json = json.loads((deriv1 / "sub-synthetic01_ses-1_task-rest_desc-stc_bold.json").read_text(encoding="utf-8"))
    assert deriv_json["SliceTimingCorrected"] is True
    assert deriv_json["SliceTimingReferenceTime"] == pytest.approx(1.25)
    assert deriv_json["StartTime"] == pytest.approx(1.25)
    assert deriv_json["TemporalOriginShiftSeconds"] == pytest.approx(1.25)
    assert "+1.250000s offset relative to volume onset" in deriv_json["TemporalOriginNote"]
    assert deriv_json["OriginalSliceTiming"] == pytest.approx(meta["SliceTiming"])


def test_24_prevent_double_slice_timing_correction(tmp_path: Path):
    """Derivative output marked SliceTimingCorrected=True must be rejected if passed to STC again."""
    img, _, meta, _ = _make_analytical_bold((5, 5, 11), n_vols=60, tr=2.0)
    bold_p = tmp_path / "sub-synthetic01_task-rest_bold.nii.gz"
    json_p = tmp_path / "sub-synthetic01_task-rest_bold.json"
    nib.save(img, str(bold_p))
    json_p.write_text(json.dumps(meta), encoding="utf-8")

    deriv_dir = tmp_path / "derivatives"
    _, prov = write_slice_timing_derivative(bold_p, json_p, deriv_dir)

    with pytest.raises(DoubleSliceTimingCorrectionError, match="double slice-timing correction"):
        correct_slice_timing_with_report(
            prov["derivative_nii_path"],
            prov["derivative_json_path"],
        )

    # Also test string 'true' even if SliceTiming key is missing
    with pytest.raises(DoubleSliceTimingCorrectionError):
        validate_slice_timing_metadata(img, {"RepetitionTime": 2.0, "SliceTimingCorrected": "true"})


def test_25_input_files_remain_byte_identical_after_derivative_write(tmp_path: Path):
    """Original input BOLD NIfTI and JSON sidecar SHA-256 checksums must remain byte-identical."""
    img, _, meta, _ = _make_analytical_bold((6, 6, 11), n_vols=60, tr=2.0)
    bold_p = tmp_path / "raw" / "sub-synthetic99_task-rest_bold.nii.gz"
    json_p = tmp_path / "raw" / "sub-synthetic99_task-rest_bold.json"
    bold_p.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(bold_p))
    json_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    sha_bold_before = compute_file_sha256(bold_p)
    sha_json_before = compute_file_sha256(json_p)

    _, prov = write_slice_timing_derivative(bold_p, json_p, tmp_path / "derivatives")
    assert prov["input_bold_sha256"] == sha_bold_before
    assert prov["input_json_sha256"] == sha_json_before
    assert compute_file_sha256(bold_p) == sha_bold_before
    assert compute_file_sha256(json_p) == sha_json_before
    assert prov["input_file_unmodified_verified"] is True
    assert Path(prov["derivative_nii_path"]).is_file()
    assert Path(prov["derivative_json_path"]).is_file()
    assert Path(prov["provenance_json_path"]).is_file()


# ==============================================================================
# 26–29. Security, 21-Stage Order Invariants & 37-Slice x 240-Vol E2E
# ==============================================================================


def test_26_27_synthetic_only_enforcement_and_real_data_rejection(tmp_path: Path):
    """Verify STC and register remain STRICTLY_SYNTHETIC_ONLY and reject real-data paths & symlinks."""
    gate = ResearchAuthorizationGate(base_dir=Path.cwd())
    reg = gate.evaluate_register()
    assert reg["execution_policy"] == "STRICTLY_SYNTHETIC_ONLY"
    assert reg["real_data_authorized"] is False
    assert reg["overall_authorization_state"] == "NOT_AUTHORIZED"

    for blocked_rel in (
        Path("external") / "NeuroADHD_Converted" / "sub-01_bold.nii.gz",
        Path("external") / "sub-2344857" / "func" / "sub-2344857_bold.nii.gz",
        Path("external") / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz",
    ):
        with pytest.raises(PermissionError, match="SAFETY VIOLATION"):
            correct_slice_timing_with_report(blocked_rel, {"RepetitionTime": 2.0, "SliceTiming": [0.0]})

    # Symlink rejection (even targeting a valid synthetic file)
    real_synth = tmp_path / "sub-synthetic_bold.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 20), dtype=np.float32), np.eye(4)), str(real_synth))
    json_synth = tmp_path / "sub-synthetic_bold.json"
    json_synth.write_text(json.dumps({"RepetitionTime": 2.0, "SliceTiming": [0.0, 0.5, 1.0, 1.5]}), encoding="utf-8")
    link_synth = tmp_path / "link_bold.nii.gz"
    link_synth.symlink_to(real_synth)

    with pytest.raises(PermissionError, match="Symlink"):
        correct_slice_timing_with_report(link_synth, json_synth)
    with pytest.raises(PermissionError, match="Symlink"):
        write_slice_timing_derivative(link_synth, json_synth, tmp_path / "deriv")

    sw = detect_stc_software()
    assert sw["default_backend"] == "scipy.interpolate.CubicSpline"
    assert DEFAULT_STC_METHOD == "cubic_spline"


def test_run_poc_v2_pipeline_stage3_stc_applied_skipped_and_double_blocked(tmp_path: Path):
    """End-to-end 22-stage runner tests: STC APPLIED, SKIPPED_MISSING_METADATA, SKIPPED_UNKNOWN_SLICE_AXIS, and double-STC fail-closed."""
    with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": "step4_integration_test_key"}, clear=False):
        os.environ.pop("NEUROADHD_MODEL_DIR", None)

        # Case 1: APPLIED (default generate_synthetic_phantom_session emits SliceTiming & SliceEncodingDirection='k')
        paths_ok = generate_synthetic_phantom_session(tmp_path / "in_ok", subject_id="sub-synthetic101", t_vol=55, seed=7)
        bold_sha0 = compute_sha256(paths_ok["bold_nii"])
        json_sha0 = compute_sha256(paths_ok["bold_json"])
        res_ok = run_poc_v2_pipeline(
            bold_path=paths_ok["bold_nii"],
            bold_json_path=paths_ok["bold_json"],
            t1w_path=paths_ok["t1w_nii"],
            atlas_path=paths_ok["cc200_atlas"],
            template_path=paths_ok["mni_template"],
            output_dir=tmp_path / "out_ok",
            subject_id="sub-synthetic101",
            session_id="ses-1",
            manifest_path=paths_ok["synthetic_manifest"],
        )
        assert res_ok["status"] == "SUCCESS"
        assert res_ok["provenance"]["pipeline_stages_total"] == 22
        assert res_ok["provenance"]["motion_estimation"]["estimated_before_slice_timing"] is True
        order_list = res_ok["provenance"]["preprocessing_order"]
        assert len(order_list) == 22
        assert "Motion Parameter Estimation" in order_list[1]
        assert "Slice-Timing Correction" in order_list[2]
        assert "Susceptibility Distortion Correction" in order_list[3]
        assert "Motion Transform" in order_list[4]
        stc_p = res_ok["provenance"]["slice_timing_correction"]
        assert stc_p["status"] == "APPLIED"
        assert stc_p["corrected"] is True
        assert stc_p["input_bold_sha256"] == bold_sha0
        assert stc_p["input_json_sha256"] == json_sha0
        assert stc_p["input_file_unmodified_verified"] is True
        assert (Path(res_ok["output_dir"]) / stc_p["derivative_nii"]).is_file()
        assert (Path(res_ok["output_dir"]) / stc_p["derivative_json"]).is_file()
        assert compute_sha256(paths_ok["bold_nii"]) == bold_sha0
        assert compute_sha256(paths_ok["bold_json"]) == json_sha0

        # Case 2: Missing SliceTiming skipped cleanly without inventing timings (allow_missing_slice_timing=True)
        paths_miss = generate_synthetic_phantom_session(
            tmp_path / "in_miss", subject_id="sub-synthetic102", t_vol=55, seed=8, include_slice_timing=False
        )
        res_miss = run_poc_v2_pipeline(
            bold_path=paths_miss["bold_nii"],
            bold_json_path=paths_miss["bold_json"],
            t1w_path=paths_miss["t1w_nii"],
            atlas_path=paths_miss["cc200_atlas"],
            template_path=paths_miss["mni_template"],
            output_dir=tmp_path / "out_miss",
            subject_id="sub-synthetic102",
            session_id="ses-1",
            manifest_path=paths_miss["synthetic_manifest"],
            allow_missing_slice_timing=True,
        )
        assert res_miss["provenance"]["slice_timing_correction"]["status"] == "SKIPPED_MISSING_METADATA"
        assert res_miss["provenance"]["slice_timing_correction"]["corrected"] is False

        # Strict mode must reject missing SliceTiming at Stage 3
        with pytest.raises(MissingSliceTimingError):
            run_poc_v2_pipeline(
                bold_path=paths_miss["bold_nii"],
                bold_json_path=paths_miss["bold_json"],
                t1w_path=paths_miss["t1w_nii"],
                atlas_path=paths_miss["cc200_atlas"],
                template_path=paths_miss["mni_template"],
                output_dir=tmp_path / "out_strict_fail",
                subject_id="sub-synthetic102",
                session_id="ses-1",
                manifest_path=paths_miss["synthetic_manifest"],
                allow_missing_slice_timing=False,
            )

        # Case 3: Double slice-timing correction blocked in Stage 3
        paths_dbl = generate_synthetic_phantom_session(
            tmp_path / "in_dbl", subject_id="sub-synthetic103", t_vol=55, seed=9
        )
        meta_dbl = json.loads(paths_dbl["bold_json"].read_text(encoding="utf-8"))
        meta_dbl["SliceTimingCorrected"] = True
        paths_dbl["bold_json"].write_text(json.dumps(meta_dbl, indent=2), encoding="utf-8")
        man_dbl = json.loads(paths_dbl["synthetic_manifest"].read_text(encoding="utf-8"))
        man_dbl["files"]["bold_json"]["sha256"] = compute_sha256(paths_dbl["bold_json"])
        man_dbl["integrity_hmac"] = compute_manifest_hmac(man_dbl)
        paths_dbl["synthetic_manifest"].write_text(json.dumps(man_dbl, indent=2), encoding="utf-8")

        with pytest.raises(DoubleSliceTimingCorrectionError):
            run_poc_v2_pipeline(
                bold_path=paths_dbl["bold_nii"],
                bold_json_path=paths_dbl["bold_json"],
                t1w_path=paths_dbl["t1w_nii"],
                atlas_path=paths_dbl["cc200_atlas"],
                template_path=paths_dbl["mni_template"],
                output_dir=tmp_path / "out_dbl_fail",
                subject_id="sub-synthetic103",
                session_id="ses-1",
                manifest_path=paths_dbl["synthetic_manifest"],
            )
        fail_reports = list((tmp_path / "out_dbl_fail" / "sub-synthetic103").glob("failed_run_*/pipeline_failure_report.json"))
        assert len(fail_reports) == 1
        assert json.loads(fail_reports[0].read_text(encoding="utf-8"))["stage"] == "Stage 3: Slice-Timing Correction"

        # Case 4: Absent SliceEncodingDirection AND unset header slice_dim skips STC cleanly with clear reason
        paths_unk = generate_synthetic_phantom_session(
            tmp_path / "in_unk", subject_id="sub-synthetic104", t_vol=55, seed=10
        )
        b_unk_img = nib.load(str(paths_unk["bold_nii"]))
        b_unk_nodim = nib.Nifti1Image(np.asanyarray(b_unk_img.dataobj), b_unk_img.affine)
        b_unk_nodim.header.set_zooms(b_unk_img.header.get_zooms())
        b_unk_nodim.header.set_xyzt_units("mm", "sec")
        nib.save(b_unk_nodim, str(paths_unk["bold_nii"]))
        meta_unk = json.loads(paths_unk["bold_json"].read_text(encoding="utf-8"))
        meta_unk.pop("SliceEncodingDirection", None)
        paths_unk["bold_json"].write_text(json.dumps(meta_unk, indent=2), encoding="utf-8")
        man_unk = json.loads(paths_unk["synthetic_manifest"].read_text(encoding="utf-8"))
        man_unk["files"]["bold_nii"]["sha256"] = compute_sha256(paths_unk["bold_nii"])
        man_unk["files"]["bold_json"]["sha256"] = compute_sha256(paths_unk["bold_json"])
        man_unk["integrity_hmac"] = compute_manifest_hmac(man_unk)
        paths_unk["synthetic_manifest"].write_text(json.dumps(man_unk, indent=2), encoding="utf-8")

        res_unk = run_poc_v2_pipeline(
            bold_path=paths_unk["bold_nii"],
            bold_json_path=paths_unk["bold_json"],
            t1w_path=paths_unk["t1w_nii"],
            atlas_path=paths_unk["cc200_atlas"],
            template_path=paths_unk["mni_template"],
            output_dir=tmp_path / "out_unk",
            subject_id="sub-synthetic104",
            session_id="ses-1",
            manifest_path=paths_unk["synthetic_manifest"],
        )
        assert res_unk["status"] == "SUCCESS"
        assert res_unk["provenance"]["slice_timing_correction"]["status"] == "SKIPPED_UNKNOWN_SLICE_AXIS"
        assert res_unk["provenance"]["slice_timing_correction"]["corrected"] is False


def test_motion_estimated_from_raw_bold_before_stc_and_single_step_resampling(tmp_path: Path):
    """Prove Stage 2 motion estimation runs on raw BOLD before Stage 3 STC, independent of STC method."""
    with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": "step4_1_order_spy_key"}, clear=False):
        os.environ.pop("NEUROADHD_MODEL_DIR", None)
        paths = generate_synthetic_phantom_session(tmp_path / "in_ord", subject_id="sub-synthetic201", t_vol=55, seed=19)
        raw_bold_arr = np.asanyarray(nib.load(str(paths["bold_nii"])).dataobj, dtype=np.float32)

        call_events: List[str] = []
        captured_motion_input_diff: List[float] = []
        captured_resample_has_motion: List[bool] = []
        captured_resample_src_max_diff_from_stc: List[float] = []
        captured_stc_out: List[np.ndarray] = []

        def spy_est(bold_img):
            call_events.append("estimate_motion_parameters")
            arr = np.asanyarray(bold_img.dataobj, dtype=np.float32)
            captured_motion_input_diff.append(float(np.max(np.abs(arr - raw_bold_arr))))
            return real_estimate_motion_parameters(bold_img)

        def spy_stc(*args, **kwargs):
            call_events.append("write_slice_timing_derivative")
            out_img, rep = real_write_slice_timing_derivative(*args, **kwargs)
            captured_stc_out.append(np.asanyarray(out_img.dataobj, dtype=np.float32))
            return out_img, rep

        def spy_apply(bold_img, rigid_transforms, interpolation="linear"):
            call_events.append("apply_motion_transforms")
            return real_apply_motion_transforms(bold_img, rigid_transforms, interpolation=interpolation)

        def spy_resample(bold_4d, composed_map, target_shape, target_affine, interpolation="linear", max_memory_gb=2.5, motion_transforms=None):
            call_events.append("resample_4d_bold")
            captured_resample_has_motion.append(motion_transforms is not None and len(motion_transforms) == 55)
            arr = np.asanyarray(bold_4d.dataobj, dtype=np.float32)
            captured_resample_src_max_diff_from_stc.append(float(np.max(np.abs(arr - captured_stc_out[-1]))))
            return real_resample_4d_bold(
                bold_4d, composed_map, target_shape, target_affine,
                interpolation=interpolation, max_memory_gb=max_memory_gb, motion_transforms=motion_transforms,
            )

        with (
            mock.patch("scripts.run_poc_v2.estimate_motion_parameters", side_effect=spy_est),
            mock.patch("scripts.run_poc_v2.write_slice_timing_derivative", side_effect=spy_stc),
            mock.patch("scripts.run_poc_v2.apply_motion_transforms", side_effect=spy_apply),
            mock.patch("scripts.run_poc_v2.resample_4d_bold", side_effect=spy_resample),
        ):
            res_cubic = run_poc_v2_pipeline(
                bold_path=paths["bold_nii"],
                bold_json_path=paths["bold_json"],
                t1w_path=paths["t1w_nii"],
                atlas_path=paths["cc200_atlas"],
                template_path=paths["mni_template"],
                output_dir=tmp_path / "out_cubic",
                subject_id="sub-synthetic201",
                session_id="ses-1",
                manifest_path=paths["synthetic_manifest"],
                slice_timing_interpolation="cubic_spline",
            )

        assert call_events == [
            "estimate_motion_parameters",
            "write_slice_timing_derivative",
            "apply_motion_transforms",
            "resample_4d_bold",
        ]
        assert captured_motion_input_diff == [0.0]  # Bit-identical raw uncorrected BOLD fed to Stage 2
        assert captured_resample_has_motion == [True]  # Single-step composed MNI + motion resampling
        assert captured_resample_src_max_diff_from_stc == [0.0]  # Unresampled STC output fed directly to Stage 12

        # Run again with Fourier STC and confirm Stage 2 motion_params & FD are bit-for-bit identical!
        res_fourier = run_poc_v2_pipeline(
            bold_path=paths["bold_nii"],
            bold_json_path=paths["bold_json"],
            t1w_path=paths["t1w_nii"],
            atlas_path=paths["cc200_atlas"],
            template_path=paths["mni_template"],
            output_dir=tmp_path / "out_fourier",
            subject_id="sub-synthetic201",
            session_id="ses-1",
            manifest_path=paths["synthetic_manifest"],
            slice_timing_interpolation="fourier",
        )
        m_cubic = pd.read_csv(res_cubic["artifacts"]["motion_tsv"], sep="\t").to_numpy()
        m_fourier = pd.read_csv(res_fourier["artifacts"]["motion_tsv"], sep="\t").to_numpy()
        np.testing.assert_array_equal(m_cubic, m_fourier)


def test_realistic_37slice_240vol_source_to_61slice_e2e(tmp_path: Path):
    """Source-acquisition 37-slice x 240-volume BOLD -> 61-slice MNI/CC200 full 22-stage E2E test."""
    with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": "step4_1_37slice_240vol_key"}, clear=False):
        os.environ.pop("NEUROADHD_MODEL_DIR", None)
        rng = np.random.default_rng(37240)

        ses_dir = tmp_path / "in" / "sub-synthetic370" / "ses-1"
        func_dir = ses_dir / "func"
        anat_dir = ses_dir / "anat"
        tpl_dir = tmp_path / "in" / "templates"
        for d in (func_dir, anat_dir, tpl_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Source EPI: 28x28x37x240 (37 slices != 61 standard target slices, TR=2.0s, interleaved)
        epi_shape = (28, 28, 37)
        epi_affine = np.diag([3.0, 3.0, 2.5, 1.0])
        epi_affine[:3, 3] = [-42.0, -42.0, -46.25]
        n_vols = 240
        tr = 2.0

        ex, ey, ez = np.ogrid[: epi_shape[0], : epi_shape[1], : epi_shape[2]]
        # World sphere radius ~32 mm centered at (0, 0, 0) mm
        wx_e = epi_affine[0, 0] * ex + epi_affine[0, 3]
        wy_e = epi_affine[1, 1] * ey + epi_affine[1, 3]
        wz_e = epi_affine[2, 2] * ez + epi_affine[2, 3]
        epi_brain = (wx_e ** 2 + wy_e ** 2 + wz_e ** 2) <= 32.0 ** 2

        order_37 = list(range(0, 37, 2)) + list(range(1, 37, 2))
        st_37 = [0.0] * 37
        for step, z_idx in enumerate(order_37):
            st_37[z_idx] = round(float(step * tr / 37.0), 8)

        vol_onsets = np.arange(n_vols, dtype=np.float64) * tr
        bold_data = np.zeros((*epi_shape, n_vols), dtype=np.float32)
        for z_idx in range(37):
            m_sl = epi_brain[:, :, z_idx]
            nv = int(np.sum(m_sl))
            if nv > 0:
                t_s = vol_onsets + st_37[z_idx]
                base_sig = (100.0 + 6.0 * np.sin(2.0 * np.pi * 0.035 * t_s + 0.1 * z_idx)).astype(np.float32)[None, :]
                bold_data[m_sl, z_idx, :] = base_sig + rng.normal(0.0, 1.2, size=(nv, n_vols)).astype(np.float32)

        bold_p = func_dir / "sub-synthetic370_ses-1_task-rest_bold.nii.gz"
        b_img = nib.Nifti1Image(bold_data, epi_affine)
        b_img.header.set_zooms((3.0, 3.0, 2.5, float(tr)))
        b_img.header.set_xyzt_units("mm", "sec")
        b_img.header.set_dim_info(slice=2)
        nib.save(b_img, str(bold_p))

        json_p = func_dir / "sub-synthetic370_ses-1_task-rest_bold.json"
        json_p.write_text(
            json.dumps(
                {
                    "RepetitionTime": tr,
                    "TaskName": "rest",
                    "SliceTiming": st_37,
                    "SliceEncodingDirection": "k",
                    "Manufacturer": "Synthetic37Slice",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Standard 61-slice MNI + CC200 grid and matched T1w geometry (32 mm sphere)
        std_shape = (30, 30, 61)
        std_affine = np.diag([3.0, 3.0, 1.5, 1.0])
        std_affine[:3, 3] = [-45.0, -45.0, -45.75]
        sx, sy, sz = np.ogrid[: std_shape[0], : std_shape[1], : std_shape[2]]
        wx_s = std_affine[0, 0] * sx + std_affine[0, 3]
        wy_s = std_affine[1, 1] * sy + std_affine[1, 3]
        wz_s = std_affine[2, 2] * sz + std_affine[2, 3]
        r_std_sq = wx_s ** 2 + wy_s ** 2 + wz_s ** 2
        std_brain = r_std_sq <= 31.0 ** 2

        atlas_data = np.zeros(std_shape, dtype=np.int32)
        # Partition core sphere (r <= 29 mm) into 200 parcels so zero border ROIs fall outside EPI FOV
        roi_idx = np.argwhere(r_std_sq <= 29.0 ** 2)
        cs = max(1, len(roi_idx) // 200)
        for i in range(200):
            s = i * cs
            e = (i + 1) * cs if i < 199 else len(roi_idx)
            for c in roi_idx[s:e]:
                atlas_data[tuple(c)] = i + 1
        atlas_p = tpl_dir / "cc200_61slice_synthetic_atlas.nii.gz"
        nib.save(nib.Nifti1Image(atlas_data, std_affine), str(atlas_p))

        mni_data = np.zeros(std_shape, dtype=np.float32)
        mni_data[std_brain] = 100.0
        mni_p = tpl_dir / "MNI152_61slice.nii.gz"
        nib.save(nib.Nifti1Image(mni_data, std_affine), str(mni_p))

        # T1w 40x40x40 at 1.9mm spacing centered at (0,0,0)mm -> 32mm brain radius
        t1_shape = (40, 40, 40)
        t1_affine = np.diag([1.9, 1.9, 1.9, 1.0])
        t1_affine[:3, 3] = [-38.0, -38.0, -38.0]
        tx, ty, tz = np.ogrid[:40, :40, :40]
        t_r_sq = ((tx - 20) * 1.9) ** 2 + ((ty - 20) * 1.9) ** 2 + ((tz - 20) * 1.9) ** 2
        t1_brain = t_r_sq <= 32.0 ** 2
        skull = (t_r_sq >= 34.0 ** 2) & (t_r_sq <= 36.0 ** 2)
        csf = t_r_sq <= 10.0 ** 2
        wm = t1_brain & (~csf) & (t_r_sq <= 24.0 ** 2)
        gm = t1_brain & (~csf) & (~wm)
        t1_data = np.zeros(t1_shape, dtype=np.float32)
        t1_data[csf] = 25.0 + rng.normal(0.0, 2.0, np.sum(csf)).astype(np.float32)
        t1_data[gm] = 85.0 + rng.normal(0.0, 3.0, np.sum(gm)).astype(np.float32)
        t1_data[wm] = 150.0 + rng.normal(0.0, 3.0, np.sum(wm)).astype(np.float32)
        t1_data[skull] = 110.0 + rng.normal(0.0, 3.0, np.sum(skull)).astype(np.float32)
        t1_p = anat_dir / "sub-synthetic370_ses-1_T1w.nii.gz"
        nib.save(nib.Nifti1Image(t1_data, t1_affine), str(t1_p))

        files_dict = {
            "bold_nii": {"filename": bold_p.name, "resolved_path": str(bold_p.resolve()), "sha256": compute_sha256(bold_p)},
            "bold_json": {"filename": json_p.name, "resolved_path": str(json_p.resolve()), "sha256": compute_sha256(json_p)},
            "t1w_nii": {"filename": t1_p.name, "resolved_path": str(t1_p.resolve()), "sha256": compute_sha256(t1_p)},
            "mni_template": {"filename": mni_p.name, "resolved_path": str(mni_p.resolve()), "sha256": compute_sha256(mni_p)},
            "cc200_atlas": {"filename": atlas_p.name, "resolved_path": str(atlas_p.resolve()), "sha256": compute_sha256(atlas_p)},
        }
        man_p = ses_dir / "synthetic_manifest.json"
        man_p.write_text(
            json.dumps(
                {
                    "manifest_magic": "NEUROADHD_SYNTHETIC_DATASET_V2",
                    "subject_id": "sub-synthetic370",
                    "session_id": "ses-1",
                    "integrity_hmac": compute_manifest_hmac("NEUROADHD_SYNTHETIC_DATASET_V2", "sub-synthetic370", files_dict),
                    "files": files_dict,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        res = run_poc_v2_pipeline(
            bold_path=bold_p,
            bold_json_path=json_p,
            t1w_path=t1_p,
            atlas_path=atlas_p,
            template_path=mni_p,
            output_dir=tmp_path / "out_37_61",
            subject_id="sub-synthetic370",
            session_id="ses-1",
            manifest_path=man_p,
        )
        assert res["status"] == "SUCCESS"
        assert res["provenance"]["pipeline_stages_total"] == 22
        assert res["provenance"]["motion_estimation"]["estimated_before_slice_timing"] is True
        stc_p = res["provenance"]["slice_timing_correction"]
        assert stc_p["status"] == "APPLIED"
        assert stc_p["num_slices"] == 37
        assert stc_p["slice_axis"] == 2

        deriv_img = nib.load(str(Path(res["output_dir"]) / stc_p["derivative_nii"]))
        assert deriv_img.shape == (28, 28, 37, 240)

        roi_df = pd.read_csv(res["artifacts"]["roi_csv"])
        assert roi_df.shape == (200, 241)  # 200 ROIs x ('node/volume' label col + 240 timepoints)
        assert roi_df.columns[0] == "node/volume"
        assert list(roi_df.columns[1:]) == [str(i) for i in range(240)]

        gate5 = res["provenance"]["validation_gates"]["gate5_roi_coverage"]
        assert gate5["n_timepoints_input"] == 240
        assert gate5["n_timepoints_retained"] == 240
        assert gate5["roi_timeseries_shape"] == [200, 240]
        assert gate5["roi_csv_shape"] == [200, 241]
        assert gate5["roi_label_column"] == "node/volume"
        assert gate5["volume_censoring_applied"] is False

        deriv_json_p = Path(res["output_dir"]) / stc_p["derivative_json"]
        deriv_meta = json.loads(deriv_json_p.read_text(encoding="utf-8"))
        assert deriv_meta["StartTime"] == pytest.approx(1.0)
        assert deriv_meta["TemporalOriginShiftSeconds"] == pytest.approx(1.0)
        assert deriv_meta["BoundaryPaddingStrategy"] == "odd_point_reflection_4tr"
        assert deriv_meta["BoundaryExtrapolationDisabled"] is True
        assert deriv_meta["BoundaryEdgeVolumesAtRisk"] == [0, 239]
        assert deriv_meta["BoundaryRiskWindowVolumes"] == [0, 1, 2, 3, 4, 235, 236, 237, 238, 239]
        assert deriv_meta["AutomaticBoundaryVolumeRemovalApplied"] is False

        feat_vec = np.load(res["artifacts"]["features_npy"])
        assert feat_vec.shape == (19900,)
        assert np.all(np.isfinite(feat_vec))


def test_run_poc_v2_240vol_explicit_boundary_censoring_opt_in(tmp_path: Path):
    """Default 240-vol run preserves 240 timepoints; opt-in censoring produces exact remaining count."""
    with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": "step4_2_censor_e2e_key"}, clear=False):
        os.environ.pop("NEUROADHD_MODEL_DIR", None)
        paths = generate_synthetic_phantom_session(tmp_path / "in_240", subject_id="sub-synthetic240", t_vol=240, seed=41)

        res_default = run_poc_v2_pipeline(
            bold_path=paths["bold_nii"],
            bold_json_path=paths["bold_json"],
            t1w_path=paths["t1w_nii"],
            atlas_path=paths["cc200_atlas"],
            template_path=paths["mni_template"],
            output_dir=tmp_path / "out_default",
            subject_id="sub-synthetic240",
            session_id="ses-1",
            manifest_path=paths["synthetic_manifest"],
        )
        df_def = pd.read_csv(res_default["artifacts"]["roi_csv"])
        assert df_def.shape == (200, 241)
        assert res_default["provenance"]["validation_gates"]["gate5_roi_coverage"]["roi_timeseries_shape"] == [200, 240]

        # Explicit censoring of the 10-volume boundary risk window [0..4, 235..239]
        censor_list = [0, 1, 2, 3, 4, 235, 236, 237, 238, 239]
        res_cen = run_poc_v2_pipeline(
            bold_path=paths["bold_nii"],
            bold_json_path=paths["bold_json"],
            t1w_path=paths["t1w_nii"],
            atlas_path=paths["cc200_atlas"],
            template_path=paths["mni_template"],
            output_dir=tmp_path / "out_censored",
            subject_id="sub-synthetic240",
            session_id="ses-1",
            manifest_path=paths["synthetic_manifest"],
            censor_volumes=censor_list,
        )
        df_cen = pd.read_csv(res_cen["artifacts"]["roi_csv"])
        assert df_cen.shape == (200, 231)
        assert df_cen.columns[0] == "node/volume"
        assert list(df_cen.columns[1:]) == [str(i) for i in range(5, 235)]
        gate5_cen = res_cen["provenance"]["validation_gates"]["gate5_roi_coverage"]
        assert gate5_cen["n_timepoints_input"] == 240
        assert gate5_cen["n_timepoints_retained"] == 230
        assert gate5_cen["roi_timeseries_shape"] == [200, 230]
        assert gate5_cen["roi_csv_shape"] == [200, 231]
        assert gate5_cen["volume_censoring_applied"] is True
        assert gate5_cen["censored_volumes"] == censor_list
