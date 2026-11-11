"""Comprehensive unit, negative, and integration tests for Step 4 Slice-Timing Correction (DEC-STC-01).

Covers all Phase B-I requirements using strictly synthetic data generated in-memory
or under pytest's temporary workspace:
- BIDS metadata validation (RepetitionTime, SliceTiming, SliceEncodingDirection)
- All 6 slice encoding directions (i, i-, j, j-, k, k-) & BIDS default 'k' warning
- Sequential, interleaved, multiband repeated timings, and simultaneous skip
- Missing metadata safety (strict rejection vs explicit skip without timing invention)
- Analytical continuous-time ground-truth accuracy across interpolation backends
- Non-periodic boundary handling (t=0 and t=T-1) with non-periodic trend step
- Geometric invariance (shape, spatial 4x4 affine, NIfTI header TR & units='sec')
- Scaled NIfTI (scl_slope/scl_inter) handling and bit-exact determinism
- BIDS derivative immutability (input SHA-256 unchanged; StartTime/+TR/2 origin shift)
- Double slice-timing correction prevention (directly and via run_poc_v2_pipeline)
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
import pytest

from scripts.run_poc_v2 import (
    compute_manifest_hmac,
    compute_sha256,
    generate_synthetic_phantom_session,
    run_poc_v2_pipeline,
)
from src.preprocessing.slice_timing import (
    DEFAULT_STC_METHOD,
    DoubleSliceTimingCorrectionError,
    InvalidSliceTimingError,
    MissingSliceTimingError,
    SliceTimingValidationError,
    compute_file_sha256,
    correct_slice_timing,
    correct_slice_timing_with_report,
    detect_stc_software,
    validate_slice_timing_metadata,
    write_slice_timing_derivative,
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
    meta = {
        "RepetitionTime": float(tr),
        "TaskName": "rest",
        "SliceTiming": [float(v) for v in slice_timing],
        "SliceEncodingDirection": direction,
    }
    return img, gt, meta, tuple(float(v) for v in voxel_times)


def _metrics(est: np.ndarray, raw: np.ndarray, gt: np.ndarray, tr: float) -> Dict[str, float]:
    rmse_raw = float(np.sqrt(np.mean((raw - gt) ** 2)))
    rmse_est = float(np.sqrt(np.mean((est - gt) ** 2)))
    rmse_interior = float(np.sqrt(np.mean((est[..., 2:-2] - gt[..., 2:-2]) ** 2)))
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
        assert m["rmse_after"] < m["rmse_before"] * 0.05
        assert m["mae_after"] < m["mae_before"] * 0.05
        assert m["phase_err_after_rad"] < m["phase_err_before_rad"] * 0.05
        assert m["pearson_r_mean"] > 0.9999


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
    assert m["rmse_after"] < 0.02
    assert m["rmse_after"] < m["rmse_before"] * 0.05
    assert m["pearson_r_mean"] > 0.9999


def test_3_valid_multiband_repeated_acquisition_times():
    """Multiband SMS sequences with repeated timings across groups are supported accurately."""
    n_slices, tr, n_groups = 24, 1.8, 8  # MB=3
    mb_timings = [float((idx % n_groups) * tr / n_groups) for idx in range(n_slices)]
    img, gt, meta, _ = _make_analytical_bold((6, 6, n_slices), n_vols=120, tr=tr, slice_timing=mb_timings)
    corr_img, report = correct_slice_timing_with_report(img, meta)
    m = _metrics(np.asarray(corr_img.dataobj, dtype=np.float64), np.asarray(img.dataobj, dtype=np.float64), gt, tr)

    assert report["status"] == "APPLIED"
    assert report["slice_order_interpretation"] == "multiband_mb3"
    assert m["rmse_after"] < m["rmse_before"] * 0.05
    assert m["pearson_r_mean"] > 0.9999


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
    assert m["rmse_after"] < m["rmse_before"] * 0.05
    assert m["rmse_after"] < 0.02
    assert m["pearson_r_mean"] > 0.9999

    # Sanity check: using unreversed positive direction on the same data must degrade accuracy dramatically
    wrong_meta = dict(meta)
    wrong_meta["SliceEncodingDirection"] = direction[0]
    wrong_img, _ = correct_slice_timing_with_report(img, wrong_meta)
    m_wrong = _metrics(np.asarray(wrong_img.dataobj, dtype=np.float64), np.asarray(img.dataobj, dtype=np.float64), gt, tr)
    assert m_wrong["rmse_after"] > m["rmse_after"] * 20.0


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
    assert m["rmse_after"] < m["rmse_before"] * 0.05
    assert m["rmse_after"] < 0.02
    assert m["pearson_r_mean"] > 0.9999


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
    assert m["rmse_after"] < m["rmse_before"] * 0.05
    assert m["rmse_after"] < 0.02
    assert m["pearson_r_mean"] > 0.9999


def test_7_slice_axis_k_and_absent_direction_defaults_to_k_with_warning():
    """Explicit 'k' works on axis 2, and absent SliceEncodingDirection defaults to 'k' with warning."""
    tr = 2.0
    timings = np.linspace(0.0, tr, 15, endpoint=False).tolist()
    img, gt, meta_k, _ = _make_analytical_bold((6, 6, 15), n_vols=100, tr=tr, slice_timing=timings, direction="k")
    corr_k, rep_k = correct_slice_timing_with_report(img, meta_k)
    assert rep_k["slice_axis"] == 2
    assert rep_k["slice_encoding_direction_assumed"] is False

    meta_absent = {k: v for k, v in meta_k.items() if k != "SliceEncodingDirection"}
    corr_def, rep_def = correct_slice_timing_with_report(img, meta_absent)
    m_def = _metrics(np.asarray(corr_def.dataobj, dtype=np.float64), np.asarray(img.dataobj, dtype=np.float64), gt, tr)

    assert rep_def["slice_axis"] == 2
    assert rep_def["slice_encoding_direction"] == "k"
    assert rep_def["slice_encoding_direction_assumed"] is True
    assert any("BIDS specification default 'k'" in w for w in rep_def["warnings"])
    np.testing.assert_array_equal(np.asarray(corr_def.dataobj), np.asarray(corr_k.dataobj))
    assert m_def["rmse_interior_after"] < 0.005
    assert m_def["rmse_after"] < m_def["rmse_before"] * 0.05
    assert m_def["rmse_after"] < 0.02


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
    """Unsupported SliceEncodingDirection values fail closed (note None triggers default 'k')."""
    img, _, meta, _ = _make_analytical_bold((4, 4, 8), n_vols=40, tr=2.0)
    if bad_dir is None:
        meta["SliceEncodingDirection"] = None
        rep = validate_slice_timing_metadata(img, meta)
        assert rep.slice_encoding_direction == "k"
        assert rep.slice_encoding_direction_assumed is True
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
        ("cubic_spline", 0.005, 0.02),
        ("quintic_bspline", 0.001, 0.005),
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
    assert m_c["rmse_after"] < 0.02
    assert m_c["rmse_after"] < m_c["rmse_before"] * 0.05

    with pytest.raises(InvalidSliceTimingError, match="reference_time"):
        correct_slice_timing_with_report(img_c, meta_c, reference_time=tr)


def test_20_boundary_handling_no_circular_wraparound_leakage():
    """Start (t=0) and end (t=T-1) remain finite and controlled without FFT wrap-around step jumps."""
    tr = 2.0
    # Add linear drift so start (t=0) and end (t=T-1) differ by ~20 intensity units!
    timings = np.linspace(0.0, tr, 13, endpoint=False).tolist()
    img, gt, meta, _ = _make_analytical_bold(
        (5, 5, 13), n_vols=90, tr=tr, slice_timing=timings, linear_drift=0.12
    )
    corr_img, _ = correct_slice_timing_with_report(img, meta, interpolation_method="cubic_spline")
    raw = np.asarray(img.dataobj, dtype=np.float64)
    est = np.asarray(corr_img.dataobj, dtype=np.float64)

    assert np.isfinite(est).all()
    # Check t=0 and t=T-1 specifically: must improve markedly over raw, proving no circular wrap-around
    err_start_est = float(np.sqrt(np.mean((est[..., 0] - gt[..., 0]) ** 2)))
    err_end_est = float(np.sqrt(np.mean((est[..., -1] - gt[..., -1]) ** 2)))
    err_start_raw = float(np.sqrt(np.mean((raw[..., 0] - gt[..., 0]) ** 2)))
    err_end_raw = float(np.sqrt(np.mean((raw[..., -1] - gt[..., -1]) ** 2)))
    assert err_start_est < err_start_raw * 0.2
    assert err_end_est < err_end_raw * 0.2


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
# 26–27. Security, Real-Data Lock & 20-Stage Pipeline Integration
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


def test_run_poc_v2_pipeline_stage2_stc_applied_skipped_and_double_blocked(tmp_path: Path):
    """End-to-end 20-stage runner tests: STC APPLIED, SKIPPED_MISSING_METADATA, and double-STC fail-closed."""
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
        assert res_ok["provenance"]["pipeline_stages_total"] == 20
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

        # Strict mode must reject missing SliceTiming at Stage 2
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

        # Case 3: Double slice-timing correction blocked in Stage 2
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
        assert json.loads(fail_reports[0].read_text(encoding="utf-8"))["stage"] == "Stage 2: Slice-Timing Correction"
