"""Metadata-driven slice-timing correction (STC) for 4D resting-state fMRI BOLD series.

Implements Step 4 (DEC-STC-01) using BIDS-compliant timing metadata:
- Validates RepetitionTime, SliceTiming, and SliceEncodingDirection (i, i-, j, j-, k, k-).
- Rejects missing/invalid timing metadata without inventing timings from TR alone.
- Detects simultaneous slice acquisition and skips unnecessary temporal resampling.
- Prevents double slice-timing correction (SliceTimingCorrected=True guard).
- Uses SciPy C2 cubic spline interpolation (not-a-knot boundary condition) by default
  when AFNI 3dTshift is not installed in the container/host environment, with optional
  quintic B-spline, odd-mirror Fourier phase-shift, and linear backends.
- Preserves spatial dimensions, spatial 4x4 affine, finite numerical stability,
  and explicit NIfTI temporal TR zoom while documenting the +reference_time (half-TR)
  temporal-origin shift relative to volume onset.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import scipy
from scipy.fft import irfft, rfft, rfftfreq
from scipy.interpolate import CubicSpline, make_interp_spline

DEFAULT_STC_METHOD = "cubic_spline"
SUPPORTED_STC_METHODS = (
    "cubic_spline",
    "quintic_bspline",
    "fourier",
    "linear",
)
SLICE_ENCODING_DIRECTION_MAP: Dict[str, Tuple[int, bool]] = {
    "i": (0, False),
    "i-": (0, True),
    "j": (1, False),
    "j-": (1, True),
    "k": (2, False),
    "k-": (2, True),
}
SIMULTANEOUS_TOL_SEC = 1e-9
TIME_UNIT_TO_SECONDS = {
    "sec": 1.0,
    "msec": 1e-3,
    "usec": 1e-6,
}


class SliceTimingValidationError(ValueError):
    """Base error raised when slice-timing metadata or BOLD geometry fails validation."""

    status: str = "REJECTED"


class MissingSliceTimingError(SliceTimingValidationError):
    """Raised when SliceTiming metadata is absent and missing metadata cannot be skipped."""


class InvalidSliceTimingError(SliceTimingValidationError):
    """Raised when RepetitionTime, SliceTiming, SliceEncodingDirection, or image geometry is invalid."""


class DoubleSliceTimingCorrectionError(SliceTimingValidationError):
    """Raised when slice-timing correction is attempted on an already slice-time-corrected volume."""


@dataclass(frozen=True)
class SliceTimingValidationReport:
    """Structured validation result for BIDS BOLD slice-timing metadata."""

    status: str  # VALIDATED | SKIPPED_MISSING_METADATA | SKIPPED_SIMULTANEOUS_ACQUISITION
    tr: float
    reference_time: Optional[float]
    slice_axis: int
    slice_encoding_direction: str
    slice_encoding_direction_reversed: bool
    slice_encoding_direction_assumed: bool
    num_slices: int
    num_volumes: int
    original_slice_timing: Optional[Tuple[float, ...]]
    voxel_slice_times: Optional[Tuple[float, ...]]
    slice_order_interpretation: str
    relative_correction_necessary: bool
    header_tr_seconds: Optional[float]
    header_time_unit: str
    warnings: List[str] = field(default_factory=list)
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "tr": float(self.tr),
            "reference_time": float(self.reference_time) if self.reference_time is not None else None,
            "slice_axis": int(self.slice_axis),
            "slice_encoding_direction": self.slice_encoding_direction,
            "slice_encoding_direction_reversed": bool(self.slice_encoding_direction_reversed),
            "slice_encoding_direction_assumed": bool(self.slice_encoding_direction_assumed),
            "num_slices": int(self.num_slices),
            "num_volumes": int(self.num_volumes),
            "original_slice_timing": list(self.original_slice_timing)
            if self.original_slice_timing is not None
            else None,
            "voxel_slice_times": list(self.voxel_slice_times)
            if self.voxel_slice_times is not None
            else None,
            "slice_order_interpretation": self.slice_order_interpretation,
            "relative_correction_necessary": bool(self.relative_correction_necessary),
            "header_tr_seconds": float(self.header_tr_seconds)
            if self.header_tr_seconds is not None
            else None,
            "header_time_unit": self.header_time_unit,
            "warnings": list(self.warnings),
            "reason": self.reason,
        }


def detect_stc_software() -> Dict[str, Any]:
    """Inspect external AFNI/FSL tools and Python signal-processing backend versions."""
    afni_bin = shutil.which("3dTshift")
    fsl_bin = shutil.which("slicetimer")
    return {
        "afni_3dtshift_available": afni_bin is not None,
        "afni_3dtshift_path": afni_bin,
        "fsl_slicetimer_available": fsl_bin is not None,
        "scipy_version": str(scipy.__version__),
        "nibabel_version": str(nib.__version__),
        "numpy_version": str(np.__version__),
        "default_backend": "scipy.interpolate.CubicSpline",
    }


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute hex SHA-256 checksum of a file on disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_real_numeric(value: Any) -> bool:
    """Return True if value is a finite non-boolean real scalar."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(float(value))


def _resolve_interpolation_method(interpolation_method: Optional[str]) -> Tuple[str, str]:
    """Normalize interpolation method alias and return (canonical_name, description)."""
    if interpolation_method is None:
        canonical = DEFAULT_STC_METHOD
    else:
        raw = str(interpolation_method).strip().lower()
        alias_map = {
            "cubic": "cubic_spline",
            "cubic_spline": "cubic_spline",
            "cspline": "cubic_spline",
            "quintic": "quintic_bspline",
            "quintic_bspline": "quintic_bspline",
            "bspline5": "quintic_bspline",
            "fourier": "fourier",
            "sinc": "fourier",
            "fft": "fourier",
            "linear": "linear",
        }
        canonical = alias_map.get(raw, raw)

    descriptions = {
        "cubic_spline": "scipy.interpolate.CubicSpline(bc_type='not-a-knot', extrapolate=True)",
        "quintic_bspline": "scipy.interpolate.make_interp_spline(k=5)",
        "fourier": "odd-mirror C1-extended scipy.fft.rfft/irfft phase shift",
        "linear": "numpy.interp piecewise linear interpolation",
    }
    if canonical not in SUPPORTED_STC_METHODS:
        raise ValueError(
            f"Unsupported slice-timing interpolation_method '{interpolation_method}'. "
            f"Supported methods: {', '.join(SUPPORTED_STC_METHODS)}."
        )
    return canonical, descriptions[canonical]


def _classify_slice_order(timings: np.ndarray) -> str:
    """Classify validated timing entries into a descriptive acquisition pattern.

    This helper is strictly descriptive for provenance and reporting; it is never
    used to infer or invent missing SliceTiming values.
    """
    n_slices = int(timings.shape[0])
    if n_slices <= 1 or float(np.ptp(timings)) <= SIMULTANEOUS_TOL_SEC:
        return "simultaneous"

    rounded = np.round(timings, 6)
    unique_vals, counts = np.unique(rounded, return_counts=True)
    max_multiband = int(np.max(counts))
    if len(unique_vals) < n_slices and max_multiband >= 2:
        return f"multiband_mb{max_multiband}"

    diffs = np.diff(timings)
    if np.all(diffs > 0.0):
        return "ascending_sequential"
    if np.all(diffs < 0.0):
        return "descending_sequential"

    evens = timings[0::2]
    odds = timings[1::2]
    if len(evens) >= 2 and len(odds) >= 2:
        evens_asc = bool(np.all(np.diff(evens) > 0.0))
        odds_asc = bool(np.all(np.diff(odds) > 0.0))
        evens_desc = bool(np.all(np.diff(evens) < 0.0))
        odds_desc = bool(np.all(np.diff(odds) < 0.0))
        if evens_asc and odds_asc and (np.max(evens) < np.min(odds) or np.max(odds) < np.min(evens)):
            return "interleaved_ascending"
        if evens_desc and odds_desc and (np.min(evens) > np.max(odds) or np.min(odds) > np.max(evens)):
            return "interleaved_descending"
        if (evens_asc and odds_asc) or (evens_desc and odds_desc):
            return "interleaved"

    return "custom"


def _enforce_no_real_data_path(candidate: Union[str, Path]) -> None:
    """Fail closed if a caller points to prohibited clinical/raw MRI directories or symlinks."""
    raw_path = Path(candidate)
    if raw_path.is_symlink():
        raise PermissionError(
            f"SAFETY VIOLATION: Symlink path rejected at '{candidate}'. Symlinks are prohibited."
        )
    paths_to_check = [str(raw_path).lower()]
    try:
        paths_to_check.append(str(raw_path.resolve()).lower())
    except OSError:
        pass
    blocked_markers = (
        "neuroadhd_converted",
        "down/outputs/adhd200",
        "sub-2344857",
        "sub-01/func",
        "sub-01/anat",
        "/sub-01/",
        "sub-01_",
        "peking",
    )
    for text in paths_to_check:
        for marker in blocked_markers:
            if marker in text:
                raise PermissionError(
                    f"SAFETY VIOLATION: Access to protected clinical/raw MRI path '{candidate}' is prohibited "
                    "under execution_policy=STRICTLY_SYNTHETIC_ONLY."
                )


def _load_bold_metadata(bold_metadata: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
    """Load and type-check BIDS BOLD metadata dictionary or JSON sidecar path."""
    if isinstance(bold_metadata, (str, Path)):
        meta_path = Path(bold_metadata)
        _enforce_no_real_data_path(meta_path)
        if not meta_path.is_file():
            raise InvalidSliceTimingError(f"BOLD JSON metadata file not found: {meta_path}")
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidSliceTimingError(f"Invalid JSON in BOLD metadata '{meta_path}': {exc}") from exc
    else:
        loaded = bold_metadata

    if not isinstance(loaded, dict):
        raise InvalidSliceTimingError("bold_metadata must be a JSON dictionary or path to a JSON sidecar.")
    return loaded


def validate_slice_timing_metadata(
    bold_image: Union[nib.Nifti1Image, Sequence[int], np.ndarray],
    bold_metadata: Union[Dict[str, Any], str, Path],
    reference_time: Optional[float] = None,
    expected_tr: Optional[float] = None,
    allow_missing_timing: bool = False,
    strict_header_tr_match: bool = False,
) -> SliceTimingValidationReport:
    """Validate RepetitionTime, SliceTiming, SliceEncodingDirection, and NIfTI geometry.

    Parameters
    ----------
    bold_image : nib.Nifti1Image | Sequence[int] | np.ndarray
        4D NIfTI BOLD image (or its 4D shape).
    bold_metadata : dict | str | Path
        BIDS JSON sidecar dictionary (or path to JSON sidecar).
    reference_time : float, optional
        Target temporal reference offset within TR in seconds. Defaults to TR / 2.0.
    expected_tr : float, optional
        Optional external TR expectation (in seconds) that RepetitionTime must match.
    allow_missing_timing : bool
        When False (default), missing SliceTiming raises MissingSliceTimingError.
        When True, missing SliceTiming returns a SKIPPED_MISSING_METADATA report while
        preserving strict rejection for malformed or out-of-range SliceTiming values.
    strict_header_tr_match : bool
        When True, require NIfTI header pixdim[4] > 0 to match JSON RepetitionTime
        even if temporal units are labeled 'unknown'.

    Returns
    -------
    SliceTimingValidationReport
    """
    metadata = _load_bold_metadata(bold_metadata)

    # 1. Prevent double slice-timing correction
    stc_flag = metadata.get("SliceTimingCorrected")
    if isinstance(stc_flag, bool) and stc_flag is True:
        raise DoubleSliceTimingCorrectionError(
            "Input BOLD metadata indicates SliceTimingCorrected=True; "
            "double slice-timing correction is prohibited to prevent temporal shifting errors."
        )
    if isinstance(stc_flag, str) and stc_flag.strip().lower() == "true":
        raise DoubleSliceTimingCorrectionError(
            "Input BOLD metadata indicates SliceTimingCorrected='true'; "
            "double slice-timing correction is prohibited."
        )

    # 2. Extract and validate 4D image geometry & NIfTI temporal header
    header_tr_sec: Optional[float] = None
    header_time_unit = "unknown"
    if isinstance(bold_image, nib.Nifti1Image):
        shape = tuple(int(dim) for dim in bold_image.shape)
        if len(shape) != 4:
            raise InvalidSliceTimingError(
                f"Slice-timing correction requires a 4D BOLD NIfTI image, got {len(shape)}D shape {shape}."
            )
        affine = np.asarray(bold_image.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise InvalidSliceTimingError("BOLD NIfTI image contains a non-finite or malformed affine matrix.")
        if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12:
            raise InvalidSliceTimingError("BOLD NIfTI spatial affine is singular (non-invertible).")

        zooms = tuple(float(z) for z in bold_image.header.get_zooms())
        _, t_unit = bold_image.header.get_xyzt_units()
        header_time_unit = str(t_unit or "unknown")
        if len(zooms) >= 4:
            raw_hdr_tr = zooms[3]
            if not math.isfinite(raw_hdr_tr) or raw_hdr_tr < 0.0:
                raise InvalidSliceTimingError(
                    f"NIfTI header temporal zoom pixdim[4]={raw_hdr_tr} is invalid."
                )
            if header_time_unit in TIME_UNIT_TO_SECONDS:
                header_tr_sec = raw_hdr_tr * TIME_UNIT_TO_SECONDS[header_time_unit]
            elif strict_header_tr_match or (raw_hdr_tr > 0.0 and not math.isclose(raw_hdr_tr, 1.0, abs_tol=1e-6)):
                header_tr_sec = raw_hdr_tr
    elif isinstance(bold_image, np.ndarray):
        shape = tuple(int(dim) for dim in bold_image.shape)
    else:
        shape = tuple(int(dim) for dim in bold_image)

    if len(shape) != 4:
        raise InvalidSliceTimingError(
            f"Slice-timing correction requires 4D BOLD shape (X, Y, Z, T), got {len(shape)}D shape {shape}."
        )
    if any(dim <= 0 for dim in shape):
        raise InvalidSliceTimingError(f"BOLD shape dimensions must all be positive, got {shape}.")

    # 3. Validate RepetitionTime (A)
    if "RepetitionTime" not in metadata or metadata["RepetitionTime"] is None:
        raise InvalidSliceTimingError("BOLD metadata missing required 'RepetitionTime' field.")
    raw_tr = metadata["RepetitionTime"]
    if not _is_real_numeric(raw_tr):
        raise InvalidSliceTimingError(
            f"Invalid 'RepetitionTime': expected finite numeric scalar, got {raw_tr!r}."
        )
    tr = float(raw_tr)
    if tr <= 0.0:
        raise InvalidSliceTimingError(
            f"Invalid 'RepetitionTime': must be strictly greater than zero seconds, got {tr}."
        )

    if expected_tr is not None:
        if not _is_real_numeric(expected_tr) or float(expected_tr) <= 0.0:
            raise InvalidSliceTimingError(f"Invalid expected_tr={expected_tr!r}.")
        if abs(tr - float(expected_tr)) > 1e-4:
            raise InvalidSliceTimingError(
                f"RepetitionTime ({tr}s) disagrees with expected acquisition TR ({float(expected_tr)}s)."
            )

    if header_tr_sec is not None and header_tr_sec > 0.0:
        # Enforce agreement when header temporal unit is explicitly configured (e.g., sec/msec/usec)
        # or non-default pixdim[4] was written alongside RepetitionTime.
        if abs(header_tr_sec - tr) > 1e-3:
            if header_time_unit in TIME_UNIT_TO_SECONDS or strict_header_tr_match:
                raise InvalidSliceTimingError(
                    f"RepetitionTime ({tr}s) disagrees with NIfTI header temporal zoom "
                    f"({header_tr_sec}s, units={header_time_unit})."
                )

    if "VolumeTiming" in metadata and metadata["VolumeTiming"] is not None:
        vol_timing = metadata["VolumeTiming"]
        if not isinstance(vol_timing, (list, tuple)) or len(vol_timing) < 2:
            raise InvalidSliceTimingError("VolumeTiming must be a numeric sequence with >= 2 entries when present.")
        if not all(_is_real_numeric(v) for v in vol_timing):
            raise InvalidSliceTimingError("VolumeTiming contains non-numeric or non-finite entries.")
        vt_arr = np.asarray([float(v) for v in vol_timing], dtype=np.float64)
        vt_diffs = np.diff(vt_arr)
        if np.any(vt_diffs <= 0.0) or not np.allclose(vt_diffs, tr, atol=1e-3, rtol=0.0):
            raise InvalidSliceTimingError(
                f"RepetitionTime ({tr}s) disagrees with VolumeTiming intervals."
            )

    # 4. Validate SliceEncodingDirection (C)
    warnings: List[str] = []
    if "SliceEncodingDirection" not in metadata or metadata["SliceEncodingDirection"] is None:
        # BIDS MRI specification defaults SliceEncodingDirection to "k" (positive 3rd spatial axis)
        direction = "k"
        direction_assumed = True
        warnings.append(
            "SliceEncodingDirection absent in metadata; applied BIDS specification default 'k' "
            "(positive index order along spatial axis 2)."
        )
    else:
        raw_dir = metadata["SliceEncodingDirection"]
        if not isinstance(raw_dir, str) or raw_dir not in SLICE_ENCODING_DIRECTION_MAP:
            raise InvalidSliceTimingError(
                f"Invalid 'SliceEncodingDirection': {raw_dir!r}. "
                f"Supported values: {', '.join(SLICE_ENCODING_DIRECTION_MAP.keys())}."
            )
        direction = raw_dir
        direction_assumed = False

    slice_axis, is_reversed = SLICE_ENCODING_DIRECTION_MAP[direction]
    num_slices = int(shape[slice_axis])
    num_volumes = int(shape[3])

    # 5. Validate reference_time (Phase D)
    if reference_time is None:
        ref_time = tr / 2.0
    else:
        if not _is_real_numeric(reference_time):
            raise InvalidSliceTimingError(
                f"Invalid reference_time={reference_time!r}; must be finite numeric seconds."
            )
        ref_time = float(reference_time)
    if not (0.0 <= ref_time < tr):
        raise InvalidSliceTimingError(
            f"Invalid reference_time={ref_time}s; must satisfy 0 <= reference_time < RepetitionTime ({tr}s)."
        )

    # 6. Validate SliceTiming (B & D)
    if "SliceTiming" not in metadata or metadata["SliceTiming"] is None:
        reason = (
            "Missing 'SliceTiming' in BOLD metadata; refusing to invent or estimate timings from TR alone. "
            "Relative slice-timing correction skipped and original BOLD series preserved."
        )
        if not allow_missing_timing:
            raise MissingSliceTimingError(reason)
        warnings.append(reason)
        return SliceTimingValidationReport(
            status="SKIPPED_MISSING_METADATA",
            tr=tr,
            reference_time=ref_time,
            slice_axis=slice_axis,
            slice_encoding_direction=direction,
            slice_encoding_direction_reversed=is_reversed,
            slice_encoding_direction_assumed=direction_assumed,
            num_slices=num_slices,
            num_volumes=num_volumes,
            original_slice_timing=None,
            voxel_slice_times=None,
            slice_order_interpretation="missing_metadata",
            relative_correction_necessary=False,
            header_tr_seconds=header_tr_sec,
            header_time_unit=header_time_unit,
            warnings=warnings,
            reason=reason,
        )

    raw_slice_timing = metadata["SliceTiming"]
    if isinstance(raw_slice_timing, (str, bytes, dict)) or not isinstance(
        raw_slice_timing, (list, tuple, np.ndarray)
    ):
        raise InvalidSliceTimingError(
            f"Invalid 'SliceTiming': expected 1D numeric array, got {type(raw_slice_timing).__name__}."
        )
    if isinstance(raw_slice_timing, np.ndarray) and raw_slice_timing.ndim != 1:
        raise InvalidSliceTimingError(
            f"Invalid 'SliceTiming': expected 1D array, got {raw_slice_timing.ndim}D."
        )
    timing_list = list(raw_slice_timing)
    if len(timing_list) == 0:
        raise InvalidSliceTimingError("Invalid 'SliceTiming': timing array must not be empty.")
    if len(timing_list) != num_slices:
        raise InvalidSliceTimingError(
            f"SliceTiming length ({len(timing_list)}) does not match number of slices ({num_slices}) "
            f"along NIfTI axis {slice_axis} for SliceEncodingDirection='{direction}'."
        )

    parsed_timings: List[float] = []
    for idx, entry in enumerate(timing_list):
        if not _is_real_numeric(entry):
            raise InvalidSliceTimingError(
                f"Invalid 'SliceTiming' value at index {idx}: {entry!r} is non-numeric, boolean, or non-finite."
            )
        t_val = float(entry)
        if t_val < 0.0 or t_val >= tr:
            raise InvalidSliceTimingError(
                f"Invalid 'SliceTiming' value at index {idx}: {t_val}s violates 0 <= timing < TR ({tr}s)."
            )
        parsed_timings.append(t_val)

    timing_arr = np.asarray(parsed_timings, dtype=np.float64)
    # Reverse index correspondence when SliceEncodingDirection ends in '-'
    voxel_times = timing_arr[::-1].copy() if is_reversed else timing_arr.copy()

    # Check simultaneous acquisition (all slices share identical acquisition time)
    is_simultaneous = (num_slices <= 1) or (float(np.ptp(voxel_times)) <= SIMULTANEOUS_TOL_SEC)
    order_interp = _classify_slice_order(timing_arr)

    if is_simultaneous:
        reason = (
            "All slices share identical acquisition time; relative inter-slice timing correction is "
            "unnecessary and temporal interpolation was skipped to avoid signal distortion."
        )
        warnings.append(reason)
        return SliceTimingValidationReport(
            status="SKIPPED_SIMULTANEOUS_ACQUISITION",
            tr=tr,
            reference_time=ref_time,
            slice_axis=slice_axis,
            slice_encoding_direction=direction,
            slice_encoding_direction_reversed=is_reversed,
            slice_encoding_direction_assumed=direction_assumed,
            num_slices=num_slices,
            num_volumes=num_volumes,
            original_slice_timing=tuple(parsed_timings),
            voxel_slice_times=tuple(float(v) for v in voxel_times),
            slice_order_interpretation="simultaneous",
            relative_correction_necessary=False,
            header_tr_seconds=header_tr_sec,
            header_time_unit=header_time_unit,
            warnings=warnings,
            reason=reason,
        )

    return SliceTimingValidationReport(
        status="VALIDATED",
        tr=tr,
        reference_time=ref_time,
        slice_axis=slice_axis,
        slice_encoding_direction=direction,
        slice_encoding_direction_reversed=is_reversed,
        slice_encoding_direction_assumed=direction_assumed,
        num_slices=num_slices,
        num_volumes=num_volumes,
        original_slice_timing=tuple(parsed_timings),
        voxel_slice_times=tuple(float(v) for v in voxel_times),
        slice_order_interpretation=order_interp,
        relative_correction_necessary=True,
        header_tr_seconds=header_tr_sec,
        header_time_unit=header_time_unit,
        warnings=warnings,
        reason=None,
    )


def _interpolate_slice_series(
    slice_y: np.ndarray,
    t_indices: np.ndarray,
    delta_samples: float,
    method: str,
) -> np.ndarray:
    """Shift a (N_vox, T) array of voxel time series by delta_samples TR units."""
    t_vol = slice_y.shape[1]
    query_indices = t_indices + delta_samples
    if method == "cubic_spline":
        spline = CubicSpline(t_indices, slice_y, axis=1, bc_type="not-a-knot", extrapolate=True)
        return spline(query_indices)
    if method == "quintic_bspline":
        spline = make_interp_spline(t_indices, slice_y, k=5, axis=1)
        return spline(query_indices)
    if method == "linear":
        # Linear interpolation clamps cleanly at boundary endpoints [0, T-1]
        out = np.empty_like(slice_y, dtype=np.float64)
        clamped = np.clip(query_indices, 0.0, float(t_vol - 1))
        for row_idx in range(slice_y.shape[0]):
            out[row_idx] = np.interp(clamped, t_indices, slice_y[row_idx])
        return out
    if method == "fourier":
        if t_vol < 3:
            # Degenerate T < 3 falls back to chord linear extrapolation
            y0 = slice_y[:, :1]
            slope = (slice_y[:, -1:] - y0) / max(1, t_vol - 1)
            return y0 + slope * (t_indices[None, :] + delta_samples)
        # Chord-detrended C1 odd-mirror periodic extension (length = 2 * (T - 1))
        # Ensures zero-value & zero-kink periodic boundary while keeping exact Nyquist spectrum
        y0 = slice_y[:, :1]
        slope = (slice_y[:, -1:] - y0) / float(t_vol - 1)
        chord = y0 + slope * t_indices[None, :]
        residual = slice_y - chord
        ext = np.concatenate([residual, -residual[:, -2:0:-1]], axis=1)
        ext_len = ext.shape[1]
        freqs = rfftfreq(ext_len, d=1.0)
        spectrum = rfft(ext, axis=1)
        phase_shift = np.exp(1j * 2.0 * np.pi * freqs * delta_samples)
        shifted_res = irfft(spectrum * phase_shift[None, :], n=ext_len, axis=1)[:, :t_vol]
        shifted_chord = y0 + slope * (t_indices[None, :] + delta_samples)
        return shifted_chord + shifted_res
    raise ValueError(f"Unsupported interpolation method: {method}")


def _build_output_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    source_header: nib.Nifti1Header,
    tr: float,
) -> nib.Nifti1Image:
    """Build output 4D NIfTI preserving spatial affine, dimensions, and TR temporal zoom."""
    out_header = source_header.copy()
    out_header.set_data_shape(data.shape)
    out_header.set_data_dtype(data.dtype)
    out_header.set_slope_inter(1.0, 0.0)
    spatial_zooms = tuple(float(z) for z in source_header.get_zooms()[:3])
    out_header.set_zooms((*spatial_zooms, float(tr)))
    spatial_unit, _ = source_header.get_xyzt_units()
    if spatial_unit == "unknown":
        spatial_unit = "mm"
    out_header.set_xyzt_units(spatial_unit, "sec")
    return nib.Nifti1Image(data, np.asarray(affine, dtype=np.float64), out_header)


def build_slice_timing_derivative_metadata(
    raw_metadata: Dict[str, Any],
    report_dict: Dict[str, Any],
    source_bold_ref: Optional[str] = None,
) -> Dict[str, Any]:
    """Create BIDS-derivative JSON metadata documenting STC and temporal-origin shift."""
    out = dict(raw_metadata)
    status = report_dict["status"]
    tr = float(report_dict["tr"])
    ref_time = report_dict.get("reference_time")
    is_applied = status == "APPLIED"

    out["RepetitionTime"] = tr
    out["SliceTimingCorrected"] = bool(is_applied)
    out["SliceTimingCorrectionStatus"] = status
    out["RelativeSliceTimingCorrectionNecessary"] = bool(
        report_dict.get("relative_correction_necessary", is_applied)
    )
    out["SliceEncodingDirection"] = report_dict.get("slice_encoding_direction")
    out["SliceEncodingDirectionAssumed"] = bool(report_dict.get("slice_encoding_direction_assumed", False))
    out["SliceAcquisitionOrder"] = report_dict.get("slice_order_interpretation")
    if report_dict.get("original_slice_timing") is not None:
        out["OriginalSliceTiming"] = list(report_dict["original_slice_timing"])

    if is_applied and ref_time is not None:
        out["SliceTimingReferenceTime"] = float(ref_time)
        out["StartTime"] = float(ref_time)
        out["TemporalOriginShiftSeconds"] = float(ref_time)
        out["TemporalOriginNote"] = (
            f"All slices aligned to reference_time={float(ref_time):.6f}s within TR={tr:.6f}s "
            f"(+{float(ref_time):.6f}s offset relative to volume onset). "
            "Downstream onset modeling must account for StartTime without re-shifting."
        )
    else:
        out["StartTime"] = 0.0
        out["TemporalOriginShiftSeconds"] = 0.0
        out["TemporalOriginNote"] = (
            f"No temporal interpolation applied (status={status}); original temporal origin preserved."
        )

    out["InterpolationMethod"] = report_dict.get("interpolation_method")
    out["SoftwareName"] = report_dict.get("software")
    out["SoftwareVersion"] = report_dict.get("software_version")
    out["ProcessingStatus"] = status
    out["NumSlices"] = int(report_dict["num_slices"])
    out["NumVolumes"] = int(report_dict["num_volumes"])
    out["NumCorrectedSlices"] = int(report_dict.get("num_corrected_slices", 0))
    if source_bold_ref:
        out["Sources"] = [source_bold_ref]
    if report_dict.get("warnings"):
        out["Warnings"] = list(report_dict["warnings"])
    return out


def correct_slice_timing_with_report(
    bold_image: Union[nib.Nifti1Image, str, Path],
    bold_metadata: Union[Dict[str, Any], str, Path],
    reference_time: Optional[float] = None,
    interpolation_method: Optional[str] = None,
    *,
    allow_missing_timing: bool = False,
    expected_tr: Optional[float] = None,
    strict_header_tr_match: bool = False,
) -> Tuple[nib.Nifti1Image, Dict[str, Any]]:
    """Run metadata-driven STC and return (corrected_nifti, detailed_report_dict)."""
    t0 = time.perf_counter()
    software_info = detect_stc_software()
    canonical_method, method_desc = _resolve_interpolation_method(interpolation_method)

    input_path: Optional[Path] = None
    if isinstance(bold_image, (str, Path)):
        input_path = Path(bold_image)
        _enforce_no_real_data_path(input_path)
        if not input_path.is_file():
            raise InvalidSliceTimingError(f"BOLD NIfTI file not found: {input_path}")
        loaded_img = nib.load(str(input_path))
    elif isinstance(bold_image, nib.Nifti1Image):
        loaded_img = bold_image
    else:
        raise InvalidSliceTimingError(
            f"Expected nibabel.Nifti1Image or file path for bold_image, got {type(bold_image).__name__}."
        )

    raw_meta = _load_bold_metadata(bold_metadata)
    val_report = validate_slice_timing_metadata(
        bold_image=loaded_img,
        bold_metadata=raw_meta,
        reference_time=reference_time,
        expected_tr=expected_tr,
        allow_missing_timing=allow_missing_timing,
        strict_header_tr_match=strict_header_tr_match,
    )

    raw_dtype = np.dtype(loaded_img.get_data_dtype())
    if not np.issubdtype(raw_dtype, np.number) or np.issubdtype(raw_dtype, np.complexfloating):
        raise InvalidSliceTimingError(f"BOLD data dtype {raw_dtype} is not a real numeric type.")
    data_arr = np.asarray(loaded_img.get_fdata(dtype=np.float64), dtype=np.float64)
    if not np.isfinite(data_arr).all():
        n_bad = int(np.count_nonzero(~np.isfinite(data_arr)))
        raise InvalidSliceTimingError(
            f"BOLD NIfTI data contains {n_bad} non-finite values (NaN or Inf)."
        )

    t_vol = int(data_arr.shape[3])
    min_vols = 6 if canonical_method == "quintic_bspline" else (4 if canonical_method == "cubic_spline" else 2)
    if t_vol < min_vols:
        raise InvalidSliceTimingError(
            f"Insufficient BOLD timepoints ({t_vol}) for {canonical_method} STC (minimum {min_vols} required)."
        )

    out_dtype = np.float64 if raw_dtype == np.float64 else np.float32

    # Early exit without interpolation when missing-metadata skip or simultaneous acquisition
    if val_report.status != "VALIDATED":
        duration_sec = float(time.perf_counter() - t0)
        passthrough_img = _build_output_nifti(
            data=np.array(data_arr, dtype=out_dtype, copy=True),
            affine=loaded_img.affine,
            source_header=loaded_img.header,
            tr=val_report.tr,
        )
        report_dict = {
            **val_report.to_dict(),
            "status": val_report.status,
            "corrected": False,
            "num_corrected_slices": 0,
            "num_zero_shift_slices": val_report.num_slices,
            "interpolation_method": "none",
            "interpolation_description": "No temporal interpolation performed.",
            "software": "neuroadhd.preprocessing.slice_timing",
            "software_version": software_info["scipy_version"],
            "afni_3dtshift_available": software_info["afni_3dtshift_available"],
            "temporal_origin_shift_seconds": 0.0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration_sec,
            "boundary_handling": "N/A (interpolation skipped)",
            "finite_output_count": int(passthrough_img.dataobj.size if hasattr(passthrough_img.dataobj, "size") else data_arr.size),
        }
        return passthrough_img, report_dict

    assert val_report.voxel_slice_times is not None and val_report.reference_time is not None
    tr = val_report.tr
    ref_time = val_report.reference_time
    voxel_times = val_report.voxel_slice_times
    slice_axis = val_report.slice_axis
    n_slices = val_report.num_slices

    corrected_data = np.empty(data_arr.shape, dtype=out_dtype)
    t_indices = np.arange(t_vol, dtype=np.float64)
    n_corrected = 0
    n_zero_shift = 0
    max_abs_shift_sec = 0.0

    for slice_idx in range(n_slices):
        slice_t = float(voxel_times[slice_idx])
        shift_sec = ref_time - slice_t
        max_abs_shift_sec = max(max_abs_shift_sec, abs(shift_sec))

        indexer: List[Any] = [slice(None), slice(None), slice(None), slice(None)]
        indexer[slice_axis] = slice_idx
        slice_selector = tuple(indexer)

        slice_view = np.asarray(data_arr[slice_selector], dtype=np.float64)
        orig_slice_shape = slice_view.shape  # 3D: (D1, D2, T)

        if abs(shift_sec) <= SIMULTANEOUS_TOL_SEC:
            corrected_slice = slice_view
            n_zero_shift += 1
        else:
            delta_samples = shift_sec / tr
            flat_voxels = slice_view.reshape(-1, t_vol)
            shifted_flat = _interpolate_slice_series(
                slice_y=flat_voxels,
                t_indices=t_indices,
                delta_samples=delta_samples,
                method=canonical_method,
            )
            corrected_slice = shifted_flat.reshape(orig_slice_shape)
            n_corrected += 1

        if not np.isfinite(corrected_slice).all():
            raise RuntimeError(
                f"Non-finite values introduced during slice-timing correction of slice {slice_idx} "
                f"(axis={slice_axis})."
            )
        corrected_data[slice_selector] = corrected_slice.astype(out_dtype, copy=False)

    if not np.isfinite(corrected_data).all():
        raise RuntimeError("Slice-timing corrected BOLD output contains non-finite values.")

    out_img = _build_output_nifti(
        data=corrected_data,
        affine=loaded_img.affine,
        source_header=loaded_img.header,
        tr=tr,
    )
    duration_sec = float(time.perf_counter() - t0)

    report_dict = {
        **val_report.to_dict(),
        "status": "APPLIED",
        "corrected": True,
        "num_corrected_slices": n_corrected,
        "num_zero_shift_slices": n_zero_shift,
        "max_abs_temporal_shift_seconds": max_abs_shift_sec,
        "interpolation_method": canonical_method,
        "interpolation_description": method_desc,
        "software": f"neuroadhd.preprocessing.slice_timing ({canonical_method})",
        "software_version": f"scipy=={software_info['scipy_version']};nibabel=={software_info['nibabel_version']}",
        "afni_3dtshift_available": software_info["afni_3dtshift_available"],
        "temporal_origin_shift_seconds": float(ref_time),
        "half_tr_shift_applied": bool(math.isclose(ref_time, tr / 2.0, abs_tol=1e-6)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_sec,
        "boundary_handling": (
            "Non-periodic boundary extrapolation clamped to sub-TR index shift |delta| < 1 sample "
            "via not-a-knot cubic spline polynomial continuation (avoids FFT wrap-around artifacts)."
        ),
        "finite_output_count": int(corrected_data.size),
    }
    return out_img, report_dict


def correct_slice_timing(
    bold_image: Union[nib.Nifti1Image, str, Path],
    bold_metadata: Union[Dict[str, Any], str, Path],
    reference_time: Optional[float] = None,
    interpolation_method: Optional[str] = None,
    *,
    allow_missing_timing: bool = False,
    return_report: bool = False,
    expected_tr: Optional[float] = None,
    strict_header_tr_match: bool = False,
) -> Union[nib.Nifti1Image, Tuple[nib.Nifti1Image, Dict[str, Any]]]:
    """Apply metadata-driven slice-timing correction to a 4D BOLD image.

    Parameters
    ----------
    bold_image : nib.Nifti1Image | str | Path
        Input 4D resting-state BOLD NIfTI image or path.
    bold_metadata : dict | str | Path
        BIDS JSON sidecar dictionary or path containing RepetitionTime, SliceTiming,
        and optional SliceEncodingDirection.
    reference_time : float, optional
        Target temporal reference within TR in seconds (default: TR / 2.0).
    interpolation_method : str, optional
        Temporal interpolation algorithm ('cubic_spline' default, 'quintic_bspline',
        'fourier', or 'linear').
    allow_missing_timing : bool
        If True and SliceTiming is absent, skip correction and preserve input image
        with status 'SKIPPED_MISSING_METADATA'. Invalid SliceTiming values still fail closed.
    return_report : bool
        When True, return (corrected_image, report_dict); when False, return corrected_image.
    expected_tr : float, optional
        Optional external TR constraint that RepetitionTime must satisfy.
    strict_header_tr_match : bool
        Require NIfTI pixdim[4] > 0 to match JSON RepetitionTime strictly.

    Returns
    -------
    nib.Nifti1Image | Tuple[nib.Nifti1Image, Dict[str, Any]]
    """
    out_img, report_dict = correct_slice_timing_with_report(
        bold_image=bold_image,
        bold_metadata=bold_metadata,
        reference_time=reference_time,
        interpolation_method=interpolation_method,
        allow_missing_timing=allow_missing_timing,
        expected_tr=expected_tr,
        strict_header_tr_match=strict_header_tr_match,
    )
    if return_report:
        return out_img, report_dict
    return out_img


def write_slice_timing_derivative(
    bold_path: Union[str, Path],
    bold_json_path: Union[str, Path],
    derivative_dir: Union[str, Path],
    *,
    reference_time: Optional[float] = None,
    interpolation_method: Optional[str] = None,
    allow_missing_timing: bool = False,
    expected_tr: Optional[float] = None,
    output_prefix: Optional[str] = None,
) -> Tuple[nib.Nifti1Image, Dict[str, Any]]:
    """Correct a BIDS BOLD NIfTI on disk and write separate BIDS derivative files.

    Never modifies or overwrites the original BOLD NIfTI or JSON sidecar.
    Records input/output SHA-256 checksums, full parameters, software versions,
    and timestamp in the returned provenance report and derivative JSON sidecar.
    """
    _enforce_no_real_data_path(bold_path)
    _enforce_no_real_data_path(bold_json_path)
    bold_p = Path(bold_path).resolve()
    json_p = Path(bold_json_path).resolve()
    _enforce_no_real_data_path(bold_p)
    _enforce_no_real_data_path(json_p)

    if not bold_p.is_file():
        raise FileNotFoundError(f"Input BOLD file not found: {bold_p}")
    if not json_p.is_file():
        raise FileNotFoundError(f"Input BOLD JSON sidecar not found: {json_p}")

    input_bold_sha256 = compute_file_sha256(bold_p)
    input_json_sha256 = compute_file_sha256(json_p)
    raw_meta = _load_bold_metadata(json_p)

    corrected_img, report_dict = correct_slice_timing_with_report(
        bold_image=bold_p,
        bold_metadata=raw_meta,
        reference_time=reference_time,
        interpolation_method=interpolation_method,
        allow_missing_timing=allow_missing_timing,
        expected_tr=expected_tr,
    )

    out_dir = Path(derivative_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = bold_p.name
    if stem.lower().endswith(".nii.gz"):
        base = stem[:-7]
    elif stem.lower().endswith(".nii"):
        base = stem[:-4]
    else:
        base = bold_p.stem

    if base.endswith("_bold"):
        deriv_stem = f"{base[:-5]}_desc-stc_bold"
    else:
        deriv_stem = f"{base}_desc-stc_bold"

    if output_prefix:
        deriv_stem = f"{output_prefix}_desc-stc_bold"

    out_nii_path = (out_dir / f"{deriv_stem}.nii.gz").resolve()
    out_json_path = (out_dir / f"{deriv_stem}.json").resolve()
    report_json_path = (out_dir / f"{deriv_stem}_provenance.json").resolve()

    if out_nii_path == bold_p or out_json_path == json_p:
        raise RuntimeError(
            "Output integrity violation: derivative output paths must never overwrite original BOLD inputs."
        )

    deriv_meta = build_slice_timing_derivative_metadata(
        raw_metadata=raw_meta,
        report_dict=report_dict,
        source_bold_ref=f"bids:raw:{bold_p.name}",
    )

    nib.save(corrected_img, str(out_nii_path))
    with out_json_path.open("w", encoding="utf-8") as jf:
        json.dump(deriv_meta, jf, indent=2)

    output_bold_sha256 = compute_file_sha256(out_nii_path)
    output_json_sha256 = compute_file_sha256(out_json_path)

    provenance_record = {
        "input_bold_filename": bold_p.name,
        "input_json_filename": json_p.name,
        "input_bold_sha256": input_bold_sha256,
        "input_json_sha256": input_json_sha256,
        "output_derivative_nii": out_nii_path.name,
        "output_derivative_json": out_json_path.name,
        "output_bold_sha256": output_bold_sha256,
        "output_json_sha256": output_json_sha256,
        "input_metadata": raw_meta,
        **report_dict,
        "input_file_unmodified_verified": (compute_file_sha256(bold_p) == input_bold_sha256)
        and (compute_file_sha256(json_p) == input_json_sha256),
    }

    with report_json_path.open("w", encoding="utf-8") as pf:
        json.dump(provenance_record, pf, indent=2)

    provenance_record["derivative_nii_path"] = str(out_nii_path)
    provenance_record["derivative_json_path"] = str(out_json_path)
    provenance_record["provenance_json_path"] = str(report_json_path)

    return corrected_img, provenance_record
