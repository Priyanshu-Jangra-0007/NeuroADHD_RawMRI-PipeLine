"""B0 Susceptibility Distortion Correction (SDC) module for NeuroADHD RawMRI PoC V2 (DEC-SDC-02).

Provides BIDS fieldmap discovery and association (B0FieldSource / B0FieldIdentifier
and IntendedFor), strict phase-encoding and readout timing validation, odd-slice-safe
PEPOLAR TOPUP configuration and subprocess orchestration, explicit unit/metadata
checks for direct B0 fieldmaps and phase-difference scans, Jacobian folding QC,
analytical unwarping, and fail-closed provenance reporting.

Notice on External Tools & Scientific Honesty
---------------------------------------------
- When FSL ``topup`` (or ``prelude`` / ``fugue`` for phase-difference maps) is not
  installed, this module never fabricates a fieldmap or substitutes generic SyN/affine
  warps as synthetic susceptibility correction. Instead it returns
  ``SKIPPED_TOOL_UNAVAILABLE`` (or raises in strict mode), preserves the original
  geometry, and records exact planned parameters and missing executables.
- Fieldmap-less nonlinear registration (SyN-b0 fallback) is explicitly reported as
  ``SKIPPED_UNSUPPORTED_METHOD`` unless a formally validated estimator is configured.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# ---------------------------------------------------------------------------
# Status & Method Constants
# ---------------------------------------------------------------------------

SDC_STATUS_APPLIED_PEPOLAR_TOPUP = "APPLIED_PEPOLAR_TOPUP"
SDC_STATUS_APPLIED_DIRECT_FIELDMAP = "APPLIED_DIRECT_FIELDMAP"
SDC_STATUS_APPLIED_PHASEDIFF = "APPLIED_PHASEDIFF"
SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP = "SKIPPED_NO_ASSOCIATED_FIELDMAP"
SDC_STATUS_SKIPPED_INSUFFICIENT_METADATA = "SKIPPED_INSUFFICIENT_METADATA"
SDC_STATUS_SKIPPED_UNSUPPORTED_METHOD = "SKIPPED_UNSUPPORTED_METHOD"
SDC_STATUS_SKIPPED_TOOL_UNAVAILABLE = "SKIPPED_TOOL_UNAVAILABLE"
SDC_STATUS_FAILED_INVALID_INPUT = "FAILED_INVALID_INPUT"

APPLIED_SDC_STATUSES = frozenset({
    SDC_STATUS_APPLIED_PEPOLAR_TOPUP,
    SDC_STATUS_APPLIED_DIRECT_FIELDMAP,
    SDC_STATUS_APPLIED_PHASEDIFF,
})

SDC_METHOD_PEPOLAR_TOPUP = "PEPOLAR_TOPUP"
SDC_METHOD_DIRECT_FIELDMAP = "DIRECT_FIELDMAP"
SDC_METHOD_PHASE_DIFFERENCE = "PHASE_DIFFERENCE"
SDC_METHOD_FIELDMAP_LESS_SYN = "FIELDMAP_LESS_SYN"
SDC_METHOD_NONE = "NONE"

SUPPORTED_SDC_REQUESTED_METHODS = (
    "auto",
    "pepolar_topup",
    "direct_fieldmap",
    "phasediff",
    "fieldmap_less_syn",
    "none",
)

# BIDS PhaseEncodingDirection -> (axis_index, polarity, unit_vector_ijk)
PHASE_ENCODING_DIRECTION_MAP: Dict[str, Tuple[int, int, Tuple[int, int, int]]] = {
    "i":  (0, +1, (1, 0, 0)),
    "i-": (0, -1, (-1, 0, 0)),
    "j":  (1, +1, (0, 1, 0)),
    "j-": (1, -1, (0, -1, 0)),
    "k":  (2, +1, (0, 0, 1)),
    "k-": (2, -1, (0, 0, -1)),
}

# Safety blocklist patterns mirroring scripts/run_poc_v2.py to enforce fail-closed clinical isolation
_BLOCKED_CLINICAL_PATTERNS = (
    "down/outputs/adhd200",
    "sub-2344857",
    "peking",
    "kki",
    "nyu",
    "abide",
    "adhd-200",
    "adhd200",
    "neuroadhd_converted",
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SDCError(ValueError):
    """Base error for Susceptibility Distortion Correction validation and execution."""


class InvalidSDCInputError(SDCError):
    """Raised when fieldmap geometry, NIfTI integrity, readout timing, or paths are invalid."""


class MissingSDCMetadataError(SDCError):
    """Raised when mandatory BIDS SDC metadata (PE direction, readout time, units, TEs) is missing."""


class AmbiguousFieldmapAssociationError(SDCError):
    """Raised when BIDS fieldmap association rules match multiple conflicting fieldmaps."""


class SDCFoldingError(SDCError):
    """Raised when a displacement field violates diffeomorphic invertibility (Jacobian <= 0)."""


@dataclass
class SDCValidationResult:
    """Structured result emitted by ``run_susceptibility_distortion_correction``."""

    status: str
    method: str
    corrected: bool
    bold_pe_direction: Optional[str] = None
    bold_pe_axis: Optional[int] = None
    bold_pe_polarity: Optional[int] = None
    bold_total_readout_time_s: Optional[float] = None
    bold_readout_time_source: Optional[str] = None
    association_mechanism: str = "NONE"
    associated_fieldmap_files: List[Dict[str, Any]] = field(default_factory=list)
    unassociated_fmap_files: List[str] = field(default_factory=list)
    acqparams_lines: List[str] = field(default_factory=list)
    topup_config: Optional[Dict[str, Any]] = None
    planned_command: Optional[List[str]] = None
    executed_command: Optional[List[str]] = None
    tool_used: Optional[str] = None
    tool_version: Optional[str] = None
    missing_tools: List[str] = field(default_factory=list)
    software: Dict[str, Any] = field(default_factory=dict)
    displacement_stats: Optional[Dict[str, Any]] = None
    jacobian_qc: Optional[Dict[str, Any]] = None
    correction_qc: Optional[Dict[str, Any]] = None
    derivative_files: Dict[str, Optional[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    field_hz: Optional[np.ndarray] = None
    displacement_vox: Optional[np.ndarray] = None
    displacement_mm: Optional[np.ndarray] = None
    unwarped_mean_3d: Optional[np.ndarray] = None

    def to_provenance_dict(self) -> Dict[str, Any]:
        """Serialize structured SDC outcome for ``provenance.json`` (omits large voxel arrays)."""
        return build_sdc_provenance(self)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nifti_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _is_nifti(path: Path) -> bool:
    low = path.name.lower()
    return low.endswith(".nii.gz") or low.endswith(".nii")


def _check_safe_path(path: Path, label: str = "path") -> None:
    if path.is_symlink():
        raise InvalidSDCInputError(
            f"SAFETY VIOLATION: Symlink substitution detected at {label} '{path}'."
        )
    p_str = str(path).lower()
    if ".." in path.parts:
        raise InvalidSDCInputError(f"Unsafe path traversal in {label}: '{path}'")
    for kw in _BLOCKED_CLINICAL_PATTERNS:
        if kw in p_str:
            raise InvalidSDCInputError(
                f"SAFETY VIOLATION: Blocked clinical path pattern '{kw}' in {label} '{path}'."
            )


def _normalize_str_or_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _extract_bids_suffix(stem: str) -> str:
    parts = stem.split("_")
    return parts[-1] if parts else stem


def _bids_entities_without_suffix(stem: str) -> Dict[str, str]:
    entities: Dict[str, str] = {}
    parts = stem.split("_")
    for token in parts[:-1]:
        if "-" in token:
            k, v = token.split("-", 1)
            entities[k] = v
    return entities


def _load_json_safe(json_path: Path, label: str = "JSON sidecar") -> Dict[str, Any]:
    _check_safe_path(json_path, label=label)
    if not json_path.is_file():
        raise MissingSDCMetadataError(f"Missing {label}: {json_path}")
    try:
        with json_path.open("r", encoding="utf-8") as stream:
            obj = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidSDCInputError(f"Corrupted {label} '{json_path}': {exc}") from exc
    if not isinstance(obj, dict):
        raise InvalidSDCInputError(f"{label} '{json_path}' must contain a JSON object.")
    return obj


# ---------------------------------------------------------------------------
# 1. Software Availability & Capability Probing
# ---------------------------------------------------------------------------


def _probe_tool_version(binary_path: str, tool_name: str) -> Optional[str]:
    try:
        args = [binary_path, "--version"] if tool_name in ("topup", "applytopup", "fugue", "prelude") else [binary_path, "-version"]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
        combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        for line in combined.splitlines():
            clean = line.strip()
            if clean:
                return clean[:120]
    except Exception:
        pass
    return "installed"


def detect_sdc_software(topup_executable: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Probe available external neuroimaging binaries and Python packages without faking availability."""
    tools = ("topup", "applytopup", "prelude", "fugue", "fslmaths", "antsRegistration", "3dQwarp")
    resolved_bins: Dict[str, Optional[str]] = {}
    for tool in tools:
        if tool == "topup" and topup_executable is not None:
            resolved_bins[tool] = str(topup_executable) if Path(topup_executable).exists() else None
        else:
            resolved_bins[tool] = shutil.which(tool)

    py_pkgs: Dict[str, Optional[str]] = {}
    for pkg in ("numpy", "scipy", "nibabel", "dipy", "nilearn", "sdcflows"):
        try:
            mod = __import__(pkg)
            py_pkgs[pkg] = str(getattr(mod, "__version__", "installed"))
        except ImportError:
            py_pkgs[pkg] = None

    return {
        "topup_available": bool(resolved_bins["topup"]),
        "applytopup_available": bool(resolved_bins["applytopup"]),
        "prelude_available": bool(resolved_bins["prelude"]),
        "fugue_available": bool(resolved_bins["fugue"]),
        "fslmaths_available": bool(resolved_bins["fslmaths"]),
        "ants_registration_available": bool(resolved_bins["antsRegistration"]),
        "afni_3dqwarp_available": bool(resolved_bins["3dQwarp"]),
        "fsldir_configured": bool(os.environ.get("FSLDIR")),
        "topup_path": resolved_bins["topup"],
        "prelude_path": resolved_bins["prelude"],
        "fugue_path": resolved_bins["fugue"],
        "available_external_tools": [k for k, v in resolved_bins.items() if v],
        "python_packages": py_pkgs,
    }


# ---------------------------------------------------------------------------
# 2. BIDS Phase Encoding & TotalReadoutTime Validation
# ---------------------------------------------------------------------------


def parse_phase_encoding_direction(direction: Any) -> Dict[str, Any]:
    """Validate and resolve a BIDS ``PhaseEncodingDirection`` string into voxel axis and polarity.

    Only strict BIDS voxel-axis tokens (``i``, ``i-``, ``j``, ``j-``, ``k``, ``k-``)
    are permitted; ambiguous anatomical labels (``AP``, ``PA``, ``RL``) are rejected.
    """
    if direction is None:
        raise MissingSDCMetadataError("Missing required BIDS 'PhaseEncodingDirection' metadata.")
    if not isinstance(direction, str) or not direction.strip():
        raise InvalidSDCInputError(f"Invalid 'PhaseEncodingDirection': {direction!r}")
    token = direction.strip()
    if token not in PHASE_ENCODING_DIRECTION_MAP:
        raise InvalidSDCInputError(
            f"Unsupported PhaseEncodingDirection '{direction}'. Expected one of "
            f"{list(PHASE_ENCODING_DIRECTION_MAP.keys())} (BIDS voxel-index convention; "
            "anatomical strings like AP/PA are ambiguous without header orientation context)."
        )
    axis, polarity, vec = PHASE_ENCODING_DIRECTION_MAP[token]
    return {
        "code": token,
        "axis": int(axis),
        "polarity": int(polarity),
        "vector": tuple(int(x) for x in vec),
    }


def compute_total_readout_time(
    metadata: Dict[str, Any],
    spatial_shape: Optional[Sequence[int]] = None,
    pe_axis: Optional[int] = None,
) -> Tuple[float, str, List[str]]:
    """Resolve and validate ``TotalReadoutTime`` (in seconds) strictly from BIDS metadata.

    Rules
    -----
    1. If ``TotalReadoutTime`` is present: validate numeric, finite, ``0.0 < val < 1.0`` s
       (never TR).
    2. Else require BOTH ``EffectiveEchoSpacing`` (s) and integer ``ReconMatrixPE >= 2`` and
       compute ``TotalReadoutTime = EffectiveEchoSpacing * (ReconMatrixPE - 1)``.
    3. Never fall back to ``RepetitionTime`` or guessed defaults.
    """
    if not isinstance(metadata, dict):
        raise InvalidSDCInputError("Sidecar metadata must be a dictionary.")

    warnings: List[str] = []
    tr_val = metadata.get("RepetitionTime")
    trt_raw = metadata.get("TotalReadoutTime")

    if trt_raw is not None:
        if isinstance(trt_raw, bool) or not isinstance(trt_raw, (int, float)):
            raise InvalidSDCInputError(f"Invalid TotalReadoutTime type/value: {trt_raw!r}")
        trt = float(trt_raw)
        if not math.isfinite(trt) or trt <= 0.0:
            raise InvalidSDCInputError(
                f"TotalReadoutTime must be finite and strictly positive (> 0s), got {trt!r}."
            )
        if trt >= 1.0:
            raise InvalidSDCInputError(
                f"TotalReadoutTime={trt}s is non-physical (>= 1.0s; never substitute RepetitionTime)."
            )
        if tr_val is not None and not isinstance(tr_val, bool) and isinstance(tr_val, (int, float)):
            if math.isclose(trt, float(tr_val), rel_tol=1e-3, abs_tol=1e-4):
                raise InvalidSDCInputError(
                    f"TotalReadoutTime ({trt}s) equals RepetitionTime ({tr_val}s), indicating TR confusion."
                )
        return trt, "supplied", warnings

    # Derive from EffectiveEchoSpacing * (ReconMatrixPE - 1)
    ees_raw = metadata.get("EffectiveEchoSpacing")
    recon_raw = metadata.get("ReconMatrixPE")
    if ees_raw is None or recon_raw is None:
        missing_fields = [f for f, v in (("TotalReadoutTime", trt_raw), ("EffectiveEchoSpacing", ees_raw), ("ReconMatrixPE", recon_raw)) if v is None]
        raise MissingSDCMetadataError(
            "Insufficient EPI readout timing metadata: provide valid 'TotalReadoutTime' or BOTH "
            f"'EffectiveEchoSpacing' and 'ReconMatrixPE' (missing: {', '.join(missing_fields)}). "
            "Refusing to substitute RepetitionTime or arbitrary defaults."
        )

    if isinstance(ees_raw, bool) or not isinstance(ees_raw, (int, float)):
        raise InvalidSDCInputError(f"Invalid EffectiveEchoSpacing: {ees_raw!r}")
    ees = float(ees_raw)
    if not math.isfinite(ees) or ees <= 0.0 or ees >= 0.1:
        raise InvalidSDCInputError(
            f"EffectiveEchoSpacing must be finite in (0, 0.1) s, got {ees!r}."
        )

    if isinstance(recon_raw, bool) or not isinstance(recon_raw, (int, float)):
        raise InvalidSDCInputError(f"Invalid ReconMatrixPE: {recon_raw!r}")
    recon_float = float(recon_raw)
    if not math.isfinite(recon_float) or abs(recon_float - round(recon_float)) > 1e-6:
        raise InvalidSDCInputError(f"ReconMatrixPE must be an integer >= 2, got {recon_raw!r}.")
    recon = int(round(recon_float))
    if recon < 2:
        raise InvalidSDCInputError(f"ReconMatrixPE must be >= 2, got {recon}.")

    if spatial_shape is not None and pe_axis is not None and 0 <= pe_axis < len(spatial_shape):
        axis_dim = int(spatial_shape[pe_axis])
        if recon != axis_dim:
            warnings.append(
                f"ReconMatrixPE ({recon}) differs from spatial dimension along PE axis {pe_axis} ({axis_dim}); "
                "using metadata ReconMatrixPE for EffectiveEchoSpacing * (ReconMatrixPE - 1)."
            )

    trt_derived = float(ees * (recon - 1))
    if not math.isfinite(trt_derived) or trt_derived <= 0.0 or trt_derived >= 1.0:
        raise InvalidSDCInputError(
            f"Derived TotalReadoutTime={trt_derived}s from EES={ees} and ReconMatrixPE={recon} is invalid."
        )

    return trt_derived, "derived_effective_echo_spacing_recon_matrix_pe", warnings


# ---------------------------------------------------------------------------
# 3. BIDS Fieldmap Discovery & Association
# ---------------------------------------------------------------------------


def _find_session_and_bids_roots(bold_path: Path) -> Tuple[Path, Path, Path]:
    """Return ``(session_dir, subject_dir, bids_root)`` conservatively without climbing outside dataset."""
    resolved = bold_path.resolve()
    func_dir = resolved.parent
    parent = func_dir.parent if func_dir.name == "func" else func_dir
    if parent.name.startswith("ses-"):
        session_dir = parent
        subject_dir = parent.parent
    elif parent.name.startswith("sub-"):
        session_dir = parent
        subject_dir = parent
    else:
        session_dir = parent
        subject_dir = parent.parent if parent.parent != parent else parent

    bids_root = subject_dir.parent if subject_dir.name.startswith("sub-") else subject_dir
    return session_dir, subject_dir, bids_root


def _intended_for_matches_bold(
    entry: str,
    bold_path: Path,
    session_dir: Path,
    subject_dir: Path,
    bids_root: Path,
) -> bool:
    """Resolve BIDS ``IntendedFor`` strictly to ``bold_path`` (never by bare basename across subjects)."""
    raw = entry.strip()
    if not raw or "\x00" in raw:
        return False
    if raw.startswith("bids::"):
        raw = raw[6:]
    raw = raw.lstrip("/")
    pure = PurePosixPath(raw)
    if any(part in ("", ".", "..") for part in pure.parts):
        raise InvalidSDCInputError(f"Unsafe path component in BIDS IntendedFor entry: {entry!r}")

    target_resolved = bold_path.resolve()
    candidate_bases = [subject_dir, session_dir, bids_root]
    for base in candidate_bases:
        cand = (base / Path(*pure.parts))
        if cand.is_symlink():
            raise InvalidSDCInputError(f"Symlink detected via IntendedFor target '{cand}'.")
        try:
            if cand.exists() and cand.resolve() == target_resolved:
                return True
        except OSError:
            continue

    # Support normalized path suffix equivalence only when relative prefix matches subject/session hierarchy
    try:
        rel_sub = target_resolved.relative_to(subject_dir.resolve()).as_posix()
        if raw == rel_sub:
            return True
    except ValueError:
        pass
    try:
        rel_ses = target_resolved.relative_to(session_dir.resolve()).as_posix()
        if raw == rel_ses:
            return True
    except ValueError:
        pass
    try:
        rel_bids = target_resolved.relative_to(bids_root.resolve()).as_posix()
        if raw == rel_bids:
            return True
    except ValueError:
        pass

    return False


def discover_sdc_inputs(
    bold_path: Union[str, Path],
    bold_json_path: Optional[Union[str, Path]] = None,
    fmap_dir: Optional[Union[str, Path]] = None,
    session_dir: Optional[Union[str, Path]] = None,
    *,
    raise_on_ambiguous: bool = True,
    raise_on_invalid: bool = True,
) -> Dict[str, Any]:
    """Discover and validate BIDS fieldmaps strictly associated with a target BOLD scan.

    Priority & Matching Rules
    -------------------------
    1. ``B0FieldSource`` (on BOLD) matching ``B0FieldIdentifier`` (on fmap)
    2. ``IntendedFor`` (on fmap) resolving to exact canonical path of ``bold_path``
    3. Unassociated files in ``fmap/`` are never silently selected; ambiguous multi-run
       associations are rejected cleanly rather than guessing.
    """
    bold_p = Path(bold_path)
    _check_safe_path(bold_p, label="bold_path")
    if not bold_p.is_file():
        raise InvalidSDCInputError(f"Target BOLD NIfTI not found: '{bold_p}'")

    if bold_json_path is None:
        bold_json_p = bold_p.with_name(_nifti_stem(bold_p) + ".json")
    else:
        bold_json_p = Path(bold_json_path)
    _check_safe_path(bold_json_p, label="bold_json_path")

    bold_meta = _load_json_safe(bold_json_p, label="BOLD JSON sidecar") if bold_json_p.is_file() else {}

    inferred_session_dir, subject_dir, bids_root = _find_session_and_bids_roots(bold_p)
    active_session_dir = Path(session_dir) if session_dir is not None else inferred_session_dir
    active_fmap_dir = Path(fmap_dir) if fmap_dir is not None else (active_session_dir / "fmap")
    _check_safe_path(active_session_dir, label="session_dir")

    bold_b0_sources = set(_normalize_str_or_list(bold_meta.get("B0FieldSource")))

    discovery: Dict[str, Any] = {
        "associated": False,
        "association_mechanism": "NONE",
        "candidate_modality_group": "none",
        "fmap_dir": str(active_fmap_dir),
        "fmap_dir_exists": active_fmap_dir.is_dir(),
        "associated_files": [],
        "unassociated_fmap_files": [],
        "ambiguous": False,
        "ambiguity_reasons": [],
        "warnings": [],
        "error": None,
        "bold_b0_sources": sorted(bold_b0_sources),
    }

    if active_fmap_dir.is_symlink():
        msg = f"SAFETY VIOLATION: fmap directory '{active_fmap_dir}' is a symlink."
        if raise_on_invalid:
            raise InvalidSDCInputError(msg)
        discovery["error"] = msg
        return discovery

    if not active_fmap_dir.is_dir():
        return discovery

    nifti_candidates = sorted(
        [p for p in active_fmap_dir.iterdir() if _is_nifti(p)],
        key=lambda p: p.name,
    )

    # First check ALL files in fmap_dir for safety violations (symlinks / clinical paths)
    for item in sorted(active_fmap_dir.iterdir(), key=lambda p: p.name):
        _check_safe_path(item, label="fmap_entry")

    b0_matched: List[Dict[str, Any]] = []
    intended_matched: List[Dict[str, Any]] = []
    companion_mag_map: Dict[Path, Dict[str, Any]] = {}

    all_records: List[Dict[str, Any]] = []
    for nii_p in nifti_candidates:
        _check_safe_path(nii_p, label="fmap_nifti")
        stem = _nifti_stem(nii_p)
        suffix = _extract_bids_suffix(stem).lower()
        entities = _bids_entities_without_suffix(stem)
        json_p = nii_p.with_name(stem + ".json")

        meta: Dict[str, Any] = {}
        meta_error: Optional[str] = None
        has_json = json_p.exists() or json_p.is_symlink()
        if has_json:
            try:
                meta = _load_json_safe(json_p, label=f"fieldmap JSON '{json_p.name}'")
            except SDCError as exc:
                meta_error = str(exc)

        rec = {
            "nifti_path": nii_p,
            "json_path": json_p if json_p.is_file() else None,
            "filename": nii_p.name,
            "stem": stem,
            "suffix": suffix,
            "entities": entities,
            "metadata": meta,
            "metadata_error": meta_error,
            "b0_identifiers": set(_normalize_str_or_list(meta.get("B0FieldIdentifier"))),
            "intended_for_entries": _normalize_str_or_list(meta.get("IntendedFor")),
            "nifti_sha256": _sha256_file(nii_p),
            "json_sha256": _sha256_file(json_p) if json_p.is_file() else None,
        }
        all_records.append(rec)
        if suffix in ("magnitude", "magnitude1", "magnitude2"):
            companion_mag_map[nii_p] = rec

        # Evaluate B0FieldSource / B0FieldIdentifier match
        b0_hit = bool(bold_b0_sources and rec["b0_identifiers"] and (bold_b0_sources & rec["b0_identifiers"]))
        if b0_hit:
            if meta_error and raise_on_invalid:
                raise InvalidSDCInputError(meta_error)
            rec_copy = dict(rec)
            rec_copy["matched_via"] = "B0FieldSource_B0FieldIdentifier"
            b0_matched.append(rec_copy)
            continue

        # Evaluate IntendedFor match
        try:
            int_hit = any(
                _intended_for_matches_bold(item, bold_p, active_session_dir, subject_dir, bids_root)
                for item in rec["intended_for_entries"]
            )
        except InvalidSDCInputError:
            if raise_on_invalid:
                raise
            int_hit = False

        if int_hit:
            if meta_error and raise_on_invalid:
                raise InvalidSDCInputError(meta_error)
            rec_copy = dict(rec)
            rec_copy["matched_via"] = "IntendedFor"
            intended_matched.append(rec_copy)

    # Resolve B0FieldSource priority over IntendedFor, checking explicit B0FieldSource mismatches
    if bold_b0_sources:
        if b0_matched:
            selected = b0_matched
            mech = "B0FieldSource_B0FieldIdentifier"
            if intended_matched:
                b0_names = {r["filename"] for r in b0_matched}
                extra_intended = [r["filename"] for r in intended_matched if r["filename"] not in b0_names]
                if extra_intended:
                    discovery["ambiguity_reasons"].append(
                        f"Conflicting associations: BOLD B0FieldSource matched {[r['filename'] for r in b0_matched]} "
                        f"while IntendedFor separately targeted {extra_intended}."
                    )
        else:
            # Explicit B0FieldSource did not find its declared identifier
            if intended_matched:
                discovery["ambiguity_reasons"].append(
                    f"BOLD specifies B0FieldSource={sorted(bold_b0_sources)} with no matching B0FieldIdentifier, "
                    f"conflicting with IntendedFor candidates {[r['filename'] for r in intended_matched]}."
                )
                selected = intended_matched
                mech = "IntendedFor"
            else:
                selected = []
                mech = "NONE"
                discovery["warnings"].append(
                    f"BOLD specifies B0FieldSource={sorted(bold_b0_sources)} but no fieldmap declares matching B0FieldIdentifier."
                )
    else:
        selected = intended_matched
        mech = "IntendedFor" if selected else "NONE"

    # Separate fieldmap primary volumes vs companion magnitudes
    # Companion magnitudes sometimes don't have B0FieldIdentifier/IntendedFor; find them cleanly by stem if primary selected
    selected_names = {r["filename"] for r in selected}
    for rec in list(selected):
        if rec["suffix"] == "phasediff":
            prefix = rec["stem"][:-len("phasediff")]
            for mag_sfx in ("magnitude1", "magnitude2"):
                mag_cand = active_fmap_dir / f"{prefix}{mag_sfx}.nii.gz"
                if not mag_cand.is_file():
                    mag_cand = active_fmap_dir / f"{prefix}{mag_sfx}.nii"
                if mag_cand in companion_mag_map and mag_cand.name not in selected_names:
                    mag_rec = dict(companion_mag_map[mag_cand])
                    mag_rec["matched_via"] = "companion_stem"
                    selected.append(mag_rec)
                    selected_names.add(mag_cand.name)
        elif rec["suffix"] in ("phase1", "phase2"):
            prefix = rec["stem"][:-len(rec["suffix"])]
            for mag_sfx in ("magnitude1", "magnitude2"):
                mag_cand = active_fmap_dir / f"{prefix}{mag_sfx}.nii.gz"
                if not mag_cand.is_file():
                    mag_cand = active_fmap_dir / f"{prefix}{mag_sfx}.nii"
                if mag_cand in companion_mag_map and mag_cand.name not in selected_names:
                    mag_rec = dict(companion_mag_map[mag_cand])
                    mag_rec["matched_via"] = "companion_stem"
                    selected.append(mag_rec)
                    selected_names.add(mag_cand.name)
        elif rec["suffix"] == "fieldmap":
            prefix = rec["stem"][:-len("fieldmap")]
            mag_cand = active_fmap_dir / f"{prefix}magnitude.nii.gz"
            if not mag_cand.is_file():
                mag_cand = active_fmap_dir / f"{prefix}magnitude.nii"
            if mag_cand in companion_mag_map and mag_cand.name not in selected_names:
                mag_rec = dict(companion_mag_map[mag_cand])
                mag_rec["matched_via"] = "companion_stem"
                selected.append(mag_rec)
                selected_names.add(mag_cand.name)

    primary_recs = [r for r in selected if r["suffix"] in ("epi", "fieldmap", "phasediff", "phase1", "phase2")]
    unassociated = [r["filename"] for r in all_records if r["filename"] not in selected_names]
    discovery["unassociated_fmap_files"] = unassociated

    if not primary_recs:
        if selected:
            # Only magnitudes matched without a fieldmap volume -> ambiguous / incomplete
            discovery["ambiguity_reasons"].append(
                f"Only magnitude image(s) {[r['filename'] for r in selected]} matched without a primary B0/EPI fieldmap."
            )
        else:
            return discovery

    # Determine candidate modality group & detect ambiguities
    suffixes = {r["suffix"] for r in primary_recs}
    modality_groups: set[str] = set()
    if "epi" in suffixes:
        modality_groups.add("pepolar_epi")
    if "fieldmap" in suffixes:
        modality_groups.add("direct_fieldmap")
    if "phasediff" in suffixes:
        modality_groups.add("phasediff")
    if "phase1" in suffixes or "phase2" in suffixes:
        modality_groups.add("phase12")

    if len(modality_groups) > 1:
        discovery["ambiguity_reasons"].append(
            f"Multiple incompatible fieldmap modalities associated with BOLD: {sorted(modality_groups)} "
            f"({[r['filename'] for r in primary_recs]})."
        )

    if "direct_fieldmap" in modality_groups and sum(1 for r in primary_recs if r["suffix"] == "fieldmap") > 1:
        discovery["ambiguity_reasons"].append(
            f"Multiple direct *_fieldmap scans associated without disambiguation: {[r['filename'] for r in primary_recs if r['suffix'] == 'fieldmap']}."
        )

    if "phasediff" in modality_groups and sum(1 for r in primary_recs if r["suffix"] == "phasediff") > 1:
        discovery["ambiguity_reasons"].append(
            f"Multiple *_phasediff scans associated without disambiguation: {[r['filename'] for r in primary_recs if r['suffix'] == 'phasediff']}."
        )

    if "phase12" in modality_groups:
        n1 = sum(1 for r in primary_recs if r["suffix"] == "phase1")
        n2 = sum(1 for r in primary_recs if r["suffix"] == "phase2")
        if n1 != 1 or n2 != 1:
            discovery["ambiguity_reasons"].append(
                f"Incomplete or multiple *_phase1/*_phase2 pair associated (phase1={n1}, phase2={n2})."
            )

    if "pepolar_epi" in modality_groups:
        epi_recs = [r for r in primary_recs if r["suffix"] == "epi"]
        if len(epi_recs) > 2:
            # Ambiguous unless cleanly partitioned? Fail closed rather than guessing between multiple SE-EPI pairs.
            discovery["ambiguity_reasons"].append(
                f"More than 2 *_epi fieldmaps ({[r['filename'] for r in epi_recs]}) match this BOLD scan; association is ambiguous."
            )

    discovery["associated"] = True
    discovery["association_mechanism"] = mech
    discovery["candidate_modality_group"] = next(iter(modality_groups)) if len(modality_groups) == 1 else "ambiguous"
    discovery["associated_files"] = selected

    if discovery["ambiguity_reasons"]:
        discovery["ambiguous"] = True
        msg = "Ambiguous fieldmap association: " + " | ".join(discovery["ambiguity_reasons"])
        discovery["error"] = msg
        if raise_on_ambiguous:
            raise AmbiguousFieldmapAssociationError(msg)

    return discovery


# ---------------------------------------------------------------------------
# 4. Metadata & Geometry Compatibility Validation
# ---------------------------------------------------------------------------


def _fov_overlap_fraction(
    bold_shape: Tuple[int, int, int],
    bold_affine: np.ndarray,
    fmap_shape: Tuple[int, int, int],
    fmap_affine: np.ndarray,
    bold_mask: Optional[np.ndarray] = None,
) -> float:
    """Compute fraction of BOLD voxels (or brain mask voxels) falling inside the fieldmap voxel bounding box."""
    inv_fmap = np.linalg.inv(fmap_affine)
    xform = inv_fmap @ bold_affine
    if bold_mask is not None and np.any(bold_mask):
        idx = np.argwhere(bold_mask)
        vox_h = np.hstack([idx.astype(np.float64), np.ones((len(idx), 1), dtype=np.float64)])
    else:
        gi, gj, gk = np.indices(bold_shape, dtype=np.float64)
        vox_h = np.column_stack([gi.ravel(), gj.ravel(), gk.ravel(), np.ones(gi.size, dtype=np.float64)])

    mapped = (xform @ vox_h.T).T[:, :3]
    inside = (
        (mapped[:, 0] >= -0.5) & (mapped[:, 0] <= fmap_shape[0] - 0.5)
        & (mapped[:, 1] >= -0.5) & (mapped[:, 1] <= fmap_shape[1] - 0.5)
        & (mapped[:, 2] >= -0.5) & (mapped[:, 2] <= fmap_shape[2] - 0.5)
    )
    return float(np.mean(inside)) if len(inside) else 0.0


def validate_sdc_metadata(
    bold_image: nib.Nifti1Image,
    bold_metadata: Dict[str, Any],
    discovery: Dict[str, Any],
    *,
    allow_incomplete_metadata: bool = False,
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    """Validate BOLD and discovered fieldmap geometry, PE directions, readout timings, and units."""
    bold_shape_full = tuple(int(x) for x in bold_image.shape)
    if len(bold_shape_full) < 3 or any(d <= 0 for d in bold_shape_full[:3]):
        raise InvalidSDCInputError(f"Invalid BOLD spatial shape: {bold_shape_full}")
    bold_shape = bold_shape_full[:3]
    bold_affine = np.asarray(bold_image.affine, dtype=np.float64)
    if bold_affine.shape != (4, 4) or not np.isfinite(bold_affine).all() or abs(float(np.linalg.det(bold_affine[:3, :3]))) < 1e-10:
        raise InvalidSDCInputError("BOLD image affine matrix is non-finite or singular.")

    result: Dict[str, Any] = {
        "valid": False,
        "status_hint": SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP,
        "bold_shape": list(bold_shape),
        "bold_pe": None,
        "bold_total_readout_time_s": None,
        "bold_readout_time_source": None,
        "fieldmap_entries": [],
        "warnings": [],
        "reason": None,
        "error_type": None,
    }

    if discovery.get("ambiguous"):
        msg = discovery.get("error") or "Ambiguous BIDS fieldmap association."
        result["status_hint"] = SDC_STATUS_FAILED_INVALID_INPUT
        result["reason"] = msg
        result["error_type"] = "AmbiguousFieldmapAssociationError"
        if raise_on_error:
            raise AmbiguousFieldmapAssociationError(msg)
        return result

    if not discovery.get("associated"):
        result["valid"] = True
        result["status_hint"] = SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP
        result["reason"] = "No BIDS fieldmap associated with this BOLD scan via B0FieldSource/B0FieldIdentifier or IntendedFor."
        return result

    # Validate BOLD PE & readout timing whenever SDC metadata is evaluated against a fieldmap
    try:
        bold_pe = parse_phase_encoding_direction(bold_metadata.get("PhaseEncodingDirection"))
        bold_trt, bold_trt_src, bold_trt_warn = compute_total_readout_time(
            bold_metadata, spatial_shape=bold_shape, pe_axis=bold_pe["axis"]
        )
        result["bold_pe"] = bold_pe
        result["bold_total_readout_time_s"] = bold_trt
        result["bold_readout_time_source"] = bold_trt_src
        result["warnings"].extend(bold_trt_warn)
    except MissingSDCMetadataError as exc:
        result["status_hint"] = SDC_STATUS_SKIPPED_INSUFFICIENT_METADATA
        result["reason"] = f"BOLD metadata incomplete for SDC: {exc}"
        result["error_type"] = type(exc).__name__
        if raise_on_error and not allow_incomplete_metadata:
            raise
        return result
    except InvalidSDCInputError as exc:
        result["status_hint"] = SDC_STATUS_FAILED_INVALID_INPUT
        result["reason"] = f"Invalid BOLD SDC metadata: {exc}"
        result["error_type"] = type(exc).__name__
        if raise_on_error:
            raise
        return result

    group = discovery.get("candidate_modality_group")
    selected = discovery.get("associated_files", [])
    primary = [r for r in selected if r["suffix"] in ("epi", "fieldmap", "phasediff", "phase1", "phase2")]

    # Inspect NIfTI images & metadata for all associated fieldmaps
    validated_entries: List[Dict[str, Any]] = []
    for rec in primary:
        nii_p = Path(rec["nifti_path"])
        try:
            _check_safe_path(nii_p, label="fieldmap_nii")
            if not rec.get("json_path"):
                raise MissingSDCMetadataError(f"Fieldmap '{nii_p.name}' is missing its required JSON sidecar.")
            fmap_img = nib.load(str(nii_p))
            fmap_arr = np.asanyarray(fmap_img.dataobj, dtype=np.float32)
            if fmap_arr.ndim not in (3, 4) or any(d <= 0 for d in fmap_arr.shape[:3]):
                raise InvalidSDCInputError(f"Fieldmap '{nii_p.name}' must be 3D or 4D; got shape {fmap_arr.shape}.")
            if not np.isfinite(fmap_arr).all():
                raise InvalidSDCInputError(f"Fieldmap '{nii_p.name}' contains non-finite (NaN/Inf) voxels.")
            fmap_affine = np.asarray(fmap_img.affine, dtype=np.float64)
            if fmap_affine.shape != (4, 4) or not np.isfinite(fmap_affine).all() or abs(float(np.linalg.det(fmap_affine[:3, :3]))) < 1e-10:
                raise InvalidSDCInputError(f"Fieldmap '{nii_p.name}' has a singular or non-finite affine.")

            fmap_shape3 = tuple(int(x) for x in fmap_arr.shape[:3])
            meta = rec.get("metadata", {})
            entry_info: Dict[str, Any] = {
                **rec,
                "shape": list(fmap_arr.shape),
                "shape3": fmap_shape3,
                "affine": fmap_affine,
            }

            if rec["suffix"] == "epi":
                if np.count_nonzero(fmap_arr) == 0 or float(np.ptp(fmap_arr)) == 0.0:
                    raise InvalidSDCInputError(f"PEPOLAR fieldmap '{nii_p.name}' is all-zero or constant.")
                # TOPUP PEPOLAR strictly requires exact grid + affine match with BOLD (same EPI matrix & geometry)
                if fmap_shape3 != bold_shape:
                    raise InvalidSDCInputError(
                        f"PEPOLAR fieldmap '{nii_p.name}' spatial shape {fmap_shape3} != BOLD shape {bold_shape}."
                    )
                if not np.allclose(fmap_affine, bold_affine, atol=1e-3):
                    raise InvalidSDCInputError(
                        f"PEPOLAR fieldmap '{nii_p.name}' affine/orientation/origin/zooms mismatch BOLD affine."
                    )
                fmap_pe = parse_phase_encoding_direction(meta.get("PhaseEncodingDirection"))
                fmap_trt, fmap_trt_src, fmap_trt_warn = compute_total_readout_time(
                    meta, spatial_shape=fmap_shape3, pe_axis=fmap_pe["axis"]
                )
                entry_info["pe"] = fmap_pe
                entry_info["total_readout_time_s"] = fmap_trt
                entry_info["readout_time_source"] = fmap_trt_src
                result["warnings"].extend(fmap_trt_warn)

            elif rec["suffix"] == "fieldmap":
                units = meta.get("Units")
                if units is None:
                    raise MissingSDCMetadataError(
                        f"Direct fieldmap '{nii_p.name}' missing mandatory 'Units' metadata (expected 'Hz' or 'rad/s')."
                    )
                if not isinstance(units, str) or units.strip() not in ("Hz", "rad/s"):
                    raise InvalidSDCInputError(
                        f"Direct fieldmap '{nii_p.name}' has unsupported Units={units!r}; expected 'Hz' or 'rad/s'."
                    )
                overlap = _fov_overlap_fraction(bold_shape, bold_affine, fmap_shape3, fmap_affine)
                if overlap < 0.50:
                    raise InvalidSDCInputError(
                        f"Direct fieldmap '{nii_p.name}' overlaps only {overlap * 100:.1f}% of BOLD FOV (< 50% required)."
                    )
                entry_info["units"] = units.strip()
                entry_info["fov_overlap_fraction"] = overlap

            elif rec["suffix"] in ("phasediff", "phase1", "phase2"):
                te1 = meta.get("EchoTime1")
                te2 = meta.get("EchoTime2")
                if rec["suffix"] == "phasediff":
                    if te1 is None or te2 is None:
                        raise MissingSDCMetadataError(
                            f"Phase-difference fieldmap '{nii_p.name}' missing EchoTime1/EchoTime2 metadata."
                        )
                    if isinstance(te1, bool) or isinstance(te2, bool) or not isinstance(te1, (int, float)) or not isinstance(te2, (int, float)):
                        raise InvalidSDCInputError("EchoTime1 and EchoTime2 must be numeric.")
                    delta_te = float(te2) - float(te1)
                    if not math.isfinite(delta_te) or float(te1) <= 0.0 or delta_te <= 0.0:
                        raise InvalidSDCInputError(
                            f"Invalid EchoTime1={te1}, EchoTime2={te2} (requires 0 < EchoTime1 < EchoTime2)."
                        )
                    entry_info["echo_time_1"] = float(te1)
                    entry_info["echo_time_2"] = float(te2)
                    entry_info["delta_te_s"] = delta_te
                else:
                    te = meta.get("EchoTime")
                    if te is None:
                        raise MissingSDCMetadataError(f"Phase image '{nii_p.name}' missing EchoTime.")
                    if isinstance(te, bool) or not isinstance(te, (int, float)) or float(te) <= 0.0:
                        raise InvalidSDCInputError(f"Invalid EchoTime in '{nii_p.name}': {te!r}")
                    entry_info["echo_time"] = float(te)

                units = meta.get("Units")
                if units is None:
                    raise MissingSDCMetadataError(f"Phase image '{nii_p.name}' missing mandatory 'Units'.")
                entry_info["units"] = str(units).strip()
                overlap = _fov_overlap_fraction(bold_shape, bold_affine, fmap_shape3, fmap_affine)
                if overlap < 0.50:
                    raise InvalidSDCInputError(
                        f"Phase fieldmap '{nii_p.name}' overlaps only {overlap * 100:.1f}% of BOLD FOV."
                    )
                entry_info["fov_overlap_fraction"] = overlap

            validated_entries.append(entry_info)

        except MissingSDCMetadataError as exc:
            result["status_hint"] = SDC_STATUS_SKIPPED_INSUFFICIENT_METADATA
            result["reason"] = str(exc)
            result["error_type"] = type(exc).__name__
            if raise_on_error and not allow_incomplete_metadata:
                raise
            return result
        except InvalidSDCInputError as exc:
            result["status_hint"] = SDC_STATUS_FAILED_INVALID_INPUT
            result["reason"] = str(exc)
            result["error_type"] = type(exc).__name__
            if raise_on_error:
                raise
            return result

    # Validate group-level PEPOLAR pair consistency
    if group == "pepolar_epi":
        try:
            axes = {e["pe"]["axis"] for e in validated_entries}
            if len(axes) != 1:
                raise InvalidSDCInputError(
                    f"PEPOLAR *_epi fieldmaps must encode along the same voxel axis; got {[(e['filename'], e['pe']['code']) for e in validated_entries]}."
                )
            fmap_axis = next(iter(axes))
            if fmap_axis != bold_pe["axis"]:
                raise InvalidSDCInputError(
                    f"PEPOLAR fieldmap axis ({fmap_axis}) does not match BOLD PhaseEncodingDirection axis "
                    f"({bold_pe['axis']}='{bold_pe['code']}')."
                )
            if len(validated_entries) == 1:
                # Single reverse-PE SE-EPI paired with BOLD reference volume (b0/first vol)
                only_e = validated_entries[0]
                if (
                    only_e["pe"]["polarity"] == bold_pe["polarity"]
                    and abs(only_e["total_readout_time_s"] - bold_trt) <= 1e-6
                ):
                    raise InvalidSDCInputError(
                        "Single PEPOLAR *_epi fieldmap has identical PE polarity and identical TotalReadoutTime "
                        f"as BOLD ({only_e['pe']['code']}, {only_e['total_readout_time_s']}s); cannot solve B0 field."
                    )
            else:
                e0, e1 = validated_entries[0], validated_entries[1]
                opposite_pol = e0["pe"]["polarity"] == -e1["pe"]["polarity"]
                distinct_rt = abs(e0["total_readout_time_s"] - e1["total_readout_time_s"]) > 1e-6
                if not (opposite_pol or distinct_rt):
                    raise InvalidSDCInputError(
                        "PEPOLAR *_epi pair has identical PhaseEncodingDirection polarity and identical TotalReadoutTime "
                        f"({e0['pe']['code']}, {e0['total_readout_time_s']}s); differential B0 estimation is singular."
                    )
        except InvalidSDCInputError as exc:
            result["status_hint"] = SDC_STATUS_FAILED_INVALID_INPUT
            result["reason"] = str(exc)
            result["error_type"] = type(exc).__name__
            if raise_on_error:
                raise
            return result

    elif group == "phase12":
        try:
            by_sfx = {e["suffix"]: e for e in validated_entries}
            if abs(by_sfx["phase2"]["echo_time"] - by_sfx["phase1"]["echo_time"]) <= 0.0:
                raise InvalidSDCInputError("phase1 and phase2 EchoTime values must differ (TE2 > TE1).")
        except InvalidSDCInputError as exc:
            result["status_hint"] = SDC_STATUS_FAILED_INVALID_INPUT
            result["reason"] = str(exc)
            result["error_type"] = type(exc).__name__
            if raise_on_error:
                raise
            return result

    result["valid"] = True
    result["status_hint"] = "VALIDATED_READY"
    result["fieldmap_entries"] = validated_entries
    return result


# ---------------------------------------------------------------------------
# 5. Method Resolution & TOPUP Acquisition / Odd-Slice Configuration
# ---------------------------------------------------------------------------


def resolve_sdc_method(
    discovery: Dict[str, Any],
    validation: Dict[str, Any],
    software: Optional[Dict[str, Any]] = None,
    requested_method: str = "auto",
) -> Dict[str, Any]:
    """Select the exact SDC method and readiness status without fabricating fallbacks."""
    if requested_method not in SUPPORTED_SDC_REQUESTED_METHODS:
        raise InvalidSDCInputError(
            f"Unsupported requested SDC method '{requested_method}'. Expected one of {SUPPORTED_SDC_REQUESTED_METHODS}."
        )
    sw = software or detect_sdc_software()

    if requested_method == "none":
        return {
            "method": SDC_METHOD_NONE,
            "expected_status": SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP,
            "ready_to_apply": False,
            "missing_tools": [],
            "reason": "Susceptibility distortion correction explicitly disabled (requested_method='none').",
        }

    if requested_method == "fieldmap_less_syn":
        return {
            "method": SDC_METHOD_FIELDMAP_LESS_SYN,
            "expected_status": SDC_STATUS_SKIPPED_UNSUPPORTED_METHOD,
            "ready_to_apply": False,
            "missing_tools": [],
            "reason": (
                "Fieldmap-less nonlinear (SyN-b0) estimation is not supported/validated here; "
                "generic SyN or affine warps are never substituted for B0 susceptibility correction."
            ),
        }

    if validation.get("status_hint") == SDC_STATUS_FAILED_INVALID_INPUT:
        return {
            "method": SDC_METHOD_NONE,
            "expected_status": SDC_STATUS_FAILED_INVALID_INPUT,
            "ready_to_apply": False,
            "missing_tools": [],
            "reason": validation.get("reason"),
        }

    if validation.get("status_hint") == SDC_STATUS_SKIPPED_INSUFFICIENT_METADATA:
        return {
            "method": SDC_METHOD_NONE,
            "expected_status": SDC_STATUS_SKIPPED_INSUFFICIENT_METADATA,
            "ready_to_apply": False,
            "missing_tools": [],
            "reason": validation.get("reason"),
        }

    if not discovery.get("associated"):
        return {
            "method": SDC_METHOD_NONE,
            "expected_status": SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP,
            "ready_to_apply": False,
            "missing_tools": [],
            "reason": "No BIDS-associated fieldmap available; skipping SDC without substituting generic warps.",
        }

    group = discovery.get("candidate_modality_group")
    if requested_method == "pepolar_topup" and group != "pepolar_epi":
        return {
            "method": SDC_METHOD_PEPOLAR_TOPUP,
            "expected_status": SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP,
            "ready_to_apply": False,
            "missing_tools": [],
            "reason": f"Requested pepolar_topup but discovered modality is '{group}'.",
        }
    if requested_method == "direct_fieldmap" and group != "direct_fieldmap":
        return {
            "method": SDC_METHOD_DIRECT_FIELDMAP,
            "expected_status": SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP,
            "ready_to_apply": False,
            "missing_tools": [],
            "reason": f"Requested direct_fieldmap but discovered modality is '{group}'.",
        }
    if requested_method == "phasediff" and group not in ("phasediff", "phase12"):
        return {
            "method": SDC_METHOD_PHASE_DIFFERENCE,
            "expected_status": SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP,
            "ready_to_apply": False,
            "missing_tools": [],
            "reason": f"Requested phasediff but discovered modality is '{group}'.",
        }

    if group == "pepolar_epi":
        if not sw.get("topup_available"):
            return {
                "method": SDC_METHOD_PEPOLAR_TOPUP,
                "expected_status": SDC_STATUS_SKIPPED_TOOL_UNAVAILABLE,
                "ready_to_apply": False,
                "missing_tools": ["topup"],
                "reason": (
                    "Valid PEPOLAR BIDS fieldmap pair and acquisition parameters verified, "
                    "but external FSL 'topup' binary is not installed; skipping SDC honestly without fabricating a fieldmap."
                ),
            }
        return {
            "method": SDC_METHOD_PEPOLAR_TOPUP,
            "expected_status": SDC_STATUS_APPLIED_PEPOLAR_TOPUP,
            "ready_to_apply": True,
            "missing_tools": [],
            "reason": None,
        }

    if group == "direct_fieldmap":
        # Calibrated direct fieldmap in Hz or rad/s uses pure physical B0->displacement conversion
        return {
            "method": SDC_METHOD_DIRECT_FIELDMAP,
            "expected_status": SDC_STATUS_APPLIED_DIRECT_FIELDMAP,
            "ready_to_apply": True,
            "missing_tools": [],
            "reason": None,
        }

    if group in ("phasediff", "phase12"):
        missing = [t for t in ("prelude", "fugue") if not sw.get(f"{t}_available")]
        if missing:
            return {
                "method": SDC_METHOD_PHASE_DIFFERENCE,
                "expected_status": SDC_STATUS_SKIPPED_TOOL_UNAVAILABLE,
                "ready_to_apply": False,
                "missing_tools": missing,
                "reason": (
                    f"Phase-difference B0 fieldmap requires external phase unwrapping tool(s) ({', '.join(missing)}), "
                    "which are not installed; raw wrapped phase is never silently misinterpreted as Hz."
                ),
            }
        return {
            "method": SDC_METHOD_PHASE_DIFFERENCE,
            "expected_status": SDC_STATUS_APPLIED_PHASEDIFF,
            "ready_to_apply": True,
            "missing_tools": [],
            "reason": None,
        }

    return {
        "method": SDC_METHOD_NONE,
        "expected_status": SDC_STATUS_SKIPPED_UNSUPPORTED_METHOD,
        "ready_to_apply": False,
        "missing_tools": [],
        "reason": f"Unsupported BIDS fieldmap modality group '{group}'.",
    }


def build_topup_acquisition_parameters(
    pe_directions: Sequence[str],
    total_readout_times: Sequence[float],
) -> List[str]:
    """Construct validated FSL TOPUP ``acqparams.txt`` lines (``px py pz TotalReadoutTime``)."""
    if len(pe_directions) != len(total_readout_times) or len(pe_directions) < 2:
        raise InvalidSDCInputError(
            "TOPUP acquisition parameters require >= 2 matching PhaseEncodingDirection and TotalReadoutTime entries."
        )

    lines: List[str] = []
    parsed_list: List[Tuple[Dict[str, Any], float]] = []
    for pe_raw, trt_raw in zip(pe_directions, total_readout_times):
        pe_info = parse_phase_encoding_direction(pe_raw)
        if isinstance(trt_raw, bool) or not isinstance(trt_raw, (int, float)):
            raise InvalidSDCInputError(f"Invalid TOPUP readout time: {trt_raw!r}")
        trt = float(trt_raw)
        if not math.isfinite(trt) or trt <= 0.0 or trt >= 1.0:
            raise InvalidSDCInputError(f"TOPUP readout time must be in (0, 1.0) s, got {trt!r}")
        vx, vy, vz = pe_info["vector"]
        lines.append(f"{vx} {vy} {vz} {trt:.8f}")
        parsed_list.append((pe_info, trt))

    # Reject completely degenerate acquisition parameter sets (same axis + polarity + identical readout)
    axes = {p["axis"] for p, _ in parsed_list}
    if len(axes) == 1:
        distinct_signatures = {(p["polarity"], round(trt, 7)) for p, trt in parsed_list}
        if len(distinct_signatures) < 2:
            raise InvalidSDCInputError(
                "All TOPUP acquisition parameter entries have identical polarity and TotalReadoutTime; "
                "off-resonance B0 field is unidentifiable."
            )
    return lines


def select_topup_config(spatial_shape: Sequence[int]) -> Dict[str, Any]:
    """Select an odd-slice-safe TOPUP multi-resolution schedule for arbitrary ``(X, Y, Z)`` dimensions.

    Standard ``b02b0.cnf`` requires all spatial dimensions to be divisible by 2 (and often 4),
    causing fatal errors on odd slice counts such as ``37``. This selector chooses ``subsamp=1``
    when any axis is non-divisible by 2, never padding, duplicating, or removing slices.
    """
    dims = tuple(int(x) for x in spatial_shape[:3])
    if len(dims) != 3 or any(d <= 0 for d in dims):
        raise InvalidSDCInputError(f"Invalid 3D spatial shape for TOPUP config: {spatial_shape}")

    odd_dims = [axis_name for axis_name, val in zip(("i", "j", "k"), dims) if val % 2 != 0]
    if odd_dims:
        max_subsamp = 1
        schedule = [1, 1, 1, 1]
        preset = "odd_dimension_safe_subsamp1"
    elif all(d % 4 == 0 for d in dims):
        max_subsamp = 4
        schedule = [4, 2, 1, 1]
        preset = "b02b0_4mm_2mm_1mm"
    else:
        max_subsamp = 2
        schedule = [2, 1, 1, 1]
        preset = "b02b0_2mm_1mm"

    return {
        "spatial_shape": list(dims),
        "odd_dimension_detected": bool(odd_dims),
        "odd_dimensions": odd_dims,
        "slice_modification_applied": False,
        "max_safe_subsamp": max_subsamp,
        "subsamp_schedule": schedule,
        "preset_name": preset,
        "cnf_params": {
            "warpres": [20, 16, 12, 8],
            "subsamp": schedule,
            "fwhm": [6, 4, 2, 0],
            "miter": [5, 5, 5, 10],
            "lambda": [0.005, 0.001, 0.0001, 0.000015],
            "ssqlambda": 1,
            "regmod": "bending_energy",
            "estmov": [1, 1, 1, 0],
            "minmet": [0, 0, 0, 0],
            "splineorder": 3,
            "numprec": "float",
            "interp": "spline",
            "scale": 1,
        },
    }


def format_topup_cnf(config: Dict[str, Any]) -> str:
    """Render an explicit TOPUP ``.cnf`` text file from ``select_topup_config()``."""
    params = config["cnf_params"]
    lines: List[str] = []
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            lines.append(f"--{k}=" + ",".join(str(x) for x in v))
        else:
            lines.append(f"--{k}={v}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 6. Displacement Conversion, Jacobian Validation, Analytical Unwarping & QC
# ---------------------------------------------------------------------------


def field_hz_to_displacement(
    field_hz: np.ndarray,
    pe_direction: str,
    total_readout_time_s: float,
    affine: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert an off-resonance B0 field in Hz to EPI pullback voxel and world-mm displacement fields.

    Pullback Convention
    -------------------
    If true brain signal at undistorted voxel coordinate ``v`` shifts during EPI readout to
    apparent distorted voxel coordinate ``v_dist = v + (field_hz(v) * pe_polarity * total_readout_time) * e_pe``,
    then unwarping evaluates the distorted EPI image at ``v_dist = v + disp_vox[v]``.
    """
    arr_hz = np.asarray(field_hz, dtype=np.float32)
    if arr_hz.ndim != 3:
        raise InvalidSDCInputError(f"Expected 3D field_hz volume, got shape {arr_hz.shape}")
    if not np.isfinite(arr_hz).all():
        raise InvalidSDCInputError("B0 field (Hz) contains non-finite values.")

    pe_info = parse_phase_encoding_direction(pe_direction)
    if isinstance(total_readout_time_s, bool) or not math.isfinite(float(total_readout_time_s)) or float(total_readout_time_s) <= 0.0:
        raise InvalidSDCInputError(f"Invalid total_readout_time_s: {total_readout_time_s!r}")

    aff = np.asarray(affine, dtype=np.float64)
    if aff.shape != (4, 4) or not np.isfinite(aff).all():
        raise InvalidSDCInputError("Expected finite 4x4 affine matrix.")

    axis = pe_info["axis"]
    pol = pe_info["polarity"]
    disp_vox = np.zeros(arr_hz.shape + (3,), dtype=np.float32)
    disp_vox[..., axis] = arr_hz * np.float32(pol * float(total_readout_time_s))

    # World mm displacement = affine[:3, :3] @ disp_vox
    linear = aff[:3, :3].astype(np.float32)
    disp_mm = np.einsum("ij,...j->...i", linear, disp_vox).astype(np.float32)
    return disp_vox, disp_mm


def validate_displacement_field(
    disp_vox: np.ndarray,
    disp_mm: np.ndarray,
    pe_axis: int,
    affine: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    max_displacement_mm: float = 35.0,
    raise_on_folding: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate finiteness, PE-axis exclusivity, physical bounds, and Jacobian invertibility."""
    if disp_vox.ndim != 4 or disp_vox.shape[3] != 3:
        raise InvalidSDCInputError(f"Expected (X, Y, Z, 3) disp_vox field, got {disp_vox.shape}")
    if disp_mm.shape != disp_vox.shape:
        raise InvalidSDCInputError(f"disp_mm shape {disp_mm.shape} != disp_vox shape {disp_vox.shape}")

    finite_vox = bool(np.isfinite(disp_vox).all())
    finite_mm = bool(np.isfinite(disp_mm).all())
    finite_pct = float(np.mean(np.isfinite(disp_vox)) * 100.0)
    if not (finite_vox and finite_mm):
        raise InvalidSDCInputError(f"Displacement field contains non-finite values ({finite_pct:.2f}% finite).")

    # Verify non-PE axes have zero displacement when generated along pure EPI PE axis
    orth_max = 0.0
    for ax in range(3):
        if ax != pe_axis:
            orth_max = max(orth_max, float(np.max(np.abs(disp_vox[..., ax]))))

    pe_shift_vox = disp_vox[..., pe_axis]
    abs_mm = np.linalg.norm(disp_mm, axis=3)
    abs_vox = np.abs(pe_shift_vox)

    eval_mask = mask.astype(bool) if (mask is not None and np.any(mask)) else np.ones(disp_vox.shape[:3], dtype=bool)
    vals_mm = abs_mm[eval_mask]
    vals_vox = abs_vox[eval_mask]
    signed_vox = pe_shift_vox[eval_mask]

    # 1D directional Jacobian along PE axis: J = 1 + d(disp_pe_vox) / d(v_pe)
    grad_pe = np.gradient(pe_shift_vox.astype(np.float64), axis=pe_axis)
    jacobian_det = 1.0 + grad_pe
    jac_eval = jacobian_det[eval_mask]

    folding_count = int(np.count_nonzero(jac_eval <= 0.0))
    near_singular_count = int(np.count_nonzero(jac_eval <= 0.05))
    min_jac = float(np.min(jac_eval))
    max_jac = float(np.max(jac_eval))
    p01_jac = float(np.percentile(jac_eval, 1))
    max_abs_mm = float(np.max(vals_mm))

    disp_stats = {
        "pe_axis": int(pe_axis),
        "finite_percent": round(finite_pct, 4),
        "orthogonal_residual_max_vox": round(orth_max, 8),
        "min_signed_vox": round(float(np.min(signed_vox)), 6),
        "max_signed_vox": round(float(np.max(signed_vox)), 6),
        "mean_abs_vox": round(float(np.mean(vals_vox)), 6),
        "p95_abs_vox": round(float(np.percentile(vals_vox, 95)), 6),
        "max_abs_vox": round(float(np.max(vals_vox)), 6),
        "mean_abs_mm": round(float(np.mean(vals_mm)), 6),
        "p95_abs_mm": round(float(np.percentile(vals_mm, 95)), 6),
        "max_abs_mm": round(max_abs_mm, 6),
        "max_allowed_mm": float(max_displacement_mm),
    }
    jacobian_qc = {
        "min_jacobian": round(min_jac, 6),
        "p01_jacobian": round(p01_jac, 6),
        "max_jacobian": round(max_jac, 6),
        "folding_voxel_count": folding_count,
        "near_singular_voxel_count": near_singular_count,
        "folding_detected": bool(folding_count > 0),
        "invertible": bool(folding_count == 0),
    }

    if max_abs_mm > max_displacement_mm:
        raise InvalidSDCInputError(
            f"Displacement magnitude {max_abs_mm:.2f} mm exceeds physical ceiling {max_displacement_mm:.2f} mm."
        )
    if raise_on_folding and folding_count > 0:
        raise SDCFoldingError(
            f"Non-invertible Jacobian folding detected in SDC field: {folding_count} voxel(s) have J <= 0 "
            f"(min J = {min_jac:.4f})."
        )

    return disp_stats, jacobian_qc


def apply_sdc_unwarping_3d(
    distorted_vol: np.ndarray,
    disp_vox: np.ndarray,
    *,
    interpolation: str = "linear",
    jacobian_modulation: bool = False,
    pe_axis: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Unwarp a single 3D EPI volume using pullback voxel displacement ``disp_vox``."""
    vol = np.asarray(distorted_vol, dtype=np.float32)
    if vol.ndim != 3:
        raise InvalidSDCInputError(f"Expected 3D volume, got shape {vol.shape}")
    if disp_vox.shape != vol.shape + (3,):
        raise InvalidSDCInputError(f"Displacement shape {disp_vox.shape} != volume shape {vol.shape} + (3,)")

    order = 1 if interpolation == "linear" else 3 if interpolation == "cubic" else 0
    gi, gj, gk = np.indices(vol.shape, dtype=np.float64)
    ci = gi + disp_vox[..., 0]
    cj = gj + disp_vox[..., 1]
    ck = gk + disp_vox[..., 2]

    oob_mask = (
        (ci < 0.0) | (ci > vol.shape[0] - 1.0)
        | (cj < 0.0) | (cj > vol.shape[1] - 1.0)
        | (ck < 0.0) | (ck > vol.shape[2] - 1.0)
    )
    unwarped = map_coordinates(
        vol,
        np.array([ci, cj, ck], dtype=np.float64),
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=(order > 1),
    ).astype(np.float32)

    if jacobian_modulation:
        ax = int(pe_axis) if pe_axis is not None else int(np.argmax(np.max(np.abs(disp_vox), axis=(0, 1, 2))))
        jac = np.maximum(0.0, 1.0 + np.gradient(disp_vox[..., ax].astype(np.float64), axis=ax)).astype(np.float32)
        unwarped = (unwarped * jac).astype(np.float32)

    qc = {
        "oob_voxel_count": int(np.count_nonzero(oob_mask)),
        "oob_percent": round(float(np.mean(oob_mask) * 100.0), 4),
        "finite_percent": round(float(np.mean(np.isfinite(unwarped)) * 100.0), 4),
    }
    return unwarped, qc


def apply_sdc_unwarping_4d(
    bold_img: nib.Nifti1Image,
    disp_vox: np.ndarray,
    *,
    interpolation: str = "linear",
    jacobian_modulation: bool = False,
    pe_axis: Optional[int] = None,
) -> Tuple[nib.Nifti1Image, Dict[str, Any]]:
    """Apply voxelwise pullback SDC unwarping to every volume of a 4D BOLD NIfTI."""
    data = np.asanyarray(bold_img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise InvalidSDCInputError(f"Expected 4D BOLD array, got {data.ndim}D")
    out = np.zeros_like(data, dtype=np.float32)
    oob_pcts: List[float] = []
    for t in range(data.shape[3]):
        out[..., t], qc_t = apply_sdc_unwarping_3d(
            data[..., t],
            disp_vox,
            interpolation=interpolation,
            jacobian_modulation=jacobian_modulation,
            pe_axis=pe_axis,
        )
        oob_pcts.append(qc_t["oob_percent"])
    hdr = bold_img.header.copy()
    hdr.set_data_shape(out.shape)
    hdr.set_data_dtype(np.float32)
    hdr.set_slope_inter(1.0, 0.0)
    return nib.Nifti1Image(out, bold_img.affine, hdr), {
        "mean_oob_percent": round(float(np.mean(oob_pcts)), 4) if oob_pcts else 0.0,
        "max_oob_percent": round(float(np.max(oob_pcts)), 4) if oob_pcts else 0.0,
    }


def generate_synthetic_distorted_epi(
    true_vol: np.ndarray,
    field_hz: np.ndarray,
    pe_direction: str,
    total_readout_time_s: float,
    *,
    affine: Optional[np.ndarray] = None,
    max_iter: int = 35,
    tol: float = 1e-8,
    jacobian_modulation: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Forward-distort an undistorted 3D/4D phantom volume strictly from physics.

    Solves the implicit forward coordinate equation ``v_true + u(v_true) = v_dist``
    via fixed-point iteration so that pullback unwarping at ``v + u(v)`` inverts
    the synthetic distortion cleanly within interpolation bounds.
    """
    vol = np.asarray(true_vol, dtype=np.float32)
    spatial_shape = vol.shape[:3]
    aff = np.asarray(affine if affine is not None else np.diag([3.0, 3.0, 3.0, 1.0]), dtype=np.float64)
    disp_vox, disp_mm = field_hz_to_displacement(field_hz, pe_direction, total_readout_time_s, aff)
    pe_info = parse_phase_encoding_direction(pe_direction)
    ax = pe_info["axis"]
    validate_displacement_field(disp_vox, disp_mm, ax, aff, raise_on_folding=True)

    u_pe = disp_vox[..., ax].astype(np.float64)
    gi, gj, gk = np.indices(spatial_shape, dtype=np.float64)
    grids = [gi, gj, gk]
    v_dist_pe = grids[ax].copy()

    # Solve v_true + u(v_true) = v_dist along PE axis with damped fixed-point updates
    v_true_pe = v_dist_pe - u_pe
    max_res = float("inf")
    for _ in range(max_iter):
        sample_coords = [gi.copy(), gj.copy(), gk.copy()]
        sample_coords[ax] = v_true_pe
        u_eval = map_coordinates(u_pe, np.array(sample_coords), order=1, mode="nearest", prefilter=False)
        res = (v_true_pe + u_eval) - v_dist_pe
        max_res = float(np.max(np.abs(res)))
        if max_res < tol:
            break
        v_true_pe = v_true_pe - 0.8 * res

    sample_coords = [gi.copy(), gj.copy(), gk.copy()]
    sample_coords[ax] = v_true_pe
    coords_arr = np.array(sample_coords, dtype=np.float64)

    if jacobian_modulation:
        # Conservation of intensity under forward mapping: I_dist(x_dist) = I_true(x_true) / (1 + du/dx)
        jac = np.maximum(0.1, 1.0 + np.gradient(u_pe, axis=ax))
        jac_at_true = map_coordinates(jac, coords_arr, order=1, mode="nearest", prefilter=False)
        inv_jac = (1.0 / jac_at_true).astype(np.float32)
    else:
        inv_jac = np.ones(spatial_shape, dtype=np.float32)

    if vol.ndim == 3:
        distorted = (map_coordinates(vol, coords_arr, order=3, mode="constant", cval=0.0).astype(np.float32) * inv_jac).astype(np.float32)
    elif vol.ndim == 4:
        distorted = np.zeros_like(vol, dtype=np.float32)
        for t in range(vol.shape[3]):
            distorted[..., t] = (
                map_coordinates(vol[..., t], coords_arr, order=3, mode="constant", cval=0.0).astype(np.float32) * inv_jac
            ).astype(np.float32)
    else:
        raise InvalidSDCInputError(f"Expected 3D or 4D array, got shape {vol.shape}")

    return distorted, {
        "fixed_point_max_residual_vox": max_res,
        "pe_direction": pe_info["code"],
        "total_readout_time_s": float(total_readout_time_s),
    }


def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    m = mask.astype(bool) if mask is not None else ((np.abs(a) > 0) | (np.abs(b) > 0))
    if np.count_nonzero(m) < 10:
        return 0.0
    av = a[m].astype(np.float64)
    bv = b[m].astype(np.float64)
    av -= np.mean(av)
    bv -= np.mean(bv)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denom) if denom > 1e-12 else 0.0


def _normalized_mutual_information(a: np.ndarray, b: np.ndarray, nbins: int = 32) -> float:
    av = a[np.isfinite(a) & np.isfinite(b)].astype(np.float64)
    bv = b[np.isfinite(a) & np.isfinite(b)].astype(np.float64)
    if len(av) < 20 or np.ptp(av) <= 1e-9 or np.ptp(bv) <= 1e-9:
        return 0.0
    hist, _, _ = np.histogram2d(av, bv, bins=nbins)
    total = np.sum(hist)
    if total <= 0:
        return 0.0
    pxy = hist / total
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))
    return float((hx + hy) / hxy) if hxy > 1e-12 else 0.0


def evaluate_sdc_correction_qc(
    distorted_3d: np.ndarray,
    unwarped_3d: np.ndarray,
    reference_3d: np.ndarray,
    reference_affine: Optional[np.ndarray] = None,
    epi_affine: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute quantitative similarity metrics comparing distorted vs SDC-unwarped EPI against reference."""
    ref_resampled = np.asarray(reference_3d, dtype=np.float32)
    if (
        reference_affine is not None
        and epi_affine is not None
        and (reference_3d.shape != distorted_3d.shape or not np.allclose(reference_affine, epi_affine, atol=1e-3))
    ):
        # Resample anatomical reference onto native EPI grid solely for computing overlap QC metrics
        inv_ref = np.linalg.inv(np.asarray(reference_affine, dtype=np.float64))
        xform = inv_ref @ np.asarray(epi_affine, dtype=np.float64)
        gi, gj, gk = np.indices(distorted_3d.shape, dtype=np.float64)
        h = np.vstack([gi.ravel(), gj.ravel(), gk.ravel(), np.ones(gi.size, dtype=np.float64)])
        c = (xform @ h)[:3]
        ref_resampled = map_coordinates(ref_resampled, c, order=1, mode="constant", cval=0.0).reshape(distorted_3d.shape).astype(np.float32)

    mask_dist = distorted_3d > (0.2 * float(np.max(distorted_3d)) if np.max(distorted_3d) > 0 else 0.0)
    mask_unw = unwarped_3d > (0.2 * float(np.max(unwarped_3d)) if np.max(unwarped_3d) > 0 else 0.0)
    mask_ref = ref_resampled > (0.2 * float(np.max(ref_resampled)) if np.max(ref_resampled) > 0 else 0.0)

    def _dice(m1: np.ndarray, m2: np.ndarray) -> float:
        denom = int(np.count_nonzero(m1) + np.count_nonzero(m2))
        if denom == 0:
            return 1.0
        return float(2.0 * np.count_nonzero(m1 & m2) / denom)

    ncc_before = _normalized_cross_correlation(distorted_3d, ref_resampled, mask_ref | mask_dist)
    ncc_after = _normalized_cross_correlation(unwarped_3d, ref_resampled, mask_ref | mask_unw)
    nmi_before = _normalized_mutual_information(distorted_3d, ref_resampled)
    nmi_after = _normalized_mutual_information(unwarped_3d, ref_resampled)
    dice_before = _dice(mask_dist, mask_ref)
    dice_after = _dice(mask_unw, mask_ref)

    return {
        "ncc_before": round(ncc_before, 6),
        "ncc_after": round(ncc_after, 6),
        "nmi_before": round(nmi_before, 6),
        "nmi_after": round(nmi_after, 6),
        "mask_dice_before": round(dice_before, 6),
        "mask_dice_after": round(dice_after, 6),
        "ncc_improved": bool(ncc_after > ncc_before),
        "nmi_improved": bool(nmi_after > nmi_before),
        "dice_improved": bool(dice_after >= dice_before),
    }


def generate_sdc_qc_figure(
    distorted_3d: np.ndarray,
    unwarped_3d: np.ndarray,
    field_hz: np.ndarray,
    out_png_path: Union[str, Path],
    reference_3d: Optional[np.ndarray] = None,
) -> Path:
    """Save a 3-panel SDC quality-control visual inspection PNG (Distorted, Unwarped, B0 Field)."""
    out_p = Path(out_png_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    z_mid = distorted_3d.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0b1020")
    panels = [
        (distorted_3d[:, :, z_mid].T, "Distorted Mean EPI", "gray"),
        (unwarped_3d[:, :, z_mid].T, "SDC-Unwarped Mean EPI", "gray"),
        (field_hz[:, :, z_mid].T, "B0 Off-Resonance Field (Hz)", "coolwarm"),
    ]
    for ax, (img2d, title, cmap) in zip(axes, panels):
        im = ax.imshow(img2d, origin="lower", cmap=cmap)
        if reference_3d is not None and reference_3d.shape == distorted_3d.shape:
            ref_slice = reference_3d[:, :, z_mid].T
            lvl = float(np.percentile(ref_slice[ref_slice > 0], 35)) if np.any(ref_slice > 0) else 0.0
            if lvl > 0:
                ax.contour(ref_slice, levels=[lvl], colors=["#50d6a2"], linewidths=0.8)
        ax.set_title(title, color="#edf3ff", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(str(out_p), dpi=130, facecolor="#0b1020")
    plt.close(fig)
    return out_p


# ---------------------------------------------------------------------------
# 7. B0 Estimation & TOPUP Subprocess Wrapper + Provenance Serialization
# ---------------------------------------------------------------------------


def _run_topup_subprocess(
    imain_path: Path,
    acqparams_path: Path,
    cnf_path: Path,
    out_prefix: Path,
    fout_path: Path,
    iout_path: Path,
    topup_bin: str,
    runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
) -> Tuple[List[str], int, str, str]:
    cmd = [
        topup_bin,
        f"--imain={imain_path}",
        f"--datain={acqparams_path}",
        f"--config={cnf_path}",
        f"--out={out_prefix}",
        f"--fout={fout_path}",
        f"--iout={iout_path}",
    ]
    if runner is not None:
        proc = runner(cmd)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    return cmd, int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or "")


def _resample_fmap_to_bold_grid(
    fmap_arr: np.ndarray,
    fmap_affine: np.ndarray,
    bold_shape: Tuple[int, int, int],
    bold_affine: np.ndarray,
) -> np.ndarray:
    if fmap_arr.shape[:3] == bold_shape and np.allclose(fmap_affine, bold_affine, atol=1e-3):
        return fmap_arr.astype(np.float32)
    inv_fmap = np.linalg.inv(np.asarray(fmap_affine, dtype=np.float64))
    xform = inv_fmap @ np.asarray(bold_affine, dtype=np.float64)
    gi, gj, gk = np.indices(bold_shape, dtype=np.float64)
    coords_h = np.vstack([gi.ravel(), gj.ravel(), gk.ravel(), np.ones(gi.size, dtype=np.float64)])
    mapped = (xform @ coords_h)[:3]
    return map_coordinates(
        fmap_arr.astype(np.float32),
        mapped,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).reshape(bold_shape).astype(np.float32)


def estimate_susceptibility_field(
    bold_image: nib.Nifti1Image,
    bold_metadata: Dict[str, Any],
    discovery: Dict[str, Any],
    validation: Dict[str, Any],
    method_plan: Dict[str, Any],
    *,
    work_dir: Optional[Union[str, Path]] = None,
    topup_executable: Optional[Union[str, Path]] = None,
    topup_runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
    reference_img: Optional[nib.Nifti1Image] = None,
    max_displacement_mm: float = 35.0,
) -> SDCValidationResult:
    """Execute validated SDC estimation (or emit explicit skipped/failed status without fabrication)."""
    software = detect_sdc_software(topup_executable=topup_executable)
    method = method_plan["method"]
    expected = method_plan["expected_status"]
    bold_pe = validation.get("bold_pe") or {}

    base_assoc_summary = [
        {
            "filename": item["filename"],
            "suffix": item["suffix"],
            "matched_via": item.get("matched_via"),
            "nifti_sha256": item.get("nifti_sha256"),
            "json_sha256": item.get("json_sha256"),
            "phase_encoding_direction": item.get("metadata", {}).get("PhaseEncodingDirection"),
            "total_readout_time_s": item.get("total_readout_time_s"),
            "readout_time_source": item.get("readout_time_source"),
            "units": item.get("units"),
        }
        for item in discovery.get("associated_files", [])
    ]

    result = SDCValidationResult(
        status=expected,
        method=method,
        corrected=False,
        bold_pe_direction=bold_pe.get("code"),
        bold_pe_axis=bold_pe.get("axis"),
        bold_pe_polarity=bold_pe.get("polarity"),
        bold_total_readout_time_s=validation.get("bold_total_readout_time_s"),
        bold_readout_time_source=validation.get("bold_readout_time_source"),
        association_mechanism=discovery.get("association_mechanism", "NONE"),
        associated_fieldmap_files=base_assoc_summary,
        unassociated_fmap_files=list(discovery.get("unassociated_fmap_files", [])),
        missing_tools=list(method_plan.get("missing_tools", [])),
        software=software,
        warnings=list(discovery.get("warnings", [])) + list(validation.get("warnings", [])),
        reason=method_plan.get("reason"),
    )

    bold_data4d = np.asanyarray(bold_image.dataobj, dtype=np.float32)
    bold_shape = tuple(int(x) for x in bold_data4d.shape[:3])
    bold_affine = np.asarray(bold_image.affine, dtype=np.float64)
    bold_mean = np.mean(bold_data4d, axis=3) if bold_data4d.ndim == 4 else bold_data4d

    # Always prepare PEPOLAR acqparams + TOPUP config when PEPOLAR pair is validated (even if tool absent!)
    if method == SDC_METHOD_PEPOLAR_TOPUP and discovery.get("candidate_modality_group") == "pepolar_epi" and validation.get("valid"):
        entries = validation["fieldmap_entries"]
        imain_vols: List[np.ndarray] = []
        pe_dirs: List[str] = []
        trts: List[float] = []
        for entry in entries:
            arr = np.asanyarray(nib.load(str(entry["nifti_path"])).dataobj, dtype=np.float32)
            vol3 = arr[..., 0] if arr.ndim == 4 else arr
            imain_vols.append(vol3)
            pe_dirs.append(entry["pe"]["code"])
            trts.append(float(entry["total_readout_time_s"]))
        if len(entries) == 1:
            # Pair single reverse-PE SE-EPI with mean/first BOLD volume
            b0_bold = bold_data4d[..., 0] if bold_data4d.ndim == 4 else bold_data4d
            imain_vols.append(b0_bold)
            pe_dirs.append(bold_pe["code"])
            trts.append(float(validation["bold_total_readout_time_s"]))

        acq_lines = build_topup_acquisition_parameters(pe_dirs, trts)
        topup_cfg = select_topup_config(bold_shape)
        result.acqparams_lines = acq_lines
        result.topup_config = topup_cfg

        if not method_plan.get("ready_to_apply") and topup_runner is None:
            result.planned_command = [
                "topup",
                "--imain=<b0_merged.nii.gz>",
                "--datain=<acqparams.txt>",
                f"--config={topup_cfg['preset_name']}.cnf",
                "--out=<topup_results>",
                "--fout=<topup_fout_hz.nii.gz>",
                "--iout=<topup_iout.nii.gz>",
            ]
            return result

        # Execute TOPUP binary (real FSL or verified test runner/mock)
        topup_bin = str(topup_executable) if topup_executable is not None else software.get("topup_path")
        if topup_bin is None and topup_runner is None:
            result.status = SDC_STATUS_SKIPPED_TOOL_UNAVAILABLE
            return result
        if topup_bin is None:
            topup_bin = "topup"

        if work_dir is None:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="neuroadhd_sdc_")
            active_work = Path(tmp_ctx.name)
        else:
            tmp_ctx = None
            active_work = Path(work_dir)
            active_work.mkdir(parents=True, exist_ok=True)

        try:
            imain_4d = np.stack(imain_vols, axis=-1).astype(np.float32)
            imain_path = active_work / "topup_imain.nii.gz"
            nib.save(nib.Nifti1Image(imain_4d, bold_affine), str(imain_path))

            acq_path = active_work / "acqparams.txt"
            acq_path.write_text("\n".join(acq_lines) + "\n", encoding="utf-8")

            cnf_path = active_work / f"{topup_cfg['preset_name']}.cnf"
            cnf_path.write_text(format_topup_cnf(topup_cfg), encoding="utf-8")

            out_prefix = active_work / "topup_out"
            fout_path = active_work / "topup_fout_hz.nii.gz"
            iout_path = active_work / "topup_iout.nii.gz"

            cmd, rc, stdout_txt, stderr_txt = _run_topup_subprocess(
                imain_path, acq_path, cnf_path, out_prefix, fout_path, iout_path,
                topup_bin=topup_bin, runner=topup_runner,
            )
            result.planned_command = cmd
            result.executed_command = cmd
            result.tool_used = Path(topup_bin).name
            result.tool_version = _probe_tool_version(topup_bin, "topup") if Path(topup_bin).is_file() else "mocked_or_custom_runner"

            if rc != 0:
                msg = f"TOPUP exited with code {rc}: {stderr_txt.strip() or stdout_txt.strip()}"
                result.status = SDC_STATUS_FAILED_INVALID_INPUT
                result.reason = msg
                raise InvalidSDCInputError(msg)

            if not fout_path.is_file():
                msg = f"TOPUP completed (rc=0) but expected '--fout' output is missing at '{fout_path}'."
                result.status = SDC_STATUS_FAILED_INVALID_INPUT
                result.reason = msg
                raise InvalidSDCInputError(msg)

            fout_img = nib.load(str(fout_path))
            fout_arr = np.asanyarray(fout_img.dataobj, dtype=np.float32)
            if fout_arr.shape[:3] != bold_shape or not np.allclose(fout_img.affine, bold_affine, atol=1e-3):
                msg = f"TOPUP fieldmap output shape/affine {fout_arr.shape} mismatches BOLD {bold_shape}."
                result.status = SDC_STATUS_FAILED_INVALID_INPUT
                result.reason = msg
                raise InvalidSDCInputError(msg)

            fout_3d = fout_arr[..., 0] if fout_arr.ndim == 4 else fout_arr
            disp_vox, disp_mm = field_hz_to_displacement(
                fout_3d,
                bold_pe["code"],
                float(validation["bold_total_readout_time_s"]),
                bold_affine,
            )
            disp_stats, jac_qc = validate_displacement_field(
                disp_vox, disp_mm, bold_pe["axis"], bold_affine,
                max_displacement_mm=max_displacement_mm, raise_on_folding=True,
            )
            unwarped_mean, unw_qc = apply_sdc_unwarping_3d(bold_mean, disp_vox, interpolation="linear")

            result.status = SDC_STATUS_APPLIED_PEPOLAR_TOPUP
            result.corrected = True
            result.field_hz = fout_3d
            result.displacement_vox = disp_vox
            result.displacement_mm = disp_mm
            result.unwarped_mean_3d = unwarped_mean
            result.displacement_stats = {**disp_stats, **unw_qc}
            result.jacobian_qc = jac_qc
            ref_arr = np.asanyarray(reference_img.dataobj, dtype=np.float32) if reference_img is not None else bold_mean
            ref_aff = reference_img.affine if reference_img is not None else bold_affine
            if reference_img is not None:
                result.correction_qc = evaluate_sdc_correction_qc(
                    bold_mean, unwarped_mean, ref_arr, ref_aff, bold_affine
                )
            return result
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    if method == SDC_METHOD_DIRECT_FIELDMAP and method_plan.get("ready_to_apply"):
        entry = validation["fieldmap_entries"][0]
        fmap_img = nib.load(str(entry["nifti_path"]))
        fmap_arr = np.asanyarray(fmap_img.dataobj, dtype=np.float32)
        if fmap_arr.ndim == 4:
            fmap_arr = fmap_arr[..., 0]
        if entry["units"] == "rad/s":
            fmap_hz_native = (fmap_arr / np.float32(2.0 * math.pi)).astype(np.float32)
        else:
            fmap_hz_native = fmap_arr.astype(np.float32)

        fmap_hz_epi = _resample_fmap_to_bold_grid(fmap_hz_native, fmap_img.affine, bold_shape, bold_affine)
        disp_vox, disp_mm = field_hz_to_displacement(
            fmap_hz_epi,
            bold_pe["code"],
            float(validation["bold_total_readout_time_s"]),
            bold_affine,
        )
        disp_stats, jac_qc = validate_displacement_field(
            disp_vox, disp_mm, bold_pe["axis"], bold_affine,
            max_displacement_mm=max_displacement_mm, raise_on_folding=True,
        )
        unwarped_mean, unw_qc = apply_sdc_unwarping_3d(bold_mean, disp_vox, interpolation="linear")

        result.status = SDC_STATUS_APPLIED_DIRECT_FIELDMAP
        result.corrected = True
        result.tool_used = "NeuroADHD Calibrated Direct Fieldmap (SciPy/Nibabel)"
        result.tool_version = software.get("python_packages", {}).get("scipy")
        result.field_hz = fmap_hz_epi
        result.displacement_vox = disp_vox
        result.displacement_mm = disp_mm
        result.unwarped_mean_3d = unwarped_mean
        result.displacement_stats = {**disp_stats, **unw_qc}
        result.jacobian_qc = jac_qc
        if reference_img is not None:
            result.correction_qc = evaluate_sdc_correction_qc(
                bold_mean,
                unwarped_mean,
                np.asanyarray(reference_img.dataobj, dtype=np.float32),
                reference_img.affine,
                bold_affine,
            )
        return result

    # All non-applied statuses return honestly with corrected=False and zero fabricated fields
    return result


def build_sdc_provenance(result: SDCValidationResult) -> Dict[str, Any]:
    """Convert an ``SDCValidationResult`` into a clean JSON-serializable provenance summary."""
    return {
        "status": str(result.status),
        "method": str(result.method),
        "corrected": bool(result.corrected),
        "association_mechanism": str(result.association_mechanism),
        "associated_fieldmap_files": result.associated_fieldmap_files,
        "unassociated_fmap_files": list(result.unassociated_fmap_files),
        "bold_phase_encoding_direction": result.bold_pe_direction,
        "bold_pe_axis": result.bold_pe_axis,
        "bold_pe_polarity": result.bold_pe_polarity,
        "bold_total_readout_time_seconds": result.bold_total_readout_time_s,
        "bold_readout_time_source": result.bold_readout_time_source,
        "acqparams_lines": list(result.acqparams_lines),
        "topup_config": result.topup_config,
        "planned_command": result.planned_command,
        "executed_command": result.executed_command,
        "tool_used": result.tool_used,
        "tool_version": result.tool_version,
        "missing_tools": list(result.missing_tools),
        "software": result.software,
        "displacement_stats": result.displacement_stats,
        "jacobian_qc": result.jacobian_qc,
        "correction_qc": result.correction_qc,
        "derivative_files": result.derivative_files,
        "warnings": list(result.warnings),
        "reason": result.reason,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_susceptibility_distortion_correction(
    bold_path: Union[str, Path],
    bold_json_path: Optional[Union[str, Path]] = None,
    *,
    bold_image: Optional[nib.Nifti1Image] = None,
    fmap_dir: Optional[Union[str, Path]] = None,
    session_dir: Optional[Union[str, Path]] = None,
    derivative_dir: Optional[Union[str, Path]] = None,
    subject_id: Optional[str] = None,
    session_id: Optional[str] = None,
    requested_method: str = "auto",
    strict_sdc: bool = False,
    allow_incomplete_metadata: bool = True,
    topup_executable: Optional[Union[str, Path]] = None,
    topup_runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
    work_dir: Optional[Union[str, Path]] = None,
    reference_img: Optional[nib.Nifti1Image] = None,
    max_displacement_mm: float = 35.0,
) -> SDCValidationResult:
    """End-to-end BIDS SDC discovery, validation, estimation, QC, and derivative reporting."""
    bold_p = Path(bold_path)
    b_img = bold_image if bold_image is not None else nib.load(str(bold_p))
    if bold_json_path is None:
        bold_json_p = bold_p.with_name(_nifti_stem(bold_p) + ".json")
    else:
        bold_json_p = Path(bold_json_path)
    bold_meta = _load_json_safe(bold_json_p, label="BOLD JSON sidecar") if bold_json_p.is_file() else {}

    discovery = discover_sdc_inputs(
        bold_p,
        bold_json_path=bold_json_p if bold_json_p.is_file() else None,
        fmap_dir=fmap_dir,
        session_dir=session_dir,
        raise_on_ambiguous=True,
        raise_on_invalid=True,
    )

    if requested_method in ("fieldmap_less_syn", "none"):
        # Short-circuit explicit unsupported/disabled request cleanly before requiring PE metadata
        validation = {
            "valid": True,
            "status_hint": SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP,
            "bold_shape": list(b_img.shape[:3]),
            "bold_pe": None,
            "bold_total_readout_time_s": None,
            "bold_readout_time_source": None,
            "fieldmap_entries": [],
            "warnings": [],
            "reason": None,
        }
    else:
        validation = validate_sdc_metadata(
            b_img,
            bold_meta,
            discovery,
            allow_incomplete_metadata=allow_incomplete_metadata and not strict_sdc,
            raise_on_error=True,
        )

    software = detect_sdc_software(topup_executable=topup_executable)
    plan = resolve_sdc_method(discovery, validation, software=software, requested_method=requested_method)

    if strict_sdc and not plan["ready_to_apply"] and topup_runner is None:
        raise SDCError(plan.get("reason") or f"Strict SDC mode blocked: {plan['expected_status']}")

    res = estimate_susceptibility_field(
        b_img,
        bold_meta,
        discovery,
        validation,
        plan,
        work_dir=work_dir,
        topup_executable=topup_executable,
        topup_runner=topup_runner,
        reference_img=reference_img,
        max_displacement_mm=max_displacement_mm,
    )

    if derivative_dir is not None:
        d_dir = Path(derivative_dir)
        d_dir.mkdir(parents=True, exist_ok=True)
        sub_tok = subject_id or "sub-synthetic"
        ses_tok = f"_{session_id}" if session_id else ""
        base_stem = f"{sub_tok}{ses_tok}_task-rest"

        if res.corrected and res.field_hz is not None and res.displacement_vox is not None:
            fmap_out = d_dir / f"{base_stem}_desc-sdc_fieldmap.nii.gz"
            disp_out = d_dir / f"{base_stem}_desc-sdc_displacement.nii.gz"
            qc_png_out = d_dir / f"{base_stem}_desc-sdc_qc.png"
            nib.save(nib.Nifti1Image(res.field_hz.astype(np.float32), b_img.affine), str(fmap_out))
            nib.save(nib.Nifti1Image(res.displacement_vox.astype(np.float32), b_img.affine), str(disp_out))
            bold_mean = np.mean(np.asanyarray(b_img.dataobj, dtype=np.float32), axis=3) if b_img.ndim == 4 else np.asanyarray(b_img.dataobj, dtype=np.float32)
            unw_mean = res.unwarped_mean_3d if res.unwarped_mean_3d is not None else bold_mean
            generate_sdc_qc_figure(bold_mean, unw_mean, res.field_hz, qc_png_out)
            res.derivative_files["fieldmap_nii"] = f"derivatives/{fmap_out.name}"
            res.derivative_files["displacement_nii"] = f"derivatives/{disp_out.name}"
            res.derivative_files["qc_png"] = f"derivatives/{qc_png_out.name}"

        report_out = d_dir / f"{base_stem}_desc-sdc_report.json"
        res.derivative_files["report_json"] = f"derivatives/{report_out.name}"
        report_payload = res.to_provenance_dict()
        report_out.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    return res
