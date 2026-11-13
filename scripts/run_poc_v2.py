#!/usr/bin/env python3
"""NeuroADHD RawMRI PoC V2 Single-Subject Preprocessing Pipeline Runner.

Executes the complete 22-stage preprocessing pipeline on synthetic BIDS datasets.
Real participant processing is disabled by default to uphold data safety restrictions.

Pipeline Stages:
1.  BIDS Input & Sidecar Metadata Validation (TR, TaskName, Raw BOLD Load)
2.  3D Rigid Motion Parameter Estimation on Raw Uncorrected BOLD (DIPY) -> Transforms & Power FD
3.  Metadata-Driven Slice-Timing Correction (DEC-STC-01; Original Slice Geometry, Pre-Resampling)
4.  B0 Susceptibility Distortion Correction (DEC-SDC-02; Fieldmap Discovery, Validation & Estimation)
5.  Motion Transform (+ Optional SDC) Application to STC BOLD -> Native Reference Series
6.  Functional Reference Generation (Temporal Mean EPI & Functional Brain Mask)
7.  Anatomical Brain Extraction (Median Otsu with Hole Filling)
8.  Log-Polynomial Bias-Field Correction
9.  Tissue Segmentation (GMM into CSF, GM, WM Probabilities)
10. Morphological Tissue Mask Erosion & Quality Verification (Mandatory T1w Reference)
11. EPI-to-T1w 3D Rigid Coregistration
12. T1w-to-MNI SyN Nonlinear Normalization
13. Composed Single-Step Pullback Resampling of STC BOLD (Motion + SDC + Coreg + SyN into CC200 Grid)
14. Tissue Mask Standard Grid Resampling (Nearest-Neighbor)
15. aCompCor Physiological Noise Extraction (5 Components)
16. Friston-24 Motion Regressor Expansion
17. Confound Matrix Assembly & Collinearity Condition Check
18. Simultaneous Nuisance Regression & Butterworth Bandpass (0.01 - 0.10 Hz)
19. CC200 ROI Time Series Extraction (200 Parcels, resampling_target=None)
20. Pairwise Fisher-z Connectivity Vector Calculation (19,900 Edges)
21. Quality Control Visual Overlay Generation (4 Orthogonal Multi-Planar Overlays)
22. Schema-Compliant Provenance JSON Generation
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import hmac
import json
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from dipy.align.imaffine import AffineMap
from scipy.ndimage import map_coordinates

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Dedicated environment variable for HMAC integrity key
HMAC_KEY_ENV_VAR = "NEUROADHD_HMAC_KEY"

from src.features.connectivity import compute_connectivity_vector, get_feature_names
from src.parcellation.extractor import (
    CANONICAL_CC200_AFFINE,
    CANONICAL_CC200_SHAPE,
    extract_cc200_timeseries,
    resolve_censor_volumes,
    verify_canonical_cc200_atlas,
)
from src.preprocessing.motion import (
    apply_motion_transforms,
    build_friston_24_regressors,
    estimate_motion_parameters,
)
from src.preprocessing.nuisance import (
    build_confound_matrix,
    clean_functional_bold,
    extract_acompcor_components,
)
from src.preprocessing.qc import (
    generate_coregistration_qc_overlay,
    generate_normalization_qc_overlay,
    generate_parcellation_qc_overlay,
    generate_tissue_mask_qc_overlay,
)
from src.preprocessing.registration import (
    check_process_memory_gb,
    compose_transformations,
    generate_functional_reference,
    register_epi_to_t1w,
    register_t1w_to_mni,
    resample_4d_bold,
    resample_mask_to_target,
)
from src.preprocessing.segmentation import (
    correct_bias_field,
    create_eroded_tissue_masks,
    extract_anatomical_brain,
    segment_t1w_tissue,
    validate_tissue_masks,
)
from src.preprocessing.slice_timing import (
    SLICE_ENCODING_DIRECTION_MAP,
    correct_slice_timing_with_report,
    write_slice_timing_derivative,
)
from src.preprocessing.susceptibility_distortion import (
    evaluate_sdc_correction_qc,
    field_hz_to_displacement,
    generate_synthetic_distorted_epi,
    run_susceptibility_distortion_correction,
)
from src.validation.authorization_gate import (
    AuthorizationBypassAttemptError,
    AuthorizationDeniedError,
    ResearchAuthorizationGate,
)
from src.validation.bids_validator import BIDSInputValidator
from src.validation.memory_guard import (
    DEFAULT_CONTAINER_MEMORY_LIMIT_BYTES,
    get_cgroup_memory_stats,
    verify_os_memory_isolation,
)


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_integrity_key() -> str:
    """Retrieve the mandatory HMAC integrity key from the dedicated environment variable.

    Fails closed with a clear safety violation error if the key is missing or empty.
    Never exposes or logs key values.
    """
    key = os.environ.get(HMAC_KEY_ENV_VAR) or os.environ.get("NEUROADHD_INTEGRITY_KEY")
    if not key or not key.strip():
        raise RuntimeError(
            f"SAFETY VIOLATION: Mandatory HMAC integrity key not set in environment variable '{HMAC_KEY_ENV_VAR}'. "
            "Synthetic input verification cannot proceed and must fail closed."
        )
    return key.strip()


def build_canonical_manifest_payload(
    manifest_magic: str,
    subject_id: str,
    files_entry: Dict[str, Dict[str, Any]],
) -> bytes:
    """Construct deterministic canonical JSON serialization for HMAC computation.

    Protects:
    - manifest_magic
    - subject_id
    - all five primary input entries (bold_nii, bold_json, t1w_nii, mni_template, cc200_atlas)
      plus any registered optional auxiliary/fieldmap entries (e.g. fmap_*)
    - each entry's filename, resolved canonical path, and sha256 digest
    """
    required_entries = ["bold_nii", "bold_json", "t1w_nii", "mni_template", "cc200_atlas"]
    extra_keys = sorted(k for k in files_entry.keys() if k not in required_entries)
    canonical_files: Dict[str, Dict[str, str]] = {}
    for k in sorted(required_entries) + extra_keys:
        entry = files_entry.get(k, {})
        raw_res = entry.get("resolved_path", "")
        res_path_str = str(Path(raw_res).resolve()) if raw_res else ""
        canonical_files[k] = {
            "filename": str(entry.get("filename", "")),
            "resolved_path": res_path_str,
            "sha256": str(entry.get("sha256", "")),
        }

    payload = {
        "files": canonical_files,
        "manifest_magic": str(manifest_magic),
        "subject_id": str(subject_id),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_manifest_hmac(
    manifest_magic: Union[str, Dict[str, Any]],
    subject_id: Optional[str] = None,
    files_entry: Optional[Dict[str, Dict[str, Any]]] = None,
    key: Optional[str] = None,
) -> str:
    """Compute HMAC-SHA256 integrity token over synthetic manifest canonical payload.

    Note on Threat Model & Authenticity:
    ------------------------------------
    This manifest HMAC is an internal integrity mechanism protecting against accidental
    modification, file swapping, and path misconfiguration during development and testing.
    It does not claim to be a public-key digital signature or proof of clinical-data origin.
    """
    if key is None:
        key = get_integrity_key()

    if isinstance(manifest_magic, dict):
        manifest_dict = manifest_magic
        if "files" in manifest_dict:
            magic = manifest_dict.get("manifest_magic", "NEUROADHD_SYNTHETIC_DATASET_V2")
            sub_id = manifest_dict.get("subject_id", subject_id or "")
            files = manifest_dict.get("files", {})
        else:
            magic = "NEUROADHD_SYNTHETIC_DATASET_V2"
            sub_id = subject_id or ""
            files = manifest_dict
    else:
        magic = str(manifest_magic)
        sub_id = str(subject_id or "")
        files = files_entry if files_entry is not None else {}

    payload_bytes = build_canonical_manifest_payload(magic, sub_id, files)
    return hmac.new(key.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()


def check_safety_guards(
    subject_id: str,
    bold_path: Optional[Union[str, Path]] = None,
    t1w_path: Optional[Union[str, Path]] = None,
    bold_json_path: Optional[Union[str, Path]] = None,
    template_path: Optional[Union[str, Path]] = None,
    atlas_path: Optional[Union[str, Path]] = None,
    allow_real_scans: bool = False,
    manifest_path: Optional[Union[str, Path]] = None,
    fmap_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Enforce strict safety restrictions preventing raw clinical participant processing.

    In Phase A, execution on real MRI scans is prohibited. The pipeline fails closed
    by default. Execution requires verified synthetic origin via an authenticated
    synthetic integrity manifest with matching SHA-256 hashes across all 5 inputs.

    Note on Threat Model:
    ---------------------
    This manifest is an internal integrity mechanism protecting against accidental
    modification, file swapping, and path misconfiguration during development and testing.
    It does not claim public-key digital signature authentication.
    """
    # 0. Enforce research authorization gate and reject bypass environment variables
    auth_gate = ResearchAuthorizationGate(base_dir=PROJECT_ROOT)
    auth_gate.check_for_bypass_attempts()

    # 1. Reject explicit bypass attempts
    if allow_real_scans:
        auth_report = auth_gate.evaluate_register()
        raise AuthorizationDeniedError(
            "SAFETY VIOLATION: Execution on real MRI scans is strictly disabled by Phase A safety policy. "
            "'allow_real_scans=True' is prohibited and will not bypass safety guards. "
            f"The Phase A.6 research authorization gate is {auth_report['overall_authorization_state']}."
        )

    # 2. Block known real participant and cohort identifiers
    sub_lower = str(subject_id).strip().lower()
    blocked_keywords = [
        "2344857",
        "peking",
        "kki",
        "nyu",
        "adhd200",
        "adhd-200",
        "control",
        "abide",
        "fcon",
        "neuromark",
        "sub-001",
        "sub-control",
        "sub-adhd",
    ]
    for kw in blocked_keywords:
        if kw in sub_lower:
            raise RuntimeError(
                f"SAFETY VIOLATION: Execution on subject '{subject_id}' is strictly blocked. "
                "Processing real ADHD-200 or clinical participant data is prohibited."
            )

    # 3. Block raw data paths referencing known clinical directories and reject symlinks
    all_inputs = {
        "bold_nii": bold_path,
        "t1w_nii": t1w_path,
        "bold_json": bold_json_path,
        "mni_template": template_path,
        "cc200_atlas": atlas_path,
    }
    for key, p_val in all_inputs.items():
        if p_val is not None:
            p_str = str(p_val).lower()
            for kw in ["down/outputs/adhd200", "sub-2344857", "peking", "kki", "nyu", "abide", "adhd-200", "adhd200"]:
                if kw in p_str:
                    raise RuntimeError(
                        f"SAFETY VIOLATION: Attempted to process real raw data at '{p_val}'. "
                        "Processing raw clinical data remains prohibited."
                    )
            # Strict symlink prohibition
            raw_p = Path(p_val)
            if raw_p.is_symlink():
                raise RuntimeError(
                    f"SAFETY VIOLATION: Symlink substitution detected at '{p_val}'. "
                    "Symlinks are strictly prohibited for pipeline inputs."
                )

    # 4. Fail closed if any of the five primary inputs is missing
    missing_inputs = [k for k, p in all_inputs.items() if p is None]
    if missing_inputs:
        raise RuntimeError(
            f"SAFETY VIOLATION: Missing required input(s): {', '.join(missing_inputs)}. "
            "All five inputs (bold_nii, bold_json, t1w_nii, mni_template, cc200_atlas) must be provided."
        )

    bold_p = Path(bold_path)
    t1w_p = Path(t1w_path)
    if not bold_p.is_file():
        raise RuntimeError(f"SAFETY VIOLATION: Input BOLD file not found at '{bold_p}'.")
    if not t1w_p.is_file():
        raise RuntimeError(f"SAFETY VIOLATION: Input T1w file not found at '{t1w_p}'.")

    # 5. Locate and verify synthetic integrity manifest
    found_manifest: Optional[Path] = None
    if manifest_path is not None:
        p_cand = Path(manifest_path)
        if p_cand.is_symlink():
            raise RuntimeError(
                f"SAFETY VIOLATION: Symlink substitution detected at manifest path '{p_cand}'. "
                "Symlinks are strictly prohibited."
            )
        if p_cand.is_file():
            found_manifest = p_cand
    else:
        # Search candidate directories around input files
        search_dirs = [
            bold_p.parent,
            bold_p.parent.parent,
            t1w_p.parent,
            t1w_p.parent.parent,
        ]
        for d in search_dirs:
            cand = d / "synthetic_manifest.json"
            if cand.is_file():
                if cand.is_symlink():
                    raise RuntimeError(
                        f"SAFETY VIOLATION: Symlink substitution detected at '{cand}'. "
                        "Symlinks are strictly prohibited."
                    )
                found_manifest = cand
                break

    if found_manifest is None:
        raise RuntimeError(
            f"SAFETY VIOLATION: Synthetic origin manifest 'synthetic_manifest.json' not found for subject '{subject_id}'. "
            "Arbitrary, unauthenticated, or real NIfTI files are strictly blocked from execution."
        )

    try:
        with open(found_manifest, "r", encoding="utf-8") as f:
            manifest_dict = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"SAFETY VIOLATION: Failed to parse synthetic manifest at '{found_manifest}': {exc}"
        ) from exc

    if manifest_dict.get("manifest_magic") != "NEUROADHD_SYNTHETIC_DATASET_V2":
        raise RuntimeError(
            f"SAFETY VIOLATION: Invalid synthetic manifest magic marker in '{found_manifest}'."
        )

    if manifest_dict.get("subject_id") != subject_id:
        raise RuntimeError(
            f"SAFETY VIOLATION: Synthetic manifest subject_id '{manifest_dict.get('subject_id')}' "
            f"does not match requested subject_id '{subject_id}'."
        )

    files_entry = manifest_dict.get("files", {})

    # Require all 5 inputs to be present in the manifest
    required_entries = ["bold_nii", "bold_json", "t1w_nii", "mni_template", "cc200_atlas"]
    for req_key in required_entries:
        if req_key not in files_entry:
            raise RuntimeError(
                f"SAFETY VIOLATION: Synthetic manifest missing required entry '{req_key}'. "
                "All five inputs must be registered in the manifest."
            )

    # Mandatory HMAC verification
    if "integrity_hmac" not in manifest_dict:
        raise RuntimeError(
            "SAFETY VIOLATION: Synthetic manifest missing mandatory 'integrity_hmac' token. "
            "Execution cannot be authenticated and must fail closed."
        )
    token = manifest_dict.get("integrity_hmac")
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError(
            "SAFETY VIOLATION: Synthetic manifest 'integrity_hmac' token is empty or malformed."
        )
    token = token.strip()
    if len(token) != 64 or not all(c in "0123456789abcdefABCDEF" for c in token):
        raise RuntimeError(
            f"SAFETY VIOLATION: Synthetic manifest 'integrity_hmac' token '{token}' is malformed (must be 64-character hex)."
        )

    expected_hmac = compute_manifest_hmac(
        manifest_magic=manifest_dict["manifest_magic"],
        subject_id=subject_id,
        files_entry=files_entry,
    )
    if not hmac.compare_digest(token.lower(), expected_hmac.lower()):
        raise RuntimeError(
            "SAFETY VIOLATION: Synthetic manifest integrity token mismatch. "
            "The manifest has been fabricated, altered, or signed with an unauthorized key."
        )

    # Verify each provided input against its manifest entry
    verified_hashes: Dict[str, str] = {}
    for input_key, input_path in all_inputs.items():
        actual_path = Path(input_path)
        if actual_path.is_symlink():
            raise RuntimeError(
                f"SAFETY VIOLATION: Symlink substitution detected at '{input_path}'. "
                "Symlinks are strictly prohibited for pipeline inputs."
            )
        if not actual_path.is_file():
            raise RuntimeError(f"SAFETY VIOLATION: Input file '{actual_path}' not found on disk.")

        actual_resolved = actual_path.resolve()
        entry = files_entry[input_key]

        # Strict canonical path equality requirement (reject filename-only matching)
        manifest_res_raw = entry.get("resolved_path")
        if not manifest_res_raw:
            raise RuntimeError(
                f"SAFETY VIOLATION: Synthetic manifest entry '{input_key}' missing 'resolved_path'."
            )
        manifest_resolved = Path(manifest_res_raw).resolve()

        if actual_resolved != manifest_resolved:
            raise RuntimeError(
                f"SAFETY VIOLATION: Input '{input_key}' canonical path '{actual_resolved}' does not match "
                f"manifest path '{manifest_resolved}'. Path mismatch or substitution detected."
            )

        # Exact filename match
        expected_filename = entry.get("filename")
        if not expected_filename or actual_resolved.name != expected_filename:
            raise RuntimeError(
                f"SAFETY VIOLATION: Input '{input_key}' filename '{actual_resolved.name}' does not match "
                f"manifest filename '{expected_filename}'. Filename mismatch detected."
            )

        # Compute and verify SHA-256 hash
        actual_sha256 = compute_sha256(actual_resolved)
        expected_sha256 = entry.get("sha256")
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SAFETY VIOLATION: Input '{input_key}' ({actual_resolved.name}) SHA-256 mismatch against synthetic manifest. "
                f"Expected {expected_sha256}, got {actual_sha256}."
            )
        verified_hashes[input_key] = actual_sha256

    # Verify optional registered auxiliary/fieldmap manifest entries and check fmap directory hygiene
    extra_manifest_keys = sorted(k for k in files_entry.keys() if k not in required_entries)
    verified_fmap_resolved = set()
    for fk in extra_manifest_keys:
        entry = files_entry[fk]
        raw_res = entry.get("resolved_path")
        if not raw_res:
            raise RuntimeError(f"SAFETY VIOLATION: Synthetic manifest entry '{fk}' missing 'resolved_path'.")
        cand_p = Path(raw_res)
        if cand_p.is_symlink():
            raise RuntimeError(f"SAFETY VIOLATION: Symlink substitution detected at manifest fieldmap '{cand_p}'.")
        p_str = str(cand_p).lower()
        for kw in ["down/outputs/adhd200", "sub-2344857", "peking", "kki", "nyu", "abide", "adhd-200", "adhd200"]:
            if kw in p_str:
                raise RuntimeError(f"SAFETY VIOLATION: Attempted to process real raw fieldmap data at '{cand_p}'.")
        if not cand_p.is_file():
            raise RuntimeError(f"SAFETY VIOLATION: Registered synthetic fieldmap '{cand_p}' not found on disk.")
        cand_resolved = cand_p.resolve()
        if cand_resolved.name != entry.get("filename"):
            raise RuntimeError(
                f"SAFETY VIOLATION: Fieldmap '{fk}' filename '{cand_resolved.name}' does not match manifest filename '{entry.get('filename')}'."
            )
        actual_f_sha = compute_sha256(cand_resolved)
        if actual_f_sha != entry.get("sha256"):
            raise RuntimeError(
                f"SAFETY VIOLATION: Input '{fk}' ({cand_resolved.name}) SHA-256 mismatch against synthetic manifest."
            )
        verified_hashes[fk] = actual_f_sha
        verified_fmap_resolved.add(cand_resolved)

    cand_fmap_dir = Path(fmap_dir) if fmap_dir is not None else ((bold_p.parent.parent / "fmap") if bold_p.parent.name == "func" else (bold_p.parent / "fmap"))
    if cand_fmap_dir.is_symlink():
        raise RuntimeError(f"SAFETY VIOLATION: Symlink substitution detected at fmap directory '{cand_fmap_dir}'.")
    if cand_fmap_dir.is_dir():
        for item in sorted(cand_fmap_dir.iterdir(), key=lambda p: p.name):
            if item.is_symlink():
                raise RuntimeError(f"SAFETY VIOLATION: Symlink substitution detected at '{item}'.")
            item_lower = str(item).lower()
            for kw in ["down/outputs/adhd200", "sub-2344857", "peking", "kki", "nyu", "abide", "adhd-200", "adhd200"]:
                if kw in item_lower:
                    raise RuntimeError(f"SAFETY VIOLATION: Blocked clinical path pattern detected in '{item}'.")
            if fmap_dir is None and (extra_manifest_keys or found_manifest is not None):
                low_name = item.name.lower()
                if (low_name.endswith(".nii.gz") or low_name.endswith(".nii") or low_name.endswith(".json")) and item.resolve() not in verified_fmap_resolved:
                    raise RuntimeError(
                        f"SAFETY VIOLATION: Unauthenticated fieldmap file '{item.name}' in fmap directory is not registered in synthetic_manifest.json."
                    )

    return {
        "status": "VERIFIED_SYNTHETIC",
        "manifest_path": str(found_manifest),
        "subject_id": subject_id,
        "verified_hashes": verified_hashes,
    }


def generate_synthetic_phantom_session(
    work_dir: Path,
    subject_id: str = "sub-synthetic001",
    session_id: str = "ses-1",
    t_vol: int = 60,
    seed: int = 42,
    canonical_atlas_path: Optional[Union[str, Path]] = None,
    *,
    tr: float = 2.0,
    include_slice_timing: bool = True,
    slice_timing: Optional[Sequence[float]] = None,
    slice_encoding_direction: str = "k",
    phase_encoding_direction: Optional[str] = None,
    total_readout_time: Optional[float] = None,
    fieldmap_mode: Optional[str] = None,
    b0_field_hz: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Construct a clean, valid synthetic BIDS session with anatomical and functional data."""
    np.random.seed(seed)
    session_dir = work_dir / subject_id / session_id
    func_dir = session_dir / "func"
    anat_dir = session_dir / "anat"
    template_dir = work_dir / "templates"

    func_dir.mkdir(parents=True, exist_ok=True)
    anat_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)

    # 1. Functional EPI series
    epi_shape = (30, 30, 20)
    epi_affine = np.diag([3.0, 3.0, 3.0, 1.0])
    ex, ey, ez = np.ogrid[: epi_shape[0], : epi_shape[1], : epi_shape[2]]
    ec = (15, 15, 10)
    epi_brain = ((ex - ec[0]) ** 2 + (ey - ec[1]) ** 2 + (ez - ec[2]) ** 2) <= 9 ** 2

    bold_data = np.zeros((*epi_shape, t_vol), dtype=np.float32)
    for t in range(t_vol):
        vol = np.zeros(epi_shape, dtype=np.float32)
        # Low frequency physiological signal + Gaussian noise
        vol[epi_brain] = 100.0 + np.sin(0.1 * t) * 5.0 + np.random.normal(0, 2, np.sum(epi_brain))
        bold_data[..., t] = vol

    bold_nii_path = func_dir / f"{subject_id}_{session_id}_task-rest_bold.nii.gz"
    slice_axis_dim, _ = SLICE_ENCODING_DIRECTION_MAP.get(slice_encoding_direction, (2, False))
    bold_nii_img = nib.Nifti1Image(bold_data, epi_affine)
    bold_nii_img.header.set_zooms((3.0, 3.0, 3.0, float(tr)))
    bold_nii_img.header.set_xyzt_units("mm", "sec")
    bold_nii_img.header.set_dim_info(slice=slice_axis_dim)
    nib.save(bold_nii_img, str(bold_nii_path))

    # BOLD Sidecar JSON
    sidecar_path = func_dir / f"{subject_id}_{session_id}_task-rest_bold.json"
    sidecar_dict: Dict[str, Any] = {
        "RepetitionTime": float(tr),
        "TaskName": "rest",
        "FlipAngle": 75.0,
        "Manufacturer": "Synthetic",
    }

    # 2. Anatomical T1w head volume with internal ventricles and deep WM core
    # 1.6 mm isotropic spacing gives a 27.2 mm physical brain radius (17 * 1.6),
    # matching the 27.0 mm phantom EPI/MNI brain (9 * 3.0) without unphysical border stretch.
    t1_shape = (40, 40, 40)
    t1_affine = np.diag([1.6, 1.6, 1.6, 1.0])
    tx, ty, tz = np.ogrid[: t1_shape[0], : t1_shape[1], : t1_shape[2]]
    tc = (20, 20, 20)
    tr_sq = (tx - tc[0]) ** 2 + (ty - tc[1]) ** 2 + (tz - tc[2]) ** 2

    t1_brain = tr_sq <= 17 ** 2
    skull = (tr_sq >= 18 ** 2) & (tr_sq <= 19 ** 2)
    ventricles = ((tx - 20) ** 2 / 6 ** 2 + (ty - 20) ** 2 / 6 ** 2 + (tz - 20) ** 2 / 6 ** 2) <= 1.0
    wm = t1_brain & (~ventricles) & (tr_sq <= 13 ** 2)
    gm = t1_brain & (~ventricles) & (~wm)
    csf = ventricles

    t1_data = np.zeros(t1_shape, dtype=np.float32)
    t1_data[csf] = 25.0 + np.random.normal(0, 2, np.sum(csf))
    t1_data[gm] = 85.0 + np.random.normal(0, 3, np.sum(gm))
    t1_data[wm] = 150.0 + np.random.normal(0, 3, np.sum(wm))
    t1_data[skull] = 110.0 + np.random.normal(0, 3, np.sum(skull))

    t1_nii_path = anat_dir / f"{subject_id}_{session_id}_T1w.nii.gz"
    nib.save(nib.Nifti1Image(t1_data, t1_affine), str(t1_nii_path))

    # 3. Target Standard Template & CC200 Atlas
    if canonical_atlas_path is not None:
        canonical_p = Path(canonical_atlas_path).resolve()
        if not canonical_p.is_file():
            raise FileNotFoundError(
                f"Canonical CC200 atlas asset not found at '{canonical_p}'. "
                "Deployment requires authentic canonical atlas file."
            )
        verify_canonical_cc200_atlas(canonical_p)
        atlas_path = canonical_p

        # Load authentic CC200 atlas parcels to construct matching synthetic data
        canonical_img = nib.load(str(canonical_p))
        canonical_atlas_data = np.asanyarray(canonical_img.dataobj, dtype=np.int32)
        target_shape = CANONICAL_CC200_SHAPE
        target_affine = CANONICAL_CC200_AFFINE
        brain_mask = canonical_atlas_data > 0

        # Regenerate BOLD data matching canonical atlas space and all 200 parcels (vectorized)
        bold_data = np.zeros((*target_shape, t_vol), dtype=np.float32)
        t_arr = np.arange(t_vol, dtype=np.float32)
        for i in range(1, 201):
            mask_i = (canonical_atlas_data == i)
            n_vox = int(np.sum(mask_i))
            if n_vox > 0:
                base_signal = (100.0 + np.sin(0.08 * t_arr + i * 0.05) * 10.0)[None, :]
                noise = np.random.normal(0, 1.0, (n_vox, t_vol)).astype(np.float32)
                bold_data[mask_i, :] = base_signal + noise
        canon_bold_img = nib.Nifti1Image(bold_data, target_affine)
        canon_bold_img.header.set_zooms((3.0, 3.0, 3.0, float(tr)))
        canon_bold_img.header.set_xyzt_units("mm", "sec")
        canon_bold_img.header.set_dim_info(slice=slice_axis_dim)
        nib.save(canon_bold_img, str(bold_nii_path))

        # Regenerate T1w data matching canonical atlas space
        mx, my, mz = np.ogrid[: target_shape[0], : target_shape[1], : target_shape[2]]
        mc = (target_shape[0] // 2, target_shape[1] // 2, target_shape[2] // 2)
        ventricles = brain_mask & (((mx - mc[0]) ** 2 / 5 ** 2 + (my - mc[1]) ** 2 / 5 ** 2 + (mz - mc[2]) ** 2 / 5 ** 2) <= 1.0)
        csf = ventricles
        wm_core = brain_mask & (~csf) & (((mx - mc[0]) ** 2 / 12 ** 2 + (my - mc[1]) ** 2 / 15 ** 2 + (mz - mc[2]) ** 2 / 12 ** 2) <= 1.0)
        gm = brain_mask & (~csf) & (~wm_core)
        dist_sq = (mx - mc[0]) ** 2 + (my - mc[1]) ** 2 + (mz - mc[2]) ** 2
        skull = (~brain_mask) & (dist_sq >= 28 ** 2) & (dist_sq <= 30 ** 2)

        t1_data = np.zeros(target_shape, dtype=np.float32)
        t1_data[csf] = 25.0 + np.random.normal(0, 2, np.sum(csf))
        t1_data[gm] = 85.0 + np.random.normal(0, 3, np.sum(gm))
        t1_data[wm_core] = 150.0 + np.random.normal(0, 3, np.sum(wm_core))
        t1_data[skull] = 110.0 + np.random.normal(0, 3, np.sum(skull))
        nib.save(nib.Nifti1Image(t1_data, target_affine), str(t1_nii_path))

        # Standard MNI template matching canonical CC200 atlas grid
        mni_data = np.zeros(target_shape, dtype=np.float32)
        mni_data[brain_mask] = 100.0
        mni_data[wm_core] = 150.0
        mni_data[gm] = 85.0
        mni_data[csf] = 25.0
        mni_path = template_dir / "MNI152_T1_3mm.nii.gz"
        nib.save(nib.Nifti1Image(mni_data, target_affine), str(mni_path))
    else:
        target_shape = (30, 30, 20)
        target_affine = np.diag([3.0, 3.0, 3.0, 1.0])
        mni_data = np.zeros(target_shape, dtype=np.float32)
        mni_data[epi_brain] = 100.0

        mni_path = template_dir / "MNI152_T1_3mm.nii.gz"
        nib.save(nib.Nifti1Image(mni_data, target_affine), str(mni_path))

        # 200 CC200 Parcels
        atlas_data = np.zeros(target_shape, dtype=np.int32)
        brain_indices = np.argwhere(epi_brain)
        n_brain = len(brain_indices)
        chunk_size = max(1, n_brain // 200)
        for i in range(200):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < 199 else n_brain
            for coord in brain_indices[start:end]:
                atlas_data[tuple(coord)] = i + 1

        atlas_path = template_dir / "cc200_synthetic_atlas.nii.gz"
        nib.save(nib.Nifti1Image(atlas_data, target_affine), str(atlas_path))

    # Populate valid SliceTiming and SliceEncodingDirection matching final BOLD geometry
    if include_slice_timing:
        slice_axis, _ = SLICE_ENCODING_DIRECTION_MAP.get(slice_encoding_direction, (2, False))
        n_slices_axis = int(bold_data.shape[slice_axis])
        if slice_timing is not None:
            timings = [round(float(x), 8) for x in slice_timing]
        else:
            order = list(range(0, n_slices_axis, 2)) + list(range(1, n_slices_axis, 2))
            timings = [0.0] * n_slices_axis
            for acq_step, sl_idx in enumerate(order):
                timings[sl_idx] = round(float(acq_step * float(tr) / n_slices_axis), 8)
        sidecar_dict["SliceTiming"] = timings
        sidecar_dict["SliceEncodingDirection"] = slice_encoding_direction

    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_dict, f, indent=2)

    # Optional B0 fieldmap generation (forward-distorts synthetic BOLD strictly from physics)
    generated_fmap_entries: Dict[str, Dict[str, Any]] = {}
    created_fmap_dir: Optional[Path] = None
    if fieldmap_mode is not None:
        pe_code = phase_encoding_direction or "j"
        trt_val = float(total_readout_time) if total_readout_time is not None else 0.05
        active_epi_img = nib.load(str(bold_nii_path))
        bold_undist = np.asanyarray(active_epi_img.dataobj, dtype=np.float32)
        active_epi_affine = np.asarray(active_epi_img.affine, dtype=np.float64)
        s_shape = bold_undist.shape[:3]

        if b0_field_hz is not None:
            b0_hz = np.asarray(b0_field_hz, dtype=np.float32)
        else:
            gx, gy, gz = np.indices(s_shape, dtype=np.float32)
            # Smooth synthetic orbitofrontal-like off-resonance lobe in anterior-inferior quadrant (~28 Hz peak)
            cx = s_shape[0] * 0.5
            cy = s_shape[1] * 0.65
            cz = s_shape[2] * 0.45
            sx = max(2.5, s_shape[0] * 0.16)
            sy = max(2.5, s_shape[1] * 0.16)
            sz = max(2.0, s_shape[2] * 0.18)
            b0_hz = (
                28.0
                * np.exp(
                    -(((gx - cx) / sx) ** 2 + ((gy - cy) / sy) ** 2 + ((gz - cz) / sz) ** 2)
                )
            ).astype(np.float32)

        bold_dist, _ = generate_synthetic_distorted_epi(
            bold_undist,
            b0_hz,
            pe_direction=pe_code,
            total_readout_time_s=trt_val,
            affine=active_epi_affine,
        )
        dist_img = nib.Nifti1Image(bold_dist, active_epi_affine)
        dist_img.header.set_zooms((3.0, 3.0, 3.0, float(tr)))
        dist_img.header.set_xyzt_units("mm", "sec")
        dist_img.header.set_dim_info(slice=slice_axis_dim)
        nib.save(dist_img, str(bold_nii_path))

        sidecar_dict["PhaseEncodingDirection"] = pe_code
        sidecar_dict["TotalReadoutTime"] = float(trt_val)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_dict, f, indent=2)

        created_fmap_dir = session_dir / "fmap"
        created_fmap_dir.mkdir(parents=True, exist_ok=True)
        intended_rel = f"{session_id}/func/{bold_nii_path.name}"

        if fieldmap_mode == "pepolar":
            rev_pe = pe_code[:-1] if pe_code.endswith("-") else f"{pe_code}-"
            ref_3d = np.mean(bold_undist, axis=3)
            se_fwd, _ = generate_synthetic_distorted_epi(
                ref_3d, b0_hz, pe_direction=pe_code, total_readout_time_s=trt_val, affine=active_epi_affine
            )
            se_rev, _ = generate_synthetic_distorted_epi(
                ref_3d, b0_hz, pe_direction=rev_pe, total_readout_time_s=trt_val, affine=active_epi_affine
            )
            for tag, pe_item, vol_item in (("fwd", pe_code, se_fwd), ("rev", rev_pe, se_rev)):
                fn_nii = created_fmap_dir / f"{subject_id}_{session_id}_dir-{tag}_epi.nii.gz"
                fn_json = created_fmap_dir / f"{subject_id}_{session_id}_dir-{tag}_epi.json"
                nib.save(nib.Nifti1Image(vol_item.astype(np.float32), active_epi_affine), str(fn_nii))
                fn_json.write_text(
                    json.dumps(
                        {
                            "PhaseEncodingDirection": pe_item,
                            "TotalReadoutTime": float(trt_val),
                            "IntendedFor": [intended_rel],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        elif fieldmap_mode in ("direct_hz", "direct_rad_s"):
            fn_nii = created_fmap_dir / f"{subject_id}_{session_id}_fieldmap.nii.gz"
            fn_json = created_fmap_dir / f"{subject_id}_{session_id}_fieldmap.json"
            fmap_vals = b0_hz if fieldmap_mode == "direct_hz" else (b0_hz * np.float32(2.0 * np.pi))
            nib.save(nib.Nifti1Image(fmap_vals.astype(np.float32), active_epi_affine), str(fn_nii))
            fn_json.write_text(
                json.dumps(
                    {
                        "Units": "Hz" if fieldmap_mode == "direct_hz" else "rad/s",
                        "IntendedFor": [intended_rel],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            raise ValueError(f"Unsupported fieldmap_mode '{fieldmap_mode}' in synthetic session generator.")

        for idx, item in enumerate(sorted(created_fmap_dir.iterdir(), key=lambda p: p.name)):
            generated_fmap_entries[f"fmap_{idx:02d}"] = {
                "filename": item.name,
                "rel_path": str(item.relative_to(session_dir)),
                "resolved_path": str(item.resolve()),
                "sha256": compute_sha256(item),
            }

    # All 5 files (+ optional generated fieldmap files) registered in synthetic manifest
    files_dict = {
        "bold_nii": {
            "filename": bold_nii_path.name,
            "rel_path": str(bold_nii_path.relative_to(session_dir)),
            "resolved_path": str(bold_nii_path.resolve()),
            "sha256": compute_sha256(bold_nii_path),
        },
        "bold_json": {
            "filename": sidecar_path.name,
            "rel_path": str(sidecar_path.relative_to(session_dir)),
            "resolved_path": str(sidecar_path.resolve()),
            "sha256": compute_sha256(sidecar_path),
        },
        "t1w_nii": {
            "filename": t1_nii_path.name,
            "rel_path": str(t1_nii_path.relative_to(session_dir)),
            "resolved_path": str(t1_nii_path.resolve()),
            "sha256": compute_sha256(t1_nii_path),
        },
        "mni_template": {
            "filename": mni_path.name,
            "rel_path": str(mni_path.name),
            "resolved_path": str(mni_path.resolve()),
            "sha256": compute_sha256(mni_path),
        },
        "cc200_atlas": {
            "filename": atlas_path.name,
            "rel_path": str(atlas_path.name),
            "resolved_path": str(atlas_path.resolve()),
            "sha256": compute_sha256(atlas_path),
        },
    }
    files_dict.update(generated_fmap_entries)

    integrity_token = compute_manifest_hmac(
        manifest_magic="NEUROADHD_SYNTHETIC_DATASET_V2",
        subject_id=subject_id,
        files_entry=files_dict,
    )

    # Write synthetic integrity manifest
    manifest_data = {
        "manifest_magic": "NEUROADHD_SYNTHETIC_DATASET_V2",
        "manifest_type": "INTEGRITY_VERIFICATION_MANIFEST",
        "schema_version": "2.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject_id": subject_id,
        "session_id": session_id,
        "integrity_hmac": integrity_token,
        "files": files_dict,
        "note": "Integrity manifest protecting against accidental modification and path misconfiguration. Not a PKI digital signature.",
    }
    manifest_path = session_dir / "synthetic_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    result_paths: Dict[str, Any] = {
        "bold_nii": bold_nii_path,
        "bold_json": sidecar_path,
        "t1w_nii": t1_nii_path,
        "mni_template": mni_path,
        "cc200_atlas": atlas_path,
        "synthetic_manifest": manifest_path,
    }
    if created_fmap_dir is not None:
        result_paths["fmap_dir"] = created_fmap_dir
    return result_paths


def run_poc_v2_pipeline(
    bold_path: Union[str, Path],
    bold_json_path: Union[str, Path],
    t1w_path: Union[str, Path],
    atlas_path: Union[str, Path],
    template_path: Union[str, Path],
    output_dir: Union[str, Path],
    subject_id: str = "sub-synthetic001",
    session_id: str = "ses-1",
    max_memory_gb: float = 2.5,
    allow_real_scans: bool = False,
    manifest_path: Optional[Union[str, Path]] = None,
    atlas_mode: str = "synthetic_phantom",
    enforce_container_memory: bool = False,
    model_dir: Optional[Union[str, Path]] = None,
    require_model_inference: bool = False,
    slice_timing_reference_time: Optional[float] = None,
    slice_timing_interpolation: Optional[str] = None,
    allow_missing_slice_timing: bool = True,
    censor_volumes: Optional[Sequence[int]] = None,
    sdc_method: str = "auto",
    strict_sdc: bool = False,
    allow_incomplete_sdc_metadata: bool = True,
    topup_executable: Optional[Union[str, Path]] = None,
    topup_runner: Optional[Callable[[list], Any]] = None,
    fmap_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute the complete 22-stage NeuroADHD PoC V2 preprocessing pipeline."""
    bold_path = Path(bold_path)
    bold_json_path = Path(bold_json_path)
    t1w_path = Path(t1w_path)
    atlas_path = Path(atlas_path)
    template_path = Path(template_path)
    output_dir = Path(output_dir)
    if manifest_path is not None:
        manifest_path = Path(manifest_path)

    start_time = time.time()
    start_rss = check_process_memory_gb()

    # Verify operating-system memory isolation
    os_memory_info = verify_os_memory_isolation(
        max_allowed_bytes=int(max_memory_gb * 1024 * 1024 * 1024),
        require_enforcement=enforce_container_memory,
    )
    if os_memory_info["enforced"]:
        print(f"[Memory Guard] OS Cgroup Memory Isolation: ENFORCED ({os_memory_info['memory_max_mib']:.1f} MiB max, cgroup v{os_memory_info['cgroup_version']})")
    else:
        print(f"[Memory Guard] OS Cgroup Memory Isolation: {os_memory_info['status']} ({os_memory_info['reason']})")

    # Safety Enforcement - fail closed across all inputs
    check_safety_guards(
        subject_id=subject_id,
        bold_path=bold_path,
        t1w_path=t1w_path,
        bold_json_path=bold_json_path,
        template_path=template_path,
        atlas_path=atlas_path,
        allow_real_scans=allow_real_scans,
        manifest_path=manifest_path,
        fmap_dir=fmap_dir,
    )

    # Downstream ML Model verification (fail-closed early check)
    target_model_dir = model_dir or os.environ.get("NEUROADHD_MODEL_DIR")
    frozen_pipe, frozen_clf = None, None
    if require_model_inference:
        if not target_model_dir:
            raise RuntimeError(
                "SAFETY VIOLATION: Model inference was explicitly requested (--require-model-inference), "
                "but neither --model-dir was provided nor NEUROADHD_MODEL_DIR environment variable is configured."
            )
        from src.integration.model_verifier import verify_and_load_frozen_model
        try:
            frozen_pipe, frozen_clf = verify_and_load_frozen_model(target_model_dir)
        except Exception as exc:
            raise RuntimeError(
                f"SAFETY VIOLATION: Explicitly requested model inference failed verification: {exc}"
            ) from exc
    elif target_model_dir:
        from src.integration.model_verifier import verify_and_load_frozen_model
        frozen_pipe, frozen_clf = verify_and_load_frozen_model(target_model_dir)

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Active staging directory for isolation: partial outputs will never be mistaken for successful runs
    sub_out_dir = output_dir / subject_id / f".staging_run_{run_tag}"
    scratch_dir = sub_out_dir / "scratch"
    sub_out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"NEUROADHD RAW MRI POC V2 PIPELINE EXECUTION [{subject_id}]")
    print(f"Timestamp UTC : {datetime.now(timezone.utc).isoformat()}")
    print(f"Staging Root  : {sub_out_dir}")
    print("=" * 75)

    current_stage = "Initialization"
    model_prediction_path: Optional[Path] = None
    try:
        # -------------------------------------------------------------
        # Stage 1: BIDS Input & Sidecar Metadata Validation
        # -------------------------------------------------------------
        current_stage = "Stage 1: BIDS Input Validation"
        print(f"\n[Stage 1/22] Validating BIDS metadata from {bold_json_path.name}...")
        validator = BIDSInputValidator()
        bids_meta = validator.validate_sidecar_json(bold_json_path)
        tr = bids_meta["repetition_time"]
        print(f"  -> RepetitionTime: {tr:.3f}s | Task: {bids_meta['task_name']} [PASSED]")

        # Explicit atlas verification policy (no filename heuristics)
        if atlas_mode == "canonical":
            print(f"  -> Enforcing canonical CC200 atlas verification on {atlas_path.name} (independent of filename)...")
            verify_canonical_cc200_atlas(atlas_path)
            print("  -> Canonical CC200 atlas verified [PASSED]")
        elif atlas_mode == "synthetic_phantom":
            print(f"  -> Validating synthetic phantom atlas {atlas_path.name} against authenticated manifest...")
            # Note: check_safety_guards already asserted that atlas_path matches the synthetic manifest hash & path
            atlas_img_check = nib.load(str(atlas_path))
            atlas_data_test = np.asanyarray(atlas_img_check.dataobj, dtype=np.int32)
            u_rois = np.unique(atlas_data_test[atlas_data_test > 0])
            if len(u_rois) != 200:
                raise ValueError(
                    f"Synthetic atlas must contain exactly 200 non-zero ROIs, found {len(u_rois)}."
                )
        else:
            raise ValueError(
                f"Invalid atlas_mode '{atlas_mode}'. Must be 'canonical' or 'synthetic_phantom'."
            )

        # Load NIfTI volumes
        bold_img = nib.load(str(bold_path))
        t1w_img = nib.load(str(t1w_path))
        atlas_img = nib.load(str(atlas_path))
        mni_img = nib.load(str(template_path))

        t_vol = bold_img.shape[3] if bold_img.ndim == 4 else 0
        if t_vol < 50:
            raise ValueError(f"Insufficient BOLD timepoints: {t_vol} (minimum 50 required)")
        censored_sorted, kept_volume_indices = resolve_censor_volumes(
            censor_volumes, t_vol, min_retained=50
        )

        epi_shape = bold_img.shape[:3]

        # -------------------------------------------------------------
        # Stage 2: 3D Rigid Motion Parameter Estimation (Raw Uncorrected BOLD)
        # -------------------------------------------------------------
        current_stage = "Stage 2: Motion Parameter Estimation"
        print(
            f"\n[Stage 2/22] Estimating 3D rigid motion parameters on raw uncorrected BOLD ({t_vol} volumes)..."
        )
        bold_input_sha256 = compute_sha256(bold_path)
        bold_json_input_sha256 = compute_sha256(bold_json_path)
        rigid_motion_transforms, motion_params, fd_vector = estimate_motion_parameters(bold_img)
        mean_fd = float(np.mean(fd_vector))
        max_fd = float(np.max(fd_vector))
        scrubbed = int(np.count_nonzero(fd_vector > 0.5))
        print(
            f"  -> Estimated from raw BOLD (pre-STC/SDC): Mean FD: {mean_fd:.4f} mm | "
            f"Max FD: {max_fd:.4f} mm | Scrubbed (>0.5mm): {scrubbed} volumes"
        )

        # -------------------------------------------------------------
        # Stage 3: Metadata-Driven Slice-Timing Correction (DEC-STC-01)
        # -------------------------------------------------------------
        current_stage = "Stage 3: Slice-Timing Correction"
        print(
            "\n[Stage 3/22] Executing metadata-driven slice-timing correction on original slice geometry..."
        )
        with open(bold_json_path, "r", encoding="utf-8") as bj_f:
            raw_bold_sidecar = json.load(bj_f)

        deriv_dir = sub_out_dir / "derivatives"
        has_slice_timing = raw_bold_sidecar.get("SliceTiming") is not None
        if has_slice_timing or not allow_missing_slice_timing:
            stc_bold_img, stc_report = write_slice_timing_derivative(
                bold_path=bold_path,
                bold_json_path=bold_json_path,
                derivative_dir=deriv_dir,
                reference_time=slice_timing_reference_time,
                interpolation_method=slice_timing_interpolation,
                allow_missing_timing=False,
                expected_tr=tr,
            )
            derivative_nii_rel = f"derivatives/{stc_report['output_derivative_nii']}"
            derivative_json_rel = f"derivatives/{stc_report['output_derivative_json']}"
            if stc_report["status"] == "APPLIED":
                print(
                    f"  -> STC APPLIED: axis={stc_report['slice_axis']} "
                    f"dir={stc_report['slice_encoding_direction']} | "
                    f"ref={stc_report['reference_time']:.3f}s (+half-TR temporal origin shift) | "
                    f"method={stc_report['interpolation_method']} | "
                    f"slices={stc_report['num_corrected_slices']}/{stc_report['num_slices']}"
                )
            else:
                print(
                    f"  -> STC {stc_report['status']}: {stc_report.get('reason') or 'No relative shift needed'}"
                )
        else:
            _, stc_report = correct_slice_timing_with_report(
                bold_image=bold_img,
                bold_metadata=raw_bold_sidecar,
                reference_time=slice_timing_reference_time,
                interpolation_method=slice_timing_interpolation,
                allow_missing_timing=True,
                expected_tr=tr,
            )
            stc_bold_img = bold_img
            deriv_dir.mkdir(parents=True, exist_ok=True)
            report_fname = f"{subject_id}_{session_id}_task-rest_desc-stc_report.json"
            report_path_stc = deriv_dir / report_fname
            stc_skip_record = {
                "input_bold_filename": bold_path.name,
                "input_json_filename": bold_json_path.name,
                "input_bold_sha256": bold_input_sha256,
                "input_json_sha256": bold_json_input_sha256,
                "policy": "skip_when_missing_never_invent_timings",
                **stc_report,
            }
            with open(report_path_stc, "w", encoding="utf-8") as sf_out:
                json.dump(stc_skip_record, sf_out, indent=2)
            stc_report = stc_skip_record
            derivative_nii_rel = None
            derivative_json_rel = f"derivatives/{report_fname}"
            print(
                "  -> WARNING: Missing 'SliceTiming' in BIDS sidecar; "
                "refusing to invent timings from TR. STC SKIPPED and original BOLD preserved."
            )

        stc_provenance_summary: Dict[str, Any] = {
            "status": stc_report["status"],
            "corrected": bool(stc_report.get("corrected", False)),
            "tr_seconds": float(stc_report["tr"]),
            "reference_time_seconds": stc_report.get("reference_time"),
            "temporal_origin_shift_seconds": float(stc_report.get("temporal_origin_shift_seconds", 0.0)),
            "slice_axis": int(stc_report["slice_axis"]) if stc_report.get("slice_axis") is not None else None,
            "header_slice_dim": stc_report.get("header_slice_dim"),
            "slice_axis_source": stc_report.get("slice_axis_source"),
            "slice_encoding_direction": stc_report["slice_encoding_direction"],
            "slice_encoding_direction_assumed": bool(stc_report.get("slice_encoding_direction_assumed", False)),
            "slice_encoding_direction_reversed": (
                bool(stc_report["slice_encoding_direction_reversed"])
                if stc_report.get("slice_encoding_direction_reversed") is not None
                else None
            ),
            "slice_order_interpretation": stc_report.get("slice_order_interpretation"),
            "num_slices": int(stc_report["num_slices"]) if stc_report.get("num_slices") is not None else None,
            "num_corrected_slices": int(stc_report.get("num_corrected_slices", 0)),
            "interpolation_method": stc_report.get("interpolation_method"),
            "boundary_handling": stc_report.get("boundary_handling"),
            "boundary_padding_strategy": stc_report.get("boundary_padding_strategy"),
            "boundary_pad_samples": stc_report.get("boundary_pad_samples"),
            "boundary_extrapolation_disabled": stc_report.get("boundary_extrapolation_disabled"),
            "boundary_edge_volumes_at_risk": stc_report.get("boundary_edge_volumes_at_risk"),
            "boundary_risk_window_samples_per_side": stc_report.get("boundary_risk_window_samples_per_side"),
            "boundary_risk_window_volumes": stc_report.get("boundary_risk_window_volumes"),
            "boundary_regions": stc_report.get("boundary_regions"),
            "automatic_boundary_volume_removal_applied": False,
            "software": stc_report.get("software"),
            "software_version": stc_report.get("software_version"),
            "afni_3dtshift_available": bool(stc_report.get("afni_3dtshift_available", False)),
            "input_bold_sha256": bold_input_sha256,
            "input_json_sha256": bold_json_input_sha256,
            "output_bold_sha256": stc_report.get("output_bold_sha256"),
            "output_json_sha256": stc_report.get("output_json_sha256"),
            "derivative_nii": derivative_nii_rel,
            "derivative_json": derivative_json_rel,
            "warnings": list(stc_report.get("warnings", [])),
            "reason": stc_report.get("reason"),
            "input_file_unmodified_verified": True,
        }

        # -------------------------------------------------------------
        # Stage 4: B0 Susceptibility Distortion Correction (DEC-SDC-02)
        # -------------------------------------------------------------
        current_stage = "Stage 4: Susceptibility Distortion Correction"
        print(
            "\n[Stage 4/22] Auditing & estimating B0 susceptibility distortion correction (DEC-SDC-02)..."
        )
        sdc_res = run_susceptibility_distortion_correction(
            bold_path=bold_path,
            bold_json_path=bold_json_path,
            bold_image=stc_bold_img,
            fmap_dir=fmap_dir,
            derivative_dir=deriv_dir,
            subject_id=subject_id,
            session_id=session_id,
            requested_method=sdc_method,
            strict_sdc=strict_sdc,
            allow_incomplete_metadata=allow_incomplete_sdc_metadata,
            topup_executable=topup_executable,
            topup_runner=topup_runner,
            work_dir=scratch_dir / "sdc_work",
        )

        # Verify byte-level immutability of input BOLD and JSON sidecar across Stages 2-4
        if (
            compute_sha256(bold_path) != bold_input_sha256
            or compute_sha256(bold_json_path) != bold_json_input_sha256
        ):
            raise RuntimeError(
                "Input integrity violation: raw BOLD or JSON sidecar was altered during Stage 2-4."
            )

        sdc_displacement_vox = (
            sdc_res.displacement_vox
            if (sdc_res.corrected and sdc_res.displacement_vox is not None)
            else None
        )
        sdc_provenance_summary = sdc_res.to_provenance_dict()
        if sdc_res.corrected and sdc_res.displacement_stats and sdc_res.jacobian_qc:
            print(
                f"  -> SDC {sdc_res.status}: method={sdc_res.method} | PE={sdc_res.bold_pe_direction} | "
                f"max_disp={sdc_res.displacement_stats['max_abs_mm']:.3f} mm | "
                f"min_J={sdc_res.jacobian_qc['min_jacobian']:.4f}"
            )
        else:
            print(
                f"  -> SDC {sdc_res.status}: {sdc_res.reason or 'No BIDS fieldmap applied'}"
            )

        # -------------------------------------------------------------
        # Stage 5: Apply Estimated Motion (+ Optional SDC) Transforms to STC BOLD
        # -------------------------------------------------------------
        current_stage = "Stage 5: Motion Transform Application"
        print(
            f"\n[Stage 5/22] Applying pre-estimated rigid motion transforms"
            f"{' + SDC unwarping' if sdc_displacement_vox is not None else ''} to STC BOLD ({t_vol} volumes)..."
        )
        if sdc_displacement_vox is not None:
            realigned_bold = apply_motion_transforms(
                stc_bold_img,
                rigid_motion_transforms,
                interpolation="linear",
                sdc_displacement=sdc_displacement_vox,
            )
        else:
            realigned_bold = apply_motion_transforms(
                stc_bold_img,
                rigid_motion_transforms,
                interpolation="linear",
            )
        print("  -> Motion-aligned native BOLD constructed for functional reference.")

        # -------------------------------------------------------------
        # Stage 6: Functional Reference Generation
        # -------------------------------------------------------------
        current_stage = "Stage 6: Functional Reference"
        print("\n[Stage 6/22] Generating 3D functional reference image...")
        mean_epi, epi_brain_mask = generate_functional_reference(realigned_bold)
        print(f"  -> Reference EPI shape: {mean_epi.shape}")
        del realigned_bold
        gc.collect()

        # -------------------------------------------------------------
        # Stage 7: Anatomical Brain Extraction
        # -------------------------------------------------------------
        current_stage = "Stage 7: Anatomical Brain Extraction"
        print("\n[Stage 7/22] Extracting T1w anatomical brain with hole-filling...")
        t1_brain, t1_brain_mask = extract_anatomical_brain(t1w_img, median_radius=2, numpass=2)
        print(f"  -> Extracted brain voxels: {int(np.count_nonzero(t1_brain_mask.dataobj))}")

        # -------------------------------------------------------------
        # Stage 8: Log-Polynomial Bias-Field Correction
        # -------------------------------------------------------------
        current_stage = "Stage 8: Bias-Field Correction"
        print("\n[Stage 8/22] Correcting anatomical bias field (order=2)...")
        t1_corrected, bias_field = correct_bias_field(t1_brain, t1_brain_mask, order=2)
        print(f"  -> Bias field mean: {float(np.mean(bias_field.dataobj)):.3f}")

        # -------------------------------------------------------------
        # Stage 9: Tissue Segmentation (CSF, GM, WM Probabilities)
        # -------------------------------------------------------------
        current_stage = "Stage 9: Tissue Segmentation"
        print("\n[Stage 9/22] Segmenting T1w tissue into CSF, GM, WM...")
        prob_maps = segment_t1w_tissue(t1_corrected, t1_brain_mask, n_classes=3, method="gmm")
        print("  -> Probability maps: CSF, GM, WM computed successfully.")

        # -------------------------------------------------------------
        # Stage 10: Morphological Mask Erosion & Geometry Verification
        # -------------------------------------------------------------
        current_stage = "Stage 10: Tissue Mask Erosion"
        print("\n[Stage 10/22] Applying 0.96 thresholding & 1-voxel 3D erosion...")
        tissue_masks = create_eroded_tissue_masks(
            prob_maps,
            reference_t1w_img=t1_corrected,
            wm_threshold=0.96,
            csf_threshold=0.96,
            erosion_voxels=1,
        )
        val_res = validate_tissue_masks(tissue_masks, min_wm_voxels=10, min_csf_voxels=10)
        print(f"  -> Validated: {val_res['wm_voxel_count']} WM voxels, {val_res['csf_voxel_count']} CSF voxels, 0 overlap.")

        # -------------------------------------------------------------
        # Stage 11: EPI-to-T1w Rigid Coregistration
        # -------------------------------------------------------------
        current_stage = "Stage 11: Coregistration"
        print("\n[Stage 11/22] Coregistering functional reference to anatomical brain (DIPY Rigid)...")
        m_epi_to_t1w, _ = register_epi_to_t1w(mean_epi, t1_brain, level_iters=[10])
        print("  -> Coregistration rigid pullback matrix derived.")

        if sdc_res.corrected and sdc_displacement_vox is not None and sdc_res.unwarped_mean_3d is not None:
            epi_aff_64 = np.asarray(bold_img.affine, dtype=np.float64)
            t1_arr = np.asanyarray(t1_brain.dataobj, dtype=np.float32)
            xform_epi_to_t1_vox = (
                np.linalg.inv(np.asarray(t1_brain.affine, dtype=np.float64))
                @ np.linalg.inv(np.asarray(m_epi_to_t1w, dtype=np.float64))
                @ epi_aff_64
            )
            gi_e, gj_e, gk_e = np.indices(epi_shape, dtype=np.float64)
            h_e = np.vstack([gi_e.ravel(), gj_e.ravel(), gk_e.ravel(), np.ones(gi_e.size, dtype=np.float64)])
            c_t1 = (xform_epi_to_t1_vox @ h_e)[:3]
            t1_in_epi = map_coordinates(
                t1_arr, c_t1, order=1, mode="constant", cval=0.0, prefilter=False
            ).reshape(epi_shape).astype(np.float32)
            dist_mean_3d = np.mean(np.asanyarray(stc_bold_img.dataobj, dtype=np.float32), axis=3)
            sdc_res.correction_qc = evaluate_sdc_correction_qc(
                dist_mean_3d,
                sdc_res.unwarped_mean_3d,
                t1_in_epi,
            )
            sdc_provenance_summary["correction_qc"] = sdc_res.correction_qc
            qc_rel = sdc_res.derivative_files.get("report_json")
            if qc_rel:
                (sub_out_dir / qc_rel).write_text(
                    json.dumps(sdc_provenance_summary, indent=2), encoding="utf-8"
                )
            if sdc_res.field_hz is not None and sdc_res.derivative_files.get("qc_png"):
                from src.preprocessing.susceptibility_distortion import generate_sdc_qc_figure

                generate_sdc_qc_figure(
                    dist_mean_3d,
                    sdc_res.unwarped_mean_3d,
                    sdc_res.field_hz,
                    sub_out_dir / sdc_res.derivative_files["qc_png"],
                    reference_3d=t1_in_epi,
                )

        # -------------------------------------------------------------
        # Stage 12: T1w-to-MNI SyN Nonlinear Normalization
        # -------------------------------------------------------------
        current_stage = "Stage 12: Nonlinear Normalization"
        print("\n[Stage 12/22] Computing T1w-to-MNI SyN nonlinear diffeomorphic registration...")
        m_t1w_to_mni, syn_map = register_t1w_to_mni(
            t1_brain, mni_img, affine_level_iters=[10], syn_level_iters=[5]
        )
        print("  -> Diffeomorphic SyN mapping computed.")

        # -------------------------------------------------------------
        # Stage 13: Composed Pullback Resampling into CC200 Grid
        # -------------------------------------------------------------
        current_stage = "Stage 13: Composed 4D Resampling"
        print(
            f"\n[Stage 13/22] Composing motion{' + SDC' if sdc_displacement_vox is not None else ''} + "
            "coregistration + normalization and resampling STC BOLD into CC200 grid (single spatial interpolation)..."
        )
        target_shape = atlas_img.shape[:3]
        target_affine = atlas_img.affine
        epi_shape = bold_img.shape[:3]
        epi_affine = bold_img.affine

        composed_map = compose_transformations(
            rigid_epi_to_t1w=m_epi_to_t1w,
            affine_t1w_to_mni=m_t1w_to_mni,
            syn_mapping=syn_map,
            target_shape=target_shape,
            target_affine=target_affine,
            source_shape=epi_shape,
            source_affine=epi_affine,
        )

        if sdc_displacement_vox is not None:
            resampled_bold = resample_4d_bold(
                stc_bold_img,
                composed_map,
                target_shape=target_shape,
                target_affine=target_affine,
                interpolation="linear",
                max_memory_gb=max_memory_gb,
                motion_transforms=rigid_motion_transforms,
                sdc_displacement=sdc_displacement_vox,
            )
        else:
            resampled_bold = resample_4d_bold(
                stc_bold_img,
                composed_map,
                target_shape=target_shape,
                target_affine=target_affine,
                interpolation="linear",
                max_memory_gb=max_memory_gb,
                motion_transforms=rigid_motion_transforms,
            )
        print(f"  -> Resampled BOLD shape: {resampled_bold.shape} [MATCHES CC200 TARGET]")

        # -------------------------------------------------------------
        # Stage 14: Tissue Mask Standard Grid Resampling
        # -------------------------------------------------------------
        current_stage = "Stage 14: Tissue Mask Resampling"
        print("\n[Stage 14/22] Resampling eroded WM/CSF masks into CC200 grid via SyN...")
        wm_eroded_mni = resample_mask_to_target(
            tissue_masks["mask_wm_eroded"], syn_map, target_shape, target_affine
        )
        csf_eroded_mni = resample_mask_to_target(
            tissue_masks["mask_csf_eroded"], syn_map, target_shape, target_affine
        )
        print(f"  -> Standard space mask voxels: WM={int(np.count_nonzero(wm_eroded_mni.dataobj))}, "
              f"CSF={int(np.count_nonzero(csf_eroded_mni.dataobj))}")

        # -------------------------------------------------------------
        # Stage 15: aCompCor Physiological Noise Extraction
        # -------------------------------------------------------------
        current_stage = "Stage 15: aCompCor Extraction"
        print("\n[Stage 15/22] Extracting 5 physiological aCompCor components...")
        compcor_5 = extract_acompcor_components(
            resampled_bold, wm_eroded_mni, csf_eroded_mni, n_components=5
        )
        print(f"  -> aCompCor extracted: shape {compcor_5.shape}")

        # -------------------------------------------------------------
        # Stage 16: Friston-24 Motion Regressor Expansion
        # -------------------------------------------------------------
        current_stage = "Stage 16: Friston-24 Expansion"
        print("\n[Stage 16/22] Expanding 6 motion parameters to Friston-24...")
        friston_24 = build_friston_24_regressors(motion_params)
        print(f"  -> Friston-24 shape: {friston_24.shape}")

        # -------------------------------------------------------------
        # Stage 17: Confound Matrix Assembly & Collinearity Check
        # -------------------------------------------------------------
        current_stage = "Stage 17: Confound Matrix Assembly"
        print("\n[Stage 17/22] Assembling 31-column nuisance confound matrix...")
        confounds_df = build_confound_matrix(friston_24, compcor_5)
        # Compute condition number & check rank
        conf_vals = confounds_df.to_numpy()
        matrix_rank = int(np.linalg.matrix_rank(conf_vals))
        cond_num = float(np.linalg.cond(conf_vals))
        print(f"  -> Confound Matrix: {confounds_df.shape[1]} columns | Rank: {matrix_rank}/31 | Cond: {cond_num:.2f}")

        # -------------------------------------------------------------
        # Stage 18: Simultaneous Nuisance Regression & Bandpass
        # -------------------------------------------------------------
        current_stage = "Stage 18: Nuisance Regression"
        print("\n[Stage 18/22] Executing simultaneous regression & Butterworth bandpass [0.01-0.10 Hz]...")
        cleaned_bold = clean_functional_bold(
            resampled_bold, confounds_df, tr=tr, low_cutoff_hz=0.01, high_cutoff_hz=0.10
        )
        print(f"  -> Cleaned BOLD shape: {cleaned_bold.shape} (GSR=OFF, TissueMean=OFF)")

        # -------------------------------------------------------------
        # Stage 19: CC200 ROI Time Series Extraction
        # -------------------------------------------------------------
        current_stage = "Stage 19: CC200 Parcellation"
        print("\n[Stage 19/22] Extracting CC200 ROI time series across 200 parcels...")
        roi_df, signals = extract_cc200_timeseries(
            cleaned_bold,
            atlas_img,
            censor_volumes=censored_sorted if censor_volumes is not None else None,
        )
        t_retained = int(signals.shape[0])
        print(
            f"  -> Timeseries DataFrame: {roi_df.shape} (CSV: 200 ROIs x ['node/volume' + {t_retained} timepoints]; "
            f"time-axis shape: (200, {t_retained}))"
        )

        # -------------------------------------------------------------
        # Stage 20: Pairwise Fisher-z Connectivity Vector Calculation
        # -------------------------------------------------------------
        current_stage = "Stage 20: Connectivity Features"
        print("\n[Stage 20/22] Computing 19,900 Fisher-z connectivity features...")
        z_vector, edge_names = compute_connectivity_vector(signals)
        print(f"  -> Fisher-z edges: {len(z_vector)} | Finite: {np.all(np.isfinite(z_vector))} | Range: [{np.min(z_vector):.3f}, {np.max(z_vector):.3f}]")

        # -------------------------------------------------------------
        # Stage 21: Quality Control Visual Overlay Generation
        # -------------------------------------------------------------
        current_stage = "Stage 21: QC Visual Overlays"
        print("\n[Stage 21/22] Generating 4 orthogonal quality control overlays...")

        # 1. Coregistration QC Overlay
        t1_shape = t1_brain.shape[:3]
        t1_affine = t1_brain.affine
        aff_epi_to_t1w = AffineMap(
            m_epi_to_t1w,
            domain_grid_shape=t1_shape,
            domain_grid2world=t1_affine,
            codomain_grid_shape=epi_shape,
            codomain_grid2world=epi_affine,
        )
        epi_resampled_t1w = nib.Nifti1Image(
            aff_epi_to_t1w.transform(np.asanyarray(mean_epi.dataobj), interpolation="linear"),
            t1_affine,
        )
        qc_coreg_path = sub_out_dir / "qc_registration_epi_to_t1w.png"
        generate_coregistration_qc_overlay(t1_brain, epi_resampled_t1w, qc_coreg_path)

        # 2. Normalization QC Overlay
        t1w_resampled_mni = nib.Nifti1Image(
            syn_map.transform(
                np.asanyarray(t1_corrected.dataobj),
                interpolation="linear",
                out_shape=target_shape,
                out_grid2world=target_affine,
            ),
            target_affine,
        )
        qc_norm_path = sub_out_dir / "qc_registration_t1w_to_mni.png"
        generate_normalization_qc_overlay(mni_img, t1w_resampled_mni, qc_norm_path)

        # 3. Tissue Mask QC Overlay
        qc_tissue_path = sub_out_dir / "qc_tissue_masks.png"
        generate_tissue_mask_qc_overlay(
            t1_brain,
            tissue_masks["mask_wm_eroded"],
            tissue_masks["mask_csf_eroded"],
            qc_tissue_path,
        )

        # 4. CC200 Parcellation QC Overlay
        mean_cleaned = nib.Nifti1Image(
            np.mean(np.asanyarray(cleaned_bold.dataobj), axis=3), target_affine
        )
        qc_parcellation_path = sub_out_dir / "qc_parcellation_cc200.png"
        generate_parcellation_qc_overlay(
            mean_cleaned, atlas_img, qc_parcellation_path, title="QC: CC200 Parcellation Overlay"
        )
        print("  -> Overlays saved: Coregistration, Normalization, Tissue Masks, Parcellation.")

        # -------------------------------------------------------------
        # File Exports
        # -------------------------------------------------------------
        print("\n[Exporting Output Artifacts]...")
        # 1. CC200 ROI Time Series CSV
        roi_csv_path = sub_out_dir / "roi_CC200.csv"
        roi_df.to_csv(roi_csv_path, index=False)

        # 2. 19,900 Fisher-z Connectivity Features
        feat_npy_path = sub_out_dir / "features_19900_fisher_z.npy"
        np.save(feat_npy_path, z_vector)
        feat_df = pd.DataFrame({"feature_name": edge_names, "fisher_z": z_vector})
        feat_csv_path = sub_out_dir / "features_19900_fisher_z.csv"
        feat_df.to_csv(feat_csv_path, index=False)

        # Downstream ML Model Inference (optional by default, mandatory when require_model_inference=True)
        if frozen_pipe is not None and frozen_clf is not None:
            print("\n[Downstream ML Integration] Executing zero-refit frozen model inference...")
            from src.integration.model_verifier import run_frozen_inference
            pred_res = run_frozen_inference(z_vector, frozen_pipe, frozen_clf)
            pred_json_path = sub_out_dir / "model_prediction.json"
            with open(pred_json_path, "w", encoding="utf-8") as f:
                json.dump(pred_res, f, indent=2)
            model_prediction_path = pred_json_path
            print(f"  -> Frozen Model Prediction: {pred_res['class_label']} (ADHD: {pred_res['adhd_probability']:.4f}, Control: {pred_res['control_probability']:.4f})")

        # 3. Motion & Confound Reports
        motion_cols = ["trans_x_mm", "trans_y_mm", "trans_z_mm", "rot_x_deg", "rot_y_deg", "rot_z_deg"]
        motion_df = pd.DataFrame(motion_params, columns=motion_cols)
        motion_txt_path = sub_out_dir / "motion_parameters.tsv"
        motion_df.to_csv(motion_txt_path, sep="\t", index=False)

        fd_df = pd.DataFrame({"framewise_displacement_mm": fd_vector})
        fd_1d_path = sub_out_dir / "framewise_displacement.tsv"
        fd_df.to_csv(fd_1d_path, sep="\t", index=False)

        confounds_path = sub_out_dir / "confounds.tsv"
        confounds_df.to_csv(confounds_path, sep="\t", index=False)

        confound_report_path = sub_out_dir / "confound_report.json"
        with open(confound_report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_columns": int(confounds_df.shape[1]),
                    "matrix_rank": matrix_rank,
                    "condition_number": cond_num,
                    "collinear_warning": cond_num > 1e6 or matrix_rank < confounds_df.shape[1],
                },
                f,
                indent=2,
            )

        # -------------------------------------------------------------
        # Stage 22: Schema-Compliant Provenance JSON Generation
        # -------------------------------------------------------------
        current_stage = "Stage 22: Provenance Reporting"
        print("\n[Stage 22/22] Generating provenance metadata and validating schema...")
        runtime_sec = float(time.time() - start_time)
        peak_rss = check_process_memory_gb()

        import dipy
        import nilearn
        import scipy
        import sklearn
        import jsonschema

        # 1. Compute Cryptographic Checksums for Output Artifacts
        output_checksums: Dict[str, str] = {
            "roi_csv": compute_sha256(roi_csv_path),
            "features_csv": compute_sha256(feat_csv_path),
            "features_npy": compute_sha256(feat_npy_path),
            "motion_tsv": compute_sha256(motion_txt_path),
            "fd_tsv": compute_sha256(fd_1d_path),
            "confounds_tsv": compute_sha256(confounds_path),
            "confound_report_json": compute_sha256(confound_report_path),
            "qc_coreg_png": compute_sha256(qc_coreg_path),
            "qc_norm_png": compute_sha256(qc_norm_path),
            "qc_tissue_png": compute_sha256(qc_tissue_path),
            "qc_parcellation_png": compute_sha256(qc_parcellation_path),
        }
        if model_prediction_path is not None:
            output_checksums["model_prediction"] = compute_sha256(model_prediction_path)

        # 2. Measurable Assertions for Validation Gates
        n_rois = int(signals.shape[1])
        constant_rois = int(np.count_nonzero(np.std(signals, axis=0, ddof=0) == 0.0))

        cgroup_stats = get_cgroup_memory_stats()
        cgroup_peak_bytes = cgroup_stats.get("memory_peak_bytes")
        cgroup_peak_gb = (cgroup_peak_bytes / (1024.0 * 1024.0 * 1024.0)) if cgroup_peak_bytes else None

        gate1_passed = bool(
            peak_rss <= max_memory_gb
            and (not enforce_container_memory or (os_memory_info["enforced"] and (cgroup_peak_gb is None or cgroup_peak_gb <= max_memory_gb)))
        )
        gate2_passed = bool(tr > 0.0 and t_vol >= 50 and all(d > 0 for d in epi_shape))
        gate3_passed = bool(len(target_shape) == 3 and all(d > 0 for d in target_shape))
        affine_aligned = bool(np.allclose(resampled_bold.affine, target_affine, atol=1e-4))
        gate4_passed = bool(resampled_bold.shape[:3] == target_shape and affine_aligned)
        gate5_passed = bool(n_rois == 200 and constant_rois == 0)
        gate6_passed = bool(
            len(z_vector) == 19900
            and np.all(np.isfinite(z_vector))
            and (edge_names == get_feature_names())
        )

        validation_gates = {
            "gate1_resources": {
                "status": "PASSED" if gate1_passed else "FAILED",
                "peak_rss_gb": round(peak_rss, 4),
                "memory_ceiling_gb": float(max_memory_gb),
                "runtime_seconds": round(runtime_sec, 2),
            },
            "gate2_bids_input": {
                "status": "PASSED" if gate2_passed else "FAILED",
                "tr_seconds": float(tr),
                "t_vol": int(t_vol),
            },
            "gate3_atlas_grid": {
                "status": "PASSED" if gate3_passed else "FAILED",
                "atlas_shape": [int(x) for x in target_shape],
            },
            "gate4_spatial_alignment": {
                "status": "PASSED" if gate4_passed else "FAILED",
                "resampled_shape": [int(x) for x in resampled_bold.shape],
            },
            "gate5_roi_coverage": {
                "status": "PASSED" if gate5_passed else "FAILED",
                "n_rois": int(n_rois),
                "zero_variance_rois": int(constant_rois),
                "n_timepoints_input": int(t_vol),
                "n_timepoints_retained": int(signals.shape[0]),
                "roi_timeseries_shape": [int(n_rois), int(signals.shape[0])],
                "roi_csv_shape": [int(roi_df.shape[0]), int(roi_df.shape[1])],
                "roi_label_column": "node/volume",
                "volume_censoring_applied": bool(len(censored_sorted) > 0),
                "censored_volumes": [int(v) for v in censored_sorted],
            },
            "gate6_feature_schema_conformance": {
                "status": "PASSED" if gate6_passed else "FAILED",
                "n_features": int(len(z_vector)),
                "finite_features": bool(np.all(np.isfinite(z_vector))),
                "edge_names_ordered": bool(edge_names == get_feature_names()),
                "historical_concordance_assessed": False,
                "historical_concordance_status": "NOT_ASSESSED",
            },
        }

        output_artifacts_dict = {
            "roi_csv": str(roi_csv_path.name),
            "features_csv": str(feat_csv_path.name),
            "features_npy": str(feat_npy_path.name),
            "motion_tsv": str(motion_txt_path.name),
            "fd_tsv": str(fd_1d_path.name),
            "confounds_tsv": str(confounds_path.name),
            "confound_report_json": str(confound_report_path.name),
            "qc_coreg_png": str(qc_coreg_path.name),
            "qc_norm_png": str(qc_norm_path.name),
            "qc_tissue_png": str(qc_tissue_path.name),
            "qc_parcellation_png": str(qc_parcellation_path.name),
            "provenance_json": "provenance.json",
        }
        if model_prediction_path is not None:
            output_artifacts_dict["model_prediction"] = model_prediction_path.name

        provenance_dict: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "subject_id": subject_id,
            "site": "SYNTHETIC_BENCHMARK",
            "cohort_split": "synthetic",
            "pipeline_stages_total": 22,
            "preprocessing_order": [
                "1. BIDS Input & Sidecar Metadata Validation",
                "2. 3D Rigid Motion Parameter Estimation (Raw Uncorrected BOLD)",
                "3. Metadata-Driven Slice-Timing Correction (DEC-STC-01; Original Slice Geometry)",
                "4. B0 Susceptibility Distortion Correction (DEC-SDC-02; Fieldmap Discovery & Estimation)",
                "5. Motion Transform (+ Optional SDC) Application (Native Reference Generation)",
                "6. Functional Reference Generation",
                "7. Anatomical Brain Extraction",
                "8. Log-Polynomial Bias-Field Correction",
                "9. Tissue Segmentation (CSF, GM, WM)",
                "10. Morphological Tissue Mask Erosion & Quality Verification",
                "11. EPI-to-T1w 3D Rigid Coregistration",
                "12. T1w-to-MNI SyN Nonlinear Normalization",
                "13. Composed Single-Step Pullback Resampling (Motion + SDC + Coreg + MNI + SyN on STC BOLD)",
                "14. Tissue Mask Standard Grid Resampling",
                "15. aCompCor Physiological Noise Extraction (5 Components)",
                "16. Friston-24 Motion Regressor Expansion",
                "17. Confound Matrix Assembly & Collinearity Check",
                "18. Simultaneous Nuisance Regression & Bandpass [0.01-0.10 Hz]",
                "19. CC200 ROI Time Series Extraction (200 Parcels)",
                "20. Pairwise Fisher-z Connectivity Vector (19,900 Edges)",
                "21. Quality Control Visual Overlays (4 Views)",
                "22. Schema-Compliant Provenance JSON Generation",
            ],
            "motion_estimation": {
                "estimated_on_raw_uncorrected_bold": True,
                "estimated_before_slice_timing": True,
                "applied_after_slice_timing": True,
                "single_step_composed_resampling_with_stc_bold": True,
                "single_step_composed_resampling_with_sdc": bool(sdc_displacement_vox is not None),
                "mean_fd_mm": round(mean_fd, 6),
                "max_fd_mm": round(max_fd, 6),
                "volumes_fd_gt_0p5mm": scrubbed,
            },
            "software_versions": {
                "dipy": str(dipy.__version__),
                "nilearn": str(nilearn.__version__),
                "nibabel": str(nib.__version__),
                "numpy": str(np.__version__),
                "scipy": str(scipy.__version__),
                "scikit-learn": str(sklearn.__version__),
                "python": str(sys.version.split()[0]),
            },
            "hardware_execution": {
                "start_rss_gb": round(start_rss, 4),
                "peak_rss_gb": round(peak_rss, 4),
                "peak_memory_ceiling_gb": float(max_memory_gb),
                "runtime_seconds": round(runtime_sec, 2),
                "cgroup_memory": {
                    "enforced": os_memory_info["enforced"],
                    "status": os_memory_info["status"],
                    "cgroup_version": os_memory_info.get("cgroup_version"),
                    "memory_max_mib": os_memory_info.get("memory_max_mib"),
                    "memory_peak_mib": cgroup_stats.get("memory_peak_mib"),
                },
            },
            "slice_timing_correction": stc_provenance_summary,
            "susceptibility_distortion_correction": sdc_provenance_summary,
            "validation_gates": validation_gates,
            "output_artifacts": output_artifacts_dict,
            "output_checksums": output_checksums,
        }

        # 3. Mandatory Schema Validation using jsonschema
        schema_path = PROJECT_ROOT / "configs" / "provenance_schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(
                f"CRITICAL ERROR: Mandatory provenance schema file missing at '{schema_path}'. "
                "Execution cannot be validated and must fail."
            )
        with open(schema_path, "r", encoding="utf-8") as sf:
            prov_schema = json.load(sf)
        try:
            jsonschema.validate(instance=provenance_dict, schema=prov_schema)
            print("  -> Provenance validated against configs/provenance_schema.json [PASSED]")
        except jsonschema.ValidationError as s_err:
            raise ValueError(f"Provenance schema validation failed: {s_err.message}") from s_err

        provenance_path = sub_out_dir / "provenance.json"
        with open(provenance_path, "w", encoding="utf-8") as f:
            json.dump(provenance_dict, f, indent=2)

        # 4. Generate Output Manifest covering all 12 files including provenance.json
        output_manifest = {
            "status": "SUCCESS",
            "subject_id": subject_id,
            "timestamp_utc": provenance_dict["timestamp_utc"],
            "correction_status": {
                "slice_timing_correction": {
                    "status": str(stc_provenance_summary["status"]),
                    "corrected": bool(stc_provenance_summary["corrected"]),
                    "reason": stc_provenance_summary.get("reason"),
                },
                "susceptibility_distortion_correction": {
                    "status": str(sdc_provenance_summary["status"]),
                    "corrected": bool(sdc_provenance_summary["corrected"]),
                    "reason": sdc_provenance_summary.get("reason"),
                },
            },
            "artifacts": {
                key: {
                    "filename": fname,
                    "sha256": compute_sha256(sub_out_dir / fname),
                }
                for key, fname in output_artifacts_dict.items()
            },
        }
        manifest_out_path = sub_out_dir / "output_manifest.json"
        with open(manifest_out_path, "w", encoding="utf-8") as mf:
            json.dump(output_manifest, mf, indent=2)

        # 5. Post-write Output Checksum Integrity Verification
        # Validate that all 12 files on disk match their hashes in output_manifest
        for art_key, art_entry in output_manifest["artifacts"].items():
            disk_file = sub_out_dir / art_entry["filename"]
            disk_hash = compute_sha256(disk_file)
            if disk_hash != art_entry["sha256"]:
                raise RuntimeError(
                    f"Output artifact verification failed: hash mismatch on '{art_entry['filename']}'. "
                    "File corrupted or altered after generation."
                )

        # 6. Check validation gate failures
        for gate_name, gate_info in validation_gates.items():
            if gate_info["status"] != "PASSED":
                raise RuntimeError(f"Pipeline validation failed at gate '{gate_name}': {gate_info}")

        # Cleanup scratch directory
        shutil.rmtree(scratch_dir, ignore_errors=True)

        # 7. Finalize Run Directory from staging to verified successful output
        final_out_dir = output_dir / subject_id / f"run_{run_tag}"
        run_manifest = {
            "status": "SUCCESS",
            "subject_id": subject_id,
            "timestamp_utc": provenance_dict["timestamp_utc"],
            "runtime_seconds": runtime_sec,
            "peak_rss_gb": peak_rss,
            "os_memory_isolation": {
                "enforced": os_memory_info["enforced"],
                "status": os_memory_info["status"],
                "cgroup_version": os_memory_info.get("cgroup_version"),
                "memory_max_mib": os_memory_info.get("memory_max_mib"),
                "memory_peak_mib": cgroup_stats.get("memory_peak_mib"),
            },
        }
        with open(sub_out_dir / "run_manifest.json", "w", encoding="utf-8") as rf:
            json.dump(run_manifest, rf, indent=2)

        sub_out_dir.rename(final_out_dir)
        sub_out_dir = final_out_dir

        print("\n" + "=" * 75)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print(f"Total Runtime : {runtime_sec:.2f} seconds")
        print(f"Peak Memory   : {peak_rss:.2f} GB (Ceiling: {max_memory_gb:.2f} GB)")
        print(f"Results Path  : {sub_out_dir}")
        print("=" * 75)

        return {
            "status": "SUCCESS",
            "subject_id": subject_id,
            "output_dir": sub_out_dir,
            "runtime_seconds": runtime_sec,
            "peak_rss_gb": peak_rss,
            "manifest_path": sub_out_dir / "output_manifest.json",
            "run_manifest_path": sub_out_dir / "run_manifest.json",
            "artifacts": {
                "roi_csv": sub_out_dir / "roi_CC200.csv",
                "features_csv": sub_out_dir / "features_19900_fisher_z.csv",
                "features_npy": sub_out_dir / "features_19900_fisher_z.npy",
                "motion_tsv": sub_out_dir / "motion_parameters.tsv",
                "fd_tsv": sub_out_dir / "framewise_displacement.tsv",
                "confounds_tsv": sub_out_dir / "confounds.tsv",
                "confound_report_json": sub_out_dir / "confound_report.json",
                "qc_coreg_png": sub_out_dir / "qc_registration_epi_to_t1w.png",
                "qc_norm_png": sub_out_dir / "qc_registration_t1w_to_mni.png",
                "qc_tissue_png": sub_out_dir / "qc_tissue_masks.png",
                "qc_parcellation_png": sub_out_dir / "qc_parcellation_cc200.png",
                "provenance_json": sub_out_dir / "provenance.json",
            },
            "derivatives_dir": sub_out_dir / "derivatives",
            "slice_timing": stc_provenance_summary,
            "susceptibility_distortion": sdc_provenance_summary,
            "provenance": provenance_dict,
        }

    except Exception as exc:
        print(f"\n[PIPELINE FAILURE] Error occurred during {current_stage}: {exc}", file=sys.stderr)
        failure_report = {
            "status": "FAILED",
            "stage": current_stage,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        shutil.rmtree(scratch_dir, ignore_errors=True)
        # Record failure report and failed run manifest in staging directory
        try:
            with open(sub_out_dir / "pipeline_failure_report.json", "w", encoding="utf-8") as f:
                json.dump(failure_report, f, indent=2)
            with open(sub_out_dir / "run_manifest.json", "w", encoding="utf-8") as rf:
                json.dump(failure_report, rf, indent=2)
        except Exception:
            pass

        # Quarantine staging directory as failed_run_{run_tag}
        failed_out_dir = output_dir / subject_id / f"failed_run_{run_tag}"
        try:
            if sub_out_dir.exists():
                sub_out_dir.rename(failed_out_dir)
        except Exception:
            failed_out_dir = sub_out_dir

        raise exc


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="NeuroADHD RawMRI PoC V2 Single-Subject Preprocessing Pipeline Runner."
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic"],
        default="synthetic",
        help="Pipeline execution mode. Only 'synthetic' mode is enabled by default for data safety.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_poc_v2",
        help="Root destination directory for output artifacts.",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        default="sub-synthetic001",
        help="Subject identifier (must begin with 'sub-synthetic' unless real scans are explicitly permitted).",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="ses-1",
        help="Session identifier.",
    )
    parser.add_argument(
        "--t-vol",
        type=int,
        default=60,
        help="Number of functional BOLD volumes to simulate in synthetic mode (min: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic synthetic phantom generation.",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=2.5,
        help="Resident set size memory ceiling in gigabytes (default: 2.5 GB).",
    )
    parser.add_argument(
        "--allow-real-scans",
        action="store_true",
        default=False,
        help="Safety override flag. Processing real participant scans remains blocked.",
    )
    parser.add_argument(
        "--atlas-mode",
        choices=["synthetic_phantom", "canonical"],
        default="synthetic_phantom",
        help="Atlas validation policy: 'synthetic_phantom' verifies against authenticated session manifest, 'canonical' verifies canonical CC200 checksum and geometry.",
    )
    parser.add_argument(
        "--canonical-atlas-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "atlases" / "cc200_roi_atlas.nii.gz",
        help="Path to authentic canonical CC200 atlas (default: data/atlases/cc200_roi_atlas.nii.gz).",
    )
    parser.add_argument(
        "--enforce-container-memory",
        action="store_true",
        default=False,
        help="Fail closed if operating-system cgroup memory limit is unconfigured or exceeds ceiling.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional directory containing frozen model artifacts (final_preprocessing_pipeline.joblib, final_model.joblib).",
    )
    parser.add_argument(
        "--require-model-inference",
        action="store_true",
        default=False,
        help="Require downstream model inference to execute; fail closed if model artifacts are missing or unverified.",
    )
    parser.add_argument(
        "--stc-reference-time",
        type=float,
        default=None,
        help="Optional slice-timing target reference time in seconds within [0, TR). Defaults to TR/2.",
    )
    parser.add_argument(
        "--stc-interpolation",
        choices=["cubic_spline", "quintic_bspline", "fourier", "linear"],
        default="cubic_spline",
        help="Slice-timing temporal interpolation method (default: cubic_spline).",
    )
    parser.add_argument(
        "--strict-slice-timing",
        action="store_true",
        default=False,
        help="Fail closed if SliceTiming metadata is missing instead of skipping STC cleanly.",
    )
    parser.add_argument(
        "--sdc-method",
        choices=["auto", "pepolar_topup", "direct_fieldmap", "phasediff", "fieldmap_less_syn", "none"],
        default="auto",
        help="B0 susceptibility distortion correction method selection (default: auto).",
    )
    parser.add_argument(
        "--strict-sdc",
        action="store_true",
        default=False,
        help="Fail closed if B0 susceptibility distortion correction cannot be applied cleanly.",
    )

    args = parser.parse_args()

    # Enforce Phase A.6 research authorization gate on CLI execution
    auth_gate = ResearchAuthorizationGate(base_dir=PROJECT_ROOT)
    auth_gate.check_execution_allowed(
        requested_mode=args.mode,
        allow_real_scans=args.allow_real_scans,
    )

    if args.allow_real_scans:
        raise AuthorizationDeniedError(
            "SAFETY VIOLATION: Execution on real MRI scans is strictly disabled by Phase A safety policy. "
            "--allow-real-scans is not permitted."
        )

    if args.mode == "synthetic":
        print(f"[Synthetic Generation] Building synthetic BIDS session for {args.subject_id}...")
        phantom_scratch = args.output_dir / "synthetic_inputs"
        phantom_scratch.mkdir(parents=True, exist_ok=True)
        canonical_atlas = args.canonical_atlas_path if args.atlas_mode == "canonical" else None
        synthetic_paths = generate_synthetic_phantom_session(
            phantom_scratch,
            subject_id=args.subject_id,
            session_id=args.session_id,
            t_vol=args.t_vol,
            seed=args.seed,
            canonical_atlas_path=canonical_atlas,
        )

        res = run_poc_v2_pipeline(
            bold_path=synthetic_paths["bold_nii"],
            bold_json_path=synthetic_paths["bold_json"],
            t1w_path=synthetic_paths["t1w_nii"],
            atlas_path=synthetic_paths["cc200_atlas"],
            template_path=synthetic_paths["mni_template"],
            output_dir=args.output_dir,
            subject_id=args.subject_id,
            session_id=args.session_id,
            max_memory_gb=args.max_memory_gb,
            allow_real_scans=False,
            manifest_path=synthetic_paths.get("synthetic_manifest"),
            atlas_mode=args.atlas_mode,
            enforce_container_memory=args.enforce_container_memory,
            model_dir=args.model_dir,
            require_model_inference=args.require_model_inference,
            slice_timing_reference_time=args.stc_reference_time,
            slice_timing_interpolation=args.stc_interpolation,
            allow_missing_slice_timing=not args.strict_slice_timing,
            sdc_method=args.sdc_method,
            strict_sdc=args.strict_sdc,
        )
        print(f"\nExecution successful. Outputs verified at: {res['output_dir']}")
    else:
        raise NotImplementedError(f"Mode '{args.mode}' is not permitted.")


if __name__ == "__main__":
    main()
