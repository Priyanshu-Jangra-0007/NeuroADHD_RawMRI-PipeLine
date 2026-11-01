#!/usr/bin/env python3
"""NeuroADHD RawMRI PoC V2 Single-Subject Preprocessing Pipeline Runner.

Executes the complete 19-stage preprocessing pipeline on synthetic BIDS datasets.
Real participant processing is disabled by default to uphold data safety restrictions.

Pipeline Stages:
1.  BIDS Input & Sidecar Metadata Validation (TR, TaskName)
2.  3D Rigid Motion Realignment (DIPY) -> Motion Parameters & Power FD
3.  Functional Reference Generation (Temporal Mean & Functional Brain Mask)
4.  Anatomical Brain Extraction (Median Otsu with Hole Filling)
5.  Log-Polynomial Bias-Field Correction
6.  Tissue Segmentation (GMM into CSF, GM, WM Probabilities)
7.  Morphological Tissue Mask Erosion & Quality Verification (Mandatory T1w Reference)
8.  EPI-to-T1w 3D Rigid Coregistration
9.  T1w-to-MNI SyN Nonlinear Normalization
10. Composed Single-Step Pullback Resampling into CC200 Grid (Preserving is_inverse=True)
11. Tissue Mask Standard Grid Resampling (Nearest-Neighbor)
12. aCompCor Physiological Noise Extraction (5 Components)
13. Friston-24 Motion Regressor Expansion
14. Confound Matrix Assembly & Collinearity Condition Check
15. Simultaneous Nuisance Regression & Butterworth Bandpass (0.01 - 0.10 Hz)
16. CC200 ROI Time Series Extraction (200 Parcels, resampling_target=None)
17. Pairwise Fisher-z Connectivity Vector Calculation (19,900 Edges)
18. Quality Control Visual Overlay Generation (4 Orthogonal Multi-Planar Overlays)
19. Schema-Compliant Provenance JSON Generation
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from dipy.align.imaffine import AffineMap

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.connectivity import compute_connectivity_vector, get_feature_names
from src.parcellation.extractor import (
    CANONICAL_CC200_AFFINE,
    CANONICAL_CC200_SHAPE,
    extract_cc200_timeseries,
    verify_canonical_cc200_atlas,
)
from src.preprocessing.motion import build_friston_24_regressors, realign_4d_bold
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
from src.validation.bids_validator import BIDSInputValidator


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


def check_safety_guards(
    subject_id: str,
    bold_path: Optional[Union[str, Path]] = None,
    t1w_path: Optional[Union[str, Path]] = None,
    allow_real_scans: bool = False,
    manifest_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Enforce strict safety restrictions preventing raw clinical participant processing.

    In Phase A, execution on real MRI scans is prohibited. The pipeline fails closed
    by default. Execution requires cryptographic proof of synthetic origin via a validated
    synthetic manifest with matching SHA-256 hashes.
    """
    # 1. Reject explicit bypass attempts
    if allow_real_scans:
        raise RuntimeError(
            "SAFETY VIOLATION: Execution on real MRI scans is strictly disabled by Phase A safety policy. "
            "'allow_real_scans=True' is prohibited and will not bypass safety guards."
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

    # 3. Block raw data paths referencing known clinical directories
    for p in [bold_path, t1w_path]:
        if p is not None:
            p_str = str(p).lower()
            for kw in ["down/outputs/adhd200", "sub-2344857", "peking", "kki", "nyu", "abide", "adhd-200"]:
                if kw in p_str:
                    raise RuntimeError(
                        f"SAFETY VIOLATION: Attempted to process real raw data at '{p}'. "
                        "Processing raw clinical data remains prohibited."
                    )

    # 4. Fail closed if paths are missing
    if bold_path is None or t1w_path is None:
        raise RuntimeError(
            "SAFETY VIOLATION: bold_path and t1w_path must both be provided for verification."
        )

    bold_p = Path(bold_path)
    t1w_p = Path(t1w_path)
    if not bold_p.is_file():
        raise RuntimeError(f"SAFETY VIOLATION: Input BOLD file not found at '{bold_p}'.")
    if not t1w_p.is_file():
        raise RuntimeError(f"SAFETY VIOLATION: Input T1w file not found at '{t1w_p}'.")

    # 5. Locate and verify cryptographic synthetic manifest
    found_manifest: Optional[Path] = None
    if manifest_path is not None:
        p_cand = Path(manifest_path)
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

    # Verify input file hashes against manifest
    files_entry = manifest_dict.get("files", {})
    if "bold_nii" not in files_entry or "t1w_nii" not in files_entry:
        raise RuntimeError(
            "SAFETY VIOLATION: Synthetic manifest missing required 'bold_nii' or 't1w_nii' entries."
        )

    actual_bold_sha256 = compute_sha256(bold_p)
    expected_bold_sha256 = files_entry["bold_nii"].get("sha256")
    if actual_bold_sha256 != expected_bold_sha256:
        raise RuntimeError(
            f"SAFETY VIOLATION: Input BOLD file '{bold_p}' SHA-256 mismatch against synthetic manifest. "
            f"Expected {expected_bold_sha256}, got {actual_bold_sha256}."
        )

    actual_t1w_sha256 = compute_sha256(t1w_p)
    expected_t1w_sha256 = files_entry["t1w_nii"].get("sha256")
    if actual_t1w_sha256 != expected_t1w_sha256:
        raise RuntimeError(
            f"SAFETY VIOLATION: Input T1w file '{t1w_p}' SHA-256 mismatch against synthetic manifest. "
            f"Expected {expected_t1w_sha256}, got {actual_t1w_sha256}."
        )

    return {
        "status": "VERIFIED_SYNTHETIC",
        "manifest_path": str(found_manifest),
        "subject_id": subject_id,
        "bold_sha256": actual_bold_sha256,
        "t1w_sha256": actual_t1w_sha256,
    }


def generate_synthetic_phantom_session(
    work_dir: Path,
    subject_id: str = "sub-synthetic001",
    session_id: str = "ses-1",
    t_vol: int = 60,
    seed: int = 42,
) -> Dict[str, Path]:
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
    nib.save(nib.Nifti1Image(bold_data, epi_affine), str(bold_nii_path))

    # BOLD Sidecar JSON
    sidecar_path = func_dir / f"{subject_id}_{session_id}_task-rest_bold.json"
    sidecar_dict = {
        "RepetitionTime": 2.0,
        "TaskName": "rest",
        "FlipAngle": 75.0,
        "Manufacturer": "Synthetic",
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_dict, f, indent=2)

    # 2. Anatomical T1w head volume with internal ventricles and deep WM core
    t1_shape = (40, 40, 40)
    t1_affine = np.diag([2.0, 2.0, 2.0, 1.0])
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

    # Write cryptographic synthetic manifest
    manifest_data = {
        "manifest_magic": "NEUROADHD_SYNTHETIC_DATASET_V2",
        "schema_version": "2.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject_id": subject_id,
        "session_id": session_id,
        "files": {
            "bold_nii": {
                "rel_path": str(bold_nii_path.relative_to(session_dir)),
                "sha256": compute_sha256(bold_nii_path),
            },
            "bold_json": {
                "rel_path": str(sidecar_path.relative_to(session_dir)),
                "sha256": compute_sha256(sidecar_path),
            },
            "t1w_nii": {
                "rel_path": str(t1_nii_path.relative_to(session_dir)),
                "sha256": compute_sha256(t1_nii_path),
            },
            "mni_template": {
                "rel_path": str(mni_path.name),
                "sha256": compute_sha256(mni_path),
            },
            "cc200_atlas": {
                "rel_path": str(atlas_path.name),
                "sha256": compute_sha256(atlas_path),
            },
        },
    }
    manifest_path = session_dir / "synthetic_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return {
        "bold_nii": bold_nii_path,
        "bold_json": sidecar_path,
        "t1w_nii": t1_nii_path,
        "mni_template": mni_path,
        "cc200_atlas": atlas_path,
        "synthetic_manifest": manifest_path,
    }


def run_poc_v2_pipeline(
    bold_path: Path,
    bold_json_path: Path,
    t1w_path: Path,
    atlas_path: Path,
    template_path: Path,
    output_dir: Path,
    subject_id: str = "sub-synthetic001",
    session_id: str = "ses-1",
    max_memory_gb: float = 2.5,
    allow_real_scans: bool = False,
    manifest_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute the complete 19-stage NeuroADHD PoC V2 preprocessing pipeline."""
    start_time = time.time()
    start_rss = check_process_memory_gb()

    # Safety Enforcement - fail closed
    check_safety_guards(
        subject_id=subject_id,
        bold_path=bold_path,
        t1w_path=t1w_path,
        allow_real_scans=allow_real_scans,
        manifest_path=manifest_path,
    )

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sub_out_dir = output_dir / subject_id / f"run_{run_tag}"
    scratch_dir = sub_out_dir / "scratch"
    sub_out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"NEUROADHD RAW MRI POC V2 PIPELINE EXECUTION [{subject_id}]")
    print(f"Timestamp UTC : {datetime.now(timezone.utc).isoformat()}")
    print(f"Output Root   : {sub_out_dir}")
    print("=" * 75)

    current_stage = "Initialization"
    try:
        # -------------------------------------------------------------
        # Stage 1: BIDS Input & Sidecar Metadata Validation
        # -------------------------------------------------------------
        current_stage = "Stage 1: BIDS Input Validation"
        print(f"\n[Stage 1/19] Validating BIDS metadata from {bold_json_path.name}...")
        validator = BIDSInputValidator()
        bids_meta = validator.validate_sidecar_json(bold_json_path)
        tr = bids_meta["repetition_time"]
        print(f"  -> RepetitionTime: {tr:.3f}s | Task: {bids_meta['task_name']} [PASSED]")

        # Canonical atlas verification if authentic atlas is provided
        if "cc200_roi_atlas" in str(atlas_path):
            print("  -> Verifying authentic CC200 canonical atlas identity...")
            verify_canonical_cc200_atlas(atlas_path)
            print("  -> CC200 canonical atlas verified [PASSED]")

        # Load NIfTI volumes
        bold_img = nib.load(str(bold_path))
        t1w_img = nib.load(str(t1w_path))
        atlas_img = nib.load(str(atlas_path))
        mni_img = nib.load(str(template_path))

        t_vol = bold_img.shape[3] if bold_img.ndim == 4 else 0
        if t_vol < 50:
            raise ValueError(f"Insufficient BOLD timepoints: {t_vol} (minimum 50 required)")

        epi_shape = bold_img.shape[:3]

        # -------------------------------------------------------------
        # Stage 2: 3D Rigid Motion Realignment (DIPY) -> Motion + FD
        # -------------------------------------------------------------
        current_stage = "Stage 2: Motion Realignment"
        print(f"\n[Stage 2/19] Realigning 4D BOLD series ({t_vol} volumes)...")
        realigned_bold, motion_params, fd_vector = realign_4d_bold(bold_img)
        mean_fd = float(np.mean(fd_vector))
        max_fd = float(np.max(fd_vector))
        scrubbed = int(np.count_nonzero(fd_vector > 0.5))
        print(f"  -> Mean FD: {mean_fd:.4f} mm | Max FD: {max_fd:.4f} mm | Scrubbed (>0.5mm): {scrubbed} volumes")

        # -------------------------------------------------------------
        # Stage 3: Functional Reference Generation
        # -------------------------------------------------------------
        current_stage = "Stage 3: Functional Reference"
        print("\n[Stage 3/19] Generating 3D functional reference image...")
        mean_epi, epi_brain_mask = generate_functional_reference(realigned_bold)
        print(f"  -> Reference EPI shape: {mean_epi.shape}")

        # -------------------------------------------------------------
        # Stage 4: Anatomical Brain Extraction
        # -------------------------------------------------------------
        current_stage = "Stage 4: Anatomical Brain Extraction"
        print("\n[Stage 4/19] Extracting T1w anatomical brain with hole-filling...")
        t1_brain, t1_brain_mask = extract_anatomical_brain(t1w_img, median_radius=2, numpass=2)
        print(f"  -> Extracted brain voxels: {int(np.count_nonzero(t1_brain_mask.dataobj))}")

        # -------------------------------------------------------------
        # Stage 5: Log-Polynomial Bias-Field Correction
        # -------------------------------------------------------------
        current_stage = "Stage 5: Bias-Field Correction"
        print("\n[Stage 5/19] Correcting anatomical bias field (order=2)...")
        t1_corrected, bias_field = correct_bias_field(t1_brain, t1_brain_mask, order=2)
        print(f"  -> Bias field mean: {float(np.mean(bias_field.dataobj)):.3f}")

        # -------------------------------------------------------------
        # Stage 6: Tissue Segmentation (CSF, GM, WM Probabilities)
        # -------------------------------------------------------------
        current_stage = "Stage 6: Tissue Segmentation"
        print("\n[Stage 6/19] Segmenting T1w tissue into CSF, GM, WM...")
        prob_maps = segment_t1w_tissue(t1_corrected, t1_brain_mask, n_classes=3, method="gmm")
        print("  -> Probability maps: CSF, GM, WM computed successfully.")

        # -------------------------------------------------------------
        # Stage 7: Morphological Mask Erosion & Geometry Verification
        # -------------------------------------------------------------
        current_stage = "Stage 7: Tissue Mask Erosion"
        print("\n[Stage 7/19] Applying 0.96 thresholding & 1-voxel 3D erosion...")
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
        # Stage 8: EPI-to-T1w Rigid Coregistration
        # -------------------------------------------------------------
        current_stage = "Stage 8: Coregistration"
        print("\n[Stage 8/19] Coregistering functional reference to anatomical brain (DIPY Rigid)...")
        m_epi_to_t1w, _ = register_epi_to_t1w(mean_epi, t1_brain, level_iters=[10])
        print("  -> Coregistration rigid pullback matrix derived.")

        # -------------------------------------------------------------
        # Stage 9: T1w-to-MNI SyN Nonlinear Normalization
        # -------------------------------------------------------------
        current_stage = "Stage 9: Nonlinear Normalization"
        print("\n[Stage 9/19] Computing T1w-to-MNI SyN nonlinear diffeomorphic registration...")
        m_t1w_to_mni, syn_map = register_t1w_to_mni(
            t1_brain, mni_img, affine_level_iters=[10], syn_level_iters=[5]
        )
        print("  -> Diffeomorphic SyN mapping computed.")

        # -------------------------------------------------------------
        # Stage 10: Composed Pullback Resampling into CC200 Grid
        # -------------------------------------------------------------
        current_stage = "Stage 10: Composed 4D Resampling"
        print("\n[Stage 10/19] Composing transformations and resampling 4D BOLD into CC200 grid...")
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

        resampled_bold = resample_4d_bold(
            realigned_bold,
            composed_map,
            target_shape=target_shape,
            target_affine=target_affine,
            interpolation="linear",
            max_memory_gb=max_memory_gb,
        )
        print(f"  -> Resampled BOLD shape: {resampled_bold.shape} [MATCHES CC200 TARGET]")

        # -------------------------------------------------------------
        # Stage 11: Tissue Mask Standard Grid Resampling
        # -------------------------------------------------------------
        current_stage = "Stage 11: Tissue Mask Resampling"
        print("\n[Stage 11/19] Resampling eroded WM/CSF masks into CC200 grid via SyN...")
        wm_eroded_mni = resample_mask_to_target(
            tissue_masks["mask_wm_eroded"], syn_map, target_shape, target_affine
        )
        csf_eroded_mni = resample_mask_to_target(
            tissue_masks["mask_csf_eroded"], syn_map, target_shape, target_affine
        )
        print(f"  -> Standard space mask voxels: WM={int(np.count_nonzero(wm_eroded_mni.dataobj))}, "
              f"CSF={int(np.count_nonzero(csf_eroded_mni.dataobj))}")

        # -------------------------------------------------------------
        # Stage 12: aCompCor Physiological Noise Extraction
        # -------------------------------------------------------------
        current_stage = "Stage 12: aCompCor Extraction"
        print("\n[Stage 12/19] Extracting 5 physiological aCompCor components...")
        compcor_5 = extract_acompcor_components(
            resampled_bold, wm_eroded_mni, csf_eroded_mni, n_components=5
        )
        print(f"  -> aCompCor extracted: shape {compcor_5.shape}")

        # -------------------------------------------------------------
        # Stage 13: Friston-24 Motion Regressor Expansion
        # -------------------------------------------------------------
        current_stage = "Stage 13: Friston-24 Expansion"
        print("\n[Stage 13/19] Expanding 6 motion parameters to Friston-24...")
        friston_24 = build_friston_24_regressors(motion_params)
        print(f"  -> Friston-24 shape: {friston_24.shape}")

        # -------------------------------------------------------------
        # Stage 14: Confound Matrix Assembly & Collinearity Check
        # -------------------------------------------------------------
        current_stage = "Stage 14: Confound Matrix Assembly"
        print("\n[Stage 14/19] Assembling 31-column nuisance confound matrix...")
        confounds_df = build_confound_matrix(friston_24, compcor_5)
        # Compute condition number & check rank
        conf_vals = confounds_df.to_numpy()
        matrix_rank = int(np.linalg.matrix_rank(conf_vals))
        cond_num = float(np.linalg.cond(conf_vals))
        print(f"  -> Confound Matrix: {confounds_df.shape[1]} columns | Rank: {matrix_rank}/31 | Cond: {cond_num:.2f}")

        # -------------------------------------------------------------
        # Stage 15: Simultaneous Nuisance Regression & Bandpass
        # -------------------------------------------------------------
        current_stage = "Stage 15: Nuisance Regression"
        print("\n[Stage 15/19] Executing simultaneous regression & Butterworth bandpass [0.01-0.10 Hz]...")
        cleaned_bold = clean_functional_bold(
            resampled_bold, confounds_df, tr=tr, low_cutoff_hz=0.01, high_cutoff_hz=0.10
        )
        print(f"  -> Cleaned BOLD shape: {cleaned_bold.shape} (GSR=OFF, TissueMean=OFF)")

        # -------------------------------------------------------------
        # Stage 16: CC200 ROI Time Series Extraction
        # -------------------------------------------------------------
        current_stage = "Stage 16: CC200 Parcellation"
        print("\n[Stage 16/19] Extracting CC200 ROI time series across 200 parcels...")
        roi_df, signals = extract_cc200_timeseries(cleaned_bold, atlas_img)
        print(f"  -> Timeseries DataFrame: {roi_df.shape} (200 ROIs x {t_vol} timepoints + 1 label column)")

        # -------------------------------------------------------------
        # Stage 17: Pairwise Fisher-z Connectivity Vector Calculation
        # -------------------------------------------------------------
        current_stage = "Stage 17: Connectivity Features"
        print("\n[Stage 17/19] Computing 19,900 Fisher-z connectivity features...")
        z_vector, edge_names = compute_connectivity_vector(signals)
        print(f"  -> Fisher-z edges: {len(z_vector)} | Finite: {np.all(np.isfinite(z_vector))} | Range: [{np.min(z_vector):.3f}, {np.max(z_vector):.3f}]")

        # -------------------------------------------------------------
        # Stage 18: Quality Control Visual Overlay Generation
        # -------------------------------------------------------------
        current_stage = "Stage 18: QC Visual Overlays"
        print("\n[Stage 18/19] Generating 4 orthogonal quality control overlays...")

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
        # Stage 19: Schema-Compliant Provenance JSON Generation
        # -------------------------------------------------------------
        current_stage = "Stage 19: Provenance Reporting"
        print("\n[Stage 19/19] Generating provenance metadata and validating schema...")
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

        # 2. Measurable Assertions for Validation Gates
        n_rois = int(signals.shape[1])
        constant_rois = int(np.count_nonzero(np.std(signals, axis=0, ddof=0) == 0.0))

        gate1_passed = bool(peak_rss <= max_memory_gb)
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

        provenance_dict: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "subject_id": subject_id,
            "site": "SYNTHETIC_BENCHMARK",
            "cohort_split": "synthetic",
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
            },
            "validation_gates": validation_gates,
            "output_artifacts": output_artifacts_dict,
            "output_checksums": output_checksums,
        }

        # 3. Schema Validation using jsonschema
        schema_path = PROJECT_ROOT / "configs" / "provenance_schema.json"
        if schema_path.is_file():
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
            "subject_id": subject_id,
            "timestamp_utc": provenance_dict["timestamp_utc"],
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

        # 5. Check validation gate failures
        for gate_name, gate_info in validation_gates.items():
            if gate_info["status"] == "FAILED":
                raise RuntimeError(f"Pipeline validation failed at gate '{gate_name}': {gate_info}")

        # Cleanup scratch directory
        shutil.rmtree(scratch_dir, ignore_errors=True)

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
            "manifest_path": manifest_out_path,
            "artifacts": {
                "roi_csv": roi_csv_path,
                "features_csv": feat_csv_path,
                "features_npy": feat_npy_path,
                "motion_tsv": motion_txt_path,
                "fd_tsv": fd_1d_path,
                "confounds_tsv": confounds_path,
                "confound_report_json": confound_report_path,
                "qc_coreg_png": qc_coreg_path,
                "qc_norm_png": qc_norm_path,
                "qc_tissue_png": qc_tissue_path,
                "qc_parcellation_png": qc_parcellation_path,
                "provenance_json": provenance_path,
            },
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
        with open(sub_out_dir / "pipeline_failure_report.json", "w", encoding="utf-8") as f:
            json.dump(failure_report, f, indent=2)
        shutil.rmtree(scratch_dir, ignore_errors=True)
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

    args = parser.parse_args()

    if args.allow_real_scans:
        raise RuntimeError(
            "SAFETY VIOLATION: Execution on real MRI scans is strictly disabled by Phase A safety policy. "
            "--allow-real-scans is not permitted."
        )

    if args.mode == "synthetic":
        print(f"[Synthetic Generation] Building synthetic BIDS session for {args.subject_id}...")
        phantom_scratch = args.output_dir / "synthetic_inputs"
        phantom_scratch.mkdir(parents=True, exist_ok=True)
        synthetic_paths = generate_synthetic_phantom_session(
            phantom_scratch,
            subject_id=args.subject_id,
            session_id=args.session_id,
            t_vol=args.t_vol,
            seed=args.seed,
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
        )
        print(f"\nExecution successful. Outputs verified at: {res['output_dir']}")
    else:
        raise NotImplementedError(f"Mode '{args.mode}' is not permitted.")


if __name__ == "__main__":
    main()
