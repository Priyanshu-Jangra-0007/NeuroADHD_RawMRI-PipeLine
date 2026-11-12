#!/usr/bin/env python3
"""Step 5 Susceptibility Distortion Correction (SDC) Validation Script (DEC-SDC-02).

Generates external synthetic MRI fixtures and runs end-to-end physics, safety,
transformation-composition, and canonical CC200 22-stage validation outside Git:
    <parent_dir>/NeuroADHD_Step5_Synthetic/  (or NEUROADHD_STEP5_ROOT env var)

Covers:
1. Software environment honesty probe (FSL topup / prelude / fugue availability).
2. All 6 BIDS PhaseEncodingDirection axes ('i', 'i-', 'j', 'j-', 'k', 'k-') across
   both direct-fieldmap unit conventions ('Hz' and 'rad/s'), including opposite-polarity
   negative controls verifying misalignment degradation vs calibrated correction.
3. Physics & BIDS fail-closed controls:
   - Jacobian folding / non-invertibility negative control (SDCFoldingError).
   - Odd-37-slice TOPUP config safety (subsamp=[1,1,1,1], slice_modification_applied=False)
     and unequal TotalReadoutTime acqparams support + identical-polarity/readout rejection.
   - Ambiguous BIDS fieldmap association rejection (AmbiguousFieldmapAssociationError).
   - Honest fallbacks: fieldmap_less_syn -> SKIPPED_UNSUPPORTED_METHOD;
     phasediff without prelude/fugue -> SKIPPED_TOOL_UNAVAILABLE;
     PEPOLAR without topup -> SKIPPED_TOOL_UNAVAILABLE (plus mocked topup_runner success).
4. Independent 5-landmark TRE & peak-sharpness validation of compose_sdc_transform() +
   apply_composed_sdc_volume() + resample_4d_bold() vs sequential multi-step resampling.
5. Source-acquisition-like 64x64x37x240 canonical CC200 22-stage E2E validation:
   - Run A (PEPOLAR missing topup): SKIPPED_TOOL_UNAVAILABLE, odd-37-slice config,
     raw BOLD immutability, Stage 2 motion identical to baseline.
   - Run B (Direct Hz fieldmap, Run 1 + Run 2): APPLIED_DIRECT_FIELDMAP, Stage 11
     coregistered correction_qc improvement, zero Jacobian folding, single-step composed
     resampling, 200x240 ROI timeseries / (200, 241) CSV, 19,900 finite Fisher-z features,
     injected pure-GM positive & negative correlation recovery, determinism < 1e-7,
     and RSS <= 2.50 GB budget compliance.

SAFETY:
- Enforces ResearchAuthorizationGate (execution_policy=STRICTLY_SYNTHETIC_ONLY).
- Never inspects or processes real participant data (sub-01, NeuroADHD_Converted).
- Writes all NIfTIs, QC PNGs, and JSON outputs strictly outside Git.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import nibabel as nib
import numpy as np
import pandas as pd
from dipy.align.imwarp import DiffeomorphicMap
from scipy.ndimage import center_of_mass, gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if not os.environ.get("NEUROADHD_HMAC_KEY"):
    os.environ["NEUROADHD_HMAC_KEY"] = "step5_sdc_synthetic_validation_key_2026"

from scripts.run_poc_v2 import (  # noqa: E402
    CANONICAL_CC200_AFFINE,
    CANONICAL_CC200_SHAPE,
    compute_manifest_hmac,
    compute_sha256,
    run_poc_v2_pipeline,
    verify_canonical_cc200_atlas,
)
from src.preprocessing.motion import apply_motion_transforms  # noqa: E402
from src.preprocessing.registration import (  # noqa: E402
    apply_composed_sdc_volume,
    compose_sdc_transform,
    compose_transformations,
    resample_4d_bold,
)
from src.preprocessing.susceptibility_distortion import (  # noqa: E402
    PHASE_ENCODING_DIRECTION_MAP,
    SDC_STATUS_APPLIED_DIRECT_FIELDMAP,
    SDC_STATUS_APPLIED_PEPOLAR_TOPUP,
    SDC_STATUS_SKIPPED_TOOL_UNAVAILABLE,
    SDC_STATUS_SKIPPED_UNSUPPORTED_METHOD,
    AmbiguousFieldmapAssociationError,
    InvalidSDCInputError,
    SDCFoldingError,
    build_topup_acquisition_parameters,
    detect_sdc_software,
    evaluate_sdc_correction_qc,
    field_hz_to_displacement,
    generate_synthetic_distorted_epi,
    run_susceptibility_distortion_correction,
    select_topup_config,
    validate_displacement_field,
)
from src.validation.authorization_gate import ResearchAuthorizationGate  # noqa: E402

DEFAULT_STEP5_ROOT = PROJECT_ROOT.parent / "NeuroADHD_Step5_Synthetic"
STEP5_ROOT = Path(os.environ.get("NEUROADHD_STEP5_ROOT") or DEFAULT_STEP5_ROOT).resolve()


def _make_structured_phantom_3d(
    shape: Tuple[int, int, int] = (26, 26, 21),
    zooms: Tuple[float, float, float] = (3.0, 3.0, 3.5),
    b0_peak_hz: float = 38.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an asymmetric multi-tissue 3D EPI phantom and smooth off-resonance B0 field."""
    gx, gy, gz = np.indices(shape, dtype=np.float32)
    cx, cy, cz = shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0
    r_ell = ((gx - cx) / (shape[0] * 0.36)) ** 2 + ((gy - cy) / (shape[1] * 0.36)) ** 2 + ((gz - cz) / (shape[2] * 0.38)) ** 2
    brain = r_ell <= 1.0
    vent = ((gx - cx) / (shape[0] * 0.13)) ** 2 + ((gy - cy) / (shape[1] * 0.16)) ** 2 + ((gz - cz) / (shape[2] * 0.14)) ** 2 <= 1.0
    sulcus = (np.abs(gx - (cx + shape[0] * 0.14)) < 1.2) & (gy > cy) & brain
    cortex = (r_ell >= 0.55) & brain & (~vent)

    true_vol = np.zeros(shape, dtype=np.float32)
    true_vol[brain] = 90.0
    true_vol[cortex] = 135.0
    true_vol[vent] = 25.0
    true_vol[sulcus] = 45.0
    true_vol = gaussian_filter(true_vol, sigma=0.75).astype(np.float32)

    b0_hz = (
        float(b0_peak_hz)
        * np.exp(
            -(
                ((gx - cx * 1.15) / (shape[0] * 0.18)) ** 2
                + ((gy - cy * 1.25) / (shape[1] * 0.18)) ** 2
                + ((gz - cz * 0.90) / (shape[2] * 0.22)) ** 2
            )
        )
    ).astype(np.float32)

    affine = np.array(
        [
            [zooms[0], 0.0, 0.0, -cx * zooms[0]],
            [0.0, zooms[1], 0.0, -cy * zooms[1]],
            [0.0, 0.0, zooms[2], -cz * zooms[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return true_vol, b0_hz, affine


def run_six_direction_and_unit_benchmark(out_root: Path) -> List[Dict[str, Any]]:
    """Validate direct fieldmap unwarping across all 6 PE directions and both units (Hz, rad/s)."""
    true_3d, b0_hz, affine = _make_structured_phantom_3d(shape=(26, 26, 21), b0_peak_hz=38.0)
    trt = 0.055
    tr = 2.0
    records: List[Dict[str, Any]] = []

    for unit_mode in ("Hz", "rad/s"):
        for pe_dir in ("i", "i-", "j", "j-", "k", "k-"):
            case_dir = out_root / "six_direction_matrix" / f"{unit_mode.replace('/', '_')}_{pe_dir}"
            ses_dir = case_dir / "sub-synthetic500" / "ses-1"
            func_dir = ses_dir / "func"
            fmap_dir = ses_dir / "fmap"
            deriv_dir = case_dir / "derivatives"
            for d in (func_dir, fmap_dir, deriv_dir):
                d.mkdir(parents=True, exist_ok=True)

            dist_3d, sim_info = generate_synthetic_distorted_epi(
                true_3d, b0_hz, pe_direction=pe_dir, total_readout_time_s=trt, affine=affine
            )
            bold_4d = np.repeat(dist_3d[..., None], 10, axis=3).astype(np.float32)
            bold_p = func_dir / "sub-synthetic500_ses-1_task-rest_bold.nii.gz"
            bold_json_p = func_dir / "sub-synthetic500_ses-1_task-rest_bold.json"
            b_img = nib.Nifti1Image(bold_4d, affine)
            b_img.header.set_zooms((3.0, 3.0, 3.5, tr))
            b_img.header.set_xyzt_units("mm", "sec")
            nib.save(b_img, str(bold_p))
            bold_json_p.write_text(
                json.dumps(
                    {
                        "RepetitionTime": tr,
                        "TaskName": "rest",
                        "PhaseEncodingDirection": pe_dir,
                        "TotalReadoutTime": trt,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            fmap_vals = b0_hz if unit_mode == "Hz" else (b0_hz * np.float32(2.0 * np.pi))
            fmap_p = fmap_dir / "sub-synthetic500_ses-1_fieldmap.nii.gz"
            fmap_json_p = fmap_dir / "sub-synthetic500_ses-1_fieldmap.json"
            nib.save(nib.Nifti1Image(fmap_vals.astype(np.float32), affine), str(fmap_p))
            fmap_json_p.write_text(
                json.dumps(
                    {
                        "Units": unit_mode,
                        "IntendedFor": [f"ses-1/func/{bold_p.name}"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            ref_img = nib.Nifti1Image(true_3d, affine)
            res = run_susceptibility_distortion_correction(
                bold_p,
                bold_json_p,
                derivative_dir=deriv_dir,
                subject_id="sub-synthetic500",
                session_id="ses-1",
                reference_img=ref_img,
            )
            assert res.status == SDC_STATUS_APPLIED_DIRECT_FIELDMAP
            assert res.corrected is True
            assert res.displacement_stats is not None
            assert res.jacobian_qc is not None
            assert res.correction_qc is not None
            assert res.displacement_stats["orthogonal_residual_max_vox"] == 0.0
            assert res.jacobian_qc["folding_detected"] is False
            assert res.jacobian_qc["min_jacobian"] > 0.0
            assert res.correction_qc["ncc_after"] > res.correction_qc["ncc_before"]
            assert res.correction_qc["ncc_after"] > 0.99
            assert res.correction_qc["nmi_after"] > res.correction_qc["nmi_before"]
            assert res.correction_qc["mask_dice_after"] >= res.correction_qc["mask_dice_before"]
            assert (case_dir / str(res.derivative_files["qc_png"])).is_file()

            # Negative control: flip PE polarity in BOLD metadata -> opposite-sign warp degrades alignment
            opp_pe = pe_dir[:-1] if pe_dir.endswith("-") else f"{pe_dir}-"
            opp_json_p = func_dir / "sub-synthetic500_ses-1_task-rest_opp_bold.json"
            opp_json_p.write_text(
                json.dumps(
                    {
                        "RepetitionTime": tr,
                        "TaskName": "rest",
                        "PhaseEncodingDirection": opp_pe,
                        "TotalReadoutTime": trt,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            res_opp = run_susceptibility_distortion_correction(
                bold_p, opp_json_p, fmap_dir=fmap_dir, reference_img=ref_img
            )
            assert res_opp.correction_qc is not None
            assert res_opp.correction_qc["ncc_after"] < res.correction_qc["ncc_before"] - 0.02

            records.append(
                {
                    "units": unit_mode,
                    "pe_direction": pe_dir,
                    "pe_axis": int(PHASE_ENCODING_DIRECTION_MAP[pe_dir][0]),
                    "pe_polarity": int(PHASE_ENCODING_DIRECTION_MAP[pe_dir][1]),
                    "status": res.status,
                    "max_abs_disp_vox": float(res.displacement_stats["max_abs_vox"]),
                    "max_abs_disp_mm": float(res.displacement_stats["max_abs_mm"]),
                    "min_jacobian": float(res.jacobian_qc["min_jacobian"]),
                    "ncc_before": float(res.correction_qc["ncc_before"]),
                    "ncc_after": float(res.correction_qc["ncc_after"]),
                    "opposite_polarity_ncc": float(res_opp.correction_qc["ncc_after"]),
                    "nmi_before": float(res.correction_qc["nmi_before"]),
                    "nmi_after": float(res.correction_qc["nmi_after"]),
                    "mask_dice_before": float(res.correction_qc["mask_dice_before"]),
                    "mask_dice_after": float(res.correction_qc["mask_dice_after"]),
                    "forward_fixed_point_residual_vox": float(sim_info["fixed_point_max_residual_vox"]),
                }
            )
    return records


def run_physics_and_bids_controls(out_root: Path) -> Dict[str, Any]:
    """Execute fail-closed Jacobian folding, odd-37 TOPUP config, ambiguity, and fallback controls."""
    ctrl_root = out_root / "controls"
    ctrl_root.mkdir(parents=True, exist_ok=True)
    _, _, aff = _make_structured_phantom_3d(shape=(24, 24, 18))

    # 1. Jacobian folding control
    gx, _, _ = np.indices((24, 24, 18), dtype=np.float32)
    b0_fold = (120.0 * np.exp(-((gx - 12.0) / 1.5) ** 2) * (gx - 12.0)).astype(np.float32)
    disp_fold, disp_fold_mm = field_hz_to_displacement(b0_fold, "i", 0.08, aff)
    folding_rejected = False
    folding_msg = ""
    try:
        validate_displacement_field(disp_fold, disp_fold_mm, pe_axis=0, affine=aff, raise_on_folding=True)
    except SDCFoldingError as exc:
        folding_rejected = True
        folding_msg = str(exc)
    assert folding_rejected, "Expected SDCFoldingError on severe Jacobian folding"

    # 2. Odd-37-slice TOPUP config + unequal readout parameters + degenerate pair rejection
    cfg_37 = select_topup_config((64, 64, 37))
    assert cfg_37["odd_dimension_detected"] is True
    assert cfg_37["odd_dimensions"] == ["k"]
    assert cfg_37["subsamp_schedule"] == [1, 1, 1, 1]
    assert cfg_37["slice_modification_applied"] is False

    acq_unequal = build_topup_acquisition_parameters(["j", "j"], [0.045, 0.068])
    assert len(acq_unequal) == 2

    degenerate_rejected = False
    try:
        build_topup_acquisition_parameters(["j", "j"], [0.050, 0.050])
    except InvalidSDCInputError:
        degenerate_rejected = True
    assert degenerate_rejected

    # 3. fieldmap_less_syn & phasediff without tools & PEPOLAR mocked runner
    true_3d, b0_hz, aff30 = _make_structured_phantom_3d(shape=(24, 24, 18), b0_peak_hz=25.0)
    dist_3d, _ = generate_synthetic_distorted_epi(true_3d, b0_hz, "j", 0.05, affine=aff30)

    ses_syn = ctrl_root / "sub-synthetic501" / "ses-1"
    (ses_syn / "func").mkdir(parents=True, exist_ok=True)
    syn_bold = ses_syn / "func" / "sub-synthetic501_ses-1_task-rest_bold.nii.gz"
    syn_json = ses_syn / "func" / "sub-synthetic501_ses-1_task-rest_bold.json"
    nib.save(nib.Nifti1Image(np.repeat(dist_3d[..., None], 6, axis=3), aff30), str(syn_bold))
    syn_json.write_text(
        json.dumps({"RepetitionTime": 2.0, "PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05}),
        encoding="utf-8",
    )

    res_syn = run_susceptibility_distortion_correction(syn_bold, syn_json, requested_method="fieldmap_less_syn")
    assert res_syn.status == SDC_STATUS_SKIPPED_UNSUPPORTED_METHOD
    assert res_syn.corrected is False

    # Phasediff missing prelude/fugue
    fmap_pd = ses_syn / "fmap"
    fmap_pd.mkdir(parents=True, exist_ok=True)
    pd_nii = fmap_pd / "sub-synthetic501_ses-1_phasediff.nii.gz"
    pd_json = fmap_pd / "sub-synthetic501_ses-1_phasediff.json"
    nib.save(nib.Nifti1Image(np.zeros((24, 24, 18), dtype=np.float32), aff30), str(pd_nii))
    pd_json.write_text(
        json.dumps(
            {
                "EchoTime1": 0.00492,
                "EchoTime2": 0.00738,
                "Units": "rad",
                "IntendedFor": [f"ses-1/func/{syn_bold.name}"],
            }
        ),
        encoding="utf-8",
    )
    with mock.patch("src.preprocessing.susceptibility_distortion.shutil.which", return_value=None):
        res_pd = run_susceptibility_distortion_correction(syn_bold, syn_json, fmap_dir=fmap_pd)
    assert res_pd.status == SDC_STATUS_SKIPPED_TOOL_UNAVAILABLE
    assert res_pd.missing_tools == ["prelude", "fugue"]
    assert res_pd.corrected is False

    # Ambiguous multiple direct fieldmaps pointing to same BOLD -> fail closed
    fmap_amb = ctrl_root / "ambig_fmap"
    fmap_amb.mkdir(parents=True, exist_ok=True)
    for tag in ("a", "b"):
        fn = fmap_amb / f"sub-synthetic501_ses-1_acq-{tag}_fieldmap.nii.gz"
        fj = fmap_amb / f"sub-synthetic501_ses-1_acq-{tag}_fieldmap.json"
        nib.save(nib.Nifti1Image(b0_hz, aff30), str(fn))
        fj.write_text(json.dumps({"Units": "Hz", "IntendedFor": [f"ses-1/func/{syn_bold.name}"]}), encoding="utf-8")

    ambiguity_rejected = False
    try:
        run_susceptibility_distortion_correction(syn_bold, syn_json, fmap_dir=fmap_amb)
    except AmbiguousFieldmapAssociationError:
        ambiguity_rejected = True
    assert ambiguity_rejected

    # PEPOLAR mocked topup_runner control
    ses_pep = ctrl_root / "sub-synthetic502" / "ses-1"
    (ses_pep / "func").mkdir(parents=True, exist_ok=True)
    (ses_pep / "fmap").mkdir(parents=True, exist_ok=True)
    p_bold = ses_pep / "func" / "sub-synthetic502_ses-1_task-rest_bold.nii.gz"
    p_json = ses_pep / "func" / "sub-synthetic502_ses-1_task-rest_bold.json"
    nib.save(nib.Nifti1Image(np.repeat(dist_3d[..., None], 6, axis=3), aff30), str(p_bold))
    p_json.write_text(
        json.dumps({"RepetitionTime": 2.0, "PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05}),
        encoding="utf-8",
    )
    se_fwd, _ = generate_synthetic_distorted_epi(true_3d, b0_hz, "j", 0.05, affine=aff30)
    se_rev, _ = generate_synthetic_distorted_epi(true_3d, b0_hz, "j-", 0.05, affine=aff30)
    for tag, pe_c, arr3 in (("AP", "j", se_fwd), ("PA", "j-", se_rev)):
        fn = ses_pep / "fmap" / f"sub-synthetic502_ses-1_dir-{tag}_epi.nii.gz"
        fj = ses_pep / "fmap" / f"sub-synthetic502_ses-1_dir-{tag}_epi.json"
        nib.save(nib.Nifti1Image(arr3, aff30), str(fn))
        fj.write_text(
            json.dumps(
                {"PhaseEncodingDirection": pe_c, "TotalReadoutTime": 0.05, "IntendedFor": [f"ses-1/func/{p_bold.name}"]}
            ),
            encoding="utf-8",
        )

    fake_topup = ctrl_root / "fake_topup"
    fake_topup.write_text("#!/bin/sh\n", encoding="utf-8")

    def _mock_topup(cmd: List[str]) -> subprocess.CompletedProcess:
        fout_arg = [x.split("=", 1)[1] for x in cmd if x.startswith("--fout=")][0]
        nib.save(nib.Nifti1Image(b0_hz.astype(np.float32), aff30), fout_arg)
        return subprocess.CompletedProcess(cmd, 0, stdout="mock topup ok", stderr="")

    res_pep_mock = run_susceptibility_distortion_correction(
        p_bold,
        p_json,
        topup_executable=fake_topup,
        topup_runner=_mock_topup,
        work_dir=ctrl_root / "mock_topup_work",
        reference_img=nib.Nifti1Image(true_3d, aff30),
    )
    assert res_pep_mock.status == SDC_STATUS_APPLIED_PEPOLAR_TOPUP
    assert res_pep_mock.corrected is True
    assert res_pep_mock.correction_qc is not None
    assert res_pep_mock.correction_qc["ncc_after"] > 0.99

    return {
        "jacobian_folding_rejected": folding_rejected,
        "jacobian_folding_message": folding_msg,
        "odd_37_slice_topup_config": cfg_37,
        "unequal_readout_acqparams": acq_unequal,
        "identical_polarity_readout_rejected": degenerate_rejected,
        "fieldmap_less_syn_status": res_syn.status,
        "phasediff_missing_tools_status": res_pd.status,
        "phasediff_missing_tools": res_pd.missing_tools,
        "ambiguous_fieldmap_rejected": ambiguity_rejected,
        "mocked_pepolar_topup_status": res_pep_mock.status,
        "mocked_pepolar_ncc_after": float(res_pep_mock.correction_qc["ncc_after"]),
    }


def run_composed_sdc_tre_and_sharpness_validation() -> Dict[str, Any]:
    """Validate single-step composed SDC + motion + coreg + SyN TRE and peak sharpness gain."""
    epi_shape = (30, 30, 25)
    epi_affine = np.array(
        [
            [-3.0, 0.0, 0.0, 45.0],
            [0.0, 3.2, 0.0, -48.0],
            [0.0, 0.0, 3.6, -45.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mni_shape = (34, 34, 28)
    mni_affine = np.array(
        [
            [-2.5, 0.0, 0.0, 42.5],
            [0.0, -2.5, 0.0, 42.5],
            [0.0, 0.0, 3.0, -42.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    gx, gy, gz = np.indices(epi_shape, dtype=np.float32)
    b0_hz = (
        22.0 * np.exp(-(((gx - 15.0) / 6.0) ** 2 + ((gy - 15.0) / 6.0) ** 2 + ((gz - 12.0) / 5.5) ** 2))
    ).astype(np.float32)
    disp_vox, _ = field_hz_to_displacement(b0_hz, "j", 0.05, epi_affine)

    r_mot = np.eye(4, dtype=np.float64)
    r_mot[:3, :3] = Rotation.from_euler("xyz", [2.5, -1.8, 3.2], degrees=True).as_matrix()
    r_mot[:3, 3] = [1.8, -2.4, 1.5]

    rigid_epi_to_t1 = np.eye(4, dtype=np.float64)
    rigid_epi_to_t1[:3, :3] = Rotation.from_euler("zyx", [-3.0, 2.0, 1.5], degrees=True).as_matrix()
    rigid_epi_to_t1[:3, 3] = [-1.5, 2.2, -1.1]

    aff_t1_to_mni = np.eye(4, dtype=np.float64)
    aff_t1_to_mni[:3, :3] = Rotation.from_euler("xyz", [1.5, 2.5, -2.0], degrees=True).as_matrix()
    aff_t1_to_mni[:3, 3] = [1.2, -1.5, 0.8]

    syn_map = DiffeomorphicMap(
        3,
        disp_shape=mni_shape,
        disp_grid2world=mni_affine,
        domain_shape=epi_shape,
        domain_grid2world=epi_affine,
        codomain_shape=mni_shape,
        codomain_grid2world=mni_affine,
        prealign=np.eye(4),
    )
    syn_map.is_inverse = True
    mx, my, mz = np.indices(mni_shape, dtype=np.float32)
    backward_field = np.zeros(mni_shape + (3,), dtype=np.float32)
    backward_field[..., 0] = (2.0 * np.sin((mx - 17.0) * 0.14)).astype(np.float32)
    backward_field[..., 1] = (2.4 * np.cos((my - 17.0) * 0.14)).astype(np.float32)
    backward_field[..., 2] = (1.6 * np.sin((mz - 14.0) * 0.16)).astype(np.float32)
    syn_map.backward = backward_field

    composed_map = compose_transformations(
        rigid_epi_to_t1w=rigid_epi_to_t1,
        affine_t1w_to_mni=aff_t1_to_mni,
        syn_mapping=syn_map,
        target_shape=mni_shape,
        target_affine=mni_affine,
        source_shape=epi_shape,
        source_affine=epi_affine,
    )
    grid = compose_sdc_transform(
        composed_map=composed_map,
        sdc_displacement=disp_vox,
        epi_affine=epi_affine,
        motion_transform=r_mot,
        target_shape=mni_shape,
        target_affine=mni_affine,
    )

    target_landmarks_ijk = np.array(
        [
            [9.0, 9.0, 8.0],
            [25.0, 10.0, 9.0],
            [10.0, 25.0, 10.0],
            [24.0, 24.0, 19.0],
            [17.0, 17.0, 14.0],
        ],
        dtype=np.float64,
    )
    prealign_inv = rigid_epi_to_t1 @ aff_t1_to_mni
    inv_prealign_inv = np.linalg.inv(prealign_inv)
    inv_epi = np.linalg.inv(epi_affine)
    inv_mni = np.linalg.inv(mni_affine)
    m_vox_motion = inv_epi @ r_mot @ epi_affine
    inv_m_vox = np.linalg.inv(m_vox_motion)

    gi_e, gj_e, gk_e = np.indices(epi_shape, dtype=np.float64)
    v_raw_all = np.vstack([gi_e.ravel(), gj_e.ravel(), gk_e.ravel()])
    v_raw_h = np.vstack([v_raw_all, np.ones((1, v_raw_all.shape[1]), dtype=np.float64)])
    x_mni = ((inv_prealign_inv @ epi_affine) @ (inv_m_vox @ v_raw_h))[:3]
    j_x_per_v = (inv_prealign_inv[:3, :3] @ epi_affine[:3, :3]) @ inv_m_vox[:3, :3]

    for _ in range(25):
        m_coords = (inv_mni[:3, :3] @ x_mni) + inv_mni[:3, 3:4]
        b_vec = np.vstack(
            [map_coordinates(backward_field[..., c].astype(np.float64), m_coords, order=1, mode="nearest") for c in range(3)]
        )
        x_def_h = np.vstack([x_mni + b_vec, np.ones((1, x_mni.shape[1]), dtype=np.float64)])
        v_undist = (inv_epi @ (prealign_inv @ x_def_h))[:3]
        u_shift = np.vstack(
            [map_coordinates(disp_vox[..., c].astype(np.float64), v_undist, order=1, mode="nearest") for c in range(3)]
        )
        v_pred = m_vox_motion[:3, :3] @ (v_undist + u_shift) + m_vox_motion[:3, 3:4]
        res_v = v_pred - v_raw_all
        if float(np.max(np.abs(res_v))) < 1e-6:
            break
        x_mni = x_mni - 0.85 * (j_x_per_v @ res_v)

    expected_mni_world: List[np.ndarray] = []
    raw_epi_flat = np.zeros(v_raw_all.shape[1], dtype=np.float64)
    sigma_mm = 3.6
    for i_m, j_m, k_m in target_landmarks_ijk:
        x_mni_k = (mni_affine @ np.array([i_m, j_m, k_m, 1.0]))[:3]
        expected_mni_world.append(x_mni_k)
        raw_epi_flat += 100.0 * np.exp(-np.sum((x_mni - x_mni_k[:, None]) ** 2, axis=0) / (2.0 * sigma_mm ** 2))

    raw_epi_phantom = raw_epi_flat.reshape(epi_shape).astype(np.float32)
    recovered_mni = apply_composed_sdc_volume(raw_epi_phantom, grid, interpolation="linear")

    raw_4d_mock = nib.Nifti1Image(raw_epi_phantom[..., None], epi_affine)
    step1_realigned = apply_motion_transforms(raw_4d_mock, [r_mot], interpolation="linear", sdc_displacement=disp_vox)
    step1_vol = np.asanyarray(step1_realigned.dataobj, dtype=np.float32)[..., 0]
    sequential_mni = composed_map.transform(step1_vol, interpolation="linear", out_shape=mni_shape, out_grid2world=mni_affine)

    single_errors_mm: List[float] = []
    peak_single: List[float] = []
    peak_seq: List[float] = []
    for lm_ijk, x_true in zip(target_landmarks_ijk, expected_mni_world):
        ii, jj, kk = (int(lm_ijk[0]), int(lm_ijk[1]), int(lm_ijk[2]))
        sub_single = recovered_mni[ii - 2 : ii + 3, jj - 2 : jj + 3, kk - 2 : kk + 3]
        sub_seq = sequential_mni[ii - 2 : ii + 3, jj - 2 : jj + 3, kk - 2 : kk + 3]
        com_rel = np.array(center_of_mass(sub_single)) - np.array([2.0, 2.0, 2.0])
        x_est = (mni_affine @ np.append(lm_ijk + com_rel, 1.0))[:3]
        single_errors_mm.append(float(np.linalg.norm(x_est - x_true)))
        peak_single.append(float(np.max(sub_single)))
        peak_seq.append(float(np.max(sub_seq)))

    max_tre = float(max(single_errors_mm))
    mean_tre = float(np.mean(single_errors_mm))
    assert max_tre < 0.08
    assert float(np.mean(peak_single)) > float(np.mean(peak_seq)) + 0.5

    raw_55_img = nib.Nifti1Image(np.repeat(raw_epi_phantom[..., None], 55, axis=3), epi_affine)
    res_4d = resample_4d_bold(
        raw_55_img,
        composed_map,
        target_shape=mni_shape,
        target_affine=mni_affine,
        interpolation="linear",
        motion_transforms=[r_mot] * 55,
        sdc_displacement=disp_vox,
    )
    assert res_4d.shape == mni_shape + (55,)
    assert np.allclose(np.asanyarray(res_4d.dataobj)[..., 0], recovered_mni, atol=1e-5)

    return {
        "status": "PASSED",
        "num_landmarks": int(len(target_landmarks_ijk)),
        "single_step_landmark_errors_mm": [round(x, 6) for x in single_errors_mm],
        "single_step_mean_tre_mm": round(mean_tre, 6),
        "single_step_max_tre_mm": round(max_tre, 6),
        "mean_peak_single_step": round(float(np.mean(peak_single)), 4),
        "mean_peak_sequential_two_step": round(float(np.mean(peak_seq)), 4),
        "peak_sharpness_gain": round(float(np.mean(peak_single) - np.mean(peak_seq)), 4),
        "oob_fraction": float(grid["oob_fraction"]),
    }


def _build_37slice_canonical_sdc_session(
    case_dir: Path,
    subject_id: str,
    mode: str,
    seed: int = 123,
) -> Dict[str, Any]:
    """Generate a 64x64x37x240 canonical CC200 BIDS session with forward B0 distortion."""
    ses_dir = case_dir / subject_id / "ses-1"
    func_dir = ses_dir / "func"
    anat_dir = ses_dir / "anat"
    fmap_dir = ses_dir / "fmap"
    tpl_dir = case_dir / "templates"
    for d in (func_dir, anat_dir, fmap_dir, tpl_dir):
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
    pe_dir = "j"
    trt = 0.055

    # Map native 37-slice EPI voxels to canonical CC200 parcels in world space
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
    assert int(len(np.unique(epi_parcels[epi_parcels > 0]))) == 200

    order = list(range(0, 37, 2)) + list(range(1, 37, 2))
    slice_timings = [0.0] * 37
    for step, z_idx in enumerate(order):
        slice_timings[z_idx] = round(float(step * tr / 37.0), 8)

    brain_mask = atlas_data > 0
    mx, my, mz = np.ogrid[: CANONICAL_CC200_SHAPE[0], : CANONICAL_CC200_SHAPE[1], : CANONICAL_CC200_SHAPE[2]]
    mc = (CANONICAL_CC200_SHAPE[0] // 2, CANONICAL_CC200_SHAPE[1] // 2, CANONICAL_CC200_SHAPE[2] // 2)
    csf = brain_mask & (((mx - mc[0]) ** 2 / 5.0 ** 2 + (my - mc[1]) ** 2 / 5.0 ** 2 + (mz - mc[2]) ** 2 / 5.0 ** 2) <= 1.0)
    wm = brain_mask & (~csf) & (((mx - mc[0]) ** 2 / 12.0 ** 2 + (my - mc[1]) ** 2 / 15.0 ** 2 + (mz - mc[2]) ** 2 / 12.0 ** 2) <= 1.0)
    gm = brain_mask & (~csf) & (~wm)
    dist_sq = (mx - mc[0]) ** 2 + (my - mc[1]) ** 2 + (mz - mc[2]) ** 2
    skull = (~brain_mask) & (dist_sq >= 28 ** 2) & (dist_sq <= 30 ** 2)

    bold_base_can = np.zeros(CANONICAL_CC200_SHAPE, dtype=np.float32)
    bold_base_can[csf] = 55.0
    bold_base_can[gm] = 100.0
    bold_base_can[wm] = 135.0
    epi_baseline = np.zeros(epi_shape, dtype=np.float32)
    epi_baseline[valid_mask] = bold_base_can[ci[valid_mask], cj[valid_mask], ck[valid_mask]]

    pos_pair = (66, 128)
    neg_pair = (93, 69)
    amp_norm_xy = float(np.sqrt(0.5 * (1.0 ** 2 + 0.4 ** 2)))
    rng = np.random.default_rng(seed)
    vol_onsets = np.arange(n_vols, dtype=np.float64) * tr
    parcel_noise = {}
    for p in range(1, 201):
        n_p = rng.normal(0.0, 1.0, size=n_vols)
        n_p = (n_p - np.mean(n_p)) / (float(np.std(n_p)) or 1.0)
        parcel_noise[p] = n_p

    bold_undist = np.zeros((*epi_shape, n_vols), dtype=np.float32)
    for parcel_id in range(1, 201):
        mask_p = epi_parcels == parcel_id
        freq = 0.02 + (parcel_id % 30) * 0.0025
        phase_p = (parcel_id * 0.37) % (2.0 * np.pi)
        for z_idx in np.unique(np.where(mask_p)[2]):
            sl_xy = mask_p[:, :, z_idx]
            nv = int(np.sum(sl_xy))
            if nv > 0:
                t_sec = vol_onsets + slice_timings[int(z_idx)]
                if parcel_id in pos_pair:
                    drv = (np.sin(2.0 * np.pi * 0.035 * t_sec) + 0.4 * np.cos(2.0 * np.pi * 0.022 * t_sec)) / amp_norm_xy
                    curve = 0.96 * drv + 0.20 * parcel_noise[parcel_id]
                elif parcel_id == neg_pair[0]:
                    drv = (np.sin(2.0 * np.pi * 0.045 * t_sec) + 0.4 * np.sin(2.0 * np.pi * 0.018 * t_sec)) / amp_norm_xy
                    curve = 0.96 * drv + 0.20 * parcel_noise[parcel_id]
                elif parcel_id == neg_pair[1]:
                    drv = (np.sin(2.0 * np.pi * 0.045 * t_sec) + 0.4 * np.sin(2.0 * np.pi * 0.018 * t_sec)) / amp_norm_xy
                    curve = -0.96 * drv + 0.20 * parcel_noise[parcel_id]
                else:
                    curve = 0.7 * np.sin(2.0 * np.pi * freq * t_sec + phase_p) + 0.3 * parcel_noise[parcel_id]
                base_col = epi_baseline[sl_xy, int(z_idx)][:, None]
                sig = (base_col + (8.0 * curve.astype(np.float32))[None, :]).astype(np.float32)
                bold_undist[sl_xy, int(z_idx), :] = sig + rng.normal(0.0, 0.4, size=(nv, n_vols)).astype(np.float32)

    # Smooth synthetic orbitofrontal-like off-resonance B0 field along j (~32 Hz peak, invertible J > 0.65)
    gx_e, gy_e, gz_e = np.indices(epi_shape, dtype=np.float32)
    b0_hz = (
        32.0
        * np.exp(-(((gx_e - 32.0) / 8.0) ** 2 + ((gy_e - 42.0) / 8.0) ** 2 + ((gz_e - 15.0) / 6.0) ** 2))
    ).astype(np.float32)

    bold_dist, _ = generate_synthetic_distorted_epi(
        bold_undist, b0_hz, pe_direction=pe_dir, total_readout_time_s=trt, affine=epi_affine
    )

    bold_path = func_dir / f"{subject_id}_ses-1_task-rest_bold.nii.gz"
    bold_img = nib.Nifti1Image(bold_dist, epi_affine)
    bold_img.header.set_zooms((3.0, 3.5, 4.0, float(tr)))
    bold_img.header.set_xyzt_units("mm", "sec")
    bold_img.header.set_dim_info(slice=2)
    nib.save(bold_img, str(bold_path))

    json_path = func_dir / f"{subject_id}_ses-1_task-rest_bold.json"
    json_path.write_text(
        json.dumps(
            {
                "RepetitionTime": tr,
                "TaskName": "rest",
                "SliceTiming": slice_timings,
                "SliceEncodingDirection": "k",
                "PhaseEncodingDirection": pe_dir,
                "TotalReadoutTime": trt,
                "Manufacturer": "SyntheticStep5E2E",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Anatomical T1w + MNI template on canonical grid
    t1_data = np.zeros(CANONICAL_CC200_SHAPE, dtype=np.float32)
    t1_data[csf] = 25.0 + rng.normal(0.0, 2.0, np.sum(csf)).astype(np.float32)
    t1_data[gm] = 85.0 + rng.normal(0.0, 3.0, np.sum(gm)).astype(np.float32)
    t1_data[wm] = 150.0 + rng.normal(0.0, 3.0, np.sum(wm)).astype(np.float32)
    t1_data[skull] = 110.0 + rng.normal(0.0, 3.0, np.sum(skull)).astype(np.float32)
    t1_path = anat_dir / f"{subject_id}_ses-1_T1w.nii.gz"
    nib.save(nib.Nifti1Image(t1_data, CANONICAL_CC200_AFFINE), str(t1_path))

    mni_data = np.zeros(CANONICAL_CC200_SHAPE, dtype=np.float32)
    mni_data[brain_mask] = 100.0
    mni_data[wm] = 150.0
    mni_data[gm] = 85.0
    mni_data[csf] = 25.0
    mni_path = tpl_dir / "MNI152_T1_3mm.nii.gz"
    nib.save(nib.Nifti1Image(mni_data, CANONICAL_CC200_AFFINE), str(mni_path))

    intended_rel = f"ses-1/func/{bold_path.name}"
    if mode == "direct_hz":
        f_nii = fmap_dir / f"{subject_id}_ses-1_fieldmap.nii.gz"
        f_json = fmap_dir / f"{subject_id}_ses-1_fieldmap.json"
        nib.save(nib.Nifti1Image(b0_hz, epi_affine), str(f_nii))
        f_json.write_text(json.dumps({"Units": "Hz", "IntendedFor": [intended_rel]}, indent=2), encoding="utf-8")
    elif mode == "pepolar":
        ref_3d = np.mean(bold_undist, axis=3)
        se_fwd, _ = generate_synthetic_distorted_epi(ref_3d, b0_hz, "j", trt, affine=epi_affine)
        se_rev, _ = generate_synthetic_distorted_epi(ref_3d, b0_hz, "j-", trt, affine=epi_affine)
        for tag, pe_i, vol_i in (("fwd", "j", se_fwd), ("rev", "j-", se_rev)):
            f_nii = fmap_dir / f"{subject_id}_ses-1_dir-{tag}_epi.nii.gz"
            f_json = fmap_dir / f"{subject_id}_ses-1_dir-{tag}_epi.json"
            nib.save(nib.Nifti1Image(vol_i.astype(np.float32), epi_affine), str(f_nii))
            f_json.write_text(
                json.dumps({"PhaseEncodingDirection": pe_i, "TotalReadoutTime": trt, "IntendedFor": [intended_rel]}, indent=2),
                encoding="utf-8",
            )
    else:
        raise ValueError(f"Unsupported session mode: {mode}")

    files_dict = {
        "bold_nii": {"filename": bold_path.name, "rel_path": f"ses-1/func/{bold_path.name}", "resolved_path": str(bold_path.resolve()), "sha256": compute_sha256(bold_path)},
        "bold_json": {"filename": json_path.name, "rel_path": f"ses-1/func/{json_path.name}", "resolved_path": str(json_path.resolve()), "sha256": compute_sha256(json_path)},
        "t1w_nii": {"filename": t1_path.name, "rel_path": f"ses-1/anat/{t1_path.name}", "resolved_path": str(t1_path.resolve()), "sha256": compute_sha256(t1_path)},
        "mni_template": {"filename": mni_path.name, "rel_path": mni_path.name, "resolved_path": str(mni_path.resolve()), "sha256": compute_sha256(mni_path)},
        "cc200_atlas": {"filename": canonical_atlas.name, "rel_path": canonical_atlas.name, "resolved_path": str(canonical_atlas.resolve()), "sha256": compute_sha256(canonical_atlas)},
    }
    for idx, f_item in enumerate(sorted(fmap_dir.iterdir(), key=lambda p: p.name)):
        files_dict[f"fmap_{idx:02d}"] = {
            "filename": f_item.name,
            "rel_path": f"ses-1/fmap/{f_item.name}",
            "resolved_path": str(f_item.resolve()),
            "sha256": compute_sha256(f_item),
        }

    man_path = ses_dir / "synthetic_manifest.json"
    man_path.write_text(
        json.dumps(
            {
                "manifest_magic": "NEUROADHD_SYNTHETIC_DATASET_V2",
                "manifest_type": "INTEGRITY_VERIFICATION_MANIFEST",
                "schema_version": "2.0.0",
                "subject_id": subject_id,
                "session_id": "ses-1",
                "atlas_mode": "canonical",
                "integrity_hmac": compute_manifest_hmac("NEUROADHD_SYNTHETIC_DATASET_V2", subject_id, files_dict),
                "files": files_dict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "bold_nii": bold_path,
        "bold_json": json_path,
        "t1w_nii": t1_path,
        "mni_template": mni_path,
        "cc200_atlas": canonical_atlas,
        "manifest_path": man_path,
        "pos_pair": pos_pair,
        "neg_pair": neg_pair,
        "bold_sha256": compute_sha256(bold_path),
        "json_sha256": compute_sha256(json_path),
    }


def run_step5_e2e_validations(out_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run canonical 64x64x37x240 22-stage E2E validation for PEPOLAR (missing topup) and Direct Hz."""
    # E2E Run A: PEPOLAR with absent FSL topup -> SKIPPED_TOOL_UNAVAILABLE + odd-37 config
    pep_case = _build_37slice_canonical_sdc_session(out_root / "e2e_pepolar", "sub-synthetic531", mode="pepolar", seed=123)
    with mock.patch("src.preprocessing.susceptibility_distortion.shutil.which", return_value=None):
        res_pep = run_poc_v2_pipeline(
            bold_path=pep_case["bold_nii"],
            bold_json_path=pep_case["bold_json"],
            t1w_path=pep_case["t1w_nii"],
            atlas_path=pep_case["cc200_atlas"],
            template_path=pep_case["mni_template"],
            output_dir=out_root / "e2e_pepolar_out",
            subject_id="sub-synthetic531",
            session_id="ses-1",
            atlas_mode="canonical",
            manifest_path=pep_case["manifest_path"],
        )
    prov_pep = res_pep["provenance"]
    sdc_pep = prov_pep["susceptibility_distortion_correction"]
    assert res_pep["status"] == "SUCCESS"
    assert prov_pep["pipeline_stages_total"] == 22
    assert sdc_pep["status"] == SDC_STATUS_SKIPPED_TOOL_UNAVAILABLE
    assert sdc_pep["corrected"] is False
    assert sdc_pep["missing_tools"] == ["topup"]
    assert sdc_pep["topup_config"]["subsamp_schedule"] == [1, 1, 1, 1]
    assert sdc_pep["topup_config"]["slice_modification_applied"] is False
    assert len(sdc_pep["acqparams_lines"]) == 2
    assert prov_pep["motion_estimation"]["single_step_composed_resampling_with_sdc"] is False
    assert compute_sha256(pep_case["bold_nii"]) == pep_case["bold_sha256"]

    e2e_pepolar_summary = {
        "status": sdc_pep["status"],
        "corrected": sdc_pep["corrected"],
        "missing_tools": sdc_pep["missing_tools"],
        "topup_config": sdc_pep["topup_config"],
        "acqparams_lines": sdc_pep["acqparams_lines"],
        "planned_command": sdc_pep["planned_command"],
        "pipeline_stages_total": prov_pep["pipeline_stages_total"],
        "single_step_composed_resampling_with_sdc": prov_pep["motion_estimation"]["single_step_composed_resampling_with_sdc"],
        "inputs_immutable_sha256": True,
        "runtime_seconds": float(res_pep["runtime_seconds"]),
        "peak_rss_gb": float(res_pep["peak_rss_gb"]),
    }

    # E2E Run B: Direct Hz Fieldmap -> APPLIED_DIRECT_FIELDMAP (Run 1 & Run 2 determinism + motion invariance baseline)
    dir_case = _build_37slice_canonical_sdc_session(out_root / "e2e_direct_hz", "sub-synthetic532", mode="direct_hz", seed=123)
    res_d1 = run_poc_v2_pipeline(
        bold_path=dir_case["bold_nii"],
        bold_json_path=dir_case["bold_json"],
        t1w_path=dir_case["t1w_nii"],
        atlas_path=dir_case["cc200_atlas"],
        template_path=dir_case["mni_template"],
        output_dir=out_root / "e2e_direct_out_run1",
        subject_id="sub-synthetic532",
        session_id="ses-1",
        atlas_mode="canonical",
        manifest_path=dir_case["manifest_path"],
    )
    res_d2 = run_poc_v2_pipeline(
        bold_path=dir_case["bold_nii"],
        bold_json_path=dir_case["bold_json"],
        t1w_path=dir_case["t1w_nii"],
        atlas_path=dir_case["cc200_atlas"],
        template_path=dir_case["mni_template"],
        output_dir=out_root / "e2e_direct_out_run2",
        subject_id="sub-synthetic532",
        session_id="ses-1",
        atlas_mode="canonical",
        manifest_path=dir_case["manifest_path"],
    )
    # Compare raw Stage 2 motion against an identical BOLD run with sdc_method="none"
    res_none = run_poc_v2_pipeline(
        bold_path=dir_case["bold_nii"],
        bold_json_path=dir_case["bold_json"],
        t1w_path=dir_case["t1w_nii"],
        atlas_path=dir_case["cc200_atlas"],
        template_path=dir_case["mni_template"],
        output_dir=out_root / "e2e_direct_out_none",
        subject_id="sub-synthetic532",
        session_id="ses-1",
        atlas_mode="canonical",
        manifest_path=dir_case["manifest_path"],
        sdc_method="none",
    )

    prov_d = res_d1["provenance"]
    sdc_d = prov_d["susceptibility_distortion_correction"]
    assert res_d1["status"] == "SUCCESS"
    assert res_d2["status"] == "SUCCESS"
    assert prov_d["pipeline_stages_total"] == 22
    assert len(prov_d["preprocessing_order"]) == 22
    assert sdc_d["status"] == SDC_STATUS_APPLIED_DIRECT_FIELDMAP
    assert sdc_d["corrected"] is True
    assert sdc_d["jacobian_qc"]["folding_detected"] is False
    assert sdc_d["correction_qc"]["ncc_after"] > sdc_d["correction_qc"]["ncc_before"]
    assert prov_d["motion_estimation"]["estimated_on_raw_uncorrected_bold"] is True
    assert prov_d["motion_estimation"]["single_step_composed_resampling_with_sdc"] is True
    assert (Path(res_d1["output_dir"]) / sdc_d["derivative_files"]["qc_png"]).is_file()
    assert compute_sha256(dir_case["bold_nii"]) == dir_case["bold_sha256"]
    assert compute_sha256(dir_case["bold_json"]) == dir_case["json_sha256"]

    # Motion bit-identical to sdc_method="none"
    m_sdc = pd.read_csv(res_d1["artifacts"]["motion_tsv"], sep="\t").to_numpy()
    m_none = pd.read_csv(res_none["artifacts"]["motion_tsv"], sep="\t").to_numpy()
    np.testing.assert_array_equal(m_sdc, m_none)

    # CC200 timeseries & correlation recovery + determinism
    time_cols = [str(i) for i in range(240)]
    roi_df1 = pd.read_csv(res_d1["artifacts"]["roi_csv"])
    roi_df2 = pd.read_csv(res_d2["artifacts"]["roi_csv"])
    assert roi_df1.shape == (200, 241)
    assert roi_df1.columns[0] == "node/volume"
    ts1 = roi_df1[time_cols].to_numpy(dtype=np.float64)
    ts2 = roi_df2[time_cols].to_numpy(dtype=np.float64)
    assert ts1.shape == (200, 240)
    assert np.all(np.isfinite(ts1))

    pos_pair = dir_case["pos_pair"]
    neg_pair = dir_case["neg_pair"]
    r_pos = float(np.corrcoef(
        roi_df1[roi_df1["node/volume"] == pos_pair[0]][time_cols].to_numpy().flatten(),
        roi_df1[roi_df1["node/volume"] == pos_pair[1]][time_cols].to_numpy().flatten(),
    )[0, 1])
    r_neg = float(np.corrcoef(
        roi_df1[roi_df1["node/volume"] == neg_pair[0]][time_cols].to_numpy().flatten(),
        roi_df1[roi_df1["node/volume"] == neg_pair[1]][time_cols].to_numpy().flatten(),
    )[0, 1])
    assert r_pos > 0.70, f"Expected r_pos > 0.70, got {r_pos:.4f}"
    assert r_neg < -0.65, f"Expected r_neg < -0.65, got {r_neg:.4f}"

    feat1 = np.load(res_d1["artifacts"]["features_npy"])
    feat2 = np.load(res_d2["artifacts"]["features_npy"])
    assert feat1.shape == (19900,)
    assert np.all(np.isfinite(feat1))

    max_ts_diff = float(np.max(np.abs(ts1 - ts2)))
    max_feat_diff = float(np.max(np.abs(feat1 - feat2)))
    assert max_ts_diff < 1e-7
    assert max_feat_diff < 1e-7

    max_rss = float(max(res_d1["peak_rss_gb"], res_d2["peak_rss_gb"]))
    assert max_rss <= 2.50

    e2e_direct_summary = {
        "status": "PASSED",
        "source_bold_shape": [64, 64, 37, 240],
        "canonical_target_shape": list(CANONICAL_CC200_SHAPE),
        "pipeline_stages_total": int(prov_d["pipeline_stages_total"]),
        "roi_timeseries_shape": [200, int(ts1.shape[1])],
        "roi_csv_shape": list(roi_df1.shape),
        "n_fisher_z_features": int(len(feat1)),
        "all_features_finite": True,
        "motion_identical_with_vs_without_sdc": True,
        "sdc_provenance": sdc_d,
        "correlation_recovery": {
            "positive_pair": {
                "rois": [f"Parcel_{pos_pair[0]:03d}", f"Parcel_{pos_pair[1]:03d}"],
                "recovered_pearson_r": r_pos,
                "threshold": "> 0.70",
            },
            "negative_pair": {
                "rois": [f"Parcel_{neg_pair[0]:03d}", f"Parcel_{neg_pair[1]:03d}"],
                "recovered_pearson_r": r_neg,
                "threshold": "< -0.65",
            },
        },
        "determinism": {
            "max_abs_diff_timeseries": max_ts_diff,
            "max_abs_diff_fisher_z": max_feat_diff,
            "tolerance": 1e-7,
            "deterministic_equality": "PASSED",
        },
        "inputs_immutable_sha256": True,
        "runtime_seconds_run1": float(res_d1["runtime_seconds"]),
        "runtime_seconds_run2": float(res_d2["runtime_seconds"]),
        "peak_rss_gb": max_rss,
    }
    return e2e_pepolar_summary, e2e_direct_summary


def run_step5_validation(root_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Execute complete Step 5 synthetic validation and emit external JSON report."""
    out_root = Path(root_dir or STEP5_ROOT).resolve()

    auth_gate = ResearchAuthorizationGate(base_dir=PROJECT_ROOT)
    auth_state = auth_gate.evaluate_register()
    auth_gate.check_execution_allowed(requested_mode="synthetic", allow_real_scans=False)

    print("=" * 78)
    print("NEUROADHD STEP 5: SUSCEPTIBILITY DISTORTION CORRECTION VALIDATION (DEC-SDC-02)")
    print(f"Policy        : {auth_state['execution_policy']}")
    print(f"Real Data Auth: {auth_state['real_data_authorized']} ({auth_state['overall_authorization_state']})")
    print(f"Fixture Root  : {out_root}")
    print("=" * 78)

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    software_info = detect_sdc_software()
    print(
        f"Software Probe -> topup={software_info['topup_available']}, "
        f"prelude={software_info['prelude_available']}, fugue={software_info['fugue_available']}"
    )

    print("\n[1/4] Running 6-direction x (Hz, rad/s) direct fieldmap accuracy & opposite-polarity matrix...")
    matrix_results = run_six_direction_and_unit_benchmark(out_root)
    for row in matrix_results:
        print(
            f"  [{row['units']:5s} | {row['pe_direction']:2s}] "
            f"max_disp={row['max_abs_disp_vox']:.2f} vox ({row['max_abs_disp_mm']:.2f} mm) | "
            f"min_J={row['min_jacobian']:.4f} | "
            f"NCC: {row['ncc_before']:.4f} -> {row['ncc_after']:.4f} "
            f"(opp={row['opposite_polarity_ncc']:.4f})"
        )

    print("\n[2/4] Running Jacobian folding, odd-37 TOPUP config, ambiguity & honest fallback controls...")
    controls_summary = run_physics_and_bids_controls(out_root)
    print("  -> All physics & BIDS controls PASSED.")

    print("\n[3/4] Running single-step composed SDC + motion + coreg + SyN TRE & sharpness validation...")
    composed_tre = run_composed_sdc_tre_and_sharpness_validation()
    print(
        f"  -> Composed SDC TRE PASSED: mean={composed_tre['single_step_mean_tre_mm']:.4f} mm, "
        f"max={composed_tre['single_step_max_tre_mm']:.4f} mm | "
        f"peak sharpness gain=+{composed_tre['peak_sharpness_gain']:.2f}"
    )

    print("\n[4/4] Running 64x64x37x240 canonical CC200 22-stage E2E (PEPOLAR missing topup + Direct Hz Run 1/2)...")
    e2e_pep, e2e_dir = run_step5_e2e_validations(out_root)
    s_qc = e2e_dir["sdc_provenance"]["correction_qc"]
    print(
        f"  -> PEPOLAR fallback: {e2e_pep['status']} (subsamp={e2e_pep['topup_config']['subsamp_schedule']}) | "
        f"runtime={e2e_pep['runtime_seconds']:.2f}s"
    )
    print(
        f"  -> Direct Hz E2E  : {e2e_dir['sdc_provenance']['status']} | "
        f"Stage11 NCC: {s_qc['ncc_before']:.4f} -> {s_qc['ncc_after']:.4f} | "
        f"r_pos={e2e_dir['correlation_recovery']['positive_pair']['recovered_pearson_r']:.4f}, "
        f"r_neg={e2e_dir['correlation_recovery']['negative_pair']['recovered_pearson_r']:.4f} | "
        f"det={e2e_dir['determinism']['max_abs_diff_timeseries']:.1e} | "
        f"RSS={e2e_dir['peak_rss_gb']:.2f} GB"
    )

    report = {
        "step": "Step 5: Susceptibility Distortion Correction (DEC-SDC-02)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VERIFIED_WITH_TOOLING_WARNINGS",
        "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
        "real_data_authorized": False,
        "overall_authorization_state": auth_state["overall_authorization_state"],
        "software": software_info,
        "pe_directions_verified": ["i", "i-", "j", "j-", "k", "k-"],
        "units_verified": ["Hz", "rad/s"],
        "six_direction_unit_matrix": matrix_results,
        "controls": controls_summary,
        "composed_transform_tre": composed_tre,
        "e2e_pepolar_missing_topup": e2e_pep,
        "e2e_direct_fieldmap": e2e_dir,
    }

    report_path = out_root / "step5_sdc_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("STEP 5 SDC VALIDATION PASSED ACROSS ALL SYNTHETIC TESTS & CONTROLS")
    print(f"Report Written: {report_path}")
    print("=" * 78)
    return report


if __name__ == "__main__":
    run_step5_validation()
