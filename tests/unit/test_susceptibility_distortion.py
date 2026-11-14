"""Comprehensive unit, negative-control, composition, and E2E tests for Step 5 SDC (DEC-SDC-02).

Covers all Step 5 scientific, BIDS, safety, and architectural requirements using strictly
synthetic data generated in-memory or inside pytest's temporary workspace:
- External neuroimaging software detection (honest reporting when FSL topup/prelude/fugue are absent)
- BIDS PhaseEncodingDirection parsing (i, i-, j, j-, k, k-; rejects AP/PA/RL/invalid)
- TotalReadoutTime resolution & safety (supplied vs EffectiveEchoSpacing*(ReconMatrixPE-1); never TR)
- BIDS fieldmap discovery & association (B0FieldSource/B0FieldIdentifier > IntendedFor, unassociated
  fmap exclusion, ambiguity fail-closed checks, symlink & clinical path blocking)
- Geometry & FOV compatibility checks (shape/affine mismatch for PEPOLAR, <50% FOV overlap, NaNs,
  phasediff EchoTime1/2 checks)
- Odd-slice TOPUP acquisition parameter & config generation (subsamp=[1,1,1,1] on 37 slices without slicing)
- Honest fallback behavior (SKIPPED_NO_ASSOCIATED_FIELDMAP, SKIPPED_TOOL_UNAVAILABLE, SKIPPED_UNSUPPORTED_METHOD)
- Mocked TOPUP runner subprocess path (success, nonzero exit code, missing --fout fail-closed)
- Calibrated Direct Fieldmap (Hz and rad/s) across all 6 PE directions (i, i-, j, j-, k, k-)
- Negative controls: opposite-polarity worsening, Jacobian folding (J <= 0 -> SDCFoldingError),
  and excessive displacement (> max_displacement_mm -> InvalidSDCInputError)
- Single-step nonlinear spatial composition (compose_sdc_transform, apply_composed_sdc_volume,
  apply_motion_transforms(..., sdc_displacement=...), resample_4d_bold(..., sdc_displacement=...))
  achieving sub-voxel landmark TRE and strictly less blur than sequential interpolation
- Full 22-stage run_poc_v2_pipeline E2E tests & Dashboard SDC API exposure.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import mock

import jsonschema
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from dipy.align.imaffine import AffineMap
from dipy.align.imwarp import DiffeomorphicMap
from scipy.ndimage import center_of_mass, gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation

from scripts.run_poc_v2 import (
    CANONICAL_CC200_AFFINE,
    CANONICAL_CC200_SHAPE,
    compute_manifest_hmac,
    compute_sha256,
    generate_synthetic_phantom_session,
    run_poc_v2_pipeline,
)
from scripts.serve_dashboard import DashboardApplication
from src.dashboard.service import DashboardDataService
from src.preprocessing.motion import apply_motion_transforms
from src.preprocessing.registration import (
    apply_composed_sdc_volume,
    compose_sdc_transform,
    compose_transformations,
    resample_4d_bold,
)
from src.preprocessing.susceptibility_distortion import (
    PHASE_ENCODING_DIRECTION_MAP,
    SDC_METHOD_DIRECT_FIELDMAP,
    SDC_METHOD_FIELDMAP_LESS_SYN,
    SDC_METHOD_NONE,
    SDC_METHOD_PEPOLAR_TOPUP,
    SDC_METHOD_PHASE_DIFFERENCE,
    SDC_STATUS_APPLIED_DIRECT_FIELDMAP,
    SDC_STATUS_APPLIED_PEPOLAR_TOPUP,
    SDC_STATUS_SKIPPED_INSUFFICIENT_METADATA,
    SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP,
    SDC_STATUS_SKIPPED_TOOL_UNAVAILABLE,
    SDC_STATUS_SKIPPED_UNSUPPORTED_METHOD,
    AmbiguousFieldmapAssociationError,
    InvalidSDCInputError,
    MissingSDCMetadataError,
    SDCError,
    SDCFoldingError,
    apply_sdc_unwarping_3d,
    apply_sdc_unwarping_4d,
    build_topup_acquisition_parameters,
    compute_total_readout_time,
    detect_sdc_software,
    discover_sdc_inputs,
    evaluate_sdc_correction_qc,
    field_hz_to_displacement,
    format_topup_cnf,
    generate_synthetic_distorted_epi,
    parse_phase_encoding_direction,
    resolve_sdc_method,
    run_susceptibility_distortion_correction,
    select_topup_config,
    validate_displacement_field,
    validate_sdc_metadata,
)


def _make_structured_phantom_3d(
    shape: Tuple[int, int, int] = (28, 28, 21),
    affine: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a rich multi-tissue 3D phantom with sharp cortex/ventricle/gyral structure and smooth B0 lobe."""
    if affine is None:
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
    gx, gy, gz = np.indices(shape, dtype=np.float32)
    cx, cy, cz = (d / 2.0 for d in shape)
    r_ell = ((gx - cx) / (shape[0] * 0.36)) ** 2 + ((gy - cy) / (shape[1] * 0.36)) ** 2 + ((gz - cz) / (shape[2] * 0.38)) ** 2
    brain = r_ell <= 1.0
    vent = ((gx - cx) / (shape[0] * 0.13)) ** 2 + ((gy - cy) / (shape[1] * 0.16)) ** 2 + ((gz - cz) / (shape[2] * 0.14)) ** 2 <= 1.0
    sulcus = (np.abs(gx - (cx + shape[0] * 0.14)) < 1.2) & (gy > cy) & brain
    cortex = (r_ell >= 0.55) & brain & (~vent)

    vol = np.zeros(shape, dtype=np.float32)
    vol[brain] = 90.0
    vol[cortex] = 135.0
    vol[vent] = 25.0
    vol[sulcus] = 45.0
    # Smooth slightly so analytical gradient & cubic/linear sampling behave cleanly like real MRI SNR
    vol = gaussian_filter(vol, sigma=0.75).astype(np.float32)

    # Asymmetric smooth Gaussian B0 off-resonance field peaking around ~38 Hz (well within invertible Jacobian)
    b0_hz = (
        38.0
        * np.exp(
            -(
                ((gx - cx * 1.15) / (shape[0] * 0.18)) ** 2
                + ((gy - cy * 1.25) / (shape[1] * 0.18)) ** 2
                + ((gz - cz * 0.90) / (shape[2] * 0.22)) ** 2
            )
        )
    ).astype(np.float32)
    return vol, b0_hz, affine


def _write_synthetic_bids_bold(
    session_dir: Path,
    subject_id: str = "sub-synthetic501",
    session_id: str = "ses-1",
    run_tag: str = "run-01",
    vol_3d: np.ndarray | None = None,
    n_vols: int = 8,
    tr: float = 2.0,
    affine: np.ndarray | None = None,
    bold_meta_extra: Dict[str, Any] | None = None,
) -> Tuple[Path, Path]:
    func_dir = session_dir / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    if vol_3d is None:
        vol_3d, _, default_aff = _make_structured_phantom_3d()
        affine = default_aff if affine is None else affine
    if affine is None:
        affine = np.diag([3.0, 3.0, 3.0, 1.0])

    data4d = np.repeat(vol_3d[..., None], n_vols, axis=3).astype(np.float32)
    bold_p = func_dir / f"{subject_id}_{session_id}_task-rest_{run_tag}_bold.nii.gz"
    img = nib.Nifti1Image(data4d, affine)
    img.header.set_zooms((abs(float(affine[0, 0])), abs(float(affine[1, 1])), abs(float(affine[2, 2])), float(tr)))
    img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(bold_p))

    meta: Dict[str, Any] = {"RepetitionTime": float(tr), "TaskName": "rest"}
    if bold_meta_extra:
        meta.update(bold_meta_extra)
    json_p = func_dir / f"{subject_id}_{session_id}_task-rest_{run_tag}_bold.json"
    json_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return bold_p, json_p


# ---------------------------------------------------------------------------
# 1. Software Detection, Phase Encoding, and TotalReadoutTime Validation
# ---------------------------------------------------------------------------


def test_detect_sdc_software_honesty(tmp_path: Path):
    """detect_sdc_software reflects real system state and never fakes missing executables."""
    with mock.patch("src.preprocessing.susceptibility_distortion.shutil.which", return_value=None):
        sw = detect_sdc_software()
        assert sw["topup_available"] is False
        assert sw["applytopup_available"] is False
        assert sw["prelude_available"] is False
        assert sw["fugue_available"] is False
        assert sw["ants_registration_available"] is False
        assert sw["afni_3dqwarp_available"] is False
        assert sw["available_external_tools"] == []
        assert sw["python_packages"]["scipy"] is not None
        assert sw["python_packages"]["nibabel"] is not None

    dummy_topup = tmp_path / "topup"
    dummy_topup.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sw_custom = detect_sdc_software(topup_executable=dummy_topup)
    assert sw_custom["topup_available"] is True
    assert sw_custom["topup_path"] == str(dummy_topup)

    sw_missing = detect_sdc_software(topup_executable=tmp_path / "nonexistent_topup")
    assert sw_missing["topup_available"] is False
    assert sw_missing["topup_path"] is None


@pytest.mark.parametrize(
    "code,axis,polarity,vec",
    [
        ("i", 0, 1, (1, 0, 0)),
        ("i-", 0, -1, (-1, 0, 0)),
        ("j", 1, 1, (0, 1, 0)),
        ("j-", 1, -1, (0, -1, 0)),
        ("k", 2, 1, (0, 0, 1)),
        ("k-", 2, -1, (0, 0, -1)),
    ],
)
def test_parse_phase_encoding_direction_valid(code: str, axis: int, polarity: int, vec: Tuple[int, int, int]):
    parsed = parse_phase_encoding_direction(code)
    assert parsed["code"] == code
    assert parsed["axis"] == axis
    assert parsed["polarity"] == polarity
    assert parsed["vector"] == vec


@pytest.mark.parametrize("bad_code", ["AP", "PA", "RL", "LR", "z", "x+", " ", "", 123])
def test_parse_phase_encoding_direction_rejects_ambiguous_or_invalid(bad_code: Any):
    with pytest.raises(InvalidSDCInputError):
        parse_phase_encoding_direction(bad_code)


def test_parse_phase_encoding_direction_missing_raises():
    with pytest.raises(MissingSDCMetadataError):
        parse_phase_encoding_direction(None)


def test_compute_total_readout_time_supplied_and_derived():
    trt, src, warn = compute_total_readout_time({"TotalReadoutTime": 0.045, "RepetitionTime": 2.0})
    assert math.isclose(trt, 0.045)
    assert src == "supplied"
    assert warn == []

    trt_d, src_d, warn_d = compute_total_readout_time(
        {"EffectiveEchoSpacing": 0.00075, "ReconMatrixPE": 64, "RepetitionTime": 2.0},
        spatial_shape=(64, 64, 37),
        pe_axis=1,
    )
    assert math.isclose(trt_d, 0.00075 * 63, rel_tol=1e-9)
    assert src_d == "derived_effective_echo_spacing_recon_matrix_pe"
    assert warn_d == []

    # Warning emitted when ReconMatrixPE != spatial matrix dim (e.g. interpolation/cropping)
    _, _, warn_mismatch = compute_total_readout_time(
        {"EffectiveEchoSpacing": 0.00075, "ReconMatrixPE": 80},
        spatial_shape=(64, 64, 37),
        pe_axis=1,
    )
    assert len(warn_mismatch) == 1
    assert "ReconMatrixPE" in warn_mismatch[0]


@pytest.mark.parametrize(
    "bad_meta,exc_type",
    [
        ({"RepetitionTime": 2.0}, MissingSDCMetadataError),
        ({"EffectiveEchoSpacing": 0.0007}, MissingSDCMetadataError),
        ({"ReconMatrixPE": 64}, MissingSDCMetadataError),
        ({"TotalReadoutTime": 0.0}, InvalidSDCInputError),
        ({"TotalReadoutTime": -0.05}, InvalidSDCInputError),
        ({"TotalReadoutTime": float("nan")}, InvalidSDCInputError),
        ({"TotalReadoutTime": True}, InvalidSDCInputError),
        ({"TotalReadoutTime": 2.0, "RepetitionTime": 2.0}, InvalidSDCInputError),
        ({"TotalReadoutTime": 1.2}, InvalidSDCInputError),
        ({"EffectiveEchoSpacing": 0.0007, "ReconMatrixPE": 1}, InvalidSDCInputError),
        ({"EffectiveEchoSpacing": 0.0007, "ReconMatrixPE": 64.5}, InvalidSDCInputError),
    ],
)
def test_compute_total_readout_time_fail_closed(bad_meta: Dict[str, Any], exc_type: type):
    with pytest.raises(exc_type):
        compute_total_readout_time(bad_meta)


# ---------------------------------------------------------------------------
# 2. BIDS Fieldmap Discovery, Priority, Ambiguity, & Safety Hygiene
# ---------------------------------------------------------------------------


def test_discover_sdc_inputs_priority_unassociated_and_ambiguity(tmp_path: Path):
    ses_dir = tmp_path / "bids" / "sub-synthetic502" / "ses-1"
    bold_p, json_p = _write_synthetic_bids_bold(
        ses_dir,
        subject_id="sub-synthetic502",
        run_tag="run-01",
        bold_meta_extra={
            "PhaseEncodingDirection": "j",
            "TotalReadoutTime": 0.05,
            "B0FieldSource": "pepolar_run01",
        },
    )
    # Also write an unrelated run-02 BOLD whose fmap should stay unassociated for run-01
    bold_r2, _ = _write_synthetic_bids_bold(
        ses_dir,
        subject_id="sub-synthetic502",
        run_tag="run-02",
        bold_meta_extra={"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05},
    )

    fmap_dir = ses_dir / "fmap"
    fmap_dir.mkdir(parents=True, exist_ok=True)
    vol, b0_hz, aff = _make_structured_phantom_3d()

    # Associated PEPOLAR pair via B0FieldIdentifier + companion IntendedFor for run-02 untouched
    for tag, pe in (("AP", "j"), ("PA", "j-")):
        f_nii = fmap_dir / f"sub-synthetic502_ses-1_dir-{tag}_epi.nii.gz"
        f_json = fmap_dir / f"sub-synthetic502_ses-1_dir-{tag}_epi.json"
        nib.save(nib.Nifti1Image(vol, aff), str(f_nii))
        f_json.write_text(
            json.dumps(
                {
                    "PhaseEncodingDirection": pe,
                    "TotalReadoutTime": 0.05,
                    "B0FieldIdentifier": ["pepolar_run01"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Extra unassociated fieldmap intended only for run-02 + orphan random scan
    r2_nii = fmap_dir / "sub-synthetic502_ses-1_acq-run02_fieldmap.nii.gz"
    r2_json = fmap_dir / "sub-synthetic502_ses-1_acq-run02_fieldmap.json"
    nib.save(nib.Nifti1Image(b0_hz, aff), str(r2_nii))
    r2_json.write_text(
        json.dumps({"Units": "Hz", "IntendedFor": [f"bids::sub-synthetic502/ses-1/func/{bold_r2.name}"]}),
        encoding="utf-8",
    )

    disc = discover_sdc_inputs(bold_p, bold_json_path=json_p)
    assert disc["associated"] is True
    assert disc["association_mechanism"] == "B0FieldSource_B0FieldIdentifier"
    assert disc["candidate_modality_group"] == "pepolar_epi"
    assert sorted(disc["unassociated_fmap_files"]) == [r2_nii.name]

    # Now add IntendedFor pointing to run-01 on r2_json -> conflicting modality ambiguity!
    r2_json.write_text(
        json.dumps({"Units": "Hz", "IntendedFor": [f"ses-1/func/{bold_p.name}"]}),
        encoding="utf-8",
    )
    # Clear B0FieldSource on bold so IntendedFor sees two AP/PA empty -> actually both r2 AND add r1 fieldmap conflict
    fmap_conf = fmap_dir / "sub-synthetic502_ses-1_acq-conflict_fieldmap.nii.gz"
    fmap_conf_j = fmap_dir / "sub-synthetic502_ses-1_acq-conflict_fieldmap.json"
    nib.save(nib.Nifti1Image(b0_hz, aff), str(fmap_conf))
    fmap_conf_j.write_text(
        json.dumps({"Units": "Hz", "B0FieldIdentifier": "pepolar_run01"}),
        encoding="utf-8",
    )
    with pytest.raises(AmbiguousFieldmapAssociationError):
        discover_sdc_inputs(bold_p, bold_json_path=json_p)


def test_discover_sdc_blocks_symlinks_clinical_paths_and_traversal(tmp_path: Path):
    ses_dir = tmp_path / "sub-synthetic503" / "ses-1"
    bold_p, json_p = _write_synthetic_bids_bold(
        ses_dir,
        subject_id="sub-synthetic503",
        bold_meta_extra={"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05},
    )
    fmap_dir = ses_dir / "fmap"
    fmap_dir.mkdir(parents=True, exist_ok=True)
    vol, _, aff = _make_structured_phantom_3d()

    real_file = tmp_path / "external_fmap.nii.gz"
    nib.save(nib.Nifti1Image(vol, aff), str(real_file))
    sym_file = fmap_dir / "sub-synthetic503_ses-1_dir-AP_epi.nii.gz"
    sym_file.symlink_to(real_file)
    with pytest.raises(InvalidSDCInputError, match="Symlink"):
        discover_sdc_inputs(bold_p, json_p)

    sym_file.unlink()
    bad_trav_nii = fmap_dir / "sub-synthetic503_ses-1_fieldmap.nii.gz"
    bad_trav_json = fmap_dir / "sub-synthetic503_ses-1_fieldmap.json"
    nib.save(nib.Nifti1Image(vol, aff), str(bad_trav_nii))
    bad_trav_json.write_text(
        json.dumps({"Units": "Hz", "IntendedFor": ["../func/escape.nii.gz"]}),
        encoding="utf-8",
    )
    with pytest.raises(InvalidSDCInputError, match="Unsafe path"):
        discover_sdc_inputs(bold_p, json_p)

    with pytest.raises(InvalidSDCInputError, match="Blocked clinical"):
        discover_sdc_inputs(Path("/tmp/adhd200/sub-2344857_bold.nii.gz"))


# ---------------------------------------------------------------------------
# 3. Geometry & Metadata Compatibility, TOPUP acqparams, Odd-Slice Config
# ---------------------------------------------------------------------------


def test_pepolar_unequal_readout_times_valid_and_degenerate_rejected(tmp_path: Path):
    lines = build_topup_acquisition_parameters(["j", "j-"], [0.05, 0.085])
    assert lines == ["0 1 0 0.05000000", "0 -1 0 0.08500000"]

    with pytest.raises(InvalidSDCInputError, match="unidentifiable"):
        build_topup_acquisition_parameters(["j", "j"], [0.05, 0.05])

    # Same-polarity unequal readout times must still be rejected; opposite-polarity unequal readout times accepted
    with pytest.raises(InvalidSDCInputError, match="opposite phase-encoding"):
        build_topup_acquisition_parameters(["j", "j"], [0.04, 0.08])
    lines_opp_diff_trt = build_topup_acquisition_parameters(["i", "i-"], [0.04, 0.08])
    assert lines_opp_diff_trt == ["1 0 0 0.04000000", "-1 0 0 0.08000000"]

    # Test validate_sdc_metadata rejects shape mismatch, orthogonal PE axis, and <50% FOV overlap
    ses_dir = tmp_path / "sub-synthetic504" / "ses-1"
    vol, b0_hz, aff = _make_structured_phantom_3d(shape=(28, 28, 21))
    bold_p, json_p = _write_synthetic_bids_bold(
        ses_dir,
        subject_id="sub-synthetic504",
        vol_3d=vol,
        affine=aff,
        bold_meta_extra={"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05},
    )
    fmap_dir = ses_dir / "fmap"
    fmap_dir.mkdir(parents=True, exist_ok=True)

    # Shifted direct fieldmap overlapping 0% of BOLD FOV
    far_aff = aff.copy()
    far_aff[:3, 3] = [300.0, 300.0, 300.0]
    f_nii = fmap_dir / "sub-synthetic504_ses-1_fieldmap.nii.gz"
    f_json = fmap_dir / "sub-synthetic504_ses-1_fieldmap.json"
    nib.save(nib.Nifti1Image(b0_hz, far_aff), str(f_nii))
    f_json.write_text(json.dumps({"Units": "Hz", "IntendedFor": [f"ses-1/func/{bold_p.name}"]}), encoding="utf-8")

    disc = discover_sdc_inputs(bold_p, json_p)
    b_img = nib.load(str(bold_p))
    b_meta = json.loads(json_p.read_text(encoding="utf-8"))
    with pytest.raises(InvalidSDCInputError, match="FOV"):
        validate_sdc_metadata(b_img, b_meta, disc)


def test_select_topup_config_odd_37_slices_never_crops_or_pads():
    cfg_odd = select_topup_config((64, 64, 37))
    assert cfg_odd["odd_dimension_detected"] is True
    assert cfg_odd["odd_dimensions"] == ["k"]
    assert cfg_odd["slice_modification_applied"] is False
    assert cfg_odd["max_safe_subsamp"] == 1
    assert cfg_odd["subsamp_schedule"] == [1, 1, 1, 1]
    cnf_text = format_topup_cnf(cfg_odd)
    assert "--subsamp=1,1,1,1" in cnf_text

    cfg_div4 = select_topup_config((64, 64, 36))
    assert cfg_div4["odd_dimension_detected"] is False
    assert cfg_div4["subsamp_schedule"] == [4, 2, 1, 1]

    cfg_div2 = select_topup_config((64, 64, 38))
    assert cfg_div2["odd_dimension_detected"] is False
    assert cfg_div2["subsamp_schedule"] == [2, 1, 1, 1]


# ---------------------------------------------------------------------------
# 4. Honest Fallbacks, Missing Tools, & Phasediff Protection
# ---------------------------------------------------------------------------


def test_honest_fallbacks_no_fmap_unsupported_syn_phasediff_and_missing_topup(tmp_path: Path):
    # A. No fmap present
    ses_a = tmp_path / "a" / "sub-synthetic505" / "ses-1"
    bold_a, json_a = _write_synthetic_bids_bold(
        ses_a, subject_id="sub-synthetic505", bold_meta_extra={"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05}
    )
    res_none = run_susceptibility_distortion_correction(bold_a, json_a)
    assert res_none.status == SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP
    assert res_none.corrected is False
    assert res_none.displacement_vox is None

    # B. Explicitly request fieldmap_less_syn -> SKIPPED_UNSUPPORTED_METHOD (never fake with generic SyN!)
    res_syn = run_susceptibility_distortion_correction(bold_a, json_a, requested_method="fieldmap_less_syn")
    assert res_syn.status == SDC_STATUS_SKIPPED_UNSUPPORTED_METHOD
    assert res_syn.method == SDC_METHOD_FIELDMAP_LESS_SYN
    assert res_syn.corrected is False

    # C. PEPOLAR 64x64x37 without topup binary -> SKIPPED_TOOL_UNAVAILABLE + valid acqparams & subsamp=1 config
    vol37, _, aff37 = _make_structured_phantom_3d(shape=(24, 24, 37))
    ses_c = tmp_path / "c" / "sub-synthetic506" / "ses-1"
    bold_c, json_c = _write_synthetic_bids_bold(
        ses_c,
        subject_id="sub-synthetic506",
        vol_3d=vol37,
        affine=aff37,
        bold_meta_extra={"PhaseEncodingDirection": "j", "EffectiveEchoSpacing": 0.00075, "ReconMatrixPE": 24},
    )
    fmap_c = ses_c / "fmap"
    fmap_c.mkdir(parents=True, exist_ok=True)
    for tag, pe, trt in (("AP", "j", 0.04), ("PA", "j-", 0.06)):
        nib.save(nib.Nifti1Image(vol37, aff37), str(fmap_c / f"sub-synthetic506_ses-1_dir-{tag}_epi.nii.gz"))
        (fmap_c / f"sub-synthetic506_ses-1_dir-{tag}_epi.json").write_text(
            json.dumps(
                {"PhaseEncodingDirection": pe, "TotalReadoutTime": trt, "IntendedFor": [f"ses-1/func/{bold_c.name}"]}
            ),
            encoding="utf-8",
        )

    with mock.patch("src.preprocessing.susceptibility_distortion.shutil.which", return_value=None):
        res_pep_skip = run_susceptibility_distortion_correction(bold_c, json_c)
        assert res_pep_skip.status == SDC_STATUS_SKIPPED_TOOL_UNAVAILABLE
        assert res_pep_skip.corrected is False
        assert res_pep_skip.missing_tools == ["topup"]
        assert res_pep_skip.acqparams_lines == ["0 1 0 0.04000000", "0 -1 0 0.06000000"]
        assert res_pep_skip.topup_config is not None
        assert res_pep_skip.topup_config["subsamp_schedule"] == [1, 1, 1, 1]
        assert res_pep_skip.topup_config["slice_modification_applied"] is False
        assert res_pep_skip.planned_command is not None
        assert "topup" in res_pep_skip.planned_command[0]

        with pytest.raises(SDCError):
            run_susceptibility_distortion_correction(bold_c, json_c, strict_sdc=True)

    # D. Phasediff without prelude/fugue -> SKIPPED_TOOL_UNAVAILABLE + never treats wrapped phase as Hz
    ses_d = tmp_path / "d" / "sub-synthetic507" / "ses-1"
    bold_d, json_d = _write_synthetic_bids_bold(
        ses_d,
        subject_id="sub-synthetic507",
        vol_3d=vol37,
        affine=aff37,
        bold_meta_extra={"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05},
    )
    fmap_d = ses_d / "fmap"
    fmap_d.mkdir(parents=True, exist_ok=True)
    pd_arr = (np.ones(vol37.shape, dtype=np.float32) * 1.5)
    nib.save(nib.Nifti1Image(pd_arr, aff37), str(fmap_d / "sub-synthetic507_ses-1_phasediff.nii.gz"))
    (fmap_d / "sub-synthetic507_ses-1_phasediff.json").write_text(
        json.dumps(
            {
                "EchoTime1": 0.00492,
                "EchoTime2": 0.00738,
                "Units": "rad",
                "IntendedFor": [f"ses-1/func/{bold_d.name}"],
            }
        ),
        encoding="utf-8",
    )
    with mock.patch("src.preprocessing.susceptibility_distortion.shutil.which", return_value=None):
        res_pd = run_susceptibility_distortion_correction(bold_d, json_d)
        assert res_pd.status == SDC_STATUS_SKIPPED_TOOL_UNAVAILABLE
        assert res_pd.method == SDC_METHOD_PHASE_DIFFERENCE
        assert res_pd.corrected is False
        assert set(res_pd.missing_tools) == {"prelude", "fugue"}

    # Negative check: EchoTime1 == EchoTime2 invalidates phasediff metadata
    (fmap_d / "sub-synthetic507_ses-1_phasediff.json").write_text(
        json.dumps(
            {
                "EchoTime1": 0.005,
                "EchoTime2": 0.005,
                "Units": "rad",
                "IntendedFor": [f"ses-1/func/{bold_d.name}"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(InvalidSDCInputError, match="EchoTime"):
        run_susceptibility_distortion_correction(bold_d, json_d)


# ---------------------------------------------------------------------------
# 5. Mocked TOPUP Subprocess Path & Failure Modes
# ---------------------------------------------------------------------------


def test_pepolar_topup_runner_success_and_failure_controls(tmp_path: Path):
    true_3d, b0_hz, aff = _make_structured_phantom_3d(shape=(28, 28, 21))
    trt = 0.055
    pe = "j"
    dist_3d, _ = generate_synthetic_distorted_epi(true_3d, b0_hz, pe_direction=pe, total_readout_time_s=trt, affine=aff)
    se_ap, _ = generate_synthetic_distorted_epi(true_3d, b0_hz, pe_direction="j", total_readout_time_s=trt, affine=aff)
    se_pa, _ = generate_synthetic_distorted_epi(true_3d, b0_hz, pe_direction="j-", total_readout_time_s=trt, affine=aff)

    ses_dir = tmp_path / "sub-synthetic508" / "ses-1"
    bold_p, json_p = _write_synthetic_bids_bold(
        ses_dir,
        subject_id="sub-synthetic508",
        vol_3d=dist_3d,
        affine=aff,
        bold_meta_extra={"PhaseEncodingDirection": pe, "TotalReadoutTime": trt},
    )
    fmap_dir = ses_dir / "fmap"
    fmap_dir.mkdir(parents=True, exist_ok=True)
    for tag, d_pe, vol in (("AP", "j", se_ap), ("PA", "j-", se_pa)):
        nib.save(nib.Nifti1Image(vol, aff), str(fmap_dir / f"sub-synthetic508_ses-1_dir-{tag}_epi.nii.gz"))
        (fmap_dir / f"sub-synthetic508_ses-1_dir-{tag}_epi.json").write_text(
            json.dumps(
                {"PhaseEncodingDirection": d_pe, "TotalReadoutTime": trt, "IntendedFor": [f"ses-1/func/{bold_p.name}"]}
            ),
            encoding="utf-8",
        )

    def _mock_topup_runner(cmd: List[str]) -> subprocess.CompletedProcess:
        fout_arg = [arg.split("=", 1)[1] for arg in cmd if arg.startswith("--fout=")][0]
        iout_arg = [arg.split("=", 1)[1] for arg in cmd if arg.startswith("--iout=")][0]
        nib.save(nib.Nifti1Image(b0_hz, aff), fout_arg)
        nib.save(nib.Nifti1Image(np.stack([true_3d, true_3d], axis=-1), aff), iout_arg)
        return subprocess.CompletedProcess(cmd, 0, stdout="mock topup converged", stderr="")

    fake_topup = tmp_path / "fake_topup"
    fake_topup.write_text("#!/bin/sh\n", encoding="utf-8")
    ref_img = nib.Nifti1Image(true_3d, aff)

    res = run_susceptibility_distortion_correction(
        bold_p,
        json_p,
        derivative_dir=ses_dir / "derivatives",
        subject_id="sub-synthetic508",
        session_id="ses-1",
        topup_executable=fake_topup,
        topup_runner=_mock_topup_runner,
        reference_img=ref_img,
    )
    assert res.status == SDC_STATUS_APPLIED_PEPOLAR_TOPUP
    assert res.corrected is True
    assert res.executed_command is not None
    assert res.topup_config is not None
    assert res.topup_config["subsamp_schedule"] == [1, 1, 1, 1]
    assert res.jacobian_qc is not None and res.jacobian_qc["folding_detected"] is False
    assert res.correction_qc is not None
    assert res.correction_qc["ncc_after"] > res.correction_qc["ncc_before"]
    assert res.correction_qc["nmi_after"] > res.correction_qc["nmi_before"]
    assert res.correction_qc["ncc_after"] > 0.96

    # Negative: nonzero rc
    def _failing_runner(cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="TOPUP failed")

    with pytest.raises(InvalidSDCInputError, match="exited with code 2"):
        run_susceptibility_distortion_correction(
            bold_p, json_p, topup_executable=fake_topup, topup_runner=_failing_runner
        )

    # Negative: missing --fout output
    def _no_fout_runner(cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    with pytest.raises(InvalidSDCInputError, match="missing"):
        run_susceptibility_distortion_correction(
            bold_p, json_p, topup_executable=fake_topup, topup_runner=_no_fout_runner
        )


# ---------------------------------------------------------------------------
# 6. Calibrated Direct Fieldmap Physics (Hz & rad/s) Across All 6 PE Directions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pe_dir", list(PHASE_ENCODING_DIRECTION_MAP.keys()))
@pytest.mark.parametrize("units", ["Hz", "rad/s"])
def test_direct_fieldmap_all_six_pe_directions_and_units(tmp_path: Path, pe_dir: str, units: str):
    true_3d, b0_hz, aff = _make_structured_phantom_3d(shape=(26, 26, 21))
    trt = 0.055
    dist_3d, fwd_qc = generate_synthetic_distorted_epi(
        true_3d, b0_hz, pe_direction=pe_dir, total_readout_time_s=trt, affine=aff
    )
    assert fwd_qc["fixed_point_max_residual_vox"] < 1e-6

    ses_dir = tmp_path / f"sub-pe_{pe_dir}_{units.replace('/', '_')}" / "ses-1"
    bold_p, json_p = _write_synthetic_bids_bold(
        ses_dir,
        subject_id="sub-synthetic509",
        vol_3d=dist_3d,
        affine=aff,
        bold_meta_extra={"PhaseEncodingDirection": pe_dir, "TotalReadoutTime": trt},
    )
    fmap_dir = ses_dir / "fmap"
    fmap_dir.mkdir(parents=True, exist_ok=True)
    stored_fmap = b0_hz if units == "Hz" else (b0_hz * np.float32(2.0 * math.pi))
    nib.save(nib.Nifti1Image(stored_fmap, aff), str(fmap_dir / "sub-synthetic509_ses-1_fieldmap.nii.gz"))
    (fmap_dir / "sub-synthetic509_ses-1_fieldmap.json").write_text(
        json.dumps({"Units": units, "IntendedFor": [f"ses-1/func/{bold_p.name}"]}),
        encoding="utf-8",
    )

    ref_img = nib.Nifti1Image(true_3d, aff)
    res = run_susceptibility_distortion_correction(
        bold_p,
        json_p,
        derivative_dir=ses_dir / "derivatives",
        subject_id="sub-synthetic509",
        reference_img=ref_img,
    )
    assert res.status == SDC_STATUS_APPLIED_DIRECT_FIELDMAP
    assert res.corrected is True
    assert res.bold_pe_direction == pe_dir
    assert res.displacement_stats["orthogonal_residual_max_vox"] == 0.0
    assert res.displacement_stats["finite_percent"] == 100.0
    assert res.jacobian_qc["folding_voxel_count"] == 0
    assert res.jacobian_qc["invertible"] is True
    assert res.correction_qc is not None
    assert res.correction_qc["ncc_after"] > res.correction_qc["ncc_before"]
    assert res.correction_qc["nmi_after"] > res.correction_qc["nmi_before"]
    assert res.correction_qc["ncc_after"] > 0.96

    # Negative polarity control: inverting PE sign must strictly DEGRADE alignment vs distorted & true unwarping
    opp_pe = pe_dir[:-1] if pe_dir.endswith("-") else f"{pe_dir}-"
    wrong_disp_vox, _ = field_hz_to_displacement(b0_hz, opp_pe, trt, aff)
    wrong_unwarped, _ = apply_sdc_unwarping_3d(dist_3d, wrong_disp_vox)
    wrong_qc = evaluate_sdc_correction_qc(dist_3d, wrong_unwarped, true_3d)
    assert wrong_qc["ncc_after"] < res.correction_qc["ncc_before"]
    assert wrong_qc["ncc_after"] < res.correction_qc["ncc_after"] - 0.05


def test_jacobian_folding_and_excessive_displacement_rejected():
    _, _, aff = _make_structured_phantom_3d(shape=(24, 24, 21))
    gx, _, _ = np.indices((24, 24, 21), dtype=np.float32)
    # Steep negative gradient along x -> J = 1 + du/dx < 0 (physical signal folding)
    folding_field_hz = (-120.0 * np.exp(-((gx - 12.0) / 1.5) ** 2)).astype(np.float32) * np.ones((24, 24, 21), dtype=np.float32)
    disp_vox, disp_mm = field_hz_to_displacement(folding_field_hz, "i", 0.08, aff)
    with pytest.raises(SDCFoldingError, match="folding"):
        validate_displacement_field(disp_vox, disp_mm, pe_axis=0, affine=aff, raise_on_folding=True)

    # Smooth extreme field exceeding max_displacement_mm ceiling -> InvalidSDCInputError
    huge_hz = np.full((24, 24, 21), 500.0, dtype=np.float32)
    disp_huge_v, disp_huge_m = field_hz_to_displacement(huge_hz, "j", 0.05, aff)
    with pytest.raises(InvalidSDCInputError, match="physical ceiling"):
        validate_displacement_field(disp_huge_v, disp_huge_m, pe_axis=1, affine=aff, max_displacement_mm=35.0)


def test_apply_sdc_unwarping_4d_preserves_shape_affine_and_slope():
    true_3d, b0_hz, aff = _make_structured_phantom_3d(shape=(20, 20, 15))
    dist_3d, _ = generate_synthetic_distorted_epi(true_3d, b0_hz, "j", 0.05, affine=aff)
    b4d = nib.Nifti1Image(np.repeat(dist_3d[..., None], 10, axis=3), aff)
    disp_vox, _ = field_hz_to_displacement(b0_hz, "j", 0.05, aff)
    out_img, qc4 = apply_sdc_unwarping_4d(b4d, disp_vox)
    assert out_img.shape == (20, 20, 15, 10)
    assert np.allclose(out_img.affine, aff)
    assert np.isfinite(out_img.get_fdata()).all()
    assert qc4["mean_oob_percent"] >= 0.0


# ---------------------------------------------------------------------------
# 7. Single-Step Composed Nonlinear Spatial Transform (SDC + Motion + Coreg + SyN)
# ---------------------------------------------------------------------------


def test_single_step_composed_sdc_transform_subvoxel_tre_and_sharpness_gain():
    """Verify compose_sdc_transform + resample_4d_bold accuracy & sharpness over multi-step interpolation."""
    # Anisotropic mixed-sign affines (RAS EPI, oblique T1w, LAS MNI target) + nonzero SyN + rigid motion + B0 field
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

    # Known B0 off-resonance field along j (TRT=0.05s)
    gx, gy, gz = np.indices(epi_shape, dtype=np.float32)
    b0_hz = (
        22.0
        * np.exp(-(((gx - 15.0) / 6.0) ** 2 + ((gy - 15.0) / 6.0) ** 2 + ((gz - 12.0) / 5.5) ** 2))
    ).astype(np.float32)
    disp_vox, _ = field_hz_to_displacement(b0_hz, "j", 0.05, epi_affine)

    # Rigid motion M_ref_to_vol (world) and non-commuting EPI->T1w & T1w->MNI affines
    r_mot = np.eye(4, dtype=np.float64)
    r_mot[:3, :3] = Rotation.from_euler("xyz", [2.5, -1.8, 3.2], degrees=True).as_matrix()
    r_mot[:3, 3] = [1.8, -2.4, 1.5]

    rigid_epi_to_t1 = np.eye(4, dtype=np.float64)
    rigid_epi_to_t1[:3, :3] = Rotation.from_euler("zyx", [-3.0, 2.0, 1.5], degrees=True).as_matrix()
    rigid_epi_to_t1[:3, 3] = [-1.5, 2.2, -1.1]

    aff_t1_to_mni = np.eye(4, dtype=np.float64)
    aff_t1_to_mni[:3, :3] = Rotation.from_euler("xyz", [1.5, 2.5, -2.0], degrees=True).as_matrix()
    aff_t1_to_mni[:3, 3] = [1.2, -1.5, 0.8]

    # Smooth nonzero SyN backward field in MNI physical space
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

    # Synthesize raw distorted+moved EPI phantom by inverting the full 5-stage pullback Phi(x_MNI) -> v_raw
    # at every EPI voxel, then sampling well-separated isotropic MNI landmark spheres (sigma=3.6 mm in MNI world).
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
    v_raw_all = np.vstack([gi_e.ravel(), gj_e.ravel(), gk_e.ravel()])  # (3, N)
    # Initial guess ignoring nonlinear SDC & SyN
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
        v_dist_ref = v_undist + u_shift
        v_pred = m_vox_motion[:3, :3] @ v_dist_ref + m_vox_motion[:3, 3:4]
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
        # Verify analytical coordinate pullback at exact landmark grid point
        idx_flat = int(i_m) * (mni_shape[1] * mni_shape[2]) + int(j_m) * mni_shape[2] + int(k_m)
        b_k = backward_field[int(i_m), int(j_m), int(k_m)].astype(np.float64)
        v_u_k = (inv_epi @ (prealign_inv @ np.append(x_mni_k + b_k, 1.0)))[:3]
        u_k = np.array(
            [map_coordinates(disp_vox[..., c].astype(np.float64), v_u_k[:, None], order=1, mode="nearest")[0] for c in range(3)]
        )
        v_exact_k = m_vox_motion[:3, :3] @ (v_u_k + u_k) + m_vox_motion[:3, 3]
        assert float(np.linalg.norm(grid["v_vol"][:, idx_flat] - v_exact_k)) < 1e-11

        d2 = np.sum((x_mni - x_mni_k[:, None]) ** 2, axis=0)
        raw_epi_flat += 100.0 * np.exp(-d2 / (2.0 * sigma_mm ** 2))

    raw_epi_phantom = raw_epi_flat.reshape(epi_shape).astype(np.float32)

    recovered_mni = apply_composed_sdc_volume(raw_epi_phantom, grid, interpolation="linear")

    # Compare against sequential multi-step interpolation: step1(SDC+motion -> EPI) -> step2(EPI -> MNI)
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
        com_ijk = lm_ijk + com_rel
        x_est = (mni_affine @ np.append(com_ijk, 1.0))[:3]
        single_errors_mm.append(float(np.linalg.norm(x_est - x_true)))
        peak_single.append(float(np.max(sub_single)))
        peak_seq.append(float(np.max(sub_seq)))

    max_tre = max(single_errors_mm)
    assert max_tre < 0.08, f"Expected sub-voxel <0.08mm TRE, got {max_tre:.4f} mm"
    # Single interpolation preserves peak amplitude strictly better (less smoothing blur) than sequential resampling!
    assert np.mean(peak_single) > np.mean(peak_seq) + 0.5

    # Also verify 4D resample_4d_bold(..., motion_transforms=..., sdc_displacement=...)
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
    assert np.allclose(res_4d.affine, mni_affine)
    np.testing.assert_allclose(np.asanyarray(res_4d.dataobj)[..., 0], recovered_mni, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. End-to-End 22-Stage Pipeline Integration + Dashboard SDC Exposure
# ---------------------------------------------------------------------------


def test_run_poc_v2_22stage_with_direct_fieldmap_and_pepolar_fallback(tmp_path: Path):
    """Verify full 22-stage pipeline with Direct Fieldmap, PEPOLAR SKIPPED_TOOL_UNAVAILABLE, and strict quarantine."""
    with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": "step5_e2e_test_hmac_key_2026"}, clear=False):
        os.environ.pop("NEUROADHD_MODEL_DIR", None)

        # 1. Baseline session WITHOUT fieldmap (validates SKIPPED_NO_ASSOCIATED_FIELDMAP & baseline raw motion)
        base_paths = generate_synthetic_phantom_session(
            tmp_path / "in_base", subject_id="sub-synthetic510", t_vol=55, seed=77
        )
        res_base = run_poc_v2_pipeline(
            bold_path=base_paths["bold_nii"],
            bold_json_path=base_paths["bold_json"],
            t1w_path=base_paths["t1w_nii"],
            atlas_path=base_paths["cc200_atlas"],
            template_path=base_paths["mni_template"],
            output_dir=tmp_path / "out_base",
            subject_id="sub-synthetic510",
            manifest_path=base_paths["synthetic_manifest"],
        )
        assert res_base["provenance"]["pipeline_stages_total"] == 23
        assert res_base["provenance"]["susceptibility_distortion_correction"]["status"] == SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP
        assert res_base["provenance"]["motion_estimation"]["single_step_composed_resampling_with_sdc"] is False

        # Strictly reject --strict-sdc when no fieldmap is present (must quarantine into failed_run_*)
        with pytest.raises(SDCError):
            run_poc_v2_pipeline(
                bold_path=base_paths["bold_nii"],
                bold_json_path=base_paths["bold_json"],
                t1w_path=base_paths["t1w_nii"],
                atlas_path=base_paths["cc200_atlas"],
                template_path=base_paths["mni_template"],
                output_dir=tmp_path / "out_strict_fail",
                subject_id="sub-synthetic510",
                manifest_path=base_paths["synthetic_manifest"],
                strict_sdc=True,
            )
        assert len(list((tmp_path / "out_strict_fail" / "sub-synthetic510").glob("failed_run_*"))) == 1

        # 2. Direct Fieldmap session (forward-distorted BOLD + BIDS direct Hz fieldmap)
        dir_paths = generate_synthetic_phantom_session(
            tmp_path / "in_direct",
            subject_id="sub-synthetic511",
            t_vol=55,
            seed=77,
            phase_encoding_direction="j",
            total_readout_time=0.055,
            fieldmap_mode="direct_hz",
        )
        cohort_root = tmp_path / "dash_root"
        cohort_dir = cohort_root / "cohort_step5test"
        res_direct = run_poc_v2_pipeline(
            bold_path=dir_paths["bold_nii"],
            bold_json_path=dir_paths["bold_json"],
            t1w_path=dir_paths["t1w_nii"],
            atlas_path=dir_paths["cc200_atlas"],
            template_path=dir_paths["mni_template"],
            output_dir=cohort_dir / "runs",
            subject_id="sub-synthetic511",
            manifest_path=dir_paths["synthetic_manifest"],
        )
        prov_d = res_direct["provenance"]
        assert prov_d["pipeline_stages_total"] == 23
        assert len(prov_d["preprocessing_order"]) == 23
        assert prov_d["motion_estimation"]["estimated_on_raw_uncorrected_bold"] is True
        assert prov_d["motion_estimation"]["single_step_composed_resampling_with_sdc"] is True
        sdc_d = prov_d["susceptibility_distortion_correction"]
        assert sdc_d["status"] == SDC_STATUS_APPLIED_DIRECT_FIELDMAP
        assert sdc_d["corrected"] is True
        assert sdc_d["bold_phase_encoding_direction"] == "j"
        assert sdc_d["jacobian_qc"]["folding_detected"] is False
        assert sdc_d["correction_qc"] is not None
        assert sdc_d["correction_qc"]["ncc_after"] > sdc_d["correction_qc"]["ncc_before"]

        run_dir_d = Path(res_direct["output_dir"])
        for rel_key in ("fieldmap_nii", "displacement_nii", "qc_png", "report_json"):
            rel_val = sdc_d["derivative_files"][rel_key]
            assert (run_dir_d / rel_val).is_file()

        schema_dict = json.loads(Path("configs/provenance_schema.json").read_text(encoding="utf-8"))
        jsonschema.validate(instance=prov_d, schema=schema_dict)

        # Verify raw motion is unaffected by Requested SDC method ("none" vs direct_fieldmap on identical BOLD)
        res_none_same = run_poc_v2_pipeline(
            bold_path=dir_paths["bold_nii"],
            bold_json_path=dir_paths["bold_json"],
            t1w_path=dir_paths["t1w_nii"],
            atlas_path=dir_paths["cc200_atlas"],
            template_path=dir_paths["mni_template"],
            output_dir=tmp_path / "out_direct_none",
            subject_id="sub-synthetic511",
            manifest_path=dir_paths["synthetic_manifest"],
            sdc_method="none",
        )
        m_sdc = pd.read_csv(res_direct["artifacts"]["motion_tsv"], sep="\t").to_numpy()
        m_none = pd.read_csv(res_none_same["artifacts"]["motion_tsv"], sep="\t").to_numpy()
        np.testing.assert_array_equal(m_sdc, m_none)

        # 3. PEPOLAR session when topup binary is missing -> SKIPPED_TOOL_UNAVAILABLE (never fabricates field)
        pep_paths = generate_synthetic_phantom_session(
            tmp_path / "in_pepolar",
            subject_id="sub-synthetic512",
            t_vol=55,
            seed=78,
            phase_encoding_direction="j",
            total_readout_time=0.055,
            fieldmap_mode="pepolar",
        )
        with mock.patch("src.preprocessing.susceptibility_distortion.shutil.which", return_value=None):
            res_pep = run_poc_v2_pipeline(
                bold_path=pep_paths["bold_nii"],
                bold_json_path=pep_paths["bold_json"],
                t1w_path=pep_paths["t1w_nii"],
                atlas_path=pep_paths["cc200_atlas"],
                template_path=pep_paths["mni_template"],
                output_dir=tmp_path / "out_pepolar",
                subject_id="sub-synthetic512",
                manifest_path=pep_paths["synthetic_manifest"],
            )
        sdc_pep = res_pep["provenance"]["susceptibility_distortion_correction"]
        assert sdc_pep["status"] == SDC_STATUS_SKIPPED_TOOL_UNAVAILABLE
        assert sdc_pep["corrected"] is False
        assert sdc_pep["missing_tools"] == ["topup"]
        assert sdc_pep["topup_config"]["subsamp_schedule"] == [2, 1, 1, 1]
        assert res_pep["provenance"]["motion_estimation"]["single_step_composed_resampling_with_sdc"] is False

        # 3b. PEPOLAR session with mocked topup_runner -> APPLIED_PEPOLAR_TOPUP
        b0_synth = np.zeros((30, 30, 20), dtype=np.float32)
        gx_p, gy_p, gz_p = np.indices((30, 30, 20), dtype=np.float32)
        b0_synth = (25.0 * np.exp(-(((gx_p - 15.0) / 5.0) ** 2 + ((gy_p - 19.0) / 5.0) ** 2 + ((gz_p - 9.0) / 4.0) ** 2))).astype(np.float32)
        epi_aff_30 = np.diag([3.0, 3.0, 3.0, 1.0])

        def _e2e_topup_runner(cmd: List[str]) -> subprocess.CompletedProcess:
            fout_p = [a.split("=", 1)[1] for a in cmd if a.startswith("--fout=")][0]
            nib.save(nib.Nifti1Image(b0_synth, epi_aff_30), fout_p)
            return subprocess.CompletedProcess(cmd, 0, stdout="mock e2e topup ok", stderr="")

        fake_topup_bin = tmp_path / "topup"
        fake_topup_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        res_pep_ok = run_poc_v2_pipeline(
            bold_path=pep_paths["bold_nii"],
            bold_json_path=pep_paths["bold_json"],
            t1w_path=pep_paths["t1w_nii"],
            atlas_path=pep_paths["cc200_atlas"],
            template_path=pep_paths["mni_template"],
            output_dir=tmp_path / "out_pepolar_ok",
            subject_id="sub-synthetic512",
            manifest_path=pep_paths["synthetic_manifest"],
            topup_executable=fake_topup_bin,
            topup_runner=_e2e_topup_runner,
        )
        sdc_pep_ok = res_pep_ok["provenance"]["susceptibility_distortion_correction"]
        assert sdc_pep_ok["status"] == SDC_STATUS_APPLIED_PEPOLAR_TOPUP
        assert sdc_pep_ok["corrected"] is True
        assert res_pep_ok["provenance"]["motion_estimation"]["single_step_composed_resampling_with_sdc"] is True

        # 4. Dashboard service & endpoint exposure of Step 5 SDC data
        cohort_dir.mkdir(parents=True, exist_ok=True)
        (cohort_dir / "cohort_manifest.json").write_text(
            json.dumps(
                {
                    "cohort_run_id": "cohort_step5test",
                    "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
                    "real_data_authorized": False,
                    "run_type": "SYNTHETIC_COHORT_ONLY",
                    "status": "COMPLETED",
                    "subjects": [
                        {
                            "subject_id": "sub-synthetic511",
                            "status": "SUCCESS",
                            "run_output_dir": str(run_dir_d),
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        svc = DashboardDataService(cohort_root)
        detail = svc.get_subject_detail("cohort_step5test", "sub-synthetic511")
        assert detail["susceptibility_distortion_correction"]["status"] == SDC_STATUS_APPLIED_DIRECT_FIELDMAP
        app = DashboardApplication()
        ver_status = app.pipeline_verification_status()
        assert ver_status["stages_total"] == 23
        assert "susceptibility_distortion" in ver_status
        assert "susceptibility_distortion_badge" in ver_status


def test_apply_motion_transforms_composed_with_sdc_and_incomplete_metadata(tmp_path: Path):
    """Verify apply_motion_transforms(..., sdc_displacement=...) & SKIPPED_INSUFFICIENT_METADATA."""
    true_3d, b0_hz, aff = _make_structured_phantom_3d(shape=(22, 22, 17))
    dist_3d, _ = generate_synthetic_distorted_epi(true_3d, b0_hz, "j", 0.05, affine=aff)
    disp_vox, _ = field_hz_to_displacement(b0_hz, "j", 0.05, aff)

    # Shift volume in world coordinates so rigid identity-free transform + SDC unwarps back to true_3d
    t_mat = np.eye(4, dtype=np.float64)
    shifted_b4d = nib.Nifti1Image(np.repeat(dist_3d[..., None], 4, axis=3), aff)
    realigned = apply_motion_transforms(
        shifted_b4d,
        [AffineMap(t_mat) for _ in range(4)],
        interpolation="linear",
        sdc_displacement=disp_vox,
    )
    rec_3d = np.asanyarray(realigned.dataobj, dtype=np.float32)[..., 0]
    qc = evaluate_sdc_correction_qc(dist_3d, rec_3d, true_3d)
    assert qc["ncc_after"] > qc["ncc_before"]
    assert qc["ncc_after"] > 0.99

    # Missing BOLD PhaseEncodingDirection when a valid fmap is present -> SKIPPED_INSUFFICIENT_METADATA
    ses_dir = tmp_path / "sub-synthetic514" / "ses-1"
    bold_p, json_p = _write_synthetic_bids_bold(ses_dir, subject_id="sub-synthetic514", vol_3d=dist_3d, affine=aff)
    fmap_dir = ses_dir / "fmap"
    fmap_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(b0_hz, aff), str(fmap_dir / "sub-synthetic514_ses-1_fieldmap.nii.gz"))
    (fmap_dir / "sub-synthetic514_ses-1_fieldmap.json").write_text(
        json.dumps({"Units": "Hz", "IntendedFor": [f"ses-1/func/{bold_p.name}"]}),
        encoding="utf-8",
    )
    res_inc = run_susceptibility_distortion_correction(bold_p, json_p, allow_incomplete_metadata=True)
    assert res_inc.status == SDC_STATUS_SKIPPED_INSUFFICIENT_METADATA
    assert res_inc.corrected is False


def test_fmap_safety_unauthenticated_file_rejected(tmp_path: Path):
    """Dropping an unregistered file into session fmap/ must trigger fail-closed RuntimeError."""
    with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": "step5_fmap_safety_key"}, clear=False):
        paths = generate_synthetic_phantom_session(
            tmp_path / "in_fmap_sec",
            subject_id="sub-synthetic513",
            t_vol=55,
            fieldmap_mode="direct_hz",
        )
        rogue_fmap = paths["fmap_dir"] / "sub-synthetic513_ses-1_acq-rogue_fieldmap.nii.gz"
        rogue_fmap.write_bytes(b"rogue unauthenticated data")
        with pytest.raises(RuntimeError, match="Unauthenticated fieldmap file"):
            run_poc_v2_pipeline(
                bold_path=paths["bold_nii"],
                bold_json_path=paths["bold_json"],
                t1w_path=paths["t1w_nii"],
                atlas_path=paths["cc200_atlas"],
                template_path=paths["mni_template"],
                output_dir=tmp_path / "out_fmap_sec",
                subject_id="sub-synthetic513",
                manifest_path=paths["synthetic_manifest"],
            )
