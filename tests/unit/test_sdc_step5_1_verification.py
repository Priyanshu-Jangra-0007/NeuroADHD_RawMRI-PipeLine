"""Step 5.1 Susceptibility Distortion Correction (SDC) verification tests.

Independently verifies Step 5.1 scientific, geometric, acquisition, composition,
QC-wording, and skipped-correction visibility requirements under synthetic-only
execution without modifying preprocessing code:

1. Fieldmap-to-BOLD spatial registration (known shift recovered)
2. Different fieldmap resolutions and NIfTI affines (anisotropic, flipped axes, oblique)
3. Physical displacement and unit conversions (Hz, rad/s, mm) against hand-computed values
4. TOPUP phase-encoding acquisition requirements (opposite-PE pair required; same-direction pair rejected)
5. Nonlinear transform composition order (motion + SDC + coreg + SyN) with point-tracking
6. SDC quality-control wording: what QC metrics mean and what they do NOT prove
7. Skipped-correction visibility (1 test per item):
   - provenance.json
   - output_manifest.json
   - dashboard HTTP API response (/api/runs/<cohort_id>/subjects/<subject_id>)
"""
from __future__ import annotations

import json
import math
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest import mock

import nibabel as nib
import numpy as np
import pytest
from dipy.align.imwarp import DiffeomorphicMap
from scipy.ndimage import center_of_mass, gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation

import scripts.serve_dashboard as serve_dashboard
from scripts.run_poc_v2 import (
    generate_synthetic_phantom_session,
    run_poc_v2_pipeline,
)
from scripts.run_synthetic_cohort import _verify_run_artifacts
from src.dashboard.service import DashboardDataService
from src.preprocessing.registration import (
    apply_composed_sdc_volume,
    compose_sdc_transform,
    compose_transformations,
    resample_4d_bold,
)
from src.preprocessing.susceptibility_distortion import (
    PHASE_ENCODING_DIRECTION_MAP,
    QC_METRIC_INTERPRETATION,
    SDC_STATUS_APPLIED_DIRECT_FIELDMAP,
    SDC_STATUS_REJECTED_QC_DEGRADED,
    SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP,
    InvalidSDCInputError,
    SDCError,
    build_topup_acquisition_parameters,
    discover_sdc_inputs,
    evaluate_sdc_correction_qc,
    field_hz_to_displacement,
    generate_synthetic_distorted_epi,
    run_susceptibility_distortion_correction,
    validate_sdc_metadata,
)


def _make_asymmetric_brain_3d(
    shape: Tuple[int, int, int] = (28, 28, 20),
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an asymmetric multi-tissue synthetic 3D brain and binary mask."""
    gx, gy, gz = np.indices(shape, dtype=np.float32)
    cx, cy, cz = (d / 2.0 for d in shape)
    r_ell = (
        ((gx - cx) / (shape[0] * 0.35)) ** 2
        + ((gy - cy) / (shape[1] * 0.35)) ** 2
        + ((gz - cz) / (shape[2] * 0.37)) ** 2
    )
    brain = r_ell <= 1.0
    vent = (
        ((gx - cx * 0.95) / (shape[0] * 0.12)) ** 2
        + ((gy - cy * 1.08) / (shape[1] * 0.15)) ** 2
        + ((gz - cz) / (shape[2] * 0.14)) ** 2
    ) <= 1.0
    sulcus = (np.abs(gx - (cx + shape[0] * 0.14)) < 1.2) & (gy > cy) & brain
    gyrus_lobe = (
        ((gx - (cx - shape[0] * 0.16)) / (shape[0] * 0.10)) ** 2
        + ((gy - (cy + shape[1] * 0.18)) / (shape[1] * 0.09)) ** 2
        + ((gz - cz) / (shape[2] * 0.16)) ** 2
    ) <= 1.0

    vol = np.zeros(shape, dtype=np.float32)
    vol[brain] = 90.0
    vol[(r_ell >= 0.55) & brain] = 140.0
    vol[vent] = 20.0
    vol[sulcus] = 40.0
    vol[gyrus_lobe] = 165.0
    vol = gaussian_filter(vol, sigma=0.75).astype(np.float32)
    return vol, brain


def _world_b0_field_hz(world_xyz: np.ndarray, peak_hz: float = 36.0) -> np.ndarray:
    """Smooth continuous world-space B0 off-resonance field (Hz) evaluated at (3, N) mm coords."""
    x, y, z = world_xyz[0], world_xyz[1], world_xyz[2]
    return (
        float(peak_hz)
        * np.exp(
            -(
                ((x - 5.0) / 14.0) ** 2
                + ((y - 9.0) / 14.0) ** 2
                + ((z + 2.0) / 13.0) ** 2
            )
        )
    ).astype(np.float32)


def _eval_world_b0_on_grid(
    shape: Tuple[int, int, int],
    affine: np.ndarray,
    peak_hz: float = 36.0,
) -> np.ndarray:
    gi, gj, gk = np.indices(shape, dtype=np.float64)
    coords_h = np.vstack([gi.ravel(), gj.ravel(), gk.ravel(), np.ones(gi.size, dtype=np.float64)])
    world_xyz = (np.asarray(affine, dtype=np.float64) @ coords_h)[:3]
    return _world_b0_field_hz(world_xyz, peak_hz=peak_hz).reshape(shape)


def _write_direct_fmap_session(
    session_dir: Path,
    subject_id: str,
    bold_3d: np.ndarray,
    bold_affine: np.ndarray,
    fmap_3d: np.ndarray,
    fmap_affine: np.ndarray,
    *,
    pe_direction: str = "j",
    total_readout_time_s: float = 0.055,
    units: str = "Hz",
    magnitude_3d: np.ndarray | None = None,
    magnitude_affine: np.ndarray | None = None,
) -> Tuple[Path, Path]:
    func_dir = session_dir / "func"
    fmap_dir = session_dir / "fmap"
    func_dir.mkdir(parents=True, exist_ok=True)
    fmap_dir.mkdir(parents=True, exist_ok=True)

    bold_4d = np.repeat(bold_3d[..., None], 6, axis=3).astype(np.float32)
    bold_p = func_dir / f"{subject_id}_ses-1_task-rest_bold.nii.gz"
    bold_j = func_dir / f"{subject_id}_ses-1_task-rest_bold.json"
    b_img = nib.Nifti1Image(bold_4d, bold_affine)
    zooms = tuple(float(np.linalg.norm(bold_affine[:3, c])) for c in range(3)) + (2.0,)
    b_img.header.set_zooms(zooms)
    b_img.header.set_xyzt_units("mm", "sec")
    nib.save(b_img, str(bold_p))
    bold_j.write_text(
        json.dumps(
            {
                "RepetitionTime": 2.0,
                "TaskName": "rest",
                "PhaseEncodingDirection": pe_direction,
                "TotalReadoutTime": float(total_readout_time_s),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fmap_p = fmap_dir / f"{subject_id}_ses-1_fieldmap.nii.gz"
    fmap_j = fmap_dir / f"{subject_id}_ses-1_fieldmap.json"
    nib.save(nib.Nifti1Image(fmap_3d.astype(np.float32), fmap_affine), str(fmap_p))
    fmap_j.write_text(
        json.dumps({"Units": units, "IntendedFor": [f"ses-1/func/{bold_p.name}"]}, indent=2),
        encoding="utf-8",
    )

    if magnitude_3d is not None:
        mag_aff = magnitude_affine if magnitude_affine is not None else fmap_affine
        mag_p = fmap_dir / f"{subject_id}_ses-1_magnitude.nii.gz"
        nib.save(nib.Nifti1Image(magnitude_3d.astype(np.float32), mag_aff), str(mag_p))

    return bold_p, bold_j


# ---------------------------------------------------------------------------
# Item 1: Fieldmap-to-BOLD Spatial Registration (Known Shift Recovered)
# ---------------------------------------------------------------------------


def test_step5_1_item1_header_affine_shifted_fieldmap_recovered(tmp_path: Path):
    """When NIfTI header encodes a known world-space shift between fieldmap and BOLD, SDC recovers it."""
    shape = (28, 28, 20)
    true_3d, brain_mask = _make_asymmetric_brain_3d(shape)
    bold_affine = np.array(
        [
            [3.0, 0.0, 0.0, -42.0],
            [0.0, 3.0, 0.0, -42.0],
            [0.0, 0.0, 3.5, -35.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    b0_bold = _eval_world_b0_on_grid(shape, bold_affine, peak_hz=36.0)
    dist_3d, _ = generate_synthetic_distorted_epi(true_3d, b0_bold, "j", 0.055, affine=bold_affine)

    # Fieldmap grid is shifted in world space (+9.0 mm X, -7.5 mm Y, +3.5 mm Z) with larger bounding grid
    fmap_shape = (34, 34, 24)
    fmap_affine = bold_affine.copy()
    fmap_affine[:3, 3] = np.array([-42.0 - 9.0, -42.0 - 7.5, -35.0 - 3.5], dtype=np.float64)
    b0_shifted = _eval_world_b0_on_grid(fmap_shape, fmap_affine, peak_hz=36.0)

    ses_dir = tmp_path / "sub-synthetic5101" / "ses-1"
    bold_p, bold_j = _write_direct_fmap_session(
        ses_dir,
        "sub-synthetic5101",
        dist_3d,
        bold_affine,
        b0_shifted,
        fmap_affine,
    )
    res = run_susceptibility_distortion_correction(
        bold_p,
        bold_j,
        reference_img=nib.Nifti1Image(true_3d, bold_affine),
    )
    assert res.status == SDC_STATUS_APPLIED_DIRECT_FIELDMAP
    assert res.corrected is True
    assert res.field_hz is not None
    mae_hz = float(np.mean(np.abs(res.field_hz[brain_mask] - b0_bold[brain_mask])))
    assert mae_hz < 0.3, f"Expected recovered shifted fieldmap MAE < 0.3 Hz, got {mae_hz:.4f} Hz"
    assert res.correction_qc is not None
    assert res.correction_qc["ncc_after"] > res.correction_qc["ncc_before"]
    assert res.correction_qc["ncc_after"] > 0.99


def test_step5_1_item1_qc_gate_rejects_degraded_misaligned_fieldmap(tmp_path: Path):
    """Post-correction QC gate (Bug #1): reject SDC when shifted fieldmap degrades NCC vs reference.

    Fieldmap-to-BOLD alignment relies on header affines (magnitude-based registration is not
    implemented). When a companion *_magnitude.nii.gz and *_fieldmap.nii.gz have an unrecorded
    spatial shift (-12 mm X, -12 mm Y) that degrades alignment (ncc_after < ncc_before), the QC gate
    must reject the warp (corrected=False, status='REJECTED_QC_DEGRADED', reason in provenance) and
    must raise SDCError under strict_sdc=True.
    """
    shape = (28, 28, 20)
    true_3d, _ = _make_asymmetric_brain_3d(shape)
    affine = np.array(
        [
            [3.0, 0.0, 0.0, -42.0],
            [0.0, 3.0, 0.0, -42.0],
            [0.0, 0.0, 3.5, -35.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Sharp off-resonance B0 lobe centered near the shifted structure inside BOLD
    gx, gy, gz = np.indices(shape, dtype=np.float32)
    b0_true_in_bold = (
        42.0 * np.exp(-(((gx - 15.0) / 3.2) ** 2 + ((gy - 16.0) / 3.2) ** 2 + ((gz - 10.0) / 3.5) ** 2))
    ).astype(np.float32)
    dist_3d, _ = generate_synthetic_distorted_epi(true_3d, b0_true_in_bold, "j", 0.055, affine=affine)

    # Shift both magnitude and fieldmap by (-4, -4, 0) voxels = (-12, -12, 0) mm relative to BOLD
    # while writing both with nominal scanner affine (simulating inter-scan head motion between fmap and EPI)
    shift_vox = (-4, -4, 0)
    shifted_mag = np.roll(true_3d, shift=shift_vox, axis=(0, 1, 2)).astype(np.float32)
    shifted_b0 = np.roll(b0_true_in_bold, shift=shift_vox, axis=(0, 1, 2)).astype(np.float32)

    ses_dir = tmp_path / "sub-synthetic5102" / "ses-1"
    deriv_dir = tmp_path / "derivatives_rejected"
    bold_p, bold_j = _write_direct_fmap_session(
        ses_dir,
        "sub-synthetic5102",
        dist_3d,
        affine,
        shifted_b0,
        affine,
        magnitude_3d=shifted_mag,
        magnitude_affine=affine,
    )
    res = run_susceptibility_distortion_correction(
        bold_p,
        bold_j,
        derivative_dir=deriv_dir,
        subject_id="sub-synthetic5102",
        session_id="ses-1",
        reference_img=nib.Nifti1Image(true_3d, affine),
    )
    assert res.status == SDC_STATUS_REJECTED_QC_DEGRADED
    assert res.corrected is False
    assert res.displacement_vox is None
    assert res.unwarped_mean_3d is None
    assert res.correction_qc is not None
    assert res.correction_qc["ncc_after"] < res.correction_qc["ncc_before"]
    assert res.reason is not None and "REJECTED_QC_DEGRADED" in res.reason
    prov_dict = res.to_provenance_dict()
    assert prov_dict["status"] == SDC_STATUS_REJECTED_QC_DEGRADED
    assert prov_dict["corrected"] is False
    assert prov_dict["reason"] == res.reason
    assert not (deriv_dir / "sub-synthetic5102_ses-1_task-rest_desc-sdc_displacement.nii.gz").exists()

    with pytest.raises(SDCError, match="REJECTED_QC_DEGRADED"):
        run_susceptibility_distortion_correction(
            bold_p,
            bold_j,
            strict_sdc=True,
            reference_img=nib.Nifti1Image(true_3d, affine),
        )


# ---------------------------------------------------------------------------
# Item 2: Different Fieldmap Resolutions & Affines (Anisotropic, Flipped, Oblique)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_name,fmap_shape,fmap_zooms,fmap_euler_deg,bold_euler_deg",
    [
        ("anisotropic_highres", (52, 34, 16), (1.5, 2.4, 4.2), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("flipped_lps_axes", (36, 36, 20), (-2.2, -2.5, -3.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("oblique_rotated", (40, 32, 20), (-2.0, 2.5, 3.6), (9.0, -7.0, 14.0), (-5.0, 6.0, 10.0)),
    ],
)
def test_step5_1_item2_different_resolutions_and_affines(
    tmp_path: Path,
    case_name: str,
    fmap_shape: Tuple[int, int, int],
    fmap_zooms: Tuple[float, float, float],
    fmap_euler_deg: Tuple[float, float, float],
    bold_euler_deg: Tuple[float, float, float],
):
    """Verify fieldmap resampling, FOV validation, and unwarping for anisotropic, flipped, and oblique affines."""
    bold_shape = (26, 26, 20)
    true_3d, brain_mask = _make_asymmetric_brain_3d(bold_shape)

    r_bold = Rotation.from_euler("xyz", bold_euler_deg, degrees=True).as_matrix()
    bold_aff = np.eye(4, dtype=np.float64)
    bold_aff[:3, :3] = r_bold @ np.diag([3.0, 3.2, 3.5])
    bold_aff[:3, 3] = -bold_aff[:3, :3] @ np.array([13.0, 13.0, 10.0])

    b0_bold = _eval_world_b0_on_grid(bold_shape, bold_aff, peak_hz=35.0)
    dist_3d, _ = generate_synthetic_distorted_epi(true_3d, b0_bold, "j", 0.055, affine=bold_aff)

    r_fmap = Rotation.from_euler("xyz", fmap_euler_deg, degrees=True).as_matrix()
    fmap_aff = np.eye(4, dtype=np.float64)
    fmap_aff[:3, :3] = r_fmap @ np.diag(list(fmap_zooms))
    bold_center_world = (bold_aff @ np.array([13.0, 13.0, 10.0, 1.0]))[:3]
    fmap_center_ijk = np.array([fmap_shape[0] / 2.0, fmap_shape[1] / 2.0, fmap_shape[2] / 2.0])
    fmap_aff[:3, 3] = bold_center_world - fmap_aff[:3, :3] @ fmap_center_ijk

    b0_fmap = _eval_world_b0_on_grid(fmap_shape, fmap_aff, peak_hz=35.0)

    ses_dir = tmp_path / case_name / "sub-synthetic5103" / "ses-1"
    bold_p, bold_j = _write_direct_fmap_session(
        ses_dir,
        "sub-synthetic5103",
        dist_3d,
        bold_aff,
        b0_fmap,
        fmap_aff,
    )
    res = run_susceptibility_distortion_correction(
        bold_p,
        bold_j,
        reference_img=nib.Nifti1Image(true_3d, bold_aff),
    )
    assert res.status == SDC_STATUS_APPLIED_DIRECT_FIELDMAP
    assert res.corrected is True
    assert res.field_hz is not None
    mae_hz = float(np.mean(np.abs(res.field_hz[brain_mask] - b0_bold[brain_mask])))
    assert mae_hz < 0.35, f"{case_name}: resampled fieldmap MAE {mae_hz:.4f} Hz too high"
    assert res.correction_qc is not None
    assert res.correction_qc["ncc_after"] > res.correction_qc["ncc_before"]
    assert res.correction_qc["ncc_after"] > 0.99


# ---------------------------------------------------------------------------
# Item 3: Physical Displacement & Unit Conversions Against Hand-Computed Values
# ---------------------------------------------------------------------------


def test_step5_1_item3_hand_computed_displacement_and_unit_conversions(tmp_path: Path):
    """Verify Hz -> rad/s -> voxel displacement -> mm displacement matches exact analytical math."""
    shape = (12, 12, 10)
    r_oblique = Rotation.from_euler("xyz", [12.0, -8.0, 18.0], degrees=True).as_matrix()
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = r_oblique @ np.diag([-2.5, 3.2, 4.0])
    affine[:3, 3] = [15.0, -19.2, -20.0]

    # Hand-specified analytical B0 field in Hz: range [-18, +24] Hz
    gi, gj, gk = np.indices(shape, dtype=np.float64)
    field_hz_exact = (18.0 + 0.5 * (gi - 6.0) - 0.8 * (gj - 6.0) + 0.3 * (gk - 5.0)).astype(np.float32)
    trt = 0.055  # s

    for pe_dir in ("j", "j-", "i-"):
        axis, pol, _ = PHASE_ENCODING_DIRECTION_MAP[pe_dir]
        # Hand calculation: u_vox[a] = polarity * TRT * delta_f_hz; orthogonal channels == 0
        expected_vox = np.zeros(shape + (3,), dtype=np.float64)
        expected_vox[..., axis] = float(pol) * trt * field_hz_exact.astype(np.float64)
        expected_mm = np.einsum("ij,...j->...i", affine[:3, :3], expected_vox)

        got_vox, got_mm = field_hz_to_displacement(field_hz_exact, pe_dir, trt, affine)
        np.testing.assert_allclose(got_vox, expected_vox, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got_mm, expected_mm, rtol=1e-5, atol=1e-5)

        # Verify world-mm displacement magnitude equals |u_vox[a]| * ||affine[:3, a]||_2
        col_norm_mm = float(np.linalg.norm(affine[:3, axis]))
        expected_abs_mm = np.abs(expected_vox[..., axis]) * col_norm_mm
        got_abs_mm = np.linalg.norm(got_mm, axis=3)
        np.testing.assert_allclose(got_abs_mm, expected_abs_mm, rtol=1e-5, atol=1e-5)

    # End-to-end rad/s -> Hz -> voxel/mm comparison on pipeline helper
    omega_rad_s = (field_hz_exact.astype(np.float64) * (2.0 * math.pi)).astype(np.float32)
    ses_dir = tmp_path / "sub-synthetic5104" / "ses-1"
    vol_dummy = np.ones(shape, dtype=np.float32) * 100.0
    bold_p, bold_j = _write_direct_fmap_session(
        ses_dir,
        "sub-synthetic5104",
        vol_dummy,
        affine,
        omega_rad_s,
        affine,
        pe_direction="j-",
        total_readout_time_s=trt,
        units="rad/s",
    )
    res_rad = run_susceptibility_distortion_correction(bold_p, bold_j)
    assert res_rad.status == SDC_STATUS_APPLIED_DIRECT_FIELDMAP
    assert res_rad.displacement_vox is not None and res_rad.displacement_mm is not None

    exp_vox_jneg = np.zeros(shape + (3,), dtype=np.float64)
    exp_vox_jneg[..., 1] = (-1.0) * trt * field_hz_exact.astype(np.float64)
    exp_mm_jneg = np.einsum("ij,...j->...i", affine[:3, :3], exp_vox_jneg)
    np.testing.assert_allclose(res_rad.field_hz, field_hz_exact, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(res_rad.displacement_vox, exp_vox_jneg, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_rad.displacement_mm, exp_mm_jneg, rtol=1e-5, atol=1e-4)


# ---------------------------------------------------------------------------
# Item 4: TOPUP Phase-Encoding Acquisition Requirements
# ---------------------------------------------------------------------------


def test_step5_1_item4_topup_accepts_opposite_pe_and_rejects_identical_pair():
    """Opposite-PE pair (e.g. j and j-) produces valid acqparams; identical same-direction pair fails."""
    lines = build_topup_acquisition_parameters(["j", "j-"], [0.055, 0.055])
    assert lines == ["0 1 0 0.05500000", "0 -1 0 0.05500000"]

    with pytest.raises(InvalidSDCInputError):
        build_topup_acquisition_parameters(["j", "j"], [0.055, 0.055])


def test_step5_1_item4_bug_same_direction_pair_with_distinct_readout_times_must_be_rejected(tmp_path: Path):
    """Minimal failing test (Bug #2): TOPUP requires opposite-PE pair; same-direction pair must be rejected.

    Currently build_topup_acquisition_parameters(['j', 'j'], [0.04, 0.08]) and
    validate_sdc_metadata() (check: `opposite_pol or distinct_rt`) accept a same-direction
    PEPOLAR pair whenever TotalReadoutTime differs, instead of enforcing opposite PE polarities.
    """
    with pytest.raises(InvalidSDCInputError):
        build_topup_acquisition_parameters(["j", "j"], [0.04, 0.08])

    ses_dir = tmp_path / "sub-synthetic5105" / "ses-1"
    func_dir = ses_dir / "func"
    fmap_dir = ses_dir / "fmap"
    func_dir.mkdir(parents=True)
    fmap_dir.mkdir(parents=True)
    vol, _ = _make_asymmetric_brain_3d((20, 20, 15))
    aff = np.diag([3.0, 3.0, 3.5, 1.0])

    bold_p = func_dir / "sub-synthetic5105_ses-1_task-rest_bold.nii.gz"
    bold_j = func_dir / "sub-synthetic5105_ses-1_task-rest_bold.json"
    nib.save(nib.Nifti1Image(np.repeat(vol[..., None], 4, axis=3), aff), str(bold_p))
    bold_j.write_text(
        json.dumps({"RepetitionTime": 2.0, "PhaseEncodingDirection": "j", "TotalReadoutTime": 0.04}),
        encoding="utf-8",
    )

    for tag, trt in (("AP1", 0.04), ("AP2", 0.08)):
        nib.save(nib.Nifti1Image(vol, aff), str(fmap_dir / f"sub-synthetic5105_ses-1_dir-{tag}_epi.nii.gz"))
        (fmap_dir / f"sub-synthetic5105_ses-1_dir-{tag}_epi.json").write_text(
            json.dumps(
                {
                    "PhaseEncodingDirection": "j",  # Same direction on both scans
                    "TotalReadoutTime": trt,
                    "IntendedFor": [f"ses-1/func/{bold_p.name}"],
                }
            ),
            encoding="utf-8",
        )

    disc = discover_sdc_inputs(bold_p, bold_j)
    with pytest.raises(InvalidSDCInputError):
        validate_sdc_metadata(nib.load(str(bold_p)), json.loads(bold_j.read_text()), disc)


# ---------------------------------------------------------------------------
# Item 5: Nonlinear Transform Composition Order (Motion + SDC + Coreg + SyN) Point Tracking
# ---------------------------------------------------------------------------


def test_step5_1_item5_nonlinear_composition_order_point_tracking():
    """Verify Motion + SDC + Coreg + SyN pullback composition order via analytical point tracking."""
    epi_shape = (28, 28, 22)
    epi_aff = np.diag([-3.0, 3.2, 3.6, 1.0])
    epi_aff[:3, 3] = [42.0, -44.8, -39.6]

    mni_shape = (30, 30, 24)
    mni_aff = np.diag([-2.5, -2.5, 3.0, 1.0])
    mni_aff[:3, 3] = [37.5, 37.5, -36.0]

    # Spatially varying nonlinear SDC pullback shift along j
    gx, gy, gz = np.indices(epi_shape, dtype=np.float32)
    b0_hz = (
        28.0 * np.exp(-(((gx - 14.0) / 5.0) ** 2 + ((gy - 14.0) / 5.0) ** 2 + ((gz - 11.0) / 4.5) ** 2))
    ).astype(np.float32)
    disp_vox, _ = field_hz_to_displacement(b0_hz, "j", 0.055, epi_aff)

    # Non-commuting 3D rigid motion, EPI->T1w coreg, T1w->MNI affine, and smooth nonzero SyN warp
    m_motion = np.eye(4, dtype=np.float64)
    m_motion[:3, :3] = Rotation.from_euler("xyz", [4.5, -3.5, 8.0], degrees=True).as_matrix()
    m_motion[:3, 3] = [-2.2, 3.1, -1.4]

    rigid_epi_to_t1 = np.eye(4, dtype=np.float64)
    rigid_epi_to_t1[:3, :3] = Rotation.from_euler("zyx", [-3.0, 2.5, -2.0], degrees=True).as_matrix()
    rigid_epi_to_t1[:3, 3] = [1.4, -1.8, 0.9]

    aff_t1_to_mni = np.eye(4, dtype=np.float64)
    aff_t1_to_mni[:3, :3] = Rotation.from_euler("xyz", [2.0, 1.5, -3.0], degrees=True).as_matrix()
    aff_t1_to_mni[:3, 3] = [-1.0, 1.2, -0.7]

    syn_map = DiffeomorphicMap(
        3,
        disp_shape=mni_shape,
        disp_grid2world=mni_aff,
        domain_shape=epi_shape,
        domain_grid2world=epi_aff,
        codomain_shape=mni_shape,
        codomain_grid2world=mni_aff,
        prealign=np.eye(4),
    )
    syn_map.is_inverse = True
    mx, my, mz = np.indices(mni_shape, dtype=np.float32)
    b_field = np.zeros(mni_shape + (3,), dtype=np.float32)
    b_field[..., 0] = (2.2 * np.sin((mx - 15.0) * 0.15)).astype(np.float32)
    b_field[..., 1] = (2.5 * np.cos((my - 15.0) * 0.14)).astype(np.float32)
    b_field[..., 2] = (1.5 * np.sin((mz - 12.0) * 0.16)).astype(np.float32)
    syn_map.backward = b_field

    composed_map = compose_transformations(
        rigid_epi_to_t1w=rigid_epi_to_t1,
        affine_t1w_to_mni=aff_t1_to_mni,
        syn_mapping=syn_map,
        target_shape=mni_shape,
        target_affine=mni_aff,
        source_shape=epi_shape,
        source_affine=epi_aff,
    )
    grid = compose_sdc_transform(
        composed_map=composed_map,
        sdc_displacement=disp_vox,
        epi_affine=epi_aff,
        motion_transform=m_motion,
        target_shape=mni_shape,
        target_affine=mni_aff,
    )

    # Track 3 distinct target MNI points independently from equations
    targets_ijk = np.array([[10.0, 11.0, 9.0], [18.0, 13.0, 12.0], [13.0, 19.0, 14.0]], dtype=np.float64)
    prealign_inv = rigid_epi_to_t1 @ aff_t1_to_mni
    inv_prealign_inv = np.linalg.inv(prealign_inv)
    inv_epi = np.linalg.inv(epi_aff)
    inv_mni = np.linalg.inv(mni_aff)
    m_vox = inv_epi @ m_motion @ epi_aff
    inv_m_vox = np.linalg.inv(m_vox)

    # Invert full 4-step composition across all EPI voxels so Gaussian point beacons sit cleanly in raw space
    ge_i, ge_j, ge_k = np.indices(epi_shape, dtype=np.float64)
    v_raw = np.vstack([ge_i.ravel(), ge_j.ravel(), ge_k.ravel()])
    v_raw_h = np.vstack([v_raw, np.ones((1, v_raw.shape[1]), dtype=np.float64)])
    x_mni = ((inv_prealign_inv @ epi_aff) @ (inv_m_vox @ v_raw_h))[:3]
    jac_lin = (inv_prealign_inv[:3, :3] @ epi_aff[:3, :3]) @ inv_m_vox[:3, :3]
    for _ in range(25):
        mc = (inv_mni[:3, :3] @ x_mni) + inv_mni[:3, 3:4]
        bv = np.vstack([map_coordinates(b_field[..., c].astype(np.float64), mc, order=1, mode="nearest") for c in range(3)])
        x_def = np.vstack([x_mni + bv, np.ones((1, x_mni.shape[1]), dtype=np.float64)])
        v_u = (inv_epi @ (prealign_inv @ x_def))[:3]
        us = np.vstack([map_coordinates(disp_vox[..., c].astype(np.float64), v_u, order=1, mode="nearest") for c in range(3)])
        res = (m_vox[:3, :3] @ (v_u + us) + m_vox[:3, 3:4]) - v_raw
        if float(np.max(np.abs(res))) < 1e-6:
            break
        x_mni = x_mni - 0.85 * (jac_lin @ res)

    raw_flat = np.zeros(v_raw.shape[1], dtype=np.float64)
    expected_world = []
    order_swap_shifts_mm = []
    for i_m, j_m, k_m in targets_ijk:
        x_true = (mni_aff @ np.array([i_m, j_m, k_m, 1.0]))[:3]
        expected_world.append(x_true)
        idx_flat = int(i_m) * (mni_shape[1] * mni_shape[2]) + int(j_m) * mni_shape[2] + int(k_m)

        # Independent hand-computed pullback vs compose_sdc_transform output
        b_pt = b_field[int(i_m), int(j_m), int(k_m)].astype(np.float64)
        v_u_pt = (inv_epi @ (prealign_inv @ np.append(x_true + b_pt, 1.0)))[:3]
        u_pt = np.array([map_coordinates(disp_vox[..., c].astype(np.float64), v_u_pt[:, None], order=1, mode="nearest")[0] for c in range(3)])
        v_exact = m_vox[:3, :3] @ (v_u_pt + u_pt) + m_vox[:3, 3]
        np.testing.assert_allclose(grid["v_vol"][:, idx_flat], v_exact, atol=1e-10)

        # Verify non-commutativity: swapping motion & SDC order shifts the sampled EPI world position
        v_after_mot = m_vox[:3, :3] @ v_u_pt + m_vox[:3, 3]
        u_at_mot = np.array([map_coordinates(disp_vox[..., c].astype(np.float64), v_after_mot[:, None], order=1, mode="nearest")[0] for c in range(3)])
        v_swapped = v_after_mot + u_at_mot
        shift_mm = float(np.linalg.norm((epi_aff[:3, :3] @ (v_exact - v_swapped))))
        order_swap_shifts_mm.append(shift_mm)

        raw_flat += 100.0 * np.exp(-np.sum((x_mni - x_true[:, None]) ** 2, axis=0) / (2.0 * 3.6 ** 2))

    # Order must be non-trivial across tracked points
    assert min(order_swap_shifts_mm) > 0.25

    raw_vol = raw_flat.reshape(epi_shape).astype(np.float32)
    warped_3d = apply_composed_sdc_volume(raw_vol, grid, interpolation="linear")

    # Verify point tracking recovery at every landmark in MNI output space (< 0.12 mm sub-voxel TRE)
    for lm_ijk, x_true in zip(targets_ijk, expected_world):
        ii, jj, kk = (int(lm_ijk[0]), int(lm_ijk[1]), int(lm_ijk[2]))
        patch = warped_3d[ii - 3 : ii + 4, jj - 3 : jj + 4, kk - 3 : kk + 4]
        com_ijk = lm_ijk + (np.array(center_of_mass(patch)) - np.array([3.0, 3.0, 3.0]))
        x_rec = (mni_aff @ np.append(com_ijk, 1.0))[:3]
        assert float(np.linalg.norm(x_rec - x_true)) < 0.12

    # Also verify resample_4d_bold reproduces identical landmark tracking across volumes
    img_4d = nib.Nifti1Image(np.repeat(raw_vol[..., None], 50, axis=3), epi_aff)
    res_4d = resample_4d_bold(
        img_4d,
        composed_map,
        target_shape=mni_shape,
        target_affine=mni_aff,
        interpolation="linear",
        motion_transforms=[m_motion] * 50,
        sdc_displacement=disp_vox,
    )
    np.testing.assert_allclose(np.asanyarray(res_4d.dataobj)[..., 0], warped_3d, atol=1e-5)


# ---------------------------------------------------------------------------
# Item 6: SDC Quality-Control Wording (Meaning & What Metrics Do NOT Prove)
# ---------------------------------------------------------------------------


def test_step5_1_item6_sdc_qc_wording_explains_meaning_and_limits():
    """Verify QC_METRIC_INTERPRETATION explains ncc/nmi/mask_dice/min_jacobian and what they do NOT prove."""
    for metric_key in ("ncc", "nmi", "mask_dice", "min_jacobian"):
        assert metric_key in QC_METRIC_INTERPRETATION["measures"]
    assert QC_METRIC_INTERPRETATION["does_not_prove"]

    vol, _ = _make_asymmetric_brain_3d((20, 20, 15))
    qc = evaluate_sdc_correction_qc(vol, vol, vol)
    assert qc["metric_interpretation"] == QC_METRIC_INTERPRETATION
    qc_text = json.dumps(qc).lower()
    assert any(k in qc_text for k in ("meaning", "description", "measures", "interpretation")), (
        "evaluate_sdc_correction_qc() must include wording explaining what the QC metrics mean"
    )
    assert any(k in qc_text for k in ("does not prove", "not prove", "limitation", "does_not_prove")), (
        "evaluate_sdc_correction_qc() must state what the QC metrics do NOT prove"
    )

    verif_sdc = serve_dashboard.DashboardApplication().pipeline_verification_status()["susceptibility_distortion"]
    assert verif_sdc["qc_metric_interpretation"] == QC_METRIC_INTERPRETATION
    verif_text = json.dumps({k: v for k, v in verif_sdc.items() if k != "report"}).lower()
    assert "not prove" in verif_text or "does not prove" in verif_text or "limitations" in verif_text
    for metric_key in ("ncc", "nmi", "mask_dice", "min_jacobian"):
        assert metric_key in verif_text


# ---------------------------------------------------------------------------
# Skipped-Correction Visibility (1 Test per Item: provenance JSON, output_manifest.json, Dashboard API)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def skipped_e2e_run(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """Run 22-stage synthetic pipeline once with BOTH STC and SDC skipped (no SliceTiming, no fmap/)."""
    root = tmp_path_factory.mktemp("step5_1_skipped_visibility")
    with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": "step5_1_skipped_visibility_hmac"}, clear=False):
        os.environ.pop("NEUROADHD_MODEL_DIR", None)
        paths = generate_synthetic_phantom_session(
            root / "bids_in",
            subject_id="sub-synthetic599",
            t_vol=55,
            seed=41,
            include_slice_timing=False,
            fieldmap_mode=None,
        )

        cohort_root = root / "cohorts"
        cohort_dir = cohort_root / "cohort_skip51"
        res = run_poc_v2_pipeline(
            bold_path=paths["bold_nii"],
            bold_json_path=paths["bold_json"],
            t1w_path=paths["t1w_nii"],
            atlas_path=paths["cc200_atlas"],
            template_path=paths["mni_template"],
            output_dir=cohort_dir / "runs",
            subject_id="sub-synthetic599",
            manifest_path=paths["synthetic_manifest"],
            allow_missing_slice_timing=True,
            allow_incomplete_sdc_metadata=True,
        )
        out_dir = Path(res["output_dir"])
        cohort_dir.mkdir(parents=True, exist_ok=True)
        (cohort_dir / "cohort_manifest.json").write_text(
            json.dumps(
                {
                    "cohort_run_id": "cohort_skip51",
                    "execution_policy": "STRICTLY_SYNTHETIC_ONLY",
                    "real_data_authorized": False,
                    "run_type": "SYNTHETIC_COHORT_ONLY",
                    "status": "COMPLETED",
                    "subjects": [
                        {
                            "subject_id": "sub-synthetic599",
                            "status": "SUCCESS",
                            "run_output_dir": str(out_dir),
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "cohort_root": cohort_root,
            "cohort_id": "cohort_skip51",
            "subject_id": "sub-synthetic599",
            "output_dir": out_dir,
        }


def test_step5_1_skipped_visibility_in_provenance_json(skipped_e2e_run: Dict[str, Any]):
    """Test 1/3: Skipped STC & SDC statuses are explicitly visible inside provenance.json."""
    prov_path = skipped_e2e_run["output_dir"] / "provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))

    stc = prov["slice_timing_correction"]
    sdc = prov["susceptibility_distortion_correction"]
    assert stc["status"].startswith("SKIPPED_")
    assert stc["corrected"] is False
    assert sdc["status"] == SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP
    assert sdc["corrected"] is False


def test_step5_1_skipped_visibility_in_output_manifest_json(skipped_e2e_run: Dict[str, Any]):
    """Test 2/3: Skipped STC & SDC statuses appear in additive correction_status in output_manifest.json."""
    out_dir = skipped_e2e_run["output_dir"]
    manifest_path = out_dir / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Existing keys & 12-artifact map remain intact
    assert manifest["status"] == "SUCCESS"
    assert manifest["subject_id"] == skipped_e2e_run["subject_id"]
    assert "timestamp_utc" in manifest
    assert len(manifest["artifacts"]) == 12

    corr = manifest["correction_status"]
    assert corr["slice_timing_correction"]["status"].startswith("SKIPPED_")
    assert corr["slice_timing_correction"]["corrected"] is False
    assert corr["susceptibility_distortion_correction"]["status"] == SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP
    assert corr["susceptibility_distortion_correction"]["corrected"] is False

    # Existing consumers (cohort runner & DashboardDataService artifact hash verification) still pass
    verified = _verify_run_artifacts(out_dir)
    assert len(verified["artifacts"]) == 12
    svc = DashboardDataService(skipped_e2e_run["cohort_root"])
    detail = svc.get_subject_detail(skipped_e2e_run["cohort_id"], skipped_e2e_run["subject_id"])
    assert detail["subject_id"] == skipped_e2e_run["subject_id"]
    content, _ = svc.get_artifact(skipped_e2e_run["cohort_id"], skipped_e2e_run["subject_id"], "provenance.json")
    assert b"SKIPPED_NO_ASSOCIATED_FIELDMAP" in content


def test_step5_1_skipped_visibility_in_dashboard_api_response(skipped_e2e_run: Dict[str, Any]):
    """Test 3/3: Skipped STC & SDC statuses are visible in live Dashboard HTTP API response."""
    app = serve_dashboard.DashboardApplication()
    app.service = DashboardDataService(skipped_e2e_run["cohort_root"])
    with mock.patch.object(serve_dashboard, "APP", app):
        server = ThreadingHTTPServer(("127.0.0.1", 0), serve_dashboard.DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = (
                f"http://127.0.0.1:{server.server_port}/api/runs/"
                f"{skipped_e2e_run['cohort_id']}/subjects/{skipped_e2e_run['subject_id']}"
            )
            with urllib.request.urlopen(url, timeout=5) as resp:
                assert resp.status == 200
                payload = json.loads(resp.read().decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()

    assert payload["slice_timing_correction"]["status"].startswith("SKIPPED_")
    assert payload["slice_timing_correction"]["corrected"] is False
    assert payload["susceptibility_distortion_correction"]["status"] == SDC_STATUS_SKIPPED_NO_ASSOCIATED_FIELDMAP
    assert payload["susceptibility_distortion_correction"]["corrected"] is False
