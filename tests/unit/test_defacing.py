"""Unit and integration tests for Step 6 — MRI Defacing & Metadata De-Identification (DEC-DEF-03).

All tests use strictly synthetic 64x64x64 (and synthetic PoC) volumes.
Tests using an injected ``defacing_runner`` validate command construction, QC logic,
and downstream pipeline integration only — they never claim to perform real PyDeface
execution or real clinical anonymization.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import List
from unittest import mock

import jsonschema
import nibabel as nib
import numpy as np
import pytest

from scripts.run_poc_v2 import generate_synthetic_phantom_session, run_poc_v2_pipeline
from scripts.serve_dashboard import DashboardApplication
from src.preprocessing.defacing import (
    ALL_DEFACING_STATUSES,
    ALL_PRIVACY_EXPORT_STATUSES,
    DEFACING_STATUS_APPLIED,
    DEFACING_STATUS_FAILED_INVALID_INPUT,
    DEFACING_STATUS_FAILED_OUTPUT_VALIDATION,
    DEFACING_STATUS_QC_PENDING_MANUAL_REVIEW,
    DEFACING_STATUS_REJECTED_BRAIN_TISSUE_LOSS,
    DEFACING_STATUS_SKIPPED_TOOL_UNAVAILABLE,
    DEFACING_STATUS_SKIPPED_UNSUPPORTED_MODALITY,
    PRIVACY_STATUS_AUDIT_FAILED,
    PRIVACY_STATUS_READY_FOR_REVIEW,
    PRIVACY_STATUS_SKIPPED_TOOL_UNAVAILABLE,
    DefacingError,
    DefacingQCError,
    InvalidDefacingInputError,
    PrivacyExportGateError,
    UnsupportedModalityError,
    audit_metadata_identifiers,
    detect_defacing_software,
    evaluate_privacy_export_gate,
    run_anatomical_defacing_and_privacy,
    run_defacing,
    sanitize_metadata,
    validate_defaced_output,
    validate_defacing_input,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_synthetic_t1w_and_masks(shape=(64, 64, 64)):
    """Create a lightweight 64x64x64 synthetic anatomical head with brain + face masks."""
    gx, gy, gz = np.indices(shape, dtype=np.float32)
    cx, cy, cz = 32.0, 32.0, 34.0
    r_brain = np.sqrt(((gx - cx) / 14.0) ** 2 + ((gy - cy) / 15.0) ** 2 + ((gz - cz) / 13.0) ** 2)
    brain_mask = r_brain <= 1.0

    # Anterior/inferior facial slab disjoint from brain ellipsoid
    face_mask = (
        (gx >= 20) & (gx <= 44) &
        (gy >= 50) & (gy <= 59) &
        (gz >= 10) & (gz <= 24)
    ) & (~brain_mask)

    vol = np.zeros(shape, dtype=np.float32)
    vol[brain_mask] = 150.0
    vol[(r_brain <= 0.5)] = 210.0  # inner white-matter-like core
    vol[face_mask] = 95.0

    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [64.0, -64.0, -64.0]
    img = nib.Nifti1Image(vol, affine)
    img.header.set_qform(affine, code=1)
    img.header.set_sform(affine, code=1)
    img.header.set_xyzt_units("mm", "unknown")
    return img, vol, affine, brain_mask, face_mask


def _save_t1w(tmp_dir: Path, name: str = "sub-synthetic601_T1w.nii.gz", img=None):
    if img is None:
        img, vol, affine, brain_mask, face_mask = _make_synthetic_t1w_and_masks()
    else:
        vol = np.asanyarray(img.dataobj, dtype=np.float32)
        affine = img.affine
        brain_mask = None
        face_mask = None
    out = tmp_dir / name
    nib.save(img, str(out))
    return out, vol, affine, brain_mask, face_mask


# ---------------------------------------------------------------------------
# 1. Defacing Software Discovery
# ---------------------------------------------------------------------------


def test_detect_defacing_software_unavailable_vs_available(tmp_path: Path):
    """Detect missing tools honestly vs available pydeface + flirt binaries."""
    with mock.patch("src.preprocessing.defacing.shutil.which", return_value=None):
        sw_missing = detect_defacing_software()
    assert sw_missing["pydeface_available"] is False
    assert sw_missing["selected_algorithm"] == "none"
    assert "pydeface" in sw_missing["missing_dependencies"]
    assert "flirt (FSL)" in sw_missing["missing_dependencies"]

    fake_pydeface = tmp_path / "pydeface"
    fake_pydeface.write_text("#!/bin/sh\necho 'pydeface 2.0.2'\n", encoding="utf-8")
    fake_pydeface.chmod(0o755)
    fake_flirt = "/usr/local/fsl/bin/flirt"

    def _which_mock(cmd: str):
        if cmd == "pydeface":
            return str(fake_pydeface)
        if cmd == "flirt":
            return fake_flirt
        return None

    with mock.patch("src.preprocessing.defacing.shutil.which", side_effect=_which_mock):
        with mock.patch("src.preprocessing.defacing._probe_cli_version", return_value="pydeface 2.0.2"):
            sw_avail = detect_defacing_software()
    assert sw_avail["pydeface_available"] is True
    assert sw_avail["fsl_flirt_available"] is True
    assert sw_avail["selected_algorithm"] == "pydeface"
    assert sw_avail["selected_version"] == "pydeface 2.0.2"
    assert sw_avail["missing_dependencies"] == []


# ---------------------------------------------------------------------------
# 2 & 3. Valid Input Validation & Invalid Input / Modality Rejection
# ---------------------------------------------------------------------------


def test_validate_defacing_input_t1w_t2w_flair_and_rejects_invalid(tmp_path: Path):
    """Validate 3D T1w/T2w/FLAIR and reject BOLD, DWI, fmap, corrupted, NaN, empty, 4D."""
    t1_path, vol, aff, _, _ = _save_t1w(tmp_path)
    meta = validate_defacing_input(t1_path)
    assert meta["valid"] is True
    assert meta["modality"] == "T1w"
    assert meta["shape"] == [64, 64, 64]

    # T2w & FLAIR supported when allowed, rejected when disabled
    t2_img = nib.Nifti1Image(vol, aff)
    t2_path = tmp_path / "sub-synthetic601_T2w.nii.gz"
    nib.save(t2_img, str(t2_path))
    assert validate_defacing_input(t2_path)["modality"] == "T2w"
    with pytest.raises(UnsupportedModalityError):
        validate_defacing_input(t2_path, allow_additional_anatomical_modalities=False)

    # Reject non-existent file
    with pytest.raises(InvalidDefacingInputError):
        validate_defacing_input(tmp_path / "missing_T1w.nii.gz")

    # Reject corrupted NIfTI bytes
    bad_nii = tmp_path / "sub-synthetic601_corrupt_T1w.nii.gz"
    bad_nii.write_bytes(b"not a valid nifti gzip file")
    with pytest.raises(InvalidDefacingInputError):
        validate_defacing_input(bad_nii)

    # Reject NaN / Inf values
    nan_vol = vol.copy()
    nan_vol[10, 10, 10] = np.nan
    nan_p = tmp_path / "sub-synthetic601_nan_T1w.nii.gz"
    nib.save(nib.Nifti1Image(nan_vol, aff), str(nan_p))
    with pytest.raises(InvalidDefacingInputError, match="NaN"):
        validate_defacing_input(nan_p)

    # Reject all-zeros empty volume
    zero_p = tmp_path / "sub-synthetic601_zero_T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((64, 64, 64), dtype=np.float32), aff), str(zero_p))
    with pytest.raises(InvalidDefacingInputError, match="empty"):
        validate_defacing_input(zero_p)

    # Reject 4D BOLD, DWI, fieldmap
    bold_p = tmp_path / "sub-synthetic601_task-rest_bold.nii.gz"
    nib.save(nib.Nifti1Image(np.repeat(vol[..., None], 4, axis=3), aff), str(bold_p))
    with pytest.raises(UnsupportedModalityError):
        validate_defacing_input(bold_p)

    dwi_p = tmp_path / "sub-synthetic601_dwi.nii.gz"
    nib.save(nib.Nifti1Image(vol, aff), str(dwi_p))
    with pytest.raises(UnsupportedModalityError):
        validate_defacing_input(dwi_p)

    fmap_p = tmp_path / "sub-synthetic601_fieldmap.nii.gz"
    nib.save(nib.Nifti1Image(vol, aff), str(fmap_p))
    with pytest.raises(UnsupportedModalityError):
        validate_defacing_input(fmap_p)


# ---------------------------------------------------------------------------
# 4 & 5. Missing Tool Behavior & Original File Immutability
# ---------------------------------------------------------------------------


def test_run_defacing_missing_tool_skips_honestly_and_preserves_original(tmp_path: Path):
    """When pydeface is absent, skip with SKIPPED_TOOL_UNAVAILABLE; never fabricate outputs or skull-strip."""
    t1_path, _, _, _, _ = _save_t1w(tmp_path)
    sha_before = _sha256(t1_path)
    deriv_dir = tmp_path / "derivatives"

    with mock.patch("src.preprocessing.defacing.shutil.which", return_value=None):
        res = run_defacing(t1_path, deriv_dir, subject_id="sub-synthetic601", strict_defacing=False)

    assert res.status == DEFACING_STATUS_SKIPPED_TOOL_UNAVAILABLE
    assert res.defaced is False
    assert res.accepted_for_downstream_preprocessing is False
    assert res.output_relative_path is None
    assert res.defacing_mask_available is False
    assert res.defacing_mask_relative_path is None
    assert res.input_sha256 == sha_before
    assert res.input_sha256_after == sha_before
    assert res.input_file_unmodified_verified is True
    assert _sha256(t1_path) == sha_before
    assert not list(deriv_dir.rglob("*.nii*"))  # Zero fabricated output images or masks


# ---------------------------------------------------------------------------
# 6, 7 & 8. Mocked Command/QC Validation, Affine/Header Preservation & Masks
# ---------------------------------------------------------------------------


def test_run_defacing_with_injected_runner_qc_geometry_masks_and_brain_loss(tmp_path: Path):
    """Test PyDeface CLI construction, derivative NIfTI validation, mask handling, and brain-loss rejection."""
    t1_path, vol, aff, brain_mask, face_mask = _save_t1w(tmp_path)
    sha_before = _sha256(t1_path)
    captured_cmds: List[List[str]] = []

    # Case A: Clean face removal WITH authentic emitted binary mask
    def _runner_with_mask(cmd: List[str]) -> subprocess.CompletedProcess:
        captured_cmds.append(list(cmd))
        out_idx = cmd.index("--outfile") + 1
        out_p = Path(cmd[out_idx])
        defaced = vol.copy()
        defaced[face_mask] = 0.0
        img_out = nib.Nifti1Image(defaced, aff)
        img_out.header.set_qform(aff, code=1)
        img_out.header.set_sform(aff, code=1)
        nib.save(img_out, str(out_p))

        # Emit authentic tool binary facemask (1=preserved, 0=removed or binary face mask)
        fm_p = out_p.with_name(f"{out_p.name.replace('.nii.gz', '')}_facemask.nii.gz")
        nib.save(nib.Nifti1Image(face_mask.astype(np.uint8), aff), str(fm_p))
        return subprocess.CompletedProcess(cmd, 0, stdout="mock pydeface ok", stderr="")

    deriv_a = tmp_path / "deriv_a"
    res_a = run_defacing(
        t1_path,
        deriv_a,
        subject_id="sub-synthetic601",
        defacing_runner=_runner_with_mask,
        brain_mask=brain_mask,
        expected_face_mask=face_mask,
    )
    assert res_a.status == DEFACING_STATUS_APPLIED
    assert res_a.defaced is True
    assert res_a.accepted_for_downstream_preprocessing is True
    assert res_a.manual_visual_review_required is True
    assert res_a.manual_review_status == DEFACING_STATUS_QC_PENDING_MANUAL_REVIEW
    assert res_a.input_file_unmodified_verified is True
    assert _sha256(t1_path) == sha_before
    assert "--outfile" in captured_cmds[0] and "--force" in captured_cmds[0]
    assert res_a.defacing_mask_available is True
    assert (deriv_a / res_a.output_relative_path).is_file()
    assert (deriv_a / res_a.defacing_mask_relative_path).is_file()
    qc_a = res_a.qc_metrics
    assert qc_a["dimensions_match"] is True
    assert qc_a["affine_preserved"] is True
    assert qc_a["qform_preserved"] is True
    assert qc_a["sform_preserved"] is True
    assert qc_a["brain_retention_fraction"] == pytest.approx(1.0)
    assert qc_a["non_face_voxels_modified"] == 0

    # Case B: Runner does NOT produce a mask -> mask remains unavailable (never fabricated)
    def _runner_no_mask(cmd: List[str]) -> subprocess.CompletedProcess:
        out_p = Path(cmd[cmd.index("--outfile") + 1])
        defaced = vol.copy()
        defaced[face_mask] = 0.0
        img_out = nib.Nifti1Image(defaced, aff)
        img_out.header.set_qform(aff, code=1)
        img_out.header.set_sform(aff, code=1)
        nib.save(img_out, str(out_p))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    deriv_b = tmp_path / "deriv_b"
    res_b = run_defacing(
        t1_path,
        deriv_b,
        subject_id="sub-synthetic601",
        defacing_runner=_runner_no_mask,
        brain_mask=brain_mask,
    )
    assert res_b.status == DEFACING_STATUS_APPLIED
    assert res_b.defacing_mask_available is False
    assert res_b.defacing_mask_relative_path is None

    # Case C: Affine mismatch -> FAILED_OUTPUT_VALIDATION and output removed
    def _runner_bad_affine(cmd: List[str]) -> subprocess.CompletedProcess:
        out_p = Path(cmd[cmd.index("--outfile") + 1])
        bad_aff = aff.copy()
        bad_aff[0, 3] += 15.0
        nib.save(nib.Nifti1Image(vol, bad_aff), str(out_p))
        return subprocess.CompletedProcess(cmd, 0)

    res_c = run_defacing(
        t1_path, tmp_path / "deriv_c", subject_id="sub-synthetic601", defacing_runner=_runner_bad_affine
    )
    assert res_c.status == DEFACING_STATUS_FAILED_OUTPUT_VALIDATION
    assert res_c.defaced is False
    assert not list((tmp_path / "deriv_c").rglob("*desc-defaced*.nii.gz"))

    # Case D: Excess brain tissue erosion -> REJECTED_BRAIN_TISSUE_LOSS
    def _runner_erode_brain(cmd: List[str]) -> subprocess.CompletedProcess:
        out_p = Path(cmd[cmd.index("--outfile") + 1])
        eroded = vol.copy()
        eroded[face_mask] = 0.0
        eroded[brain_mask] = 0.0  # catastrophic brain removal!
        eroded[32, 32, 34] = 100.0  # 1 voxel left so output is non-empty
        img_out = nib.Nifti1Image(eroded, aff)
        img_out.header.set_qform(aff, code=1)
        img_out.header.set_sform(aff, code=1)
        nib.save(img_out, str(out_p))
        return subprocess.CompletedProcess(cmd, 0)

    res_d = run_defacing(
        t1_path,
        tmp_path / "deriv_d",
        subject_id="sub-synthetic601",
        defacing_runner=_runner_erode_brain,
        brain_mask=brain_mask,
        strict_defacing=False,
    )
    assert res_d.status == DEFACING_STATUS_REJECTED_BRAIN_TISSUE_LOSS
    assert res_d.defaced is False
    assert res_d.accepted_for_downstream_preprocessing is False

    # Case E: OSError raised by runner cleanly fails closed
    def _runner_raises(cmd: List[str]):
        raise OSError("binary execution failed")

    res_e = run_defacing(
        t1_path, tmp_path / "deriv_e", subject_id="sub-synthetic601", defacing_runner=_runner_raises
    )
    assert res_e.status == DEFACING_STATUS_FAILED_OUTPUT_VALIDATION
    with pytest.raises(DefacingQCError):
        run_defacing(
            t1_path,
            tmp_path / "deriv_e_strict",
            subject_id="sub-synthetic601",
            defacing_runner=_runner_raises,
            strict_defacing=True,
        )


# ---------------------------------------------------------------------------
# 9, 10 & 11. Metadata Privacy Audit, Sanitization & Scientific Preservation
# ---------------------------------------------------------------------------


def test_metadata_audit_and_sanitization_without_phi_leakage(tmp_path: Path):
    """Detect & remove direct PHI fields, preserve scientific BIDS metadata, and never log PHI values."""
    secret_phi_values = [
        "Jane Synthetic SecretPerson",
        "MRN-99887766",
        "1991-04-12",
        "ACC-445566",
        "742 Secret Clinic Way",
        "General Synthetic Hospital",
        "SN-123456-DEV",
        "MRI-BAY-09",
        "Tech OpName",
    ]
    raw_sidecar = {
        "PatientName": secret_phi_values[0],
        "PatientID": secret_phi_values[1],
        "PatientBirthDate": secret_phi_values[2],
        "AccessionNumber": secret_phi_values[3],
        "InstitutionAddress": secret_phi_values[4],
        "InstitutionName": secret_phi_values[5],
        "DeviceSerialNumber": secret_phi_values[6],
        "StationName": secret_phi_values[7],
        "OperatorsName": secret_phi_values[8],
        # Scientific BIDS fields that MUST remain intact
        "RepetitionTime": 2.0,
        "SliceTiming": [0.0, 0.5, 1.0, 1.5],
        "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.048,
        "EchoTime": 0.03,
        "EffectiveEchoSpacing": 0.0005,
        "ReconMatrixPE": 97,
        "MagneticFieldStrength": 3.0,
    }
    sidecar_p = tmp_path / "sub-synthetic601_bold.json"
    sidecar_p.write_text(json.dumps(raw_sidecar, indent=2), encoding="utf-8")
    sidecar_sha0 = _sha256(sidecar_p)

    # participants.tsv with pseudonym + denied columns
    tsv_p = tmp_path / "participants.tsv"
    tsv_p.write_text(
        "participant_id\tPatientName\tDOB\tage\tsex\n"
        "sub-synthetic601\tSecret Row Name\t1992-01-01\t10\tF\n",
        encoding="utf-8",
    )
    tsv_sha0 = _sha256(tsv_p)

    # NIfTI with PHI string inside header descrip
    img, vol, aff, _, _ = _make_synthetic_t1w_and_masks()
    img.header["descrip"] = b"PatientName: Secret Header Value"
    nii_phi_p = tmp_path / "sub-synthetic601_T1w.nii.gz"
    nib.save(img, str(nii_phi_p))
    nii_sha0 = _sha256(nii_phi_p)

    audit_before = audit_metadata_identifiers([sidecar_p, tsv_p, nii_phi_p])
    assert audit_before["clean"] is False
    assert audit_before["raw_identifier_values_logged"] is False
    assert audit_before["dicom_deidentification_separate_required"] is True
    for expected_field in (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "AccessionNumber",
        "InstitutionAddress",
        "InstitutionName",
        "DeviceSerialNumber",
        "StationName",
        "OperatorsName",
        "DOB",
        "nifti_header.descrip",
    ):
        assert expected_field in audit_before["detected_identifier_fields"]

    # Ensure raw PHI values never appear anywhere in the serialized audit report
    audit_dump = json.dumps(audit_before)
    for val in secret_phi_values + ["Secret Row Name", "Secret Header Value"]:
        assert val not in audit_dump

    # Sanitize all three into derivatives directory
    deriv_meta = tmp_path / "derivatives" / "sanitized"
    san_json_p = deriv_meta / "sub-synthetic601_bold.json"
    san_tsv_p = deriv_meta / "participants.tsv"
    san_nii_p = deriv_meta / "sub-synthetic601_T1w.nii.gz"

    rep_json = sanitize_metadata(sidecar_p, san_json_p, permitted_root=deriv_meta)
    rep_tsv = sanitize_metadata(tsv_p, san_tsv_p, permitted_root=deriv_meta)
    rep_nii = sanitize_metadata(nii_phi_p, san_nii_p, permitted_root=deriv_meta)

    # Original files must remain byte-for-byte unmodified
    assert _sha256(sidecar_p) == sidecar_sha0
    assert _sha256(tsv_p) == tsv_sha0
    assert _sha256(nii_phi_p) == nii_sha0
    assert rep_json["dicom_header_phi_removed"] is False

    # Scientific BIDS fields preserved in sanitized JSON
    cleaned_json = json.loads(san_json_p.read_text(encoding="utf-8"))
    for sci_key in (
        "RepetitionTime",
        "SliceTiming",
        "PhaseEncodingDirection",
        "TotalReadoutTime",
        "EchoTime",
        "EffectiveEchoSpacing",
        "ReconMatrixPE",
        "MagneticFieldStrength",
    ):
        assert cleaned_json[sci_key] == raw_sidecar[sci_key]

    # participant_id preserved in sanitized TSV while PHI columns removed
    tsv_lines = san_tsv_p.read_text(encoding="utf-8").splitlines()
    assert tsv_lines[0].split("\t") == ["participant_id", "age", "sex"]
    assert "sub-synthetic601" in tsv_lines[1]

    # Re-auditing sanitized derivatives passes cleanly
    audit_after = audit_metadata_identifiers([san_json_p, san_tsv_p, san_nii_p])
    assert audit_after["clean"] is True
    assert audit_after["identifier_field_count"] == 0


# ---------------------------------------------------------------------------
# 12 & 13. Path Safety / Containment & Export Readiness Privacy Gate
# ---------------------------------------------------------------------------


def test_path_traversal_symlink_and_clinical_blocklist_rejected(tmp_path: Path):
    """Reject '..' traversal, symlinks, symlinked directories, and protected clinical paths."""
    t1_path, _, _, _, _ = _save_t1w(tmp_path)
    symlink_p = tmp_path / "sym_T1w.nii.gz"
    symlink_p.symlink_to(t1_path)

    with pytest.raises(InvalidDefacingInputError, match="symlink"):
        validate_defacing_input(symlink_p)

    with pytest.raises(InvalidDefacingInputError, match="traversal"):
        validate_defacing_input(tmp_path / ".." / tmp_path.name / t1_path.name)

    with pytest.raises(InvalidDefacingInputError, match="SAFETY VIOLATION"):
        run_defacing(tmp_path / "NeuroADHD_Converted" / "sub-01_T1w.nii.gz", tmp_path / "deriv")

    # Symlinked directory pointing outside permitted root -> containment violation
    perm_root = tmp_path / "perm"
    perm_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (perm_root / "linkdir").symlink_to(outside)
    src_json = tmp_path / "bold.json"
    src_json.write_text('{"RepetitionTime": 2.0}', encoding="utf-8")
    with pytest.raises(InvalidDefacingInputError, match="containment"):
        sanitize_metadata(src_json, perm_root / "linkdir" / "out.json", permitted_root=perm_root)

    # Symlinked directory pointing inside permitted root -> symlink forbidden
    inner_real = perm_root / "inner_real"
    inner_real.mkdir()
    (perm_root / "inner_link").symlink_to(inner_real)
    with pytest.raises(InvalidDefacingInputError, match="symlink"):
        sanitize_metadata(src_json, perm_root / "inner_link" / "out.json", permitted_root=perm_root)


def test_privacy_export_gate_rejects_skipped_and_never_claims_anonymized(tmp_path: Path):
    """Skipped defacing must reject export readiness and never emit ANONYMIZED/ANONYMOUS."""
    t1_path, _, _, _, _ = _save_t1w(tmp_path)
    bold_json = tmp_path / "sub-synthetic601_bold.json"
    bold_json.write_text(
        json.dumps({"RepetitionTime": 2.0, "PatientName": "Synthetic PHI"}), encoding="utf-8"
    )
    deriv_dir = tmp_path / "deriv_gate"

    with mock.patch("src.preprocessing.defacing.shutil.which", return_value=None):
        _, prov = run_anatomical_defacing_and_privacy(
            t1w_path=t1_path,
            bold_json_path=bold_json,
            derivative_dir=deriv_dir,
            subject_id="sub-synthetic601",
            require_export_ready=False,
        )
        gate = prov["privacy_export_gate"]
        assert gate["overall_status"] == PRIVACY_STATUS_SKIPPED_TOOL_UNAVAILABLE
        assert gate["export_ready_for_review"] is False
        assert gate["overall_status"] not in ("ANONYMIZED", "ANONYMOUS")
        assert prov["metadata_sanitization"]["status"] == "SANITIZED"
        assert "PatientName" in prov["metadata_sanitization"]["removed_identifier_fields"]
        assert "Synthetic PHI" not in json.dumps(prov)

        with pytest.raises(PrivacyExportGateError):
            run_anatomical_defacing_and_privacy(
                t1w_path=t1_path,
                bold_json_path=bold_json,
                derivative_dir=tmp_path / "deriv_gate_strict",
                subject_id="sub-synthetic601",
                require_export_ready=True,
            )


# ---------------------------------------------------------------------------
# 14 & 15. End-to-End 23-Stage Pipeline Integration + Strict Failure + Dashboard
# ---------------------------------------------------------------------------


def test_run_poc_v2_23stage_defacing_applied_vs_skipped_and_strict_fail(tmp_path: Path):
    """Verify Stage 7 in 23-stage PoC V2 pipeline: skip when tool absent, strict failure, and applied via runner."""
    with mock.patch.dict(os.environ, {"NEUROADHD_HMAC_KEY": "step6_defacing_e2e_key_2026"}, clear=False):
        os.environ.pop("NEUROADHD_MODEL_DIR", None)
        ds = generate_synthetic_phantom_session(tmp_path / "in_ds", subject_id="sub-synthetic610", t_vol=55, seed=88)
        t1_sha0 = _sha256(ds["t1w_nii"])
        json_sha0 = _sha256(ds["bold_json"])

        # A. Missing tool default run -> SKIPPED_TOOL_UNAVAILABLE, Stage 7/23 in provenance + manifest
        with mock.patch("src.preprocessing.defacing.shutil.which", return_value=None):
            res_skip = run_poc_v2_pipeline(
                bold_path=ds["bold_nii"],
                bold_json_path=ds["bold_json"],
                t1w_path=ds["t1w_nii"],
                atlas_path=ds["cc200_atlas"],
                template_path=ds["mni_template"],
                output_dir=tmp_path / "out_skip",
                subject_id="sub-synthetic610",
                session_id="ses-1",
                manifest_path=ds["synthetic_manifest"],
            )
        prov_skip = res_skip["provenance"]
        assert prov_skip["pipeline_stages_total"] == 23
        assert len(prov_skip["preprocessing_order"]) == 23
        assert "Defacing" in prov_skip["preprocessing_order"][6]
        assert prov_skip["defacing"]["status"] == DEFACING_STATUS_SKIPPED_TOOL_UNAVAILABLE
        assert prov_skip["defacing"]["defaced"] is False
        assert prov_skip["defacing"]["input_file_unmodified_verified"] is True
        assert _sha256(ds["t1w_nii"]) == t1_sha0
        assert _sha256(ds["bold_json"]) == json_sha0

        manifest_skip = json.loads((Path(res_skip["output_dir"]) / "output_manifest.json").read_text(encoding="utf-8"))
        assert manifest_skip["defacing_status"]["status"] == DEFACING_STATUS_SKIPPED_TOOL_UNAVAILABLE
        assert manifest_skip["defacing_status"]["export_ready_for_review"] is False

        schema_dict = json.loads(Path("configs/provenance_schema.json").read_text(encoding="utf-8"))
        jsonschema.validate(instance=prov_skip, schema=schema_dict)

        # B. --strict-defacing fails closed & quarantines failed run when pydeface missing
        with mock.patch("src.preprocessing.defacing.shutil.which", return_value=None):
            with pytest.raises(DefacingError):
                run_poc_v2_pipeline(
                    bold_path=ds["bold_nii"],
                    bold_json_path=ds["bold_json"],
                    t1w_path=ds["t1w_nii"],
                    atlas_path=ds["cc200_atlas"],
                    template_path=ds["mni_template"],
                    output_dir=tmp_path / "out_strict_fail",
                    subject_id="sub-synthetic610",
                    session_id="ses-1",
                    manifest_path=ds["synthetic_manifest"],
                    strict_defacing=True,
                )
        assert len(list((tmp_path / "out_strict_fail" / "sub-synthetic610").glob("failed_run_*"))) == 1

        # C. Injected pydeface runner -> Stage 7 APPLIED & privacy gate PRIVACY_READY_FOR_REVIEW
        t1_img = nib.load(str(ds["t1w_nii"]))
        t1_arr = np.asanyarray(t1_img.dataobj, dtype=np.float32)

        def _e2e_deface_runner(cmd: List[str]) -> subprocess.CompletedProcess:
            out_p = Path(cmd[cmd.index("--outfile") + 1])
            defaced_arr = t1_arr.copy()
            defaced_arr[:4, :4, :4] = 0.0  # corner background/anterior corner cleared
            out_img = nib.Nifti1Image(defaced_arr, t1_img.affine)
            out_img.header.set_qform(t1_img.affine, code=1)
            out_img.header.set_sform(t1_img.affine, code=1)
            nib.save(out_img, str(out_p))
            return subprocess.CompletedProcess(cmd, 0, stdout="mock e2e pydeface", stderr="")

        fake_bin = tmp_path / "pydeface"
        fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        res_app = run_poc_v2_pipeline(
            bold_path=ds["bold_nii"],
            bold_json_path=ds["bold_json"],
            t1w_path=ds["t1w_nii"],
            atlas_path=ds["cc200_atlas"],
            template_path=ds["mni_template"],
            output_dir=tmp_path / "out_applied",
            subject_id="sub-synthetic610",
            session_id="ses-1",
            manifest_path=ds["synthetic_manifest"],
            pydeface_executable=fake_bin,
            defacing_runner=_e2e_deface_runner,
            strict_defacing=True,
            require_export_ready=True,
        )
        prov_app = res_app["provenance"]
        assert prov_app["defacing"]["status"] == DEFACING_STATUS_APPLIED
        assert prov_app["defacing"]["defaced"] is True
        assert prov_app["defacing"]["privacy_export_gate"]["overall_status"] == PRIVACY_STATUS_READY_FOR_REVIEW
        assert prov_app["defacing"]["privacy_export_gate"]["export_ready_for_review"] is True
        jsonschema.validate(instance=prov_app, schema=schema_dict)

        # D. Dashboard verification payload reflects 23 stages + Stage 7 defacing summary
        app = DashboardApplication()
        h = app.health()
        assert h["pipeline_stages_total"] == 23
        v = app.pipeline_verification_status()
        assert v["stages_total"] == 23
        assert len(v["stages"]) == 23
        assert "Defacing" in v["stages"][6]["name"]
        assert v["defacing"]["manual_visual_review_required"] is True
        assert "NOT_AUTHORIZED" in v["defacing"]["real_data_authorization_status"]
        assert all(s in ALL_DEFACING_STATUSES for s in (DEFACING_STATUS_APPLIED, DEFACING_STATUS_SKIPPED_TOOL_UNAVAILABLE, DEFACING_STATUS_REJECTED_BRAIN_TISSUE_LOSS))
        assert all(s in ALL_PRIVACY_EXPORT_STATUSES for s in (PRIVACY_STATUS_READY_FOR_REVIEW, PRIVACY_STATUS_AUDIT_FAILED, PRIVACY_STATUS_SKIPPED_TOOL_UNAVAILABLE))
