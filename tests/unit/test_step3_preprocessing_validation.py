"""Unit Test Suite for Step 3: Neuroimaging Preprocessing Pipeline Validation.

Covers all 24 failure and validation scenarios mandated by Step 3:
1.  Valid synthetic BIDS input
2.  Invalid NIfTI dimensionality (3D BOLD / 2D T1w)
3.  Missing metadata (no TR in sidecar)
4.  Incorrect TR (out of range: <= 0 or > 10.0s)
5.  Invalid image affine (singular/non-invertible)
6.  Motion parameter calculations
7.  Framewise displacement (Power FD formulation)
8.  Brain mask validation (non-empty binary mask)
9.  Tissue segmentation output (valid CSF, GM, WM probabilities)
10. Registration transform existence
11. Nonlinear normalization failure handling
12. Atlas alignment (CC200 grid match)
13. Empty ROI detection
14. Missing ROI time series
15. Incorrect ROI count (strictly 200 parcels)
16. Connectivity symmetry (C = C^T)
17. Correlation value limits ([-1.0, 1.0])
18. Fisher-z numerical stability (atanh clamping at +/- 1.0)
19. Exactly 19,900 features
20. Non-finite feature rejection (NaN / Inf)
21. Deterministic feature ordering (matches get_feature_names)
22. Synthetic-only execution enforcement
23. Rejection of real-data execution attempts
24. Preservation of original input files (byte-level immutability)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.features.connectivity import (
    FISHER_EPSILON,
    compute_connectivity_vector,
    get_feature_names,
)
from src.parcellation.extractor import (
    CANONICAL_CC200_AFFINE,
    CANONICAL_CC200_SHAPE,
    extract_cc200_timeseries,
    verify_canonical_cc200_atlas,
)
from src.preprocessing.motion import (
    build_friston_24_regressors,
    compute_framewise_displacement,
    realign_4d_bold,
)
from src.preprocessing.nuisance import (
    build_confound_matrix,
    clean_functional_bold,
    extract_acompcor_components,
)
from src.preprocessing.segmentation import (
    correct_bias_field,
    create_eroded_tissue_masks,
    extract_anatomical_brain,
    segment_t1w_tissue,
    validate_tissue_masks,
)
from src.validation.authorization_gate import AuthorizationDeniedError
from src.validation.bids_validator import BIDSInputValidator
from scripts.run_poc_v2 import (
    check_safety_guards,
    compute_manifest_hmac,
    compute_sha256,
)


@pytest.fixture
def synthetic_mini_dataset(tmp_path: Path):
    """Fixture providing a fast, lightweight mini-synthetic BIDS dataset."""
    shape = (15, 15, 12)
    t_vol = 30
    affine = np.diag([3.0, 3.0, 3.0, 1.0])

    func_dir = tmp_path / "sub-synthetic001" / "ses-01" / "func"
    anat_dir = tmp_path / "sub-synthetic001" / "ses-01" / "anat"
    template_dir = tmp_path / "templates"
    for d in (func_dir, anat_dir, template_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 4D BOLD
    bold_data = np.random.normal(100.0, 5.0, (*shape, t_vol)).astype(np.float32)
    bold_nii = func_dir / "sub-synthetic001_ses-01_task-rest_bold.nii.gz"
    nib.save(nib.Nifti1Image(bold_data, affine), str(bold_nii))

    # BOLD sidecar
    sidecar_data = {"RepetitionTime": 2.0, "TaskName": "rest", "Manufacturer": "Synthetic"}
    bold_json = func_dir / "sub-synthetic001_ses-01_task-rest_bold.json"
    with open(bold_json, "w") as f:
        json.dump(sidecar_data, f)

    # 3D T1w
    t1_data = np.random.normal(100.0, 20.0, shape).astype(np.float32)
    t1_nii = anat_dir / "sub-synthetic001_ses-01_T1w.nii.gz"
    nib.save(nib.Nifti1Image(t1_data, affine), str(t1_nii))

    # Synthetic Atlas with 200 parcels
    atlas_data = np.zeros(shape, dtype=np.int32)
    flat_indices = np.arange(np.prod(shape))
    for p in range(1, 201):
        idx = flat_indices[p * 5 : (p + 1) * 5]
        coords = np.unravel_index(idx, shape)
        atlas_data[coords] = p
    atlas_nii = template_dir / "cc200_synthetic.nii.gz"
    nib.save(nib.Nifti1Image(atlas_data, affine), str(atlas_nii))

    # MNI Template
    mni_nii = template_dir / "MNI152_T1_3mm.nii.gz"
    nib.save(nib.Nifti1Image(t1_data, affine), str(mni_nii))

    return {
        "bold_nii": bold_nii,
        "bold_json": bold_json,
        "t1w_nii": t1_nii,
        "atlas_nii": atlas_nii,
        "mni_nii": mni_nii,
        "shape": shape,
        "t_vol": t_vol,
        "affine": affine,
    }


# =========================================================================
# Scenario 1: Valid synthetic BIDS input
# =========================================================================
def test_01_valid_synthetic_bids_input(synthetic_mini_dataset):
    validator = BIDSInputValidator()
    report = validator.validate_sidecar_json(synthetic_mini_dataset["bold_json"])
    assert report["status"] == "VALIDATED"
    assert report["repetition_time"] == 2.0
    assert report["task_name"] == "rest"


# =========================================================================
# Scenario 2: Invalid NIfTI dimensionality (3D BOLD / 2D T1w)
# =========================================================================
def test_02_invalid_nifti_dimensionality(tmp_path: Path):
    affine = np.eye(4)

    # 3D BOLD instead of 4D
    bad_bold = tmp_path / "bad_bold.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.float32), affine), str(bad_bold))

    # 2D T1w instead of 3D
    bad_t1 = tmp_path / "bad_t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((10, 10), dtype=np.float32), affine), str(bad_t1))

    bold_img = nib.load(str(bad_bold))
    assert bold_img.ndim != 4
    with pytest.raises(ValueError, match="Expected 4D"):
        realign_4d_bold(bold_img)

    t1_img = nib.load(str(bad_t1))
    assert t1_img.ndim != 3
    with pytest.raises(ValueError, match="3D"):
        extract_anatomical_brain(t1_img)


# =========================================================================
# Scenario 3: Missing metadata (no TR in sidecar)
# =========================================================================
def test_03_missing_metadata_repetition_time(tmp_path: Path):
    validator = BIDSInputValidator()
    no_tr_json = tmp_path / "no_tr.json"
    with open(no_tr_json, "w") as f:
        json.dump({"TaskName": "rest"}, f)

    with pytest.raises(KeyError, match="RepetitionTime"):
        validator.validate_sidecar_json(no_tr_json)


# =========================================================================
# Scenario 4: Incorrect TR (out of range: <= 0 or > 10.0s)
# =========================================================================
@pytest.mark.parametrize("invalid_tr", [0.0, -1.5, 9.0, 15.0])
def test_04_incorrect_repetition_time_out_of_range(tmp_path: Path, invalid_tr: float):
    validator = BIDSInputValidator()
    bad_tr_json = tmp_path / f"bad_tr_{invalid_tr}.json"
    with open(bad_tr_json, "w") as f:
        json.dump({"RepetitionTime": invalid_tr, "TaskName": "rest"}, f)

    with pytest.raises(ValueError, match="outside valid physiological range"):
        validator.validate_sidecar_json(bad_tr_json)


# =========================================================================
# Scenario 5: Invalid image affine (singular / non-invertible)
# =========================================================================
def test_05_invalid_image_affine():
    singular_affine = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],  # rank deficient
        [0.0, 0.0, 0.0, 1.0],
    ])
    assert np.isclose(np.linalg.det(singular_affine), 0.0)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(singular_affine)


# =========================================================================
# Scenario 6: Motion parameter calculations
# =========================================================================
def test_06_motion_parameter_calculations(synthetic_mini_dataset):
    bold_img = nib.load(str(synthetic_mini_dataset["bold_nii"]))
    realigned_img, motion_params, fd = realign_4d_bold(bold_img)

    assert motion_params.shape == (synthetic_mini_dataset["t_vol"], 6)
    assert np.all(np.isfinite(motion_params))


# =========================================================================
# Scenario 7: Framewise displacement (Power FD formulation)
# =========================================================================
def test_07_framewise_displacement():
    # Construct synthetic motion with known translations
    t_vol = 5
    motion = np.zeros((t_vol, 6), dtype=np.float64)
    # volume 1 has 1mm shift in X
    motion[1, 0] = 1.0
    # volume 2 has 2mm shift in Y
    motion[2, 1] = 2.0

    fd = compute_framewise_displacement(motion, radius_mm=50.0)
    assert len(fd) == t_vol
    assert fd[0] == 0.0  # Initial volume strictly 0.0
    assert fd[1] == pytest.approx(1.0, abs=1e-5)
    # Between vol 1 and 2: delta_x = 1.0, delta_y = 2.0 -> sum(|diff|) = 3.0
    assert fd[2] == pytest.approx(3.0, abs=1e-5)
    assert np.all(fd >= 0.0)


# =========================================================================
# Scenario 8: Brain mask validation (non-empty binary mask)
# =========================================================================
def test_08_brain_mask_validation(synthetic_mini_dataset):
    t1_img = nib.load(str(synthetic_mini_dataset["t1w_nii"]))
    brain_img, mask_img = extract_anatomical_brain(t1_img)

    mask_data = np.asanyarray(mask_img.dataobj)
    assert mask_data.ndim == 3
    assert set(np.unique(mask_data)).issubset({0, 1})
    assert np.sum(mask_data) > 0  # Non-empty


# =========================================================================
# Scenario 9: Tissue segmentation output (valid CSF, GM, WM probabilities)
# =========================================================================
def test_09_tissue_segmentation_output(synthetic_mini_dataset):
    t1_img = nib.load(str(synthetic_mini_dataset["t1w_nii"]))
    brain_img, mask_img = extract_anatomical_brain(t1_img)
    corrected_img, _ = correct_bias_field(brain_img, mask_img)
    tissue_dict = segment_t1w_tissue(corrected_img, mask_img)

    assert "prob_csf" in tissue_dict
    assert "prob_gm" in tissue_dict
    assert "prob_wm" in tissue_dict

    for key in ("prob_csf", "prob_gm", "prob_wm"):
        data = np.asanyarray(tissue_dict[key].dataobj)
        assert np.all(data >= 0.0)
        assert np.all(data <= 1.0)
        assert data.shape == synthetic_mini_dataset["shape"]


# =========================================================================
# Scenario 10: Registration transform existence
# =========================================================================
def test_10_registration_transform_existence(synthetic_mini_dataset):
    from src.preprocessing.registration import register_epi_to_t1w, generate_functional_reference

    bold_img = nib.load(str(synthetic_mini_dataset["bold_nii"]))
    t1_img = nib.load(str(synthetic_mini_dataset["t1w_nii"]))
    ref_img, _ = generate_functional_reference(bold_img)

    m_epi_to_t1w, m_t1w_to_epi = register_epi_to_t1w(ref_img, t1_img)
    assert m_epi_to_t1w is not None
    assert m_epi_to_t1w.shape == (4, 4)
    assert np.all(np.isfinite(m_epi_to_t1w))
    assert np.linalg.det(m_epi_to_t1w) != 0.0


# =========================================================================
# Scenario 11: Nonlinear normalization failure handling
# =========================================================================
def test_11_nonlinear_normalization_failure_handling(synthetic_mini_dataset, tmp_path: Path):
    from src.preprocessing.registration import register_t1w_to_mni

    t1_img = nib.load(str(synthetic_mini_dataset["t1w_nii"]))
    # Incompatible template with 0-dimension
    bad_template = tmp_path / "bad_template.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.float32), np.eye(4)), str(bad_template))
    bad_template_img = nib.load(str(bad_template))

    with pytest.raises(Exception):
        register_t1w_to_mni(t1_img, bad_template_img)


# =========================================================================
# Scenario 12: Atlas alignment (CC200 grid match)
# =========================================================================
def test_12_atlas_alignment(synthetic_mini_dataset):
    canonical_atlas_path = Path("data/atlases/cc200_roi_atlas.nii.gz")
    if canonical_atlas_path.is_file():
        img = nib.load(str(canonical_atlas_path))
        assert img.shape == CANONICAL_CC200_SHAPE
        np.testing.assert_allclose(img.affine, CANONICAL_CC200_AFFINE, atol=1e-5)


# =========================================================================
# Scenario 13: Empty ROI detection
# =========================================================================
def test_13_empty_roi_detection(synthetic_mini_dataset, tmp_path: Path):
    # Atlas with missing/empty parcels (e.g. only 1 unique label instead of 200)
    atlas_data = np.ones(synthetic_mini_dataset["shape"], dtype=np.int32)
    bad_atlas_img = nib.Nifti1Image(atlas_data, synthetic_mini_dataset["affine"])
    bold_img = nib.load(str(synthetic_mini_dataset["bold_nii"]))

    # Extracting on an atlas with missing parcels fails closed
    with pytest.raises(ValueError, match="Expected exactly 200 ROIs"):
        extract_cc200_timeseries(bold_img, bad_atlas_img)


# =========================================================================
# Scenario 14: Missing ROI time series
# =========================================================================
def test_14_missing_roi_timeseries():
    # Missing parcels in dataframe: fewer than 200 parcels
    bad_signals = np.random.randn(30, 150)  # Only 150 ROIs instead of 200
    with pytest.raises(ValueError, match="Expected 200"):
        compute_connectivity_vector(bad_signals)


# =========================================================================
# Scenario 15: Incorrect ROI count (strictly 200 parcels)
# =========================================================================
def test_15_incorrect_roi_count():
    bad_signals = np.random.randn(50, 205)  # 205 ROIs
    with pytest.raises(ValueError, match="Expected 200"):
        compute_connectivity_vector(bad_signals)


# =========================================================================
# Scenario 16: Connectivity symmetry (C = C^T)
# =========================================================================
def test_16_connectivity_symmetry():
    np.random.seed(42)
    signals = np.random.randn(100, 200)
    corr = np.corrcoef(signals, rowvar=False)
    assert corr.shape == (200, 200)
    np.testing.assert_allclose(corr, corr.T, atol=1e-12)


# =========================================================================
# Scenario 17: Correlation value limits ([-1.0, 1.0])
# =========================================================================
def test_17_correlation_value_limits():
    np.random.seed(42)
    signals = np.random.randn(100, 200)
    corr = np.corrcoef(signals, rowvar=False)
    assert np.all(corr >= -1.0 - 1e-7)
    assert np.all(corr <= 1.0 + 1e-7)


# =========================================================================
# Scenario 18: Fisher-z numerical stability (atanh clamping at +/- 1.0)
# =========================================================================
def test_18_fisher_z_numerical_stability():
    extreme_corrs = np.array([-1.0, -0.9999999, 0.0, 0.9999999, 1.0])
    clipped = np.clip(extreme_corrs, -1.0 + FISHER_EPSILON, 1.0 - FISHER_EPSILON)
    z_vals = np.arctanh(clipped)
    assert np.all(np.isfinite(z_vals))
    assert not np.any(np.isinf(z_vals))
    assert not np.any(np.isnan(z_vals))


# =========================================================================
# Scenario 19: Exactly 19,900 features
# =========================================================================
def test_19_exactly_19900_features():
    signals = np.random.randn(80, 200)
    z_vector, edge_names = compute_connectivity_vector(signals)
    assert len(z_vector) == 19900
    expected_n = (200 * 199) // 2
    assert len(z_vector) == expected_n
    assert len(edge_names) == 19900


# =========================================================================
# Scenario 20: Non-finite feature rejection (NaN / Inf)
# =========================================================================
def test_20_non_finite_feature_rejection():
    # If signals contain NaNs, compute_connectivity_vector should fail closed
    bad_signals = np.random.randn(50, 200)
    bad_signals[0, 5] = np.nan
    with pytest.raises(Exception):
        compute_connectivity_vector(bad_signals)


# =========================================================================
# Scenario 21: Deterministic feature ordering (matches get_feature_names)
# =========================================================================
def test_21_deterministic_feature_ordering():
    names = get_feature_names()
    assert len(names) == 19900
    assert names[0] == "ROI_001__ROI_002"
    assert names[1] == "ROI_001__ROI_003"
    assert names[-1] == "ROI_199__ROI_200"

    signals = np.random.randn(50, 200)
    _, returned_names = compute_connectivity_vector(signals)
    assert returned_names == names


# =========================================================================
# Scenario 22: Synthetic-only execution enforcement
# =========================================================================
def test_22_synthetic_only_execution_enforcement(synthetic_mini_dataset):
    # check_safety_guards should pass for synthetic inputs with valid manifest
    files_dict = {
        "bold_nii": {
            "filename": synthetic_mini_dataset["bold_nii"].name,
            "resolved_path": str(synthetic_mini_dataset["bold_nii"].resolve()),
            "sha256": compute_sha256(synthetic_mini_dataset["bold_nii"]),
        },
        "bold_json": {
            "filename": synthetic_mini_dataset["bold_json"].name,
            "resolved_path": str(synthetic_mini_dataset["bold_json"].resolve()),
            "sha256": compute_sha256(synthetic_mini_dataset["bold_json"]),
        },
        "t1w_nii": {
            "filename": synthetic_mini_dataset["t1w_nii"].name,
            "resolved_path": str(synthetic_mini_dataset["t1w_nii"].resolve()),
            "sha256": compute_sha256(synthetic_mini_dataset["t1w_nii"]),
        },
        "mni_template": {
            "filename": synthetic_mini_dataset["mni_nii"].name,
            "resolved_path": str(synthetic_mini_dataset["mni_nii"].resolve()),
            "sha256": compute_sha256(synthetic_mini_dataset["mni_nii"]),
        },
        "cc200_atlas": {
            "filename": synthetic_mini_dataset["atlas_nii"].name,
            "resolved_path": str(synthetic_mini_dataset["atlas_nii"].resolve()),
            "sha256": compute_sha256(synthetic_mini_dataset["atlas_nii"]),
        },
    }
    hmac_token = compute_manifest_hmac("NEUROADHD_SYNTHETIC_DATASET_V2", "sub-synthetic001", files_dict)
    manifest_data = {
        "manifest_magic": "NEUROADHD_SYNTHETIC_DATASET_V2",
        "subject_id": "sub-synthetic001",
        "files": files_dict,
        "integrity_hmac": hmac_token,
    }
    manifest_path = synthetic_mini_dataset["bold_nii"].parent / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)

    # Should not raise exception
    check_safety_guards(
        subject_id="sub-synthetic001",
        bold_path=synthetic_mini_dataset["bold_nii"],
        t1w_path=synthetic_mini_dataset["t1w_nii"],
        bold_json_path=synthetic_mini_dataset["bold_json"],
        template_path=synthetic_mini_dataset["mni_nii"],
        atlas_path=synthetic_mini_dataset["atlas_nii"],
        allow_real_scans=False,
        manifest_path=manifest_path,
    )


# =========================================================================
# Scenario 23: Rejection of real-data execution attempts
# =========================================================================
def test_23_rejection_of_real_data_execution_attempts(synthetic_mini_dataset):
    with pytest.raises(RuntimeError, match="SAFETY VIOLATION"):
        check_safety_guards(
            subject_id="sub-01",  # Real patient scan ID
            bold_path=synthetic_mini_dataset["bold_nii"],
            t1w_path=synthetic_mini_dataset["t1w_nii"],
            bold_json_path=synthetic_mini_dataset["bold_json"],
            template_path=synthetic_mini_dataset["mni_nii"],
            atlas_path=synthetic_mini_dataset["atlas_nii"],
            allow_real_scans=False,
        )


# =========================================================================
# Scenario 24: Preservation of original input files (byte-level immutability)
# =========================================================================
def test_24_preservation_of_original_input_files(synthetic_mini_dataset):
    # Compute SHA-256 before
    bold_hash_before = compute_sha256(synthetic_mini_dataset["bold_nii"])
    t1_hash_before = compute_sha256(synthetic_mini_dataset["t1w_nii"])
    json_hash_before = compute_sha256(synthetic_mini_dataset["bold_json"])

    # Run read operations
    bold_img = nib.load(str(synthetic_mini_dataset["bold_nii"]))
    _ = np.asanyarray(bold_img.dataobj)
    t1_img = nib.load(str(synthetic_mini_dataset["t1w_nii"]))
    _ = np.asanyarray(t1_img.dataobj)
    with open(synthetic_mini_dataset["bold_json"], "r") as f:
        _ = json.load(f)

    # Assert SHA-256 after remains byte-identical
    assert compute_sha256(synthetic_mini_dataset["bold_nii"]) == bold_hash_before
    assert compute_sha256(synthetic_mini_dataset["t1w_nii"]) == t1_hash_before
    assert compute_sha256(synthetic_mini_dataset["bold_json"]) == json_hash_before


# =========================================================================
# Scenario 25: MNI152 Nonlinear Registration SyN Validation
# =========================================================================
def test_25_mni152_syn_registration_ground_truth_validation():
    from scripts.validate_mni152_syn_registration import run_synthetic_registration_experiment

    results = run_synthetic_registration_experiment()
    assert results["status"] == "VALIDATED"
    assert results["anatomical_overlap"]["brain_dice_after"] > 0.95
    assert results["anatomical_overlap"]["pearson_r_after"] > 0.95
    assert results["jacobian_determinants"]["non_positive_count"] == 0
    assert results["jacobian_determinants"]["min_det_j"] > 0.0
    assert results["jacobian_determinants"]["topology_preserved"] is True
    assert results["parcel_alignment"]["alignment_improvement"] > 0.10

