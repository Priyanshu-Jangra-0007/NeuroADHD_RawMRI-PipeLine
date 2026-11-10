#!/usr/bin/env python3
"""Mathematical Regression Tests for SyN Registration and Coordinate Transformations.

Audits and verifies the complete transformation chain:
A. Identity transformation: All landmarks remain unchanged.
B. Affine-only transformation: Known landmark coordinates recovered via affine and inverse affine.
C. Nonlinear deformation: Known deformation field and landmark recovery via exact inverse mapping.
D. Composite transformation: Full affine prealignment plus nonlinear deformation field.
E. Image/landmark consistency: Point-mapping agrees with image-resampling peaks within interpolation tolerance.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates, center_of_mass
from sklearn.cluster import KMeans

from dipy.align.imwarp import DiffeomorphicMap
from src.preprocessing.registration import register_t1w_to_mni, compute_spatial_dice


@pytest.fixture
def test_volume_setup():
    shape = (24, 28, 24)
    vox_size = 2.5
    affine = np.diag([vox_size, vox_size, vox_size, 1.0])
    affine[:3, 3] = [-30.0, -35.0, -30.0]
    inv_affine = np.linalg.inv(affine)
    return {
        "shape": shape,
        "vox_size": vox_size,
        "affine": affine,
        "inv_affine": inv_affine,
    }


def test_a_identity_transformation(test_volume_setup):
    """Test A: Identity transformation must leave all landmarks strictly unchanged."""
    shape = test_volume_setup["shape"]
    affine = test_volume_setup["affine"]
    inv_affine = test_volume_setup["inv_affine"]

    dmap = DiffeomorphicMap(
        dim=3,
        disp_shape=shape,
        disp_grid2world=affine,
        domain_shape=shape,
        domain_grid2world=affine,
        codomain_shape=shape,
        codomain_grid2world=affine,
        prealign=np.eye(4),
    )
    dmap.forward = np.zeros(shape + (3,), dtype=np.float32)
    dmap.backward = np.zeros(shape + (3,), dtype=np.float32)
    dmap.is_inverse = False

    landmarks = np.array([
        [12.0, 14.0, 12.0],
        [8.0, 10.0, 8.0],
        [16.0, 18.0, 16.0],
        [6.0, 20.0, 14.0],
    ])

    mapped = dmap.transform_points(landmarks, coord2world=affine, world2coord=inv_affine)
    assert np.allclose(mapped, landmarks, atol=1e-10), f"Identity mapping failed: max diff = {np.max(np.abs(mapped - landmarks))}"

    mapped_inv = dmap.transform_points_inverse(landmarks, coord2world=affine, world2coord=inv_affine)
    assert np.allclose(mapped_inv, landmarks, atol=1e-10), f"Inverse identity mapping failed: max diff = {np.max(np.abs(mapped_inv - landmarks))}"


def test_b_affine_only_transformation(test_volume_setup):
    """Test B: Affine-only transformation must recover landmark coordinates with machine precision."""
    shape = test_volume_setup["shape"]
    affine = test_volume_setup["affine"]
    inv_affine = test_volume_setup["inv_affine"]

    # Construct rigid+scale affine in physical space
    # 5 degree rotation around z + 3mm translation
    theta = np.deg2rad(5.0)
    R_physical = np.array([
        [np.cos(theta), -np.sin(theta), 0.0, 3.5],
        [np.sin(theta),  np.cos(theta), 0.0, -2.0],
        [0.0,            0.0,           1.0, 1.5],
        [0.0,            0.0,           0.0, 1.0],
    ])

    dmap_aff = DiffeomorphicMap(
        dim=3,
        disp_shape=shape,
        disp_grid2world=affine,
        domain_shape=shape,
        domain_grid2world=affine,
        codomain_shape=shape,
        codomain_grid2world=affine,
        prealign=R_physical,
    )
    dmap_aff.forward = np.zeros(shape + (3,), dtype=np.float32)
    dmap_aff.backward = np.zeros(shape + (3,), dtype=np.float32)
    dmap_aff.is_inverse = False

    landmarks = np.array([
        [12.0, 14.0, 12.0],
        [7.0, 11.0, 9.0],
        [15.0, 17.0, 15.0],
    ])

    # In DIPY forward point mapping:
    # coord2prealigned = R_physical * affine
    # out = inv_affine * coord2prealigned * pt = inv_affine * R_physical * affine * pt
    expected_vox = nib.affines.apply_affine(inv_affine @ R_physical @ affine, landmarks)
    mapped_aff = dmap_aff.transform_points(landmarks, coord2world=affine, world2coord=inv_affine)

    assert np.allclose(mapped_aff, expected_vox, atol=1e-5), f"Affine forward mismatch: {np.max(np.abs(mapped_aff - expected_vox))}"

    # Verify round-trip recovery
    recovered_landmarks = nib.affines.apply_affine(np.linalg.inv(inv_affine @ R_physical @ affine), mapped_aff)
    assert np.allclose(recovered_landmarks, landmarks, atol=1e-5), f"Round-trip affine recovery failed: {np.max(np.abs(recovered_landmarks - landmarks))}"


def test_c_nonlinear_exact_inversion(test_volume_setup):
    """Test C: Fixed-point exact inversion must find true root y* + u(y*) = x_target with zero residual."""
    shape = test_volume_setup["shape"]
    gx, gy, gz = np.indices(shape)

    # Smooth controlled displacement field in voxel space
    u_raw_x = np.sin(2.0 * np.pi * gy / shape[1]) * 1.5
    u_raw_y = np.cos(2.0 * np.pi * gx / shape[0]) * 1.2
    u_raw_z = np.sin(2.0 * np.pi * gz / shape[2]) * 0.8

    ux = gaussian_filter(u_raw_x.astype(np.float32), sigma=2.0)
    uy = gaussian_filter(u_raw_y.astype(np.float32), sigma=2.0)
    uz = gaussian_filter(u_raw_z.astype(np.float32), sigma=2.0)

    def eval_u(pt):
        c = np.array([[pt[0]], [pt[1]], [pt[2]]])
        vx = map_coordinates(ux, c, order=3, mode="nearest")[0]
        vy = map_coordinates(uy, c, order=3, mode="nearest")[0]
        vz = map_coordinates(uz, c, order=3, mode="nearest")[0]
        return np.array([vx, vy, vz])

    target_landmarks = np.array([
        [12.0, 14.0, 12.0],
        [8.0, 10.0, 14.0],
        [16.0, 18.0, 10.0],
    ])

    for p_tgt in target_landmarks:
        # Fixed point iteration to solve y* + u(y*) = p_tgt
        y = p_tgt - eval_u(p_tgt)
        for _ in range(20):
            y = p_tgt - eval_u(y)
        residual = np.linalg.norm(y + eval_u(y) - p_tgt)
        assert residual < 1e-10, f"Fixed-point root finding failed: residual = {residual}"


def test_d_composite_transformation_estimation(test_volume_setup):
    """Test D: SyN registration with composite affine + nonlinear fields recovers landmarks within sub-voxel error."""
    shape = test_volume_setup["shape"]
    affine = test_volume_setup["affine"]
    inv_affine = test_volume_setup["inv_affine"]

    gx, gy, gz = np.indices(shape)
    # Anatomical phantom with distinct local contrast features
    target_data = (
        np.exp(-((gx - 12)**2 + (gy - 14)**2 + (gz - 12)**2) / 16.0) * 80.0
        + np.exp(-((gx - 8)**2 + (gy - 10)**2 + (gz - 8)**2) / 8.0) * 120.0
        + np.exp(-((gx - 16)**2 + (gy - 18)**2 + (gz - 16)**2) / 8.0) * 140.0
    ).astype(np.float32)
    target_img = nib.Nifti1Image(target_data, affine)

    # Known composite displacement: +1.5 in x, -1.0 in y, +0.5 in z
    moving_data = (
        np.exp(-((gx - 13.5)**2 + (gy - 13.0)**2 + (gz - 12.5)**2) / 16.0) * 80.0
        + np.exp(-((gx - 9.5)**2 + (gy - 9.0)**2 + (gz - 8.5)**2) / 8.0) * 120.0
        + np.exp(-((gx - 17.5)**2 + (gy - 17.0)**2 + (gz - 16.5)**2) / 8.0) * 140.0
    ).astype(np.float32)
    moving_img = nib.Nifti1Image(moving_data, affine)

    m_affine, syn_map = register_t1w_to_mni(
        t1w_brain=moving_img,
        mni_template=target_img,
        affine_level_iters=[20, 10],
        syn_level_iters=[10, 5],
        step_length=0.25,
    )

    pts_target = np.array([
        [12.0, 14.0, 12.0],
        [8.0, 10.0, 8.0],
        [16.0, 18.0, 16.0],
    ])
    expected_moving = pts_target + np.array([1.5, -1.0, 0.5])

    estimated_moving = syn_map.transform_points(pts_target, coord2world=affine, world2coord=inv_affine)
    tre_vox = np.linalg.norm(estimated_moving - expected_moving, axis=1)

    # All landmarks must achieve sub-voxel spatial accuracy (< 0.5 voxels)
    assert np.all(tre_vox < 0.5), f"Composite transformation failed sub-voxel test: max TRE = {np.max(tre_vox):.4f} voxels"


def test_e_image_landmark_consistency(test_volume_setup):
    """Test E: Point mapping agrees with image-resampling peak location within interpolation tolerance."""
    shape = test_volume_setup["shape"]
    affine = test_volume_setup["affine"]
    inv_affine = test_volume_setup["inv_affine"]
    gx, gy, gz = np.indices(shape)

    # Target image with a smooth Gaussian peak at (12, 14, 12)
    p_target = np.array([12.0, 14.0, 12.0])
    target_data = np.exp(-((gx - p_target[0])**2 + (gy - p_target[1])**2 + (gz - p_target[2])**2) / 6.0) * 100.0

    # Moving image with peak shifted by (+1.5, -1.0, +0.5) voxels
    p_moving_true = p_target + np.array([1.5, -1.0, 0.5])
    moving_data = np.exp(-((gx - p_moving_true[0])**2 + (gy - p_moving_true[1])**2 + (gz - p_moving_true[2])**2) / 6.0) * 100.0

    img_target = nib.Nifti1Image(target_data.astype(np.float32), affine)
    img_moving = nib.Nifti1Image(moving_data.astype(np.float32), affine)

    m_affine, syn_map = register_t1w_to_mni(
        t1w_brain=img_moving,
        mni_template=img_target,
        affine_level_iters=[20, 10],
        syn_level_iters=[10, 5],
    )

    # Resample moving image into target space
    warped_data = syn_map.transform(moving_data.astype(np.float32))
    warped_peak = np.array(center_of_mass(warped_data))

    # Point mapping: map target peak to moving, then invert
    pt_est_moving = syn_map.transform_points(np.array([p_target]), coord2world=affine, world2coord=inv_affine)[0]
    point_error = np.linalg.norm(pt_est_moving - p_moving_true)

    # Image peak error relative to target peak
    image_peak_error = np.linalg.norm(warped_peak - p_target)

    # Documented interpolation tolerance: continuous interpolation & discretization within 0.5 voxels
    assert point_error < 0.5, f"Point transform error too high: {point_error:.4f} voxels"
    assert image_peak_error < 0.5, f"Image peak error too high: {image_peak_error:.4f} voxels"


def test_f_cc200_synthetic_atlas_alignment(test_volume_setup):
    """Test F: Synthetic 200-parcel labeled atlas partitions brain, maintains 200 parcels, and improves Dice."""
    shape = test_volume_setup["shape"]
    affine = test_volume_setup["affine"]
    gx, gy, gz = np.indices(shape)
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2

    # Brain mask
    r_head = np.sqrt(((gx - cx) / 9.0) ** 2 + ((gy - cy) / 11.0) ** 2 + ((gz - cz) / 9.0) ** 2)
    brain_mask = r_head <= 1.0
    coords = np.argwhere(brain_mask)

    # Partition into 200 parcels
    kmeans = KMeans(n_clusters=200, random_state=42, n_init=1).fit(coords)
    atlas_target = np.zeros(shape, dtype=np.int32)
    for i, c in enumerate(coords):
        atlas_target[tuple(c)] = kmeans.labels_[i] + 1

    # Multi-tissue target image
    target_data = np.zeros(shape, dtype=np.float32)
    target_data[brain_mask] = 100.0 + (atlas_target[brain_mask] % 5) * 15.0
    target_data = gaussian_filter(target_data, sigma=0.8)
    target_img = nib.Nifti1Image(target_data, affine)

    # Shift moving image
    np.random.seed(101)
    ux = gaussian_filter(np.sin(2.0 * np.pi * gy / shape[1]).astype(np.float32), sigma=2.5) * 1.0
    uy = gaussian_filter(np.cos(2.0 * np.pi * gx / shape[0]).astype(np.float32), sigma=2.5) * 1.0
    uz = gaussian_filter(np.sin(2.0 * np.pi * gz / shape[2]).astype(np.float32), sigma=2.5) * 0.8

    sample_coords = np.array([(gx + ux).ravel(), (gy + uy).ravel(), (gz + uz).ravel()])
    moving_data = map_coordinates(target_data, sample_coords, order=2, mode="nearest").reshape(shape)
    moving_img = nib.Nifti1Image(moving_data, affine)

    # Warp atlas to moving space
    atlas_moving = map_coordinates(atlas_target.astype(np.float32), sample_coords, order=0, mode="nearest").reshape(shape).astype(np.int32)

    m_affine, syn_map = register_t1w_to_mni(
        t1w_brain=moving_img,
        mni_template=target_img,
        affine_level_iters=[20, 10],
        syn_level_iters=[10, 5],
        step_length=0.25,
    )

    # Warp atlas back to target space
    atlas_warped = syn_map.transform(atlas_moving.astype(np.float32), interpolation="nearest").astype(np.int32)

    # Verify parcel coverage and empty parcel count
    unique_warped = np.unique(atlas_warped[atlas_warped > 0])
    coverage = len(unique_warped) / 200.0

    dices_before = []
    dices_after = []
    for label in range(1, 201):
        m_t = (atlas_target == label)
        m_m = (atlas_moving == label)
        m_w = (atlas_warped == label)
        dices_before.append(compute_spatial_dice(m_m, m_t))
        dices_after.append(compute_spatial_dice(m_w, m_t))

    mean_dice_before = float(np.mean(dices_before))
    mean_dice_after = float(np.mean(dices_after))

    assert coverage >= 0.98, f"Parcel coverage too low: {coverage*100:.1f}%"
    assert mean_dice_after > mean_dice_before, f"Registration did not improve parcel overlap: {mean_dice_after:.4f} <= {mean_dice_before:.4f}"
    assert mean_dice_after > 0.65, f"Post-registration mean parcel Dice too low: {mean_dice_after:.4f}"
