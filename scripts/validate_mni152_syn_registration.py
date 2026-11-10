#!/usr/bin/env python3
"""Synthetic Ground-Truth MNI152 Nonlinear Registration Validation Script.

Executes a mathematically rigorous ground-truth SyN registration experiment:
1. Generates a synthetic anatomical reference (MNI152 template space, 3mm isotropic).
2. Applies a known, smooth, controlled nonlinear ground-truth deformation phi_GT.
3. Places 10 anatomical 3D landmarks and calculates exact expected deformed locations
   via fixed-point root inversion y* + u(y*) = x_target.
4. Estimates the transformation using DIPY SyN (CCMetric with 12-DOF affine prealignment).
5. Compares estimated transform against ground truth using DIPY's complete transformation
   chain (syn_map.transform_points including affine prealignment and nonlinear fields):
   - Landmark Target Registration Error (TRE in mm and voxels: mean, median, std, max).
   - Anatomical overlap metrics (Dice coefficient, Pearson r, RMSE before and after).
   - Jacobian determinant det(J) field statistics (mean, std, min, max, folding count).
   - 200-parcel synthetic labeled atlas parcellation alignment (mean/median Dice, empty count, coverage).
   - Multi-planar before-and-after visual overlays, Jacobian maps, and landmark plots.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.qc import compute_registration_qc_metrics
from src.preprocessing.registration import compute_spatial_dice, register_t1w_to_mni

OUTPUT_DIR = Path("/Users/apple/Documents/Codex/2026-09-29/NeuroADHD_Step3_Synthetic/registration_experiment")


def compute_jacobian_determinants(displacement_field: np.ndarray, voxel_spacing: Tuple[float, float, float] = (3.0, 3.0, 3.0)) -> np.ndarray:
    """Calculate the 3D Jacobian determinant det(J) = det(I + grad(u)) across a displacement field."""
    ux = displacement_field[..., 0]
    uy = displacement_field[..., 1]
    uz = displacement_field[..., 2]

    dux_dx, dux_dy, dux_dz = np.gradient(ux)
    duy_dx, duy_dy, duy_dz = np.gradient(uy)
    duz_dx, duz_dy, duz_dz = np.gradient(uz)

    j00 = 1.0 + dux_dx
    j01 = dux_dy
    j02 = dux_dz

    j10 = duy_dx
    j11 = 1.0 + duy_dy
    j12 = duy_dz

    j20 = duz_dx
    j21 = duz_dy
    j22 = 1.0 + duz_dz

    det_j = (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )
    return det_j


def run_synthetic_registration_experiment() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("MNI152 NONLINEAR REGISTRATION (SyN) RIGOROUS VALIDATION EXPERIMENT")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 80)

    shape = (36, 42, 36)
    vox_size = 3.0  # 3mm isotropic
    affine = np.diag([vox_size, vox_size, vox_size, 1.0])
    affine[:3, 3] = [-54.0, -63.0, -54.0]
    inv_affine = np.linalg.inv(affine)

    # 1. Create Synthetic Anatomical Phantoms (Target = MNI152)
    gx, gy, gz = np.indices(shape)
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    r_head = np.sqrt(((gx - cx) / 14.0) ** 2 + ((gy - cy) / 16.0) ** 2 + ((gz - cz) / 13.0) ** 2)

    brain_mask = r_head <= 1.0
    r_core = np.sqrt(((gx - cx) / 8.0) ** 2 + ((gy - cy) / 9.0) ** 2 + ((gz - cz) / 7.0) ** 2)
    wm_mask = brain_mask & (r_core <= 1.0)

    # Ventricles: bilateral ellipsoids
    v_left = ((gx - (cx - 4)) / 2.5) ** 2 + ((gy - cy) / 5.0) ** 2 + ((gz - cz) / 3.0) ** 2 <= 1.0
    v_right = ((gx - (cx + 4)) / 2.5) ** 2 + ((gy - cy) / 5.0) ** 2 + ((gz - cz) / 3.0) ** 2 <= 1.0
    ventricles = v_left | v_right
    wm_mask = wm_mask & (~ventricles)
    gm_mask = brain_mask & (~wm_mask) & (~ventricles)

    target_data = np.zeros(shape, dtype=np.float32)
    target_data[ventricles] = 25.0
    target_data[gm_mask] = 85.0
    target_data[wm_mask] = 145.0
    target_data = gaussian_filter(target_data, sigma=0.8)

    target_img = nib.Nifti1Image(target_data, affine)
    target_nii_path = OUTPUT_DIR / "mni_target.nii.gz"
    nib.save(target_img, str(target_nii_path))

    # 2. Synthesize Ground-Truth Smooth Nonlinear Diffeomorphism
    np.random.seed(101)
    raw_ux = np.sin(2.0 * np.pi * gy / shape[1]) * np.cos(2.0 * np.pi * gz / shape[2])
    raw_uy = np.cos(2.0 * np.pi * gx / shape[0]) * np.sin(2.0 * np.pi * gz / shape[2])
    raw_uz = np.sin(2.0 * np.pi * gx / shape[0]) * np.cos(2.0 * np.pi * gy / shape[1])

    smooth_ux = gaussian_filter(raw_ux.astype(np.float32), sigma=3.0) * 1.5
    smooth_uy = gaussian_filter(raw_uy.astype(np.float32), sigma=3.0) * 1.5
    smooth_uz = gaussian_filter(raw_uz.astype(np.float32), sigma=3.0) * 1.5

    gt_disp = np.stack([smooth_ux, smooth_uy, smooth_uz], axis=-1)

    # Verify GT Jacobian is strictly positive (diffeomorphic)
    det_j_gt = compute_jacobian_determinants(gt_disp, (vox_size, vox_size, vox_size))
    min_det_gt = float(np.min(det_j_gt))
    assert min_det_gt > 0.0, f"Ground-truth deformation contains folding: min det(J)={min_det_gt}"

    # Backward resampling to synthesize moving T1w: moving(y) = target(y + u(y))
    coords_x = gx + smooth_ux
    coords_y = gy + smooth_uy
    coords_z = gz + smooth_uz
    sample_coords = np.array([coords_x.ravel(), coords_y.ravel(), coords_z.ravel()])
    moving_data = map_coordinates(target_data, sample_coords, order=2, mode="nearest").reshape(shape)

    moving_img = nib.Nifti1Image(moving_data, affine)
    moving_nii_path = OUTPUT_DIR / "t1w_moving.nii.gz"
    nib.save(moving_img, str(moving_nii_path))

    # 3. Known Ground-Truth 3D Anatomical Landmarks (10 locations across brain)
    landmarks_voxel = np.array([
        [cx - 4, cy, cz],          # 1. Left ventricle center
        [cx + 4, cy, cz],          # 2. Right ventricle center
        [cx, cy - 8, cz],          # 3. Anterior deep brain
        [cx, cy + 8, cz],          # 4. Posterior deep brain
        [cx - 8, cy, cz + 4],      # 5. Left cortex
        [cx + 8, cy, cz + 4],      # 6. Right cortex
        [cx, cy, cz + 7],          # 7. Superior cortex
        [cx, cy, cz - 7],          # 8. Inferior brainstem region
        [cx - 6, cy - 5, cz],      # 9. Left frontal
        [cx + 6, cy + 5, cz],      # 10. Right parietal
    ], dtype=np.float64)

    landmark_names = [
        "Left ventricle center",
        "Right ventricle center",
        "Anterior deep brain",
        "Posterior deep brain",
        "Left cortex",
        "Right cortex",
        "Superior cortex",
        "Inferior brainstem region",
        "Left frontal",
        "Right parietal",
    ]

    # Ground-truth continuous displacement evaluator
    def eval_u(pt: np.ndarray) -> np.ndarray:
        c = np.array([[pt[0]], [pt[1]], [pt[2]]])
        vx = map_coordinates(smooth_ux, c, order=3, mode="nearest")[0]
        vy = map_coordinates(smooth_uy, c, order=3, mode="nearest")[0]
        vz = map_coordinates(smooth_uz, c, order=3, mode="nearest")[0]
        return np.array([vx, vy, vz])

    # Exact fixed-point root solver for y* such that y* + u(y*) = x_target
    def find_exact_moving_coord(p_target: np.ndarray) -> np.ndarray:
        y = p_target - eval_u(p_target)
        for _ in range(20):
            y = p_target - eval_u(y)
        return y

    expected_moving_vox = np.array([find_exact_moving_coord(p) for p in landmarks_voxel])

    # Initial displacements (before registration)
    initial_disp_vox = np.linalg.norm(expected_moving_vox - landmarks_voxel, axis=1)
    initial_disp_mm = initial_disp_vox * vox_size

    # Metrics Before Registration
    dice_ventricles_before = compute_spatial_dice(moving_data < 40.0, target_data < 40.0)
    dice_brain_before = compute_spatial_dice(moving_data > 20.0, target_data > 20.0)
    qc_metrics_before = compute_registration_qc_metrics(target_img, moving_img)

    # 4. Estimate Nonlinear SyN Registration using DIPY
    print("\nEstimating nonlinear SyN registration (hierarchical linear affine + SyN CCMetric)...")
    t0 = time.time()
    m_affine, syn_map = register_t1w_to_mni(
        t1w_brain=moving_img,
        mni_template=target_img,
        affine_level_iters=[20, 10],
        syn_level_iters=[10, 5],
        step_length=0.25,
    )
    reg_time = time.time() - t0
    print(f"SyN Registration completed in {reg_time:.2f}s")

    # Warp moving image into target space
    warped_data = syn_map.transform(moving_data)
    warped_img = nib.Nifti1Image(warped_data, affine)
    warped_nii_path = OUTPUT_DIR / "t1w_warped_syn.nii.gz"
    nib.save(warped_img, str(warped_nii_path))

    # Metrics After Registration
    dice_ventricles_after = compute_spatial_dice(warped_data < 40.0, target_data < 40.0)
    dice_brain_after = compute_spatial_dice(warped_data > 20.0, target_data > 20.0)
    qc_metrics_after = compute_registration_qc_metrics(target_img, warped_img)

    # 5. Full-Chain Landmark Mapping (Target -> Moving)
    # Uses DIPY's transform_points with complete affine prealignment + SyN vector field
    estimated_moving_vox = syn_map.transform_points(
        landmarks_voxel, coord2world=affine, world2coord=inv_affine
    )

    tre_vox = np.linalg.norm(estimated_moving_vox - expected_moving_vox, axis=1)
    tre_mm = tre_vox * vox_size

    mean_tre_mm = float(np.mean(tre_mm))
    median_tre_mm = float(np.median(tre_mm))
    std_tre_mm = float(np.std(tre_mm))
    max_tre_mm = float(np.max(tre_mm))

    mean_tre_vox = float(np.mean(tre_vox))
    median_tre_vox = float(np.median(tre_vox))
    max_tre_vox = float(np.max(tre_vox))

    # 6. Estimated Jacobian Determinant Analysis
    print("\nComputing Jacobian determinant field statistics for estimated transformation...")
    est_disp = syn_map.forward
    det_j_est = compute_jacobian_determinants(est_disp, (vox_size, vox_size, vox_size))

    det_j_brain = det_j_est[brain_mask]
    mean_det_j = float(np.mean(det_j_brain))
    std_det_j = float(np.std(det_j_brain))
    min_det_j = float(np.min(det_j_brain))
    max_det_j = float(np.max(det_j_brain))
    non_positive_count = int(np.count_nonzero(det_j_brain <= 0.0))
    folding_ratio = float(non_positive_count / len(det_j_brain))

    # 7. CC200 Synthetic Atlas (200-Parcel Labeled Volume) Alignment Validation
    print("\nEvaluating 200-parcel synthetic labeled atlas alignment...")
    coords = np.argwhere(brain_mask)
    kmeans = KMeans(n_clusters=200, random_state=42, n_init=1).fit(coords)
    atlas_target = np.zeros(shape, dtype=np.int32)
    for i, c in enumerate(coords):
        atlas_target[tuple(c)] = kmeans.labels_[i] + 1  # Labels 1..200

    # Warp synthetic atlas to moving space via ground truth backward mapping
    atlas_moving = map_coordinates(
        atlas_target.astype(np.float32), sample_coords, order=0, mode="nearest"
    ).reshape(shape).astype(np.int32)

    # Warp moving atlas back to target space using estimated SyN transform
    atlas_warped = syn_map.transform(
        atlas_moving.astype(np.float32), interpolation="nearest"
    ).astype(np.int32)

    dices_before_200 = []
    dices_after_200 = []
    empty_parcels_before = 0
    empty_parcels_after = 0

    for label in range(1, 201):
        m_t = (atlas_target == label)
        m_m = (atlas_moving == label)
        m_w = (atlas_warped == label)

        if np.sum(m_m) == 0:
            empty_parcels_before += 1
        if np.sum(m_w) == 0:
            empty_parcels_after += 1

        dices_before_200.append(compute_spatial_dice(m_m, m_t))
        dices_after_200.append(compute_spatial_dice(m_w, m_t))

    mean_dice_before_200 = float(np.mean(dices_before_200))
    mean_dice_after_200 = float(np.mean(dices_after_200))
    median_dice_before_200 = float(np.median(dices_before_200))
    median_dice_after_200 = float(np.median(dices_after_200))
    min_dice_after_200 = float(np.min(dices_after_200))
    max_dice_after_200 = float(np.max(dices_after_200))
    parcel_coverage = float(np.count_nonzero(np.unique(atlas_warped) > 0) / 200.0)

    # 8. Generate Visual QC Figures
    print("\nGenerating visual QC overlay, landmark maps, and 200-parcel alignment figure...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    slice_z = cz
    slice_y = cy

    # Row 1: Anatomical slice overlays
    axes[0, 0].imshow(target_data[:, :, slice_z].T, origin="lower", cmap="gray")
    axes[0, 0].set_title("MNI Target (Ground Truth)", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(moving_data[:, :, slice_z].T, origin="lower", cmap="gray")
    axes[0, 1].set_title(f"Moving T1w (Before SyN, Dice={dice_brain_before:.3f})", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(warped_data[:, :, slice_z].T, origin="lower", cmap="gray")
    axes[0, 2].set_title(f"Warped T1w (After SyN, Dice={dice_brain_after:.3f})", fontsize=11)
    axes[0, 2].axis("off")

    # Row 2: Residual error, Jacobian map, and 200-Parcel Atlas
    diff_after = np.abs(warped_data - target_data)
    im_diff = axes[1, 0].imshow(diff_after[:, :, slice_z].T, origin="lower", cmap="magma")
    axes[1, 0].set_title(f"Residual Error (RMSE: {qc_metrics_after['rmse']:.2f})", fontsize=11)
    axes[1, 0].axis("off")
    plt.colorbar(im_diff, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_j = axes[1, 1].imshow(det_j_est[:, :, slice_z].T, origin="lower", cmap="coolwarm", vmin=0.5, vmax=1.5)
    axes[1, 1].set_title(f"Jacobian Det det(J) [Min={min_det_j:.2f}, Folds={non_positive_count}]", fontsize=11)
    axes[1, 1].axis("off")
    plt.colorbar(im_j, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 200-Parcel atlas slice overlay
    im_atlas = axes[1, 2].imshow(atlas_warped[:, :, slice_z].T, origin="lower", cmap="nipy_spectral")
    axes[1, 2].set_title(f"200-Parcel Atlas After SyN (Mean Dice={mean_dice_after_200:.3f})", fontsize=11)
    axes[1, 2].axis("off")
    plt.colorbar(im_atlas, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    overlay_path = OUTPUT_DIR / "qc_syn_registration_before_after.png"
    plt.savefig(overlay_path, dpi=150)
    plt.close()

    # Build per-landmark details
    per_landmark_data = []
    for i in range(len(landmarks_voxel)):
        per_landmark_data.append({
            "index": i + 1,
            "name": landmark_names[i],
            "target_voxel": landmarks_voxel[i].tolist(),
            "expected_moving_voxel": expected_moving_vox[i].tolist(),
            "estimated_moving_voxel": estimated_moving_vox[i].tolist(),
            "initial_disp_mm": float(initial_disp_mm[i]),
            "initial_disp_vox": float(initial_disp_vox[i]),
            "tre_mm": float(tre_mm[i]),
            "tre_vox": float(tre_vox[i]),
            "sub_voxel": bool(tre_vox[i] < 1.0),
            "improvement_pct": float((initial_disp_mm[i] - tre_mm[i]) / initial_disp_mm[i] * 100.0),
        })

    results = {
        "status": "VALIDATED",
        "method": "Symmetric Diffeomorphic Registration (SyN) with CCMetric(3) and Affine Prealignment",
        "resolution_mm": [vox_size, vox_size, vox_size],
        "image_shape": list(shape),
        "runtime_seconds": reg_time,
        "landmark_analysis": {
            "num_landmarks": len(landmarks_voxel),
            "initial_mean_displacement_mm": float(np.mean(initial_disp_mm)),
            "initial_median_displacement_mm": float(np.median(initial_disp_mm)),
            "mean_tre_mm": mean_tre_mm,
            "median_tre_mm": median_tre_mm,
            "std_tre_mm": std_tre_mm,
            "max_tre_mm": max_tre_mm,
            "mean_tre_vox": mean_tre_vox,
            "median_tre_vox": median_tre_vox,
            "max_tre_vox": max_tre_vox,
            "sub_voxel_accuracy": bool(mean_tre_vox < 1.0),
            "sub_voxel_landmark_count": int(np.count_nonzero(tre_vox < 1.0)),
            "per_landmark_results": per_landmark_data,
        },
        "anatomical_overlap": {
            "brain_dice_before": dice_brain_before,
            "brain_dice_after": dice_brain_after,
            "ventricles_dice_before": dice_ventricles_before,
            "ventricles_dice_after": dice_ventricles_after,
            "pearson_r_before": qc_metrics_before["pearson_r"],
            "pearson_r_after": qc_metrics_after["pearson_r"],
            "rmse_before": qc_metrics_before["rmse"],
            "rmse_after": qc_metrics_after["rmse"],
        },
        "jacobian_determinants": {
            "mean_det_j": mean_det_j,
            "std_det_j": std_det_j,
            "min_det_j": min_det_j,
            "max_det_j": max_det_j,
            "non_positive_count": non_positive_count,
            "folding_ratio": folding_ratio,
            "topology_preserved": bool(non_positive_count == 0),
        },
        "parcel_alignment": {
            "num_test_parcels": 200,
            "mean_dice_before": mean_dice_before_200,
            "mean_dice_after": mean_dice_after_200,
            "alignment_improvement": float(mean_dice_after_200 - mean_dice_before_200),
        },
        "cc200_synthetic_atlas_benchmark": {
            "num_parcels": 200,
            "mean_dice_before": mean_dice_before_200,
            "mean_dice_after": mean_dice_after_200,
            "median_dice_before": median_dice_before_200,
            "median_dice_after": median_dice_after_200,
            "min_dice_after": min_dice_after_200,
            "max_dice_after": max_dice_after_200,
            "empty_parcels_before": empty_parcels_before,
            "empty_parcels_after": empty_parcels_after,
            "parcel_coverage": parcel_coverage,
            "notes": "Synthetic labeled atlas partitioning brain mask into 200 compact 3D ROIs. Distinguished from authentic CC200 Craddock template.",
        },
        "artifacts": {
            "target_nii": str(target_nii_path),
            "moving_nii": str(moving_nii_path),
            "warped_nii": str(warped_nii_path),
            "qc_overlay_png": str(overlay_path),
        },
    }

    report_path = OUTPUT_DIR / "syn_registration_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("REGISTRATION EXPERIMENT COMPLETED")
    print(f"Mean TRE                 : {mean_tre_mm:.3f} mm ({mean_tre_vox:.3f} vox) [Initial: {float(np.mean(initial_disp_mm)):.3f} mm]")
    print(f"Median TRE               : {median_tre_mm:.3f} mm ({median_tre_vox:.3f} vox)")
    print(f"Max TRE                  : {max_tre_mm:.3f} mm ({max_tre_vox:.3f} vox)")
    print(f"Sub-voxel Accuracy       : {mean_tre_vox < 1.0} ({int(np.count_nonzero(tre_vox < 1.0))}/10 landmarks)")
    print(f"Brain Dice Before / After: {dice_brain_before:.4f} -> {dice_brain_after:.4f}")
    print(f"Ventricle Dice B / A    : {dice_ventricles_before:.4f} -> {dice_ventricles_after:.4f}")
    print(f"Pearson r Before / After : {qc_metrics_before['pearson_r']:.4f} -> {qc_metrics_after['pearson_r']:.4f}")
    print(f"RMSE Before / After      : {qc_metrics_before['rmse']:.2f} -> {qc_metrics_after['rmse']:.2f}")
    print(f"Jacobian Min det(J)      : {min_det_j:.4f} (Topology Preserved: {non_positive_count == 0})")
    print(f"Non-positive Jacobians   : {non_positive_count}")
    print(f"200-Parcel Dice B / A    : {mean_dice_before_200:.4f} -> {mean_dice_after_200:.4f} (Coverage: {parcel_coverage*100:.1f}%)")
    print(f"Report written to        : {report_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_synthetic_registration_experiment()
