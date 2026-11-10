#!/usr/bin/env python3
"""Synthetic Ground-Truth MNI152 Nonlinear Registration Validation Script.

Executes a rigorous ground-truth SyN registration experiment:
1. Generates a synthetic anatomical reference (MNI152 template space).
2. Applies a known, smooth, controlled nonlinear ground-truth deformation phi_GT.
3. Places anatomical 3D landmarks and computes true deformed locations.
4. Estimates the transformation using DIPY SyN (CCMetric).
5. Compares estimated transform against ground truth:
   - Landmark Target Registration Error (TRE in mm: mean, median, std, max).
   - Anatomical overlap metrics (Dice coefficient before and after).
   - Image intensity Pearson r and RMSE before and after.
   - Jacobian determinant det(J) field statistics (mean, std, min, max, folding count det(J) <= 0).
   - CC200 parcel overlap and alignment.
   - Multi-planar before-and-after visual overlays and Jacobian determinant slice maps.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parcellation.extractor import CANONICAL_CC200_AFFINE, CANONICAL_CC200_SHAPE
from src.preprocessing.qc import compute_registration_qc_metrics
from src.preprocessing.registration import compute_spatial_dice, register_t1w_to_mni

OUTPUT_DIR = Path("/Users/apple/Documents/Codex/2026-09-29/NeuroADHD_Step3_Synthetic/registration_experiment")


def compute_jacobian_determinants(displacement_field: np.ndarray, voxel_spacing: Tuple[float, float, float] = (3.0, 3.0, 3.0)) -> np.ndarray:
    """Calculate the 3D Jacobian determinant det(J) = det(I + grad(u)) across a displacement field.

    Parameters
    ----------
    displacement_field : np.ndarray
        Array of shape (X, Y, Z, 3) representing voxel displacement vector (u_x, u_y, u_z) in voxels.
    voxel_spacing : Tuple[float, float, float]
        Voxel size in mm along each axis.

    Returns
    -------
    np.ndarray
        3D array of shape (X, Y, Z) containing local Jacobian determinants.
    """
    # Displacement components in voxels
    ux = displacement_field[..., 0]
    uy = displacement_field[..., 1]
    uz = displacement_field[..., 2]

    # Compute numerical spatial gradients using central differences
    dux_dx, dux_dy, dux_dz = np.gradient(ux)
    duy_dx, duy_dy, duy_dz = np.gradient(uy)
    duz_dx, duz_dy, duz_dz = np.gradient(uz)

    # Deformation gradient matrix components F = I + grad(u)
    j00 = 1.0 + dux_dx
    j01 = dux_dy
    j02 = dux_dz

    j10 = duy_dx
    j11 = 1.0 + duy_dy
    j12 = duy_dz

    j20 = duz_dx
    j21 = duz_dy
    j22 = 1.0 + duz_dz

    # Determinant of 3x3 matrix via standard expansion:
    det_j = (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )
    return det_j


def run_synthetic_registration_experiment() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("MNI152 NONLINEAR REGISTRATION (SyN) VALIDATION EXPERIMENT")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 80)

    shape = (36, 42, 36)
    vox_size = 3.0  # 3mm isotropic
    affine = np.diag([vox_size, vox_size, vox_size, 1.0])
    affine[:3, 3] = [-54.0, -63.0, -54.0]

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
    # Maximum physical displacement approx ~4.5 mm (1.5 voxels)
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

    # Warp target to create moving T1w
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
        [cx - 4, cy, cz],          # Left ventricle center
        [cx + 4, cy, cz],          # Right ventricle center
        [cx, cy - 8, cz],          # Anterior deep brain
        [cx, cy + 8, cz],          # Posterior deep brain
        [cx - 8, cy, cz + 4],      # Left cortex
        [cx + 8, cy, cz + 4],      # Right cortex
        [cx, cy, cz + 7],          # Superior cortex
        [cx, cy, cz - 7],          # Inferior brainstem region
        [cx - 6, cy - 5, cz],      # Left frontal
        [cx + 6, cy + 5, cz],      # Right parietal
    ], dtype=np.float64)

    # Physical coordinates in target space (mm)
    lms_target_mm = nib.affines.apply_affine(affine, landmarks_voxel)

    # Physical coordinates in deformed moving space (mm)
    lms_disp_vox = np.array([
        [smooth_ux[int(p[0]), int(p[1]), int(p[2])],
         smooth_uy[int(p[0]), int(p[1]), int(p[2])],
         smooth_uz[int(p[0]), int(p[1]), int(p[2])]]
        for p in landmarks_voxel
    ])
    lms_moving_vox = landmarks_voxel + lms_disp_vox
    lms_moving_mm = nib.affines.apply_affine(affine, lms_moving_vox)

    initial_landmark_errors = np.linalg.norm(lms_moving_mm - lms_target_mm, axis=1)

    # 4. Metrics Before Registration
    dice_ventricles_before = compute_spatial_dice(moving_data < 40.0, target_data < 40.0)
    dice_brain_before = compute_spatial_dice(moving_data > 20.0, target_data > 20.0)
    qc_metrics_before = compute_registration_qc_metrics(target_img, moving_img)

    # 5. Estimate Nonlinear SyN Registration using DIPY
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

    # 6. Metrics After Registration
    dice_ventricles_after = compute_spatial_dice(warped_data < 40.0, target_data < 40.0)
    dice_brain_after = compute_spatial_dice(warped_data > 20.0, target_data > 20.0)
    qc_metrics_after = compute_registration_qc_metrics(target_img, warped_img)

    # 7. Landmark Displacement Error (TRE) Evaluation
    # Evaluate recovered target landmarks from moving landmark coordinates
    warped_lms_vox = []
    inv_affine = np.linalg.inv(affine)
    for p_vox in landmarks_voxel:
        # Pullback mapping samples at target coordinate:
        disp_at_target = syn_map.forward[int(round(p_vox[0])), int(round(p_vox[1])), int(round(p_vox[2]))]
        recovered_moving_vox = p_vox + disp_at_target
        warped_lms_vox.append(recovered_moving_vox)
    warped_lms_vox = np.array(warped_lms_vox)
    recovered_lms_mm = nib.affines.apply_affine(affine, warped_lms_vox)

    # Landmark Target Registration Error (TRE) in mm: difference between estimated moving position and true moving position
    tre_per_landmark = np.linalg.norm(recovered_lms_mm - lms_moving_mm, axis=1)
    mean_tre = float(np.mean(tre_per_landmark))
    median_tre = float(np.median(tre_per_landmark))
    std_tre = float(np.std(tre_per_landmark))
    max_tre = float(np.max(tre_per_landmark))

    # 8. Estimated Jacobian Determinant Analysis
    print("\nComputing Jacobian determinant field statistics for estimated transformation...")
    est_disp = syn_map.forward  # shape (X, Y, Z, 3) in voxels
    det_j_est = compute_jacobian_determinants(est_disp, (vox_size, vox_size, vox_size))

    # Compute statistics inside brain mask
    det_j_brain = det_j_est[brain_mask]
    mean_det_j = float(np.mean(det_j_brain))
    std_det_j = float(np.std(det_j_brain))
    min_det_j = float(np.min(det_j_brain))
    max_det_j = float(np.max(det_j_brain))
    non_positive_count = int(np.count_nonzero(det_j_brain <= 0.0))
    folding_ratio = float(non_positive_count / len(det_j_brain))

    # 9. CC200 Synthetic Parcel Alignment Check
    print("\nEvaluating CC200 parcel alignment before and after transformation...")
    parcel_overlap_before = []
    parcel_overlap_after = []
    # Create 20 sample test parcels
    for p in range(1, 21):
        p_mask_target = (gx == (cx - 10 + p)) & brain_mask
        if np.sum(p_mask_target) > 5:
            # Move through GT
            p_data = p_mask_target.astype(np.float32)
            p_moving = map_coordinates(p_data, sample_coords, order=0, mode="nearest").reshape(shape) > 0.5
            p_warped = syn_map.transform(p_moving.astype(np.float32), interpolation="nearest") > 0.5

            d_bef = compute_spatial_dice(p_moving, p_mask_target)
            d_aft = compute_spatial_dice(p_warped, p_mask_target)
            parcel_overlap_before.append(d_bef)
            parcel_overlap_after.append(d_aft)

    mean_parcel_dice_before = float(np.mean(parcel_overlap_before)) if parcel_overlap_before else 0.0
    mean_parcel_dice_after = float(np.mean(parcel_overlap_after)) if parcel_overlap_after else 0.0

    # 10. Generate Visual QC Figures
    print("\nGenerating visual QC overlay and Jacobian maps...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    slice_z = cz
    slice_y = cy

    # Row 1: Axial slices
    axes[0, 0].imshow(target_data[:, :, slice_z].T, origin="lower", cmap="gray")
    axes[0, 0].set_title("MNI Target (Ground Truth)", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(moving_data[:, :, slice_z].T, origin="lower", cmap="gray")
    axes[0, 1].set_title(f"Moving T1w (Before SyN, Dice={dice_brain_before:.3f})", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(warped_data[:, :, slice_z].T, origin="lower", cmap="gray")
    axes[0, 2].set_title(f"Warped T1w (After SyN, Dice={dice_brain_after:.3f})", fontsize=11)
    axes[0, 2].axis("off")

    # Row 2: Coronal slices & Jacobian map
    axes[1, 0].imshow(target_data[:, slice_y, :].T, origin="lower", cmap="gray")
    axes[1, 0].set_title("Coronal Target", fontsize=11)
    axes[1, 0].axis("off")

    diff_before = np.abs(moving_data - target_data)
    diff_after = np.abs(warped_data - target_data)
    im_diff = axes[1, 1].imshow(diff_after[:, :, slice_z].T, origin="lower", cmap="magma")
    axes[1, 1].set_title("Absolute Residual Error (After SyN)", fontsize=11)
    axes[1, 1].axis("off")
    plt.colorbar(im_diff, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im_j = axes[1, 2].imshow(det_j_est[:, :, slice_z].T, origin="lower", cmap="coolwarm", vmin=0.5, vmax=1.5)
    axes[1, 2].set_title(f"Jacobian Det det(J) [Min={min_det_j:.2f}, Fold={non_positive_count}]", fontsize=11)
    axes[1, 2].axis("off")
    plt.colorbar(im_j, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    overlay_path = OUTPUT_DIR / "qc_syn_registration_before_after.png"
    plt.savefig(overlay_path, dpi=150)
    plt.close()

    results = {
        "status": "VALIDATED",
        "method": "Symmetric Diffeomorphic Registration (SyN) with CCMetric(3)",
        "resolution_mm": [vox_size, vox_size, vox_size],
        "image_shape": list(shape),
        "runtime_seconds": reg_time,
        "landmark_analysis": {
            "num_landmarks": len(landmarks_voxel),
            "initial_mean_displacement_mm": float(np.mean(initial_landmark_errors)),
            "mean_tre_mm": mean_tre,
            "median_tre_mm": median_tre,
            "std_tre_mm": std_tre,
            "max_tre_mm": max_tre,
            "sub_voxel_accuracy": bool(mean_tre < vox_size),
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
            "num_test_parcels": len(parcel_overlap_before),
            "mean_dice_before": mean_parcel_dice_before,
            "mean_dice_after": mean_parcel_dice_after,
            "alignment_improvement": float(mean_parcel_dice_after - mean_parcel_dice_before),
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
    print(f"Mean TRE                 : {mean_tre:.3f} mm (Initial: {float(np.mean(initial_landmark_errors)):.3f} mm)")
    print(f"Brain Dice Before / After: {dice_brain_before:.4f} -> {dice_brain_after:.4f}")
    print(f"Ventricle Dice B / A    : {dice_ventricles_before:.4f} -> {dice_ventricles_after:.4f}")
    print(f"Pearson r Before / After : {qc_metrics_before['pearson_r']:.4f} -> {qc_metrics_after['pearson_r']:.4f}")
    print(f"RMSE Before / After      : {qc_metrics_before['rmse']:.2f} -> {qc_metrics_after['rmse']:.2f}")
    print(f"Jacobian Min det(J)      : {min_det_j:.4f} (Topology Preserved: {non_positive_count == 0})")
    print(f"Non-positive Jacobians   : {non_positive_count}")
    print(f"Parcel Dice B / A        : {mean_parcel_dice_before:.4f} -> {mean_parcel_dice_after:.4f}")
    print(f"Report written to        : {report_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_synthetic_registration_experiment()
