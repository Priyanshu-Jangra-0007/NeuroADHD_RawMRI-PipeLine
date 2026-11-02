"""Quality control visualization module for registration, tissue masks, and parcellation.

Enforces strict geometry validation: QC overlays must evaluate images that are already
spatially aligned and resampled into the same physical grid. No implicit interpolation
or resampling is permitted inside QC routines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import laplace


def validate_qc_grid_alignment(img_a: nib.Nifti1Image, img_b: nib.Nifti1Image, label_a: str, label_b: str) -> None:
    """Strictly verify that two images share identical spatial dimensions and grid affines."""
    shape_a = img_a.shape[:3]
    shape_b = img_b.shape[:3]
    if shape_a != shape_b:
        raise ValueError(
            f"Registration QC geometry mismatch: {label_a} shape {shape_a} != {label_b} shape {shape_b}. "
            "Images must be resampled into the same spatial grid prior to generating visual QC overlays."
        )
    if not np.allclose(img_a.affine, img_b.affine, atol=1e-4):
        raise ValueError(
            f"Registration QC affine mismatch: {label_a} affine does not match {label_b} affine. "
            "Implicit grid resampling inside QC is prohibited. Images must be explicitly registered."
        )


def generate_registration_qc_overlay(
    background_img: nib.Nifti1Image,
    contour_img: nib.Nifti1Image,
    output_png_path: Path,
    title: str = "Registration QC Overlay",
    contour_level: float = 0.5,
    contour_color: str = "red",
) -> Path:
    """Generate 3-plane orthogonal slice inspection figure showing contour overlay.

    Parameters
    ----------
    background_img : nib.Nifti1Image
        Underlying anatomical or template image (grayscale).
    contour_img : nib.Nifti1Image
        Aligned image whose contours will be overlaid.
    output_png_path : Path
        Destination file path for the output PNG.
    title : str
        Figure title.
    contour_level : float
        Iso-surface level for contour drawing.
    contour_color : str
        Matplotlib color specification for the contour.

    Returns
    -------
    Path
        Path to the saved PNG image.
    """
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    validate_qc_grid_alignment(background_img, contour_img, "background_img", "contour_img")

    bg_data = np.asanyarray(background_img.dataobj, dtype=np.float32)
    fg_data = np.asanyarray(contour_img.dataobj, dtype=np.float32)

    # Normalize foreground to [0, 1] relative to foreground max
    fg_max = float(np.max(fg_data)) if np.max(fg_data) > 0 else 1.0
    fg_norm = fg_data / fg_max

    # Determine center slice indices along each axis
    nx, ny, nz = bg_data.shape[:3]
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Sagittal (X slice, Y-Z plane)
    bg_sag = np.rot90(bg_data[cx, :, :])
    fg_sag = np.rot90(fg_norm[cx, :, :])
    axes[0].imshow(bg_sag, cmap="gray", origin="upper")
    if np.any(fg_sag >= contour_level):
        axes[0].contour(fg_sag, levels=[contour_level], colors=contour_color, linewidths=1.2)
    axes[0].set_title(f"Sagittal (x={cx})")
    axes[0].axis("off")

    # 2. Coronal (Y slice, X-Z plane)
    bg_cor = np.rot90(bg_data[:, cy, :])
    fg_cor = np.rot90(fg_norm[:, cy, :])
    axes[1].imshow(bg_cor, cmap="gray", origin="upper")
    if np.any(fg_cor >= contour_level):
        axes[1].contour(fg_cor, levels=[contour_level], colors=contour_color, linewidths=1.2)
    axes[1].set_title(f"Coronal (y={cy})")
    axes[1].axis("off")

    # 3. Axial (Z slice, X-Y plane)
    bg_ax = np.rot90(bg_data[:, :, cz])
    fg_ax = np.rot90(fg_norm[:, :, cz])
    axes[2].imshow(bg_ax, cmap="gray", origin="upper")
    if np.any(fg_ax >= contour_level):
        axes[2].contour(fg_ax, levels=[contour_level], colors=contour_color, linewidths=1.2)
    axes[2].set_title(f"Axial (z={cz})")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_png_path


def generate_coregistration_qc_overlay(
    t1w_brain: nib.Nifti1Image,
    resampled_epi: nib.Nifti1Image,
    output_png_path: Path,
) -> Path:
    """Generate QC overlay of coregistered functional EPI contours onto anatomical T1w brain."""
    return generate_registration_qc_overlay(
        background_img=t1w_brain,
        contour_img=resampled_epi,
        output_png_path=output_png_path,
        title="QC: Functional EPI to Anatomical T1w Coregistration",
        contour_color="red",
    )


def generate_normalization_qc_overlay(
    mni_template: nib.Nifti1Image,
    normalized_t1w: nib.Nifti1Image,
    output_png_path: Path,
) -> Path:
    """Generate QC overlay of normalized anatomical T1w contours onto standard MNI152 template."""
    return generate_registration_qc_overlay(
        background_img=mni_template,
        contour_img=normalized_t1w,
        output_png_path=output_png_path,
        title="QC: Subject T1w to MNI152 Standard Normalization",
        contour_color="yellow",
    )


def generate_tissue_mask_qc_overlay(
    anatomical_img: nib.Nifti1Image,
    wm_mask: nib.Nifti1Image,
    csf_mask: nib.Nifti1Image,
    output_png_path: Path,
    title: str = "QC: Anatomical Tissue Masks (WM & CSF)",
) -> Path:
    """Generate 3-plane QC overlay showing eroded WM (blue) and CSF (red) masks on anatomical background."""
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    validate_qc_grid_alignment(anatomical_img, wm_mask, "anatomical_img", "wm_mask")
    validate_qc_grid_alignment(anatomical_img, csf_mask, "anatomical_img", "csf_mask")

    bg_data = np.asanyarray(anatomical_img.dataobj, dtype=np.float32)
    wm_data = np.asanyarray(wm_mask.dataobj, dtype=bool)
    csf_data = np.asanyarray(csf_mask.dataobj, dtype=bool)

    nx, ny, nz = bg_data.shape[:3]
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    planes = [
        ("Sagittal", cx, lambda a, idx: np.rot90(a[idx, :, :])),
        ("Coronal", cy, lambda a, idx: np.rot90(a[:, idx, :])),
        ("Axial", cz, lambda a, idx: np.rot90(a[:, :, idx])),
    ]

    for ax_idx, (name, slice_idx, slicer) in enumerate(planes):
        ax = axes[ax_idx]
        bg_slice = slicer(bg_data, slice_idx)
        wm_slice = slicer(wm_data, slice_idx)
        csf_slice = slicer(csf_data, slice_idx)

        ax.imshow(bg_slice, cmap="gray", origin="upper")
        if np.any(wm_slice):
            ax.contour(wm_slice, levels=[0.5], colors="dodgerblue", linewidths=1.2)
        if np.any(csf_slice):
            ax.contour(csf_slice, levels=[0.5], colors="crimson", linewidths=1.2)
        ax.set_title(f"{name} (slice={slice_idx})")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_png_path


def generate_parcellation_qc_overlay(
    mean_bold_img: nib.Nifti1Image,
    atlas_labels_img: nib.Nifti1Image,
    output_png_path: Path,
    title: str = "CC200 Parcellation Alignment QC",
) -> Path:
    """Generate 3-plane inspection figure showing CC200 parcel boundaries over mean functional image.

    Parameters
    ----------
    mean_bold_img : nib.Nifti1Image
        Normalized 3D mean functional image.
    atlas_labels_img : nib.Nifti1Image
        CC200 atlas parcellation containing integer labels 1..200.
    output_png_path : Path
        Destination file path for the output PNG.
    title : str
        Figure title.

    Returns
    -------
    Path
        Path to the saved PNG image.
    """
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    validate_qc_grid_alignment(mean_bold_img, atlas_labels_img, "mean_bold_img", "atlas_labels_img")

    bold_data = np.asanyarray(mean_bold_img.dataobj, dtype=np.float32)
    atlas_data = np.asanyarray(atlas_labels_img.dataobj, dtype=np.int32)

    # Compute slice indices with high parcel density
    nx, ny, nz = bold_data.shape[:3]
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Sagittal
    bg_sag = np.rot90(bold_data[cx, :, :])
    edge_sag = np.abs(laplace(atlas_data[cx, :, :].astype(np.float32))) > 0
    edge_sag = np.rot90(edge_sag)
    axes[0].imshow(bg_sag, cmap="gray", origin="upper")
    axes[0].contour(edge_sag, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[0].set_title(f"Sagittal (x={cx})")
    axes[0].axis("off")

    # 2. Coronal
    bg_cor = np.rot90(bold_data[:, cy, :])
    edge_cor = np.abs(laplace(atlas_data[:, cy, :].astype(np.float32))) > 0
    edge_cor = np.rot90(edge_cor)
    axes[1].imshow(bg_cor, cmap="gray", origin="upper")
    axes[1].contour(edge_cor, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[1].set_title(f"Coronal (y={cy})")
    axes[1].axis("off")

    # 3. Axial
    bg_ax = np.rot90(bold_data[:, :, cz])
    edge_ax = np.abs(laplace(atlas_data[:, :, cz].astype(np.float32))) > 0
    edge_ax = np.rot90(edge_ax)
    axes[2].imshow(bg_ax, cmap="gray", origin="upper")
    axes[2].contour(edge_ax, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[2].set_title(f"Axial (z={cz})")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_png_path


def compute_registration_qc_metrics(
    target_img: nib.Nifti1Image,
    aligned_img: nib.Nifti1Image,
    mask: Optional[nib.Nifti1Image] = None,
    correlation_threshold: float = 0.70,
    dice_threshold: float = 0.70,
) -> Dict[str, Any]:
    """Compute quantitative registration accuracy metrics between target and aligned images.

    Evaluates:
    - Pearson correlation (NCC proxy)
    - Root-Mean-Square Error (RMSE)
    - Binary Sørensen-Dice coefficient (at 10% maximum intensity)

    Parameters
    ----------
    target_img : nib.Nifti1Image
        Reference image in target space.
    aligned_img : nib.Nifti1Image
        Resampled image aligned to the target grid.
    mask : Optional[nib.Nifti1Image]
        Optional binary mask restricting metric evaluation.
    correlation_threshold : float
        Minimum Pearson correlation coefficient required for pass.
    dice_threshold : float
        Minimum Sørensen-Dice coefficient required for pass.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - "pearson_r": float
        - "rmse": float
        - "dice": float
        - "qc_pass": bool
        - "n_voxels_evaluated": int
    """
    validate_qc_grid_alignment(target_img, aligned_img, "target_img", "aligned_img")
    if mask is not None:
        validate_qc_grid_alignment(target_img, mask, "target_img", "mask")

    t_data = np.asanyarray(target_img.dataobj, dtype=np.float32)
    a_data = np.asanyarray(aligned_img.dataobj, dtype=np.float32)

    if mask is not None:
        roi = np.asanyarray(mask.dataobj) > 0
    else:
        roi = (np.abs(t_data) > 1e-4) | (np.abs(a_data) > 1e-4)

    if not np.any(roi):
        roi = np.ones(t_data.shape, dtype=bool)

    t_vals = t_data[roi]
    a_vals = a_data[roi]

    # 1. Pearson correlation
    std_t = float(np.std(t_vals))
    std_a = float(np.std(a_vals))
    if std_t > 1e-7 and std_a > 1e-7:
        pearson_r = float(np.corrcoef(t_vals, a_vals)[0, 1])
        if np.isnan(pearson_r):
            pearson_r = 1.0 if np.allclose(t_vals, a_vals) else 0.0
    else:
        pearson_r = 1.0 if np.allclose(t_vals, a_vals) else 0.0

    # 2. RMSE
    rmse = float(np.sqrt(np.mean((t_vals - a_vals) ** 2)))

    # 3. Sørensen-Dice
    t_max = float(np.max(t_vals)) if len(t_vals) > 0 and np.max(t_vals) > 0 else 1.0
    a_max = float(np.max(a_vals)) if len(a_vals) > 0 and np.max(a_vals) > 0 else 1.0
    bin_t = t_vals > (0.1 * t_max)
    bin_a = a_vals > (0.1 * a_max)

    intersection = int(np.sum(bin_t & bin_a))
    denom = int(np.sum(bin_t) + np.sum(bin_a))
    dice = float((2.0 * intersection) / denom) if denom > 0 else 1.0

    qc_pass = bool((pearson_r >= correlation_threshold) and (dice >= dice_threshold))

    return {
        "pearson_r": pearson_r,
        "rmse": rmse,
        "dice": dice,
        "qc_pass": qc_pass,
        "n_voxels_evaluated": int(np.sum(roi)),
    }


def generate_before_after_qc_figure(
    target_img: nib.Nifti1Image,
    before_img: nib.Nifti1Image,
    after_img: nib.Nifti1Image,
    output_png_path: Path,
    title: str = "Registration Before vs After QC",
) -> Path:
    """Generate 3x3 orthogonal comparison panel: Target, Before Registration, After Registration.

    Parameters
    ----------
    target_img : nib.Nifti1Image
        Ground-truth or static reference image.
    before_img : nib.Nifti1Image
        Unregistered image on target grid (e.g., identity resample).
    after_img : nib.Nifti1Image
        Registered / transformed image on target grid.
    output_png_path : Path
        Destination PNG path.
    title : str
        Figure suptitle.

    Returns
    -------
    Path
        Path to generated QC figure.
    """
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    validate_qc_grid_alignment(target_img, before_img, "target_img", "before_img")
    validate_qc_grid_alignment(target_img, after_img, "target_img", "after_img")

    t_data = np.asanyarray(target_img.dataobj, dtype=np.float32)
    b_data = np.asanyarray(before_img.dataobj, dtype=np.float32)
    a_data = np.asanyarray(after_img.dataobj, dtype=np.float32)

    nx, ny, nz = t_data.shape[:3]
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    rows = [
        ("Target Reference", t_data),
        ("Before Registration", b_data),
        ("After Registration", a_data),
    ]

    for row_idx, (row_label, data) in enumerate(rows):
        sag = np.rot90(data[cx, :, :])
        cor = np.rot90(data[:, cy, :])
        ax_view = np.rot90(data[:, :, cz])

        axes[row_idx, 0].imshow(sag, cmap="gray", origin="upper")
        axes[row_idx, 0].set_ylabel(row_label, fontsize=12, fontweight="bold")
        if row_idx == 0:
            axes[row_idx, 0].set_title(f"Sagittal (x={cx})")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(cor, cmap="gray", origin="upper")
        if row_idx == 0:
            axes[row_idx, 1].set_title(f"Coronal (y={cy})")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(ax_view, cmap="gray", origin="upper")
        if row_idx == 0:
            axes[row_idx, 2].set_title(f"Axial (z={cz})")
        axes[row_idx, 2].axis("off")

    plt.tight_layout()
    fig.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_png_path


def generate_landmark_qc_overlay(
    background_img: nib.Nifti1Image,
    target_landmarks_mm: np.ndarray,
    warped_landmarks_mm: np.ndarray,
    output_png_path: Path,
    max_allowed_tre_mm: float = 3.0,
    title: str = "Landmark Alignment QC Overlay",
) -> Tuple[Path, Dict[str, Any]]:
    """Compute Target Registration Error (TRE) and generate visual landmark overlay.

    Parameters
    ----------
    background_img : nib.Nifti1Image
        Reference image in target space.
    target_landmarks_mm : np.ndarray
        (N, 3) coordinates of target landmarks in physical millimeters.
    warped_landmarks_mm : np.ndarray
        (N, 3) coordinates of transformed landmarks in physical millimeters.
    output_png_path : Path
        Destination PNG file path.
    max_allowed_tre_mm : float
        Maximum acceptable TRE threshold (mm).
    title : str
        Figure suptitle.

    Returns
    -------
    Tuple[Path, Dict[str, Any]]
        File path and dictionary with TRE metrics:
        - "mean_tre_mm": float
        - "max_tre_mm": float
        - "median_tre_mm": float
        - "max_allowed_tre_mm": float
        - "qc_pass": bool
        - "tre_errors_mm": List[float]
    """
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    t_lms = np.asarray(target_landmarks_mm, dtype=np.float64)
    w_lms = np.asarray(warped_landmarks_mm, dtype=np.float64)

    if t_lms.ndim != 2 or t_lms.shape[1] != 3:
        raise ValueError(f"target_landmarks_mm must have shape (N, 3), got {t_lms.shape}")
    if w_lms.ndim != 2 or w_lms.shape[1] != 3:
        raise ValueError(f"warped_landmarks_mm must have shape (N, 3), got {w_lms.shape}")
    if len(t_lms) != len(w_lms):
        raise ValueError(f"Number of landmarks mismatch: {len(t_lms)} vs {len(w_lms)}")

    diffs = w_lms - t_lms
    errors_mm = np.linalg.norm(diffs, axis=1)

    mean_tre = float(np.mean(errors_mm)) if len(errors_mm) > 0 else 0.0
    max_tre = float(np.max(errors_mm)) if len(errors_mm) > 0 else 0.0
    median_tre = float(np.median(errors_mm)) if len(errors_mm) > 0 else 0.0
    qc_pass = bool(max_tre <= max_allowed_tre_mm)

    metrics = {
        "mean_tre_mm": mean_tre,
        "max_tre_mm": max_tre,
        "median_tre_mm": median_tre,
        "max_allowed_tre_mm": float(max_allowed_tre_mm),
        "qc_pass": qc_pass,
        "tre_errors_mm": errors_mm.tolist(),
    }

    # Convert mm coordinates to voxel coordinates on background_img
    inv_affine = np.linalg.inv(background_img.affine)
    t_vox = (inv_affine[:3, :3] @ t_lms.T).T + inv_affine[:3, 3]
    w_vox = (inv_affine[:3, :3] @ w_lms.T).T + inv_affine[:3, 3]

    bg_data = np.asanyarray(background_img.dataobj, dtype=np.float32)
    nx, ny, nz = bg_data.shape[:3]
    cx = int(np.clip(np.mean(t_vox[:, 0]), 0, nx - 1)) if len(t_vox) > 0 else nx // 2
    cy = int(np.clip(np.mean(t_vox[:, 1]), 0, ny - 1)) if len(t_vox) > 0 else ny // 2
    cz = int(np.clip(np.mean(t_vox[:, 2]), 0, nz - 1)) if len(t_vox) > 0 else nz // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    status_str = "PASS" if qc_pass else "FAIL"
    fig.suptitle(
        f"{title}\nMean TRE: {mean_tre:.2f} mm | Max TRE: {max_tre:.2f} mm | Status: {status_str}",
        fontsize=13,
        fontweight="bold",
    )

    # 1. Sagittal (Y-Z plane, slice at x=cx)
    axes[0].imshow(np.rot90(bg_data[cx, :, :]), cmap="gray", origin="upper")
    axes[0].scatter(t_vox[:, 1], nz - 1 - t_vox[:, 2], c="lime", marker="o", s=40, label="Target LM")
    axes[0].scatter(w_vox[:, 1], nz - 1 - w_vox[:, 2], c="red", marker="x", s=40, label="Warped LM")
    axes[0].set_title(f"Sagittal Proj (x={cx})")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].axis("off")

    # 2. Coronal (X-Z plane, slice at y=cy)
    axes[1].imshow(np.rot90(bg_data[:, cy, :]), cmap="gray", origin="upper")
    axes[1].scatter(t_vox[:, 0], nz - 1 - t_vox[:, 2], c="lime", marker="o", s=40)
    axes[1].scatter(w_vox[:, 0], nz - 1 - w_vox[:, 2], c="red", marker="x", s=40)
    axes[1].set_title(f"Coronal Proj (y={cy})")
    axes[1].axis("off")

    # 3. Axial (X-Y plane, slice at z=cz)
    axes[2].imshow(np.rot90(bg_data[:, :, cz]), cmap="gray", origin="upper")
    axes[2].scatter(t_vox[:, 0], ny - 1 - t_vox[:, 1], c="lime", marker="o", s=40)
    axes[2].scatter(w_vox[:, 0], ny - 1 - w_vox[:, 1], c="red", marker="x", s=40)
    axes[2].set_title(f"Axial Proj (z={cz})")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_png_path, metrics

