"""Quality control visualization module for registration, tissue masks, and parcellation.

Enforces strict geometry validation: QC overlays must evaluate images that are already
spatially aligned and resampled into the same physical grid. No implicit interpolation
or resampling is permitted inside QC routines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

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
