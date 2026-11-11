"""Functional BOLD 3D rigid motion correction and Friston-24 regressor calculation."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import nibabel as nib
import numpy as np
from dipy.align.imaffine import (
    AffineMap,
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.transforms import RigidTransform3D


def compute_framewise_displacement(motion_params: np.ndarray, radius_mm: float = 50.0) -> np.ndarray:
    """Calculate Framewise Displacement (FD) using the Power et al. (2012) formulation.

    Parameters
    ----------
    motion_params : np.ndarray
        Array of shape (T, 6) containing translations (mm) [0:3] and rotations (degrees) [3:6].
    radius_mm : float
        Assumed cerebral cortex radius for converting angular rotations (degrees) to mm (default: 50.0).

    Returns
    -------
    np.ndarray
        1D array of length T with FD values in mm (first volume set to 0.0).
    """
    if motion_params.ndim != 2 or motion_params.shape[1] != 6:
        raise ValueError(f"Expected 2D array of 6 motion parameters (T, 6), got shape {motion_params.shape}")

    t = motion_params[:, :3]
    # Convert degrees to radians, then to arc length in mm: arc = radius * radians
    r_deg = motion_params[:, 3:6]
    r_rad = np.deg2rad(r_deg)
    r_mm = radius_mm * r_rad

    pos = np.hstack([t, r_mm])
    diff = np.diff(pos, axis=0)
    fd = np.sum(np.abs(diff), axis=1)
    # Prepend 0.0 for initial volume
    return np.insert(fd, 0, 0.0)


def build_friston_24_regressors(motion_params: np.ndarray) -> np.ndarray:
    """Construct the 24-parameter Friston motion regressor expansion: R, R^2, R_{t-1}, R_{t-1}^2.

    Parameters
    ----------
    motion_params : np.ndarray
        Array of shape (T, 6) containing 6 rigid realignment parameters.

    Returns
    -------
    np.ndarray
        Array of shape (T, 24) representing the full Friston-24 nuisance regressors.
    """
    if motion_params.ndim != 2 or motion_params.shape[1] != 6:
        raise ValueError(f"Expected 2D array of (T, 6) motion parameters, got shape {motion_params.shape}")

    t_vol = motion_params.shape[0]
    r = motion_params
    r_sq = r ** 2

    # Temporal derivative with backward difference (R_{t-1})
    r_prev = np.zeros_like(r)
    r_prev[1:] = r[:-1]
    r_prev_sq = r_prev ** 2

    friston_24 = np.hstack([r, r_sq, r_prev, r_prev_sq])
    if friston_24.shape != (t_vol, 24):
        raise RuntimeError(f"Expected (T, 24) Friston matrix, got {friston_24.shape}")
    return friston_24


def estimate_motion_parameters(
    bold_img: nib.Nifti1Image,
) -> Tuple[List[AffineMap], np.ndarray, np.ndarray]:
    """Estimate per-volume 3D rigid-body motion transforms without spatial resampling.

    Estimates 6-DOF rigid transforms mapping the temporal mean reference to each raw
    BOLD volume so motion estimation uses original, un-interpolated BOLD intensities
    prior to slice-timing correction and subsequent composed spatial resampling.

    Parameters
    ----------
    bold_img : nib.Nifti1Image
        Raw, uncorrected 4D functional BOLD scan.

    Returns
    -------
    Tuple[List[AffineMap], np.ndarray, np.ndarray]
        - List of T DIPY ``AffineMap`` rigid transforms (M_ref_to_vol(t)).
        - Motion parameters array of shape (T, 6) [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z].
        - Framewise displacement 1D array of shape (T,).
    """
    data = np.asanyarray(bold_img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D functional scan, got {data.ndim}D")

    t_vol = data.shape[3]
    if t_vol < 2:
        raise ValueError(f"At least 2 functional volumes required for 3D motion correction, got {t_vol}")
    affine = np.asarray(bold_img.affine, dtype=np.float64)

    # Compute reference: mean volume across all uncorrected timepoints
    static_vol = np.mean(data, axis=3)

    # Set up DIPY rigid registration
    metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
    transform = RigidTransform3D()
    affreg = AffineRegistration(metric=metric, level_iters=[100, 20], sigmas=[2.0, 0.5], factors=[4, 2])

    rigid_transforms: List[AffineMap] = []
    motion_params = np.zeros((t_vol, 6), dtype=np.float64)

    for vol_idx in range(t_vol):
        moving_vol = data[..., vol_idx]
        c_of_mass = transform_centers_of_mass(static_vol, affine, moving_vol, affine)
        rigid_mat = affreg.optimize(
            static_vol,
            moving_vol,
            transform,
            None,
            static_grid2world=affine,
            moving_grid2world=affine,
            starting_affine=c_of_mass.affine,
        )
        rigid_transforms.append(rigid_mat)

        # Extract translation (x, y, z in mm)
        t_vec = rigid_mat.affine[:3, 3]
        # Decompose rotation matrix into Euler angles in degrees
        r_mat = rigid_mat.affine[:3, :3]
        sy = np.sqrt(r_mat[0, 0] ** 2 + r_mat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            rx = np.arctan2(r_mat[2, 1], r_mat[2, 2])
            ry = np.arctan2(-r_mat[2, 0], sy)
            rz = np.arctan2(r_mat[1, 0], r_mat[0, 0])
        else:
            rx = np.arctan2(-r_mat[1, 2], r_mat[1, 1])
            ry = np.arctan2(-r_mat[2, 0], sy)
            rz = 0.0

        motion_params[vol_idx, :3] = t_vec
        motion_params[vol_idx, 3:6] = np.rad2deg([rx, ry, rz])

    fd = compute_framewise_displacement(motion_params)
    return rigid_transforms, motion_params, fd


def apply_motion_transforms(
    bold_img: nib.Nifti1Image,
    rigid_transforms: Sequence[AffineMap],
    interpolation: str = "linear",
) -> nib.Nifti1Image:
    """Apply previously estimated 3D rigid motion transforms to a 4D BOLD image.

    Parameters
    ----------
    bold_img : nib.Nifti1Image
        4D functional BOLD image (typically slice-time corrected in original slice geometry).
    rigid_transforms : Sequence[AffineMap]
        Sequence of T per-volume DIPY ``AffineMap`` transforms returned by ``estimate_motion_parameters``.
    interpolation : str
        Spatial interpolation mode for DIPY ``AffineMap.transform`` (default: 'linear').

    Returns
    -------
    nib.Nifti1Image
        Motion-realigned 4D NIfTI image in native BOLD grid geometry.
    """
    data = np.asanyarray(bold_img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D functional scan, got {data.ndim}D")

    t_vol = data.shape[3]
    if len(rigid_transforms) != t_vol:
        raise ValueError(
            f"Expected {t_vol} rigid motion transforms matching BOLD timepoints, got {len(rigid_transforms)}."
        )

    realigned_data = np.zeros_like(data, dtype=np.float32)
    for vol_idx in range(t_vol):
        realigned_data[..., vol_idx] = rigid_transforms[vol_idx].transform(
            data[..., vol_idx], interpolation=interpolation
        )

    out_header = bold_img.header.copy()
    out_header.set_data_shape(realigned_data.shape)
    out_header.set_data_dtype(np.float32)
    out_header.set_slope_inter(1.0, 0.0)
    return nib.Nifti1Image(realigned_data, bold_img.affine, out_header)


def realign_4d_bold(bold_img: nib.Nifti1Image) -> Tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    """Perform 3D rigid-body motion correction on a 4D BOLD series using mean volume reference.

    Convenience wrapper combining ``estimate_motion_parameters`` and ``apply_motion_transforms``
    for backward compatibility. When slice-timing correction is active, prefer calling
    ``estimate_motion_parameters`` before STC and ``apply_motion_transforms`` (or composed
    MNI pullback resampling) afterward.

    Parameters
    ----------
    bold_img : nib.Nifti1Image
        Raw 4D BOLD functional scan.

    Returns
    -------
    Tuple[nib.Nifti1Image, np.ndarray, np.ndarray]
        - Realigned 4D NIfTI image.
        - Motion parameters array of shape (T, 6) [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z].
        - Framewise displacement 1D array of shape (T,).
    """
    rigid_transforms, motion_params, fd = estimate_motion_parameters(bold_img)
    realigned_img = apply_motion_transforms(bold_img, rigid_transforms)
    return realigned_img, motion_params, fd
