"""Spatial coregistration, MNI152 standard space normalization, and composed resampling.

Follows the approved NeuroADHD_RawMRI PoC V2 specification (docs/CORRECTED_POC_V2_DESIGN.md).
Uses DIPY (dipy.align) for native ARM64 Apple Silicon execution within a 2.5 GB RAM ceiling.
"""
from __future__ import annotations

import gc
import resource
from typing import List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
from dipy.align.imaffine import (
    AffineMap,
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.imwarp import DiffeomorphicMap, SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.align.transforms import AffineTransform3D, RigidTransform3D
from nilearn.masking import compute_brain_mask
from scipy.ndimage import map_coordinates


def check_process_memory_gb() -> float:
    """Return maximum resident set size (RSS) in gigabytes for the current process."""
    # On macOS Darwin, ru_maxrss is reported in bytes; on Linux in kilobytes
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import platform
    if platform.system() == "Darwin":
        return raw_rss / (1024.0 ** 3)
    return raw_rss / (1024.0 ** 2)


def generate_functional_reference(
    bold_4d: nib.Nifti1Image,
    mask_threshold: float = 0.5,
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Generate a high-quality 3D functional reference image from 4D BOLD series.

    Computes the temporal mean volume across all realigned timepoints and
    derives a binary intracranial functional mask.

    Parameters
    ----------
    bold_4d : nib.Nifti1Image
        4D functional BOLD image of shape (X, Y, Z, T).
    mask_threshold : float
        Fraction of upper mean intensity used for brain masking heuristic.

    Returns
    -------
    Tuple[nib.Nifti1Image, nib.Nifti1Image]
        - 3D temporal mean functional reference image.
        - 3D binary brain mask of the functional image.
    """
    data = np.asanyarray(bold_4d.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D functional scan, got {data.ndim}D array")

    mean_data = np.mean(data, axis=3)
    mean_img = nib.Nifti1Image(mean_data, bold_4d.affine, bold_4d.header)

    # Compute functional brain mask
    try:
        brain_mask_img = compute_brain_mask(mean_img, mask_type="epi")
    except Exception:
        # Fallback to robust thresholding if nilearn mask fails on synthetic/small data
        pos = mean_data[mean_data > 0]
        thresh = np.percentile(pos, 20) if len(pos) > 0 else 0.0
        mask_data = (mean_data > thresh).astype(np.int32)
        brain_mask_img = nib.Nifti1Image(mask_data, bold_4d.affine)

    return mean_img, brain_mask_img


def register_epi_to_t1w(
    mean_epi: nib.Nifti1Image,
    t1w_brain: nib.Nifti1Image,
    level_iters: Optional[List[int]] = None,
    nbins: int = 32,
    sampling_proportion: float = 0.3,
) -> Tuple[np.ndarray, AffineMap]:
    """Perform 3D rigid coregistration (6 DOF) of functional mean EPI to anatomical T1w volume.

    Parameters
    ----------
    mean_epi : nib.Nifti1Image
        Moving functional reference image.
    t1w_brain : nib.Nifti1Image
        Static skull-stripped anatomical T1w image.
    level_iters : Optional[List[int]]
        Multi-resolution pyramid iterations (default: [40, 20, 10]).
    nbins : int
        Number of histogram bins for mutual information metric.
    sampling_proportion : float
        Proportion of voxels sampled for metric calculation.

    Returns
    -------
    Tuple[np.ndarray, AffineMap]
        - 4x4 rigid transformation matrix mapping points in static T1w space to moving EPI space.
        - DIPY AffineMap object for applying the rigid transform.
    """
    if level_iters is None:
        level_iters = [40, 20, 10]

    static_data = np.asanyarray(t1w_brain.dataobj, dtype=np.float32)
    moving_data = np.asanyarray(mean_epi.dataobj, dtype=np.float32)

    if static_data.ndim != 3 or moving_data.ndim != 3:
        raise ValueError(
            f"Expected 3D volumes for registration, got shapes static={static_data.shape}, moving={moving_data.shape}"
        )
    if np.sum(np.abs(static_data)) == 0.0 or np.sum(np.abs(moving_data)) == 0.0:
        raise ValueError("Cannot perform registration with an empty or all-zero image")

    static_affine = t1w_brain.affine
    moving_affine = mean_epi.affine

    metric = MutualInformationMetric(nbins=nbins, sampling_proportion=sampling_proportion)
    affreg = AffineRegistration(
        metric=metric,
        level_iters=level_iters,
        sigmas=[3.0, 1.0, 0.0][: len(level_iters)],
        factors=[4, 2, 1][: len(level_iters)],
    )

    com = transform_centers_of_mass(
        static_data, static_affine, moving_data, moving_affine
    )
    rigid = affreg.optimize(
        static_data,
        moving_data,
        RigidTransform3D(),
        None,
        static_grid2world=static_affine,
        moving_grid2world=moving_affine,
        starting_affine=com.affine,
    )

    return rigid.affine, rigid


def register_t1w_to_mni(
    t1w_brain: nib.Nifti1Image,
    mni_template: nib.Nifti1Image,
    affine_level_iters: Optional[List[int]] = None,
    syn_level_iters: Optional[List[int]] = None,
    step_length: float = 0.25,
) -> Tuple[np.ndarray, DiffeomorphicMap]:
    """Normalize subject skull-stripped T1w anatomical image into MNI152 standard space.

    Executes a two-stage hierarchical registration:
    1. 12 DOF linear affine registration (translation, rotation, scaling, shearing).
    2. Symmetric Diffeomorphic Registration (SyN) with cross-correlation metric.

    Parameters
    ----------
    t1w_brain : nib.Nifti1Image
        Moving skull-stripped subject anatomical volume.
    mni_template : nib.Nifti1Image
        Static reference MNI152 brain template.
    affine_level_iters : Optional[List[int]]
        Multi-resolution iterations for affine stage (default: [40, 20, 10]).
    syn_level_iters : Optional[List[int]]
        Multi-resolution iterations for nonlinear SyN stage (default: [10, 5]).
    step_length : float
        Gradient step length for SyN optimization (default: 0.25).

    Returns
    -------
    Tuple[np.ndarray, DiffeomorphicMap]
        - 4x4 affine matrix mapping MNI space to T1w space.
        - DiffeomorphicMap containing nonlinear SyN deformation fields.
    """
    if affine_level_iters is None:
        affine_level_iters = [40, 20, 10]
    if syn_level_iters is None:
        syn_level_iters = [10, 5]

    static_data = np.asanyarray(mni_template.dataobj, dtype=np.float32)
    moving_data = np.asanyarray(t1w_brain.dataobj, dtype=np.float32)

    if static_data.ndim != 3 or moving_data.ndim != 3:
        raise ValueError(
            f"Expected 3D volumes for registration, got shapes static={static_data.shape}, moving={moving_data.shape}"
        )
    if np.sum(np.abs(static_data)) == 0.0 or np.sum(np.abs(moving_data)) == 0.0:
        raise ValueError("Cannot perform registration with an empty or all-zero image")

    static_affine = mni_template.affine
    moving_affine = t1w_brain.affine

    # Stage 1: Linear 12 DOF Affine
    metric_affine = MutualInformationMetric(nbins=32, sampling_proportion=0.25)
    affreg = AffineRegistration(
        metric=metric_affine,
        level_iters=affine_level_iters,
        sigmas=[3.0, 1.0, 0.0][: len(affine_level_iters)],
        factors=[4, 2, 1][: len(affine_level_iters)],
    )

    com = transform_centers_of_mass(
        static_data, static_affine, moving_data, moving_affine
    )
    affine_result = affreg.optimize(
        static_data,
        moving_data,
        AffineTransform3D(),
        None,
        static_grid2world=static_affine,
        moving_grid2world=moving_affine,
        starting_affine=com.affine,
    )
    m_t1w_to_mni = affine_result.affine

    # Stage 2: Symmetric Diffeomorphic (SyN)
    sdr_metric = CCMetric(3)
    sdr = SymmetricDiffeomorphicRegistration(
        metric=sdr_metric,
        level_iters=syn_level_iters,
        step_length=step_length,
    )
    syn_mapping = sdr.optimize(
        static_data,
        moving_data,
        static_grid2world=static_affine,
        moving_grid2world=moving_affine,
        prealign=m_t1w_to_mni,
    )

    return m_t1w_to_mni, syn_mapping


def compose_transformations(
    rigid_epi_to_t1w: np.ndarray,
    affine_t1w_to_mni: np.ndarray,
    syn_mapping: DiffeomorphicMap,
    target_shape: Optional[Tuple[int, int, int]] = None,
    target_affine: Optional[np.ndarray] = None,
    source_shape: Optional[Tuple[int, int, int]] = None,
    source_affine: Optional[np.ndarray] = None,
    *,
    domain_shape: Optional[Tuple[int, int, int]] = None,
    domain_affine: Optional[np.ndarray] = None,
    codomain_shape: Optional[Tuple[int, int, int]] = None,
    codomain_affine: Optional[np.ndarray] = None,
) -> DiffeomorphicMap:
    """Compose rigid EPI-to-T1w, affine T1w-to-MNI, and nonlinear SyN into a single pullback mapping.

    DIPY Transformation Direction Mathematics:
    -------------------------------------------
    In DIPY's ``AffineRegistration.optimize(static, moving)``:
    - The returned affine matrix operates in physical world coordinates in the PULLBACK direction:
      x_moving = M_static_to_moving * x_static
    - Therefore:
      - ``register_epi_to_t1w(mean_epi, t1w_brain)`` (static=T1w, moving=EPI) returns:
        M_t1w_to_epi = rigid_epi_to_t1w, where x_EPI = M_t1w_to_epi * x_T1w.
        Its inverse is the forward coordinate transform: F_epi_to_t1w = inv(M_t1w_to_epi).
      - ``register_t1w_to_mni(t1w_brain, mni_template)`` (static=MNI, moving=T1w) returns:
        M_mni_to_t1w = affine_t1w_to_mni, where x_T1w = M_mni_to_t1w * x_MNI.
        Its inverse is the forward coordinate transform: F_t1w_to_mni = inv(M_mni_to_t1w).

    Pullback Sampling Composition for DiffeomorphicMap:
    ---------------------------------------------------
    With DIPY ``is_inverse = True``, ``DiffeomorphicMap.transform(epi_image)`` calls
    ``_warp_backward(epi_image)``, sampling voxel intensities from the native EPI (domain)
    image onto the MNI/CC200 (codomain) output grid. The ``prealign_inv`` matrix must map
    a physical point in target MNI space (x_MNI) into native EPI space (x_EPI):
        x_EPI = P * x_MNI
    Substituting the chain:
        x_T1w = M_mni_to_t1w * x_MNI
        x_EPI = M_t1w_to_epi * x_T1w = M_t1w_to_epi * (M_mni_to_t1w * x_MNI)
        => P = M_t1w_to_epi @ M_mni_to_t1w = rigid_epi_to_t1w @ affine_t1w_to_mni

    Forward Coordinate Transformation (for landmark verification):
    --------------------------------------------------------------
        F_epi_to_mni = F_t1w_to_mni @ F_epi_to_t1w
                     = inv(M_mni_to_t1w) @ inv(M_t1w_to_epi)
                     = inv(M_t1w_to_epi @ M_mni_to_t1w)
                     = inv(P)

    Parameters
    ----------
    rigid_epi_to_t1w : np.ndarray
        4x4 DIPY AffineMap matrix from ``register_epi_to_t1w()``.
        Physically represents M_t1w_to_epi (static T1w -> moving EPI).
    affine_t1w_to_mni : np.ndarray
        4x4 DIPY AffineMap matrix from ``register_t1w_to_mni()``.
        Physically represents M_mni_to_t1w (static MNI -> moving T1w).
    syn_mapping : DiffeomorphicMap
        SyN deformation mapping computed between T1w (moving) and MNI (static).
    target_shape : Tuple[int, int, int], optional
        Shape of the CC200/MNI output grid (e.g. (63, 75, 61)). Stored on DIPY ``codomain_shape``
        so ``is_inverse=True`` ``.transform(epi)`` outputs onto the MNI/CC200 grid.
    target_affine : np.ndarray, optional
        4x4 affine of CC200/MNI output grid. Stored on DIPY ``codomain_grid2world``.
    source_shape : Tuple[int, int, int], optional
        Shape of native functional EPI input grid. Stored on DIPY ``domain_shape``.
    source_affine : np.ndarray, optional
        4x4 affine of native functional EPI input grid. Stored on DIPY ``domain_grid2world``.
    domain_shape, domain_affine, codomain_shape, codomain_affine : optional
        Legacy keyword aliases supported for backward compatibility.

    Returns
    -------
    DiffeomorphicMap
        Composed diffeomorphic mapping. Calling ``.transform(epi_volume)`` samples native EPI
        voxels directly into the CC200/MNI target grid in a single interpolation step.
    """
    # Resolve target (output CC200 grid) and source (input EPI grid) parameters
    if target_shape is None:
        if codomain_shape is not None:
            target_shape = codomain_shape
        elif domain_shape is not None:
            target_shape = domain_shape
        else:
            raise ValueError("target_shape (output CC200 grid) must be provided")

    if target_affine is None:
        if codomain_affine is not None:
            target_affine = codomain_affine
        elif domain_affine is not None:
            target_affine = domain_affine
        else:
            raise ValueError("target_affine (output CC200 affine) must be provided")

    if source_shape is None:
        if domain_shape is not None:
            source_shape = domain_shape
        elif codomain_shape is not None:
            source_shape = codomain_shape
        else:
            raise ValueError("source_shape (native EPI input grid) must be provided")

    if source_affine is None:
        if domain_affine is not None:
            source_affine = domain_affine
        elif codomain_affine is not None:
            source_affine = codomain_affine
        else:
            raise ValueError("source_affine (native EPI input affine) must be provided")

    # Pullback sampling composition:
    # x_EPI = M_t1w_to_epi @ M_mni_to_t1w @ x_MNI
    # rigid_epi_to_t1w is M_t1w_to_epi; affine_t1w_to_mni is M_mni_to_t1w.
    prealign_composed = rigid_epi_to_t1w @ affine_t1w_to_mni

    # Preserve DIPY transformation state:
    # syn_mapping returned by dipy.align.imwarp.SymmetricDiffeomorphicRegistration.optimize
    # natively has is_inverse = True, where:
    # - domain_shape describes the moving (native EPI) input grid
    # - domain_grid2world describes the moving (native EPI) input affine
    # - codomain_shape describes the static reference (target CC200/MNI) output grid
    # - codomain_grid2world describes the static reference (target CC200/MNI) output affine
    # - disp_shape / disp_grid2world describes the static target discretization
    # - backward contains the deformation field in static space mapping static -> moving
    # - forward contains the deformation field mapping moving -> static
    #
    # Calling composed_map.transform(epi) executes _warp_backward(epi):
    #   warped[i] = image[W * Pinv * backward[Dinv * S * i] + W * Pinv * S * i]
    # where:
    #   S = codomain_grid2world = target_affine
    #   Dinv = disp_world2grid = target_affine^-1 => Dinv * S * i = i (exact static voxel lookup)
    #   Pinv = self.prealign_inv = prealign_composed
    #   W = domain_world2grid = source_affine^-1
    # Thus: x_EPI = P_composed @ (x_target + backward[i]), which samples the exact deformed coordinate.
    # Setting prealign = inv(prealign_composed) guarantees self.prealign_inv = prealign_composed.
    # Calling composed_map.transform_inverse(target) executes _warp_forward(target),
    # warping standard space images back into native functional EPI space.

    base_disp_shape = syn_mapping.disp_shape if syn_mapping.disp_shape is not None else target_shape
    base_disp_g2w = syn_mapping.disp_grid2world if syn_mapping.disp_grid2world is not None else target_affine

    composed_map = DiffeomorphicMap(
        3,
        disp_shape=base_disp_shape,
        disp_grid2world=base_disp_g2w,
        domain_shape=source_shape,         # native EPI input grid
        domain_grid2world=source_affine,   # native EPI input affine
        codomain_shape=target_shape,       # CC200/MNI target grid
        codomain_grid2world=target_affine, # CC200/MNI target affine
        prealign=np.linalg.inv(prealign_composed), # guarantees prealign_inv = prealign_composed
    )
    composed_map.is_inverse = True
    composed_map.backward = syn_mapping.backward
    composed_map.forward = syn_mapping.forward

    return composed_map


def compose_volume_transform(
    composed_map: DiffeomorphicMap,
    motion_transform: Union[AffineMap, np.ndarray],
) -> DiffeomorphicMap:
    """Compose a per-volume rigid motion transform with an EPI-to-MNI DiffeomorphicMap.

    In DIPY's pullback direction:
        - ``composed_map.prealign_inv`` maps x_MNI_warped -> x_ref (M_t1w_to_epi @ M_mni_to_t1w)
        - ``motion_transform`` maps x_ref -> x_vol(t) (estimated from raw uncorrected BOLD)
    Combining them gives:
        vol_prealign_inv = M_ref_to_vol(t) @ composed_map.prealign_inv
    so ``.transform(raw_or_stc_vol_t)`` resamples directly from native acquisition volume t
    into standard MNI/CC200 coordinates in a single spatial interpolation step.
    """
    if not isinstance(composed_map, DiffeomorphicMap):
        raise TypeError(f"Expected DiffeomorphicMap for composed_map, got {type(composed_map).__name__}")

    motion_aff = np.asarray(
        motion_transform.affine if hasattr(motion_transform, "affine") else motion_transform,
        dtype=np.float64,
    )
    if motion_aff.shape != (4, 4) or not np.isfinite(motion_aff).all():
        raise ValueError(f"Expected finite 4x4 motion transform matrix, got shape {motion_aff.shape}")
    if abs(float(np.linalg.det(motion_aff[:3, :3]))) < 1e-10:
        raise ValueError("Motion transform matrix is singular or degenerate (det ~ 0).")

    base_prealign_inv = (
        np.asarray(composed_map.prealign_inv, dtype=np.float64)
        if getattr(composed_map, "prealign_inv", None) is not None
        else np.eye(4, dtype=np.float64)
    )
    vol_prealign_inv = motion_aff @ base_prealign_inv
    if abs(float(np.linalg.det(vol_prealign_inv[:3, :3]))) < 1e-10:
        raise ValueError("Composed volume transform produced a singular matrix.")

    vol_map = DiffeomorphicMap(
        3,
        disp_shape=composed_map.disp_shape,
        disp_grid2world=composed_map.disp_grid2world,
        domain_shape=composed_map.domain_shape,
        domain_grid2world=composed_map.domain_grid2world,
        codomain_shape=composed_map.codomain_shape,
        codomain_grid2world=composed_map.codomain_grid2world,
        prealign=np.linalg.inv(vol_prealign_inv),
    )
    vol_map.is_inverse = composed_map.is_inverse
    vol_map.backward = composed_map.backward
    vol_map.forward = composed_map.forward
    return vol_map


def _interpolation_order(interpolation: str) -> int:
    key = str(interpolation).strip().lower()
    if key == "linear":
        return 1
    if key == "nearest":
        return 0
    if key == "cubic":
        return 3
    raise ValueError(f"Unsupported interpolation mode '{interpolation}'; expected 'linear', 'nearest', or 'cubic'.")


def compose_sdc_transform(
    composed_map: DiffeomorphicMap,
    sdc_displacement: np.ndarray,
    epi_affine: Optional[np.ndarray] = None,
    motion_transform: Optional[Union[AffineMap, np.ndarray]] = None,
    target_shape: Optional[Tuple[int, int, int]] = None,
    target_affine: Optional[np.ndarray] = None,
) -> dict:
    """Compose target-to-MNI SyN, T1w-to-EPI coregistration, EPI SDC pullback, and optional rigid motion.

    Computes the single-step pullback sampling coordinates from the target MNI/CC200 grid
    into the raw/STC native EPI volume without intermediate spatial resampling:

    1. ``x_MNI = target_affine @ [i_MNI, 1]``
    2. ``x_EPI_undist = prealign_inv @ (x_MNI + syn_backward(disp_affine^-1 @ x_MNI))``
    3. ``v_undist = inv(epi_affine) @ x_EPI_undist``
    4. ``v_dist_ref = v_undist + sdc_displacement(v_undist)``
    5. ``v_vol = inv(epi_affine) @ M_ref_to_vol(t) @ epi_affine @ v_dist_ref`` (when ``motion_transform`` supplied)

    Returns a dictionary containing static target coordinates ``v_dist_ref`` and active
    per-volume sampling coordinates ``v_vol`` for ``apply_composed_sdc_volume``.
    """
    if not isinstance(composed_map, DiffeomorphicMap):
        raise TypeError(f"Expected DiffeomorphicMap for composed_map, got {type(composed_map).__name__}")

    disp = np.asarray(sdc_displacement, dtype=np.float64)
    if disp.ndim != 4 or disp.shape[3] != 3 or not np.isfinite(disp).all():
        raise ValueError(
            f"Expected finite 4D sdc_displacement of shape (X, Y, Z, 3), got shape {disp.shape}"
        )
    epi_shape = tuple(int(x) for x in disp.shape[:3])
    if composed_map.domain_shape is not None and tuple(int(x) for x in composed_map.domain_shape[:3]) != epi_shape:
        raise ValueError(
            f"SDC displacement shape {epi_shape} does not match composed_map EPI domain_shape "
            f"{tuple(int(x) for x in composed_map.domain_shape[:3])}."
        )

    eff_epi_affine = np.asarray(
        epi_affine if epi_affine is not None else composed_map.domain_grid2world,
        dtype=np.float64,
    )
    if eff_epi_affine.shape != (4, 4) or not np.isfinite(eff_epi_affine).all():
        raise ValueError("Expected finite 4x4 epi_affine for SDC composition.")
    inv_epi_affine = np.linalg.inv(eff_epi_affine)

    eff_target_shape = tuple(
        int(x)
        for x in (target_shape if target_shape is not None else composed_map.codomain_shape)[:3]
    )
    eff_target_affine = np.asarray(
        target_affine if target_affine is not None else composed_map.codomain_grid2world,
        dtype=np.float64,
    )
    if len(eff_target_shape) != 3 or any(d <= 0 for d in eff_target_shape):
        raise ValueError(f"Invalid target_shape for SDC composition: {eff_target_shape}")
    if eff_target_affine.shape != (4, 4) or not np.isfinite(eff_target_affine).all():
        raise ValueError("Expected finite 4x4 target_affine for SDC composition.")

    prealign_inv = (
        np.asarray(composed_map.prealign_inv, dtype=np.float64)
        if getattr(composed_map, "prealign_inv", None) is not None
        else np.eye(4, dtype=np.float64)
    )

    # Target MNI voxel coordinates (3, N)
    gi, gj, gk = np.indices(eff_target_shape, dtype=np.float64)
    n_vox = gi.size
    mni_vox_h = np.vstack([gi.ravel(), gj.ravel(), gk.ravel(), np.ones(n_vox, dtype=np.float64)])
    x_mni_h = eff_target_affine @ mni_vox_h  # (4, N) physical target coordinates

    # Evaluate DIPY backward deformation field (target physical offset)
    backward_phys = np.zeros((3, n_vox), dtype=np.float64)
    if getattr(composed_map, "backward", None) is not None:
        b_field = np.asarray(composed_map.backward, dtype=np.float64)
        disp_g2w = (
            np.asarray(composed_map.disp_grid2world, dtype=np.float64)
            if composed_map.disp_grid2world is not None
            else eff_target_affine
        )
        if b_field.shape[:3] == eff_target_shape and np.allclose(disp_g2w, eff_target_affine, atol=1e-6):
            for c in range(3):
                backward_phys[c] = b_field[..., c].ravel()
        else:
            disp_coords = (np.linalg.inv(disp_g2w) @ x_mni_h)[:3]
            for c in range(3):
                backward_phys[c] = map_coordinates(
                    b_field[..., c],
                    disp_coords,
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )

    x_target_def_h = x_mni_h.copy()
    x_target_def_h[:3] += backward_phys
    v_undist = ((inv_epi_affine @ prealign_inv) @ x_target_def_h)[:3]

    # Sample SDC pullback shift u_SDC(v_undist) in undistorted EPI voxel space
    v_dist_ref = v_undist.copy()
    for c in range(3):
        v_dist_ref[c] += map_coordinates(
            disp[..., c],
            v_undist,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

    if motion_transform is not None:
        m_aff = np.asarray(
            motion_transform.affine if hasattr(motion_transform, "affine") else motion_transform,
            dtype=np.float64,
        )
        if m_aff.shape != (4, 4) or not np.isfinite(m_aff).all():
            raise ValueError(f"Expected finite 4x4 motion transform matrix, got {m_aff.shape}.")
        m_vox = inv_epi_affine @ m_aff @ eff_epi_affine
        v_vol = m_vox[:3, :3] @ v_dist_ref + m_vox[:3, 3:4]
    else:
        v_vol = v_dist_ref

    oob_mask = (
        (v_vol[0] < 0.0) | (v_vol[0] > epi_shape[0] - 1.0)
        | (v_vol[1] < 0.0) | (v_vol[1] > epi_shape[1] - 1.0)
        | (v_vol[2] < 0.0) | (v_vol[2] > epi_shape[2] - 1.0)
    )

    return {
        "v_undist": v_undist,
        "v_dist_ref": v_dist_ref,
        "v_vol": v_vol,
        "target_shape": eff_target_shape,
        "target_affine": eff_target_affine,
        "epi_shape": epi_shape,
        "epi_affine": eff_epi_affine,
        "inv_epi_affine": inv_epi_affine,
        "oob_fraction": float(np.mean(oob_mask)) if oob_mask.size else 0.0,
    }


def apply_composed_sdc_volume(
    vol_3d: Union[np.ndarray, nib.Nifti1Image],
    composed_grid: dict,
    interpolation: str = "linear",
) -> np.ndarray:
    """Sample a 3D EPI volume in a single pullback step using coordinates from ``compose_sdc_transform``."""
    arr = np.asanyarray(vol_3d.dataobj if isinstance(vol_3d, nib.Nifti1Image) else vol_3d, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D EPI volume, got shape {arr.shape}")
    expected_epi = tuple(composed_grid["epi_shape"])
    if tuple(arr.shape) != expected_epi:
        raise ValueError(f"EPI volume shape {arr.shape} does not match composed_grid epi_shape {expected_epi}.")
    order = _interpolation_order(interpolation)
    target_shape = tuple(composed_grid["target_shape"])
    return (
        map_coordinates(
            arr,
            composed_grid["v_vol"],
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=(order > 1),
        )
        .reshape(target_shape)
        .astype(np.float32)
    )


def resample_4d_bold(
    bold_4d: nib.Nifti1Image,
    composed_map: DiffeomorphicMap,
    target_shape: Tuple[int, int, int],
    target_affine: np.ndarray,
    interpolation: str = "linear",
    max_memory_gb: float = 2.5,
    motion_transforms: Optional[Sequence[Union[AffineMap, np.ndarray]]] = None,
    sdc_displacement: Optional[np.ndarray] = None,
) -> nib.Nifti1Image:
    """Resample 4D functional series into the target CC200 grid using the composed mapping.

    Performs volume-by-volume resampling across time to preserve temporal fidelity,
    prevent intermediate interpolation blur, and guarantee execution below the 2.5 GB RAM ceiling.
    When ``motion_transforms`` is supplied, composes each volume's rigid motion matrix with
    ``composed_map`` (EPI-to-T1w + T1w-to-MNI affine + SyN) so STC-corrected unresampled BOLD
    is spatially resampled in a single interpolation step.

    Parameters
    ----------
    bold_4d : nib.Nifti1Image
        4D functional BOLD scan of shape (X, Y, Z, T).
    composed_map : DiffeomorphicMap
        Composed spatial transformation mapping native functional coordinates to CC200 target grid.
    target_shape : Tuple[int, int, int]
        3D target grid dimensions (e.g. (63, 75, 61) for CC200).
    target_affine : np.ndarray
        4x4 target affine matrix.
    interpolation : str
        Interpolation type: 'linear' for continuous functional data.
    max_memory_gb : float
        Maximum allowed resident memory in GB (default: 2.5 GB).
    motion_transforms : Sequence[AffineMap | np.ndarray], optional
        Optional length-T sequence of per-volume rigid motion transforms estimated prior to
        resampling; when provided, per-volume motion + coregistration + MNI normalization are
        applied in a single spatial interpolation step.
    sdc_displacement : np.ndarray, optional
        Optional 4D voxel-displacement field of shape (X, Y, Z, 3) mapping undistorted EPI voxels
        to distorted reference EPI voxels. When provided, SDC unwarping is composed with
        motion + coregistration + MNI SyN in the same single spatial interpolation step.

    Returns
    -------
    nib.Nifti1Image
        Resampled 4D BOLD series in standard space of shape (*target_shape, T).
    """
    bold_data = np.asanyarray(bold_4d.dataobj, dtype=np.float32)
    if bold_data.ndim != 4:
        raise ValueError(f"Expected 4D functional scan, got {bold_data.ndim}D array")

    t_vol = bold_data.shape[3]
    if t_vol < 50:
        raise ValueError(f"Insufficient BOLD timepoints: {t_vol} (minimum 50 required)")
    if motion_transforms is not None and len(motion_transforms) != t_vol:
        raise ValueError(
            f"motion_transforms length ({len(motion_transforms)}) must match BOLD timepoints ({t_vol})."
        )

    # Preallocate output 4D array in float32
    resampled_4d = np.zeros((*target_shape, t_vol), dtype=np.float32)

    if sdc_displacement is not None:
        base_grid = compose_sdc_transform(
            composed_map=composed_map,
            sdc_displacement=sdc_displacement,
            epi_affine=np.asarray(bold_4d.affine, dtype=np.float64),
            motion_transform=None,
            target_shape=target_shape,
            target_affine=target_affine,
        )
        inv_epi = base_grid["inv_epi_affine"]
        epi_aff = base_grid["epi_affine"]
        v_dist_ref = base_grid["v_dist_ref"]

        for t in range(t_vol):
            vol_3d = bold_data[..., t]
            if motion_transforms is not None:
                rt = motion_transforms[t]
                m_aff = np.asarray(rt.affine if hasattr(rt, "affine") else rt, dtype=np.float64)
                m_vox = inv_epi @ m_aff @ epi_aff
                v_vol = m_vox[:3, :3] @ v_dist_ref + m_vox[:3, 3:4]
            else:
                v_vol = v_dist_ref
            vol_grid = {
                "v_vol": v_vol,
                "target_shape": base_grid["target_shape"],
                "epi_shape": base_grid["epi_shape"],
            }
            resampled_4d[..., t] = apply_composed_sdc_volume(vol_3d, vol_grid, interpolation=interpolation)

            if (t + 1) % 25 == 0 or t == t_vol - 1:
                current_mem = check_process_memory_gb()
                if current_mem > max_memory_gb:
                    gc.collect()
                    current_mem = check_process_memory_gb()
                    if current_mem > max_memory_gb:
                        raise MemoryError(
                            f"Memory limit exceeded during 4D resampling at volume {t}/{t_vol}: "
                            f"{current_mem:.2f} GB > {max_memory_gb:.2f} GB"
                        )
    else:
        for t in range(t_vol):
            vol_3d = bold_data[..., t]
            active_map = (
                compose_volume_transform(composed_map, motion_transforms[t])
                if motion_transforms is not None
                else composed_map
            )
            warped_vol = active_map.transform(
                vol_3d,
                interpolation=interpolation,
                out_shape=target_shape,
                out_grid2world=target_affine,
            )
            resampled_4d[..., t] = warped_vol

            # Periodically monitor memory headroom
            if (t + 1) % 25 == 0 or t == t_vol - 1:
                current_mem = check_process_memory_gb()
                if current_mem > max_memory_gb:
                    gc.collect()
                    current_mem = check_process_memory_gb()
                    if current_mem > max_memory_gb:
                        raise MemoryError(
                            f"Memory limit exceeded during 4D resampling at volume {t}/{t_vol}: "
                            f"{current_mem:.2f} GB > {max_memory_gb:.2f} GB"
                        )

    out_header = bold_4d.header.copy()
    out_header.set_data_shape(resampled_4d.shape)
    out_img = nib.Nifti1Image(resampled_4d, target_affine, out_header)
    return out_img


def resample_mask_to_target(
    mask_3d: nib.Nifti1Image,
    transform_map: Union[DiffeomorphicMap, AffineMap],
    target_shape: Tuple[int, int, int],
    target_affine: np.ndarray,
) -> nib.Nifti1Image:
    """Resample 3D binary or discrete label mask using nearest-neighbor interpolation.

    Strictly validates the input mask's 3D dimensions and affine against the transformation's
    source domain before warping, rejecting shape, translation, scaling, or orientation mismatches.

    Note on Anatomical Tissue Masks vs Functional Masks:
    ---------------------------------------------------
    - For anatomical tissue masks (eroded WM and CSF masks, which originate in structural T1w space),
      pass the T1w-to-MNI normalization transform (``syn_mapping`` or the T1w affine map).
      Do NOT pass the composed EPI-to-MNI mapping, which includes an EPI-to-T1w coregistration
      that would corrupt anatomical mask spatial alignment.
    - For functional EPI masks, pass the composed pullback mapping.

    Parameters
    ----------
    mask_3d : nib.Nifti1Image
        3D mask image in moving space (e.g. anatomical T1w space or native EPI space).
    transform_map : Union[DiffeomorphicMap, AffineMap]
        Spatial transformation mapping source coordinates to target grid.
    target_shape : Tuple[int, int, int]
        3D target grid dimensions (e.g. CC200 shape (63, 75, 61)).
    target_affine : np.ndarray
        4x4 target affine matrix.

    Returns
    -------
    nib.Nifti1Image
        Resampled 3D mask image with discrete values preserved and exact target affine.
    """
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected 3D mask image, got {mask_3d.ndim}D image of shape {mask_3d.shape}")

    # Extract source domain shape and affine from transform_map
    if isinstance(transform_map, DiffeomorphicMap):
        dom_shape = transform_map.domain_shape
        dom_affine = transform_map.domain_grid2world
        codom_shape = transform_map.codomain_shape
    elif isinstance(transform_map, AffineMap):
        dom_shape = getattr(transform_map, "codomain_shape", None) or getattr(transform_map, "domain_shape", None)
        dom_affine = getattr(transform_map, "codomain_grid2world", None) or getattr(transform_map, "domain_grid2world", None)
        codom_shape = getattr(transform_map, "domain_shape", None)
    else:
        dom_shape = getattr(transform_map, "domain_shape", None) or getattr(transform_map, "codomain_shape", None)
        dom_affine = getattr(transform_map, "domain_grid2world", None) or getattr(transform_map, "codomain_grid2world", None)
        codom_shape = getattr(transform_map, "codomain_shape", None)

    if dom_shape is None or dom_affine is None:
        raise ValueError(
            "Transformation map lacks source domain geometry specification "
            "(domain_shape or domain_grid2world is None). Cannot validate input mask spatial alignment."
        )

    dom_shape = tuple(dom_shape[:3])
    mask_shape = tuple(mask_3d.shape[:3])

    # 1. Spatial Dimension Validation
    if mask_shape != dom_shape:
        raise ValueError(
            f"Spatial dimension mismatch: mask shape {mask_shape} does not match transformation source domain shape {dom_shape}. "
            "Anatomical T1w masks must use T1w-to-MNI transform (syn_mapping), not composed EPI-to-MNI transform."
        )

    # 2. Affine Translation / Origin Validation
    mask_origin = np.asarray(mask_3d.affine[:3, 3], dtype=np.float64)
    dom_origin = np.asarray(dom_affine[:3, 3], dtype=np.float64)
    if not np.allclose(mask_origin, dom_origin, atol=1e-3):
        raise ValueError(
            f"Translation offset mismatch: mask origin {mask_origin.tolist()} differs from transformation source origin {dom_origin.tolist()}."
        )

    # 3. Voxel-Size Scaling Validation
    mask_vox = np.linalg.norm(mask_3d.affine[:3, :3], axis=0)
    dom_vox = np.linalg.norm(dom_affine[:3, :3], axis=0)
    if not np.allclose(mask_vox, dom_vox, atol=1e-3):
        raise ValueError(
            f"Voxel-size scaling mismatch: mask voxel size {mask_vox.tolist()} differs from transformation source voxel size {dom_vox.tolist()}."
        )

    # 4. Oblique Orientation / Rotation Validation
    mask_rot = mask_3d.affine[:3, :3] / mask_vox
    dom_rot = dom_affine[:3, :3] / dom_vox
    if not np.allclose(mask_rot, dom_rot, atol=1e-3):
        raise ValueError(
            "Oblique orientation / rotation mismatch between mask affine and transformation source affine."
        )

    # 5. Full Affine Matrix Validation
    if not np.allclose(mask_3d.affine, dom_affine, atol=1e-3):
        raise ValueError(
            "Affine matrix mismatch between mask and transformation source domain."
        )

    # 6. Target Grid Codomain Validation (if available)
    if codom_shape is not None and tuple(codom_shape[:3]) != tuple(target_shape[:3]):
        raise ValueError(
            f"Target shape mismatch: requested target shape {target_shape} does not match map codomain shape {tuple(codom_shape[:3])}."
        )

    mask_data = np.asanyarray(mask_3d.dataobj, dtype=np.float32)
    if isinstance(transform_map, DiffeomorphicMap):
        warped_mask = transform_map.transform(
            mask_data,
            interpolation="nearest",
            out_shape=target_shape,
            out_grid2world=target_affine,
        )
    elif isinstance(transform_map, AffineMap):
        warped_mask = transform_map.transform(
            mask_data,
            interpolation="nearest",
            sampling_grid_shape=target_shape,
            sampling_grid2world=target_affine,
        )
    else:
        warped_mask = transform_map.transform(mask_data, interpolation="nearest")

    # Cast to integer if original was integer
    if np.issubdtype(mask_3d.get_data_dtype(), np.integer):
        warped_mask = np.round(warped_mask).astype(mask_3d.get_data_dtype())

    # Guarantee exact target shape
    if warped_mask.shape != target_shape:
        raise ValueError(f"Resampled mask shape {warped_mask.shape} does not match target {target_shape}")

    out_header = mask_3d.header.copy()
    out_header.set_data_shape(warped_mask.shape)
    out_img = nib.Nifti1Image(warped_mask, target_affine, out_header)
    return out_img


def compute_spatial_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Calculate the spatial Sørensen-Dice coefficient between two binary masks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Parameters
    ----------
    mask_a : np.ndarray
        First binary mask.
    mask_b : np.ndarray
        Second binary mask.

    Returns
    -------
    float
        Dice coefficient in range [0.0, 1.0].
    """
    bool_a = mask_a > 0
    bool_b = mask_b > 0

    sum_a = np.count_nonzero(bool_a)
    sum_b = np.count_nonzero(bool_b)

    if sum_a == 0 and sum_b == 0:
        return 1.0
    if sum_a == 0 or sum_b == 0:
        return 0.0

    intersection = np.count_nonzero(bool_a & bool_b)
    return float(2.0 * intersection / (sum_a + sum_b))
