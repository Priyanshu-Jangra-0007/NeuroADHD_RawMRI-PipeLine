"""Spatial coregistration, MNI152 standard space normalization, and composed resampling.

Follows the approved NeuroADHD_RawMRI PoC V2 specification (docs/CORRECTED_POC_V2_DESIGN.md).
Uses DIPY (dipy.align) for native ARM64 Apple Silicon execution within a 2.5 GB RAM ceiling.
"""
from __future__ import annotations

import gc
import resource
from typing import List, Optional, Tuple, Union

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
    ``DiffeomorphicMap.transform(epi_image)`` samples voxel intensities into the domain (MNI/CC200)
    grid. The ``prealign`` matrix must map a physical point in domain space (x_MNI) into codomain
    space (x_EPI):
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
        Shape of the CC200/MNI output grid (e.g. (63, 75, 61)). Maps to DIPY ``domain_shape``.
    target_affine : np.ndarray, optional
        4x4 affine of CC200/MNI output grid. Maps to DIPY ``domain_grid2world``.
    source_shape : Tuple[int, int, int], optional
        Shape of native functional EPI input grid. Maps to DIPY ``codomain_shape``.
    source_affine : np.ndarray, optional
        4x4 affine of native functional EPI input grid. Maps to DIPY ``codomain_grid2world``.
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


def resample_4d_bold(
    bold_4d: nib.Nifti1Image,
    composed_map: DiffeomorphicMap,
    target_shape: Tuple[int, int, int],
    target_affine: np.ndarray,
    interpolation: str = "linear",
    max_memory_gb: float = 2.5,
) -> nib.Nifti1Image:
    """Resample 4D functional series into the target CC200 grid using the composed mapping.

    Performs volume-by-volume resampling across time to preserve temporal fidelity,
    prevent intermediate interpolation blur, and guarantee execution below the 2.5 GB RAM ceiling.

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

    # Preallocate output 4D array in float32
    resampled_4d = np.zeros((*target_shape, t_vol), dtype=np.float32)

    for t in range(t_vol):
        vol_3d = bold_data[..., t]
        warped_vol = composed_map.transform(
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
