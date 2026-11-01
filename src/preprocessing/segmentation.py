"""Anatomical tissue segmentation and physiological mask derivation.

Follows the approved NeuroADHD_RawMRI PoC V2 specification (docs/CORRECTED_POC_V2_DESIGN.md).
Implements:
- Skull stripping via DIPY median_otsu
- 3D Legendre polynomial bias-field correction
- Tissue classification into CSF, GM, and WM probability maps via multi-class GMM with spatial priors or HMRF
- Strict historical C-PAC 0.3.9.1 thresholding: P(WM) >= 0.96, P(CSF) >= 0.96
- 1-voxel 3D morphological erosion for anatomical CompCor physiological nuisance regression
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from dipy.segment.mask import median_otsu
from dipy.segment.tissue import TissueClassifierHMRF
from scipy.ndimage import binary_erosion, binary_fill_holes, gaussian_filter, generate_binary_structure
from sklearn.mixture import GaussianMixture


def extract_anatomical_brain(
    t1w_img: nib.Nifti1Image,
    median_radius: int = 4,
    numpass: int = 4,
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Perform anatomical brain extraction (skull stripping) on 3D T1w structural scan.

    Uses DIPY's median_otsu algorithm to separate brain parenchyma from skull and scalp.

    Parameters
    ----------
    t1w_img : nib.Nifti1Image
        Raw 3D T1-weighted structural image.
    median_radius : int
        Radius for the 3D median filter (default: 4).
    numpass : int
        Number of passes of the otsu thresholding algorithm (default: 4).

    Returns
    -------
    Tuple[nib.Nifti1Image, nib.Nifti1Image]
        - Skull-stripped 3D T1w brain image (background voxels set to 0.0).
        - 3D binary brain mask image.
    """
    t1w_data = np.asanyarray(t1w_img.dataobj, dtype=np.float32)
    if t1w_data.ndim != 3:
        raise ValueError(f"Expected 3D structural image, got {t1w_data.ndim}D array")

    b0_mask, mask = median_otsu(t1w_data, median_radius=median_radius, numpass=numpass)
    # Ensure internal cavities (e.g. ventricles) are retained in the intracranial brain mask
    mask = binary_fill_holes(mask)

    brain_data = (t1w_data * mask).astype(np.float32)
    brain_img = nib.Nifti1Image(brain_data, t1w_img.affine, t1w_img.header)
    mask_img = nib.Nifti1Image(mask.astype(np.int32), t1w_img.affine, t1w_img.header)

    return brain_img, mask_img


def correct_bias_field(
    t1w_brain_img: nib.Nifti1Image,
    brain_mask_img: nib.Nifti1Image,
    order: int = 2,
    parenchyma_percentile: float = 35.0,
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Perform smooth 3D polynomial bias-field correction on skull-stripped T1w brain.

    Models spatial intensity non-uniformity as a multiplicative low-frequency smooth field:
    I(x) = J(x) * B(x). Fits a 3D polynomial basis in log space on brain parenchyma voxels,
    avoiding confounding by dark CSF or partial volume edges.

    Parameters
    ----------
    t1w_brain_img : nib.Nifti1Image
        Skull-stripped 3D T1w brain image.
    brain_mask_img : nib.Nifti1Image
        3D binary brain mask image.
    order : int
        Maximum polynomial order (default: 2 for smooth low-frequency correction).
    parenchyma_percentile : float
        Percentile threshold within brain mask to isolate brain parenchyma from CSF (default: 35.0).

    Returns
    -------
    Tuple[nib.Nifti1Image, nib.Nifti1Image]
        - Bias-corrected 3D T1w brain image.
        - Estimated 3D multiplicative bias field image.
    """
    brain_data = np.asanyarray(t1w_brain_img.dataobj, dtype=np.float32)
    mask_data = np.asanyarray(brain_mask_img.dataobj, dtype=bool)

    if not np.any(mask_data):
        raise ValueError("Cannot perform bias correction on empty brain mask")

    brain_voxels = brain_data[mask_data]
    pos_voxels = brain_voxels[brain_voxels > 0]
    if len(pos_voxels) == 0:
        raise ValueError("Brain mask contains only zero-valued voxels")

    # Isolate parenchyma (excluding dark CSF and zero voxels) to estimate pure bias field
    thresh = float(np.percentile(pos_voxels, parenchyma_percentile)) if len(pos_voxels) > 10 else 0.0
    parenchyma_mask = mask_data & (brain_data >= thresh)

    coords = np.argwhere(parenchyma_mask)
    if len(coords) < 10:
        coords = np.argwhere(mask_data & (brain_data > 0))
        parenchyma_mask = mask_data & (brain_data > 0)

    intensities = brain_data[parenchyma_mask]
    log_intensities = np.log(np.maximum(intensities, 1e-4))

    # Normalize coordinates to [-1, 1] relative to the whole brain bounding box
    all_brain_coords = np.argwhere(mask_data)
    mins = all_brain_coords.min(axis=0)
    maxs = all_brain_coords.max(axis=0)
    ranges = np.maximum(maxs - mins, 1.0)
    c_norm = 2.0 * (coords - mins) / ranges - 1.0

    # Build polynomial design matrix
    terms = [np.ones(len(coords), dtype=np.float32)]
    for d in range(1, order + 1):
        for ax in range(3):
            terms.append((c_norm[:, ax] ** d).astype(np.float32))

    design_matrix = np.column_stack(terms)
    coeffs, _, _, _ = np.linalg.lstsq(design_matrix, log_intensities, rcond=None)

    # Evaluate smooth bias field across entire volume grid using separable 1D broadcasting
    # (eliminates O(N_voxels x 7) full-grid coordinate and design matrix allocation,
    # preventing ~1 GB peak RAM overhead on full-size anatomical volumes)
    nx, ny, nz = brain_data.shape
    x_idx = np.arange(nx, dtype=np.float32)
    y_idx = np.arange(ny, dtype=np.float32)
    z_idx = np.arange(nz, dtype=np.float32)

    x_norm = (2.0 * (x_idx - mins[0]) / ranges[0] - 1.0)[:, None, None]
    y_norm = (2.0 * (y_idx - mins[1]) / ranges[1] - 1.0)[None, :, None]
    z_norm = (2.0 * (z_idx - mins[2]) / ranges[2] - 1.0)[None, None, :]

    log_bias = np.full((nx, ny, nz), coeffs[0], dtype=np.float32)
    term_idx = 1
    for d in range(1, order + 1):
        log_bias += coeffs[term_idx] * (x_norm ** d)
        term_idx += 1
        log_bias += coeffs[term_idx] * (y_norm ** d)
        term_idx += 1
        log_bias += coeffs[term_idx] * (z_norm ** d)
        term_idx += 1

    # Normalize log bias so geometric mean inside brain mask is 1.0 (mean log bias = 0)
    mean_log_bias = np.mean(log_bias[mask_data])
    log_bias_centered = log_bias - mean_log_bias
    bias_field = np.exp(log_bias_centered).astype(np.float32)

    # Apply correction within brain mask
    corrected_data = np.zeros_like(brain_data)
    corrected_data[mask_data] = brain_data[mask_data] / bias_field[mask_data]

    corrected_img = nib.Nifti1Image(corrected_data, t1w_brain_img.affine, t1w_brain_img.header)
    bias_img = nib.Nifti1Image(bias_field, t1w_brain_img.affine, t1w_brain_img.header)

    return corrected_img, bias_img


def segment_t1w_tissue(
    corrected_t1w_img: nib.Nifti1Image,
    brain_mask_img: nib.Nifti1Image,
    n_classes: int = 3,
    method: str = "gmm",
    beta: float = 0.1,
    max_iter: int = 30,
) -> Dict[str, nib.Nifti1Image]:
    """Segment brain-extracted, bias-corrected T1w structural image into CSF, GM, and WM.

    Supports:
    - 'gmm' (default): Multi-class Gaussian Mixture Model with covariance regularization and spatial smoothing.
      Robust, guaranteed positive definite covariance, no division-by-zero singularities.
    - 'hmrf': DIPY TissueClassifierHMRF (ICM with Markov random field).

    Identifies tissue classes based on intensity ordering:
    - Lowest mean intensity: CSF
    - Intermediate mean intensity: GM
    - Highest mean intensity: WM

    Parameters
    ----------
    corrected_t1w_img : nib.Nifti1Image
        Brain-extracted and bias-corrected 3D T1w image.
    brain_mask_img : nib.Nifti1Image
        3D binary brain mask.
    n_classes : int
        Number of tissue classes (default: 3).
    method : str
        Segmentation algorithm: 'gmm' or 'hmrf' (default: 'gmm').
    beta : float
        Markov random field spatial smoothness weight (default: 0.1).
    max_iter : int
        Maximum number of iterations (default: 30).

    Returns
    -------
    Dict[str, nib.Nifti1Image]
        Dictionary containing:
        - 'prob_csf': 3D probability map for Cerebrospinal Fluid [0.0, 1.0].
        - 'prob_gm': 3D probability map for Gray Matter [0.0, 1.0].
        - 'prob_wm': 3D probability map for White Matter [0.0, 1.0].
        - 'segmentation': 3D discrete segmentation map (0=bg, 1=CSF, 2=GM, 3=WM).
    """
    brain_data = np.asanyarray(corrected_t1w_img.dataobj, dtype=np.float32)
    mask_data = np.asanyarray(brain_mask_img.dataobj, dtype=bool)

    if not np.any(mask_data):
        raise ValueError("Cannot perform tissue segmentation on empty brain mask")

    shape = brain_data.shape
    prob_csf = np.zeros(shape, dtype=np.float32)
    prob_gm = np.zeros(shape, dtype=np.float32)
    prob_wm = np.zeros(shape, dtype=np.float32)

    if method.lower() == "gmm":
        voxels = brain_data[mask_data].reshape(-1, 1).astype(np.float64)
        gmm = GaussianMixture(
            n_components=n_classes,
            covariance_type="full",
            reg_covar=1e-3,
            random_state=42,
            max_iter=max_iter,
        )
        gmm.fit(voxels)
        posteriors = gmm.predict_proba(voxels)

        # Order components in ascending order of intensity mean: CSF < GM < WM
        means = gmm.means_.flatten()
        order = np.argsort(means)
        csf_k, gm_k, wm_k = order[0], order[1], order[2]

        prob_csf[mask_data] = posteriors[:, csf_k].astype(np.float32)
        prob_gm[mask_data] = posteriors[:, gm_k].astype(np.float32)
        prob_wm[mask_data] = posteriors[:, wm_k].astype(np.float32)

        # Spatial smoothing prior (beta weight)
        if beta > 0.0:
            sigma_vox = float(beta)
            p_csf_s = gaussian_filter(prob_csf, sigma=sigma_vox)
            p_gm_s = gaussian_filter(prob_gm, sigma=sigma_vox)
            p_wm_s = gaussian_filter(prob_wm, sigma=sigma_vox)

            # Re-normalize to sum to 1.0 inside mask
            total_s = p_csf_s + p_gm_s + p_wm_s
            total_s[total_s == 0] = 1.0
            prob_csf[mask_data] = (p_csf_s[mask_data] / total_s[mask_data]).astype(np.float32)
            prob_gm[mask_data] = (p_gm_s[mask_data] / total_s[mask_data]).astype(np.float32)
            prob_wm[mask_data] = (p_wm_s[mask_data] / total_s[mask_data]).astype(np.float32)

    elif method.lower() == "hmrf":
        clean_brain = np.where(mask_data, np.maximum(brain_data, 0.0), 0.0).astype(np.float32)
        brain_voxels = clean_brain[mask_data]
        if np.std(brain_voxels) < 1e-2:
            clean_brain[mask_data] += np.random.normal(0, 1e-2, len(brain_voxels)).astype(np.float32)

        hmrf = TissueClassifierHMRF(verbose=False)
        _, final_seg, pve = hmrf.classify(clean_brain, nclasses=n_classes, beta=beta, max_iter=max_iter)
        pve = np.nan_to_num(pve, nan=0.0, posinf=1.0, neginf=0.0)

        class_means = []
        for c in range(1, n_classes + 1):
            class_mask = (final_seg == c) & mask_data
            if np.any(class_mask):
                class_means.append((c - 1, float(np.mean(clean_brain[class_mask]))))
            else:
                class_means.append((c - 1, 0.0))

        class_means.sort(key=lambda item: item[1])
        csf_idx = class_means[0][0]
        gm_idx = class_means[1][0]
        wm_idx = class_means[2][0]

        prob_csf[mask_data] = np.clip(pve[..., csf_idx][mask_data], 0.0, 1.0)
        prob_gm[mask_data] = np.clip(pve[..., gm_idx][mask_data], 0.0, 1.0)
        prob_wm[mask_data] = np.clip(pve[..., wm_idx][mask_data], 0.0, 1.0)

    else:
        raise ValueError(f"Unknown segmentation method '{method}'; choose 'gmm' or 'hmrf'")

    # Discrete segmentation: 1=CSF, 2=GM, 3=WM, 0=Background
    discrete_seg = np.zeros(shape, dtype=np.int32)
    # Stack probabilities to find argmax inside brain mask
    stacked_probs = np.stack([prob_csf, prob_gm, prob_wm], axis=-1)
    best_tissue = np.argmax(stacked_probs, axis=-1) + 1  # 1=CSF, 2=GM, 3=WM
    discrete_seg[mask_data] = best_tissue[mask_data]

    affine = corrected_t1w_img.affine
    header = corrected_t1w_img.header

    return {
        "prob_csf": nib.Nifti1Image(prob_csf, affine, header),
        "prob_gm": nib.Nifti1Image(prob_gm, affine, header),
        "prob_wm": nib.Nifti1Image(prob_wm, affine, header),
        "segmentation": nib.Nifti1Image(discrete_seg, affine, header),
    }


def create_eroded_tissue_masks(
    prob_maps: Dict[str, nib.Nifti1Image],
    reference_t1w_img: Optional[nib.Nifti1Image] = None,
    wm_threshold: float = 0.96,
    csf_threshold: float = 0.96,
    erosion_voxels: int = 1,
) -> Dict[str, nib.Nifti1Image]:
    """Generate thresholded and eroded anatomical White Matter and CSF masks.

    Enforces historical C-PAC 0.3.9.1 parameters:
    - WM threshold: P(WM) >= 0.96
    - CSF threshold: P(CSF) >= 0.96
    - 1-voxel 3D binary erosion to eliminate partial volume averaging with gray matter.

    Validates tissue geometry before erosion:
    - Confirms required probability maps ('prob_wm', 'prob_csf') are present.
    - Confirms WM and CSF probability maps share identical spatial dimensions and grid affines.
    - Confirms probability maps match the reference T1w volume geometry if provided.

    Parameters
    ----------
    prob_maps : Dict[str, nib.Nifti1Image]
        Dictionary of probability maps from `segment_t1w_tissue`.
    reference_t1w_img : Optional[nib.Nifti1Image]
        Optional reference T1w anatomical image to verify spatial grid alignment.
    wm_threshold : float
        Probability threshold for White Matter (default: 0.96).
    csf_threshold : float
        Probability threshold for Cerebrospinal Fluid (default: 0.96).
    erosion_voxels : int
        Number of 3D morphological erosion iterations (default: 1).

    Returns
    -------
    Dict[str, nib.Nifti1Image]
        Dictionary containing raw and eroded binary masks for WM and CSF.
    """
    # Handle backward compatibility if threshold was passed as second positional argument
    if isinstance(reference_t1w_img, (float, int)):
        wm_threshold = float(reference_t1w_img)
        reference_t1w_img = None

    if "prob_wm" not in prob_maps or "prob_csf" not in prob_maps:
        raise ValueError("prob_maps dictionary must contain 'prob_wm' and 'prob_csf' probability maps")

    wm_prob_img = prob_maps["prob_wm"]
    csf_prob_img = prob_maps["prob_csf"]

    # Verify CSF and WM probability maps have matching shapes and affines
    if wm_prob_img.shape != csf_prob_img.shape:
        raise ValueError(
            f"Tissue probability map shape mismatch: prob_wm shape {wm_prob_img.shape} "
            f"!= prob_csf shape {csf_prob_img.shape}"
        )
    if not np.allclose(wm_prob_img.affine, csf_prob_img.affine, atol=1e-4):
        raise ValueError(
            "Tissue probability map affine mismatch: prob_wm affine does not match prob_csf affine"
        )

    # Verify against reference T1w image geometry if provided
    if reference_t1w_img is not None:
        ref_shape = reference_t1w_img.shape[:3]
        if wm_prob_img.shape[:3] != ref_shape:
            raise ValueError(
                f"Tissue probability map geometry mismatch: prob_wm shape {wm_prob_img.shape[:3]} "
                f"does not match T1w reference shape {ref_shape}"
            )
        if not np.allclose(wm_prob_img.affine, reference_t1w_img.affine, atol=1e-4):
            raise ValueError(
                "Tissue probability map affine mismatch: prob_wm affine does not match T1w reference affine"
            )

    prob_wm = np.asanyarray(wm_prob_img.dataobj, dtype=np.float32)
    prob_csf = np.asanyarray(csf_prob_img.dataobj, dtype=np.float32)

    raw_wm_mask = prob_wm >= wm_threshold
    raw_csf_mask = prob_csf >= csf_threshold

    structuring_element = generate_binary_structure(3, 1)  # 3D 6-connectivity

    if erosion_voxels > 0:
        eroded_wm_mask = binary_erosion(raw_wm_mask, structure=structuring_element, iterations=erosion_voxels)
        eroded_csf_mask = binary_erosion(raw_csf_mask, structure=structuring_element, iterations=erosion_voxels)
    else:
        eroded_wm_mask = raw_wm_mask.copy()
        eroded_csf_mask = raw_csf_mask.copy()

    affine = wm_prob_img.affine
    header = wm_prob_img.header

    return {
        "mask_wm_raw": nib.Nifti1Image(raw_wm_mask.astype(np.int32), affine, header),
        "mask_csf_raw": nib.Nifti1Image(raw_csf_mask.astype(np.int32), affine, header),
        "mask_wm_eroded": nib.Nifti1Image(eroded_wm_mask.astype(np.int32), affine, header),
        "mask_csf_eroded": nib.Nifti1Image(eroded_csf_mask.astype(np.int32), affine, header),
    }


def validate_tissue_masks(
    eroded_masks: Dict[str, nib.Nifti1Image],
    min_wm_voxels: int = 50,
    min_csf_voxels: int = 20,
) -> Dict[str, Any]:
    """Validate eroded White Matter and CSF anatomical masks against quality criteria.

    Parameters
    ----------
    eroded_masks : Dict[str, nib.Nifti1Image]
        Dictionary containing 'mask_wm_eroded' and 'mask_csf_eroded'.
    min_wm_voxels : int
        Minimum permitted eroded White Matter voxels (default: 50).
    min_csf_voxels : int
        Minimum permitted eroded CSF voxels (default: 20).

    Returns
    -------
    Dict[str, Any]
        Dictionary with validation metrics.

    Raises
    ------
    ValueError
        If masks are empty, have insufficient coverage, or overlap.
    """
    if "mask_wm_eroded" not in eroded_masks or "mask_csf_eroded" not in eroded_masks:
        raise ValueError("Missing required keys 'mask_wm_eroded' and 'mask_csf_eroded' in eroded_masks")

    wm_img = eroded_masks["mask_wm_eroded"]
    csf_img = eroded_masks["mask_csf_eroded"]

    if wm_img.shape != csf_img.shape:
        raise ValueError(f"Shape mismatch between WM mask {wm_img.shape} and CSF mask {csf_img.shape}")

    if not np.allclose(wm_img.affine, csf_img.affine, atol=1e-4):
        raise ValueError("Spatial affine mismatch between WM and CSF tissue masks")

    wm_data = np.asanyarray(wm_img.dataobj, dtype=bool)
    csf_data = np.asanyarray(csf_img.dataobj, dtype=bool)

    n_wm = int(np.count_nonzero(wm_data))
    n_csf = int(np.count_nonzero(csf_data))
    overlap = int(np.count_nonzero(wm_data & csf_data))

    if n_wm < min_wm_voxels:
        raise ValueError(
            f"Eroded White Matter mask has insufficient voxels ({n_wm} < {min_wm_voxels}). "
            "CompCor cannot extract reliable physiological regressors."
        )

    if n_csf < min_csf_voxels:
        raise ValueError(
            f"Eroded CSF mask has insufficient voxels ({n_csf} < {min_csf_voxels}). "
            "CompCor cannot extract reliable physiological regressors."
        )

    if overlap > 0:
        raise ValueError(f"Eroded WM and CSF masks overlap by {overlap} voxels; must be strictly disjoint")

    return {
        "status": "VALIDATED",
        "wm_voxel_count": n_wm,
        "csf_voxel_count": n_csf,
        "overlap_voxel_count": overlap,
    }
