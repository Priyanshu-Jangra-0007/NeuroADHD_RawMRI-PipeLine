"""Nuisance confound regression and bandpass temporal filtering calibrated to historical C-PAC 0.3.9.1.

Implements:
- 24 Friston motion regressors (R, R^2, R_{t-1}, R_{t-1}^2)
- 5 anatomical CompCor (aCompCor) components derived strictly from eroded White Matter and CSF masks
- Linear and quadratic polynomial detrending
- Simultaneous nuisance regression and 0.01 - 0.10 Hz bandpass filtering at the subject's actual TR
- Global Signal Regression: STRICTLY OFF (global0)
- Mean tissue signal regression: STRICTLY OFF (wm0.gm0.csf0)
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import clean_img
from sklearn.decomposition import PCA


def extract_acompcor_components(
    bold_data: Union[nib.Nifti1Image, np.ndarray],
    eroded_wm_mask: Union[nib.Nifti1Image, np.ndarray],
    eroded_csf_mask: Union[nib.Nifti1Image, np.ndarray],
    n_components: int = 5,
    random_state: int = 42,
    bold_affine: Optional[np.ndarray] = None,
    wm_affine: Optional[np.ndarray] = None,
    csf_affine: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract anatomical CompCor (aCompCor) components from eroded WM and CSF tissue masks.

    Follows historical C-PAC 0.3.9.1 methodology:
    - Verifies exact NIfTI spatial dimensions and affine grid alignment of WM/CSF masks against BOLD.
    - Combines eroded anatomical White Matter and CSF masks into a physiological noise region.
    - Standardizes voxel time series within the physiological mask.
    - Uses Principal Component Analysis (PCA) to extract the top 5 eigenvectors.

    Parameters
    ----------
    bold_data : Union[nib.Nifti1Image, np.ndarray]
        4D BOLD data of shape (X, Y, Z, T) or 4D NIfTI image in standard CC200 space.
    eroded_wm_mask : Union[nib.Nifti1Image, np.ndarray]
        3D binary mask of eroded White Matter in standard CC200 space.
    eroded_csf_mask : Union[nib.Nifti1Image, np.ndarray]
        3D binary mask of eroded CSF in standard CC200 space.
    n_components : int
        Number of principal components to extract (default: 5).
    random_state : int
        Random seed for PCA repeatability (default: 42).
    bold_affine, wm_affine, csf_affine : Optional[np.ndarray]
        Explicit 4x4 spatial affines to verify grid alignment when passing raw arrays.

    Returns
    -------
    np.ndarray
        Array of shape (T, n_components) representing the 5 aCompCor physiological regressors.
    """
    # Extract affines from NIfTI objects if provided
    if isinstance(bold_data, nib.Nifti1Image):
        if bold_affine is None:
            bold_affine = bold_data.affine
        bold_data = np.asanyarray(bold_data.dataobj, dtype=np.float32)

    if isinstance(eroded_wm_mask, nib.Nifti1Image):
        if wm_affine is None:
            wm_affine = eroded_wm_mask.affine
        eroded_wm_mask = np.asanyarray(eroded_wm_mask.dataobj)

    if isinstance(eroded_csf_mask, nib.Nifti1Image):
        if csf_affine is None:
            csf_affine = eroded_csf_mask.affine
        eroded_csf_mask = np.asanyarray(eroded_csf_mask.dataobj)

    # Priority 3: Strictly verify NIfTI grid affine alignment against BOLD image
    if bold_affine is not None and wm_affine is not None:
        if not np.allclose(bold_affine, wm_affine, atol=1e-4):
            raise ValueError(
                "WM mask affine does not match BOLD image affine. "
                "Ensure WM mask has been mapped to standard CC200 space using the T1w-to-MNI transform."
            )
    if bold_affine is not None and csf_affine is not None:
        if not np.allclose(bold_affine, csf_affine, atol=1e-4):
            raise ValueError(
                "CSF mask affine does not match BOLD image affine. "
                "Ensure CSF mask has been mapped to standard CC200 space using the T1w-to-MNI transform."
            )

    if bold_data.ndim != 4:
        raise ValueError(f"Expected 4D functional data, got {bold_data.ndim}D array")

    t_vol = bold_data.shape[3]
    if t_vol < 50:
        raise ValueError(f"Insufficient BOLD timepoints: {t_vol} (minimum 50 required)")

    # Validate mask dimensionality
    if eroded_wm_mask.ndim != 3:
        raise ValueError(
            f"Expected 3D WM mask, got {eroded_wm_mask.ndim}D array. "
            "Masks must be registered to the BOLD spatial grid before CompCor extraction."
        )
    if eroded_csf_mask.ndim != 3:
        raise ValueError(
            f"Expected 3D CSF mask, got {eroded_csf_mask.ndim}D array. "
            "Masks must be registered to the BOLD spatial grid before CompCor extraction."
        )

    # Validate mask-BOLD spatial alignment
    bold_spatial = bold_data.shape[:3]
    if eroded_wm_mask.shape != bold_spatial:
        raise ValueError(
            f"WM mask shape {eroded_wm_mask.shape} does not match "
            f"BOLD spatial dimensions {bold_spatial}. "
            "Ensure tissue masks are registered into the same MNI/CC200 grid as the BOLD image."
        )
    if eroded_csf_mask.shape != bold_spatial:
        raise ValueError(
            f"CSF mask shape {eroded_csf_mask.shape} does not match "
            f"BOLD spatial dimensions {bold_spatial}. "
            "Ensure tissue masks are registered into the same MNI/CC200 grid as the BOLD image."
        )

    # Reject non-finite BOLD values before CompCor extraction
    if not np.isfinite(bold_data).all():
        n_bad = int(np.count_nonzero(~np.isfinite(bold_data)))
        raise ValueError(
            f"BOLD data contains {n_bad} non-finite values (NaN or Inf) prior to aCompCor extraction. "
            "Ensure spatial normalization and motion correction completed without errors."
        )

    # Union of eroded physiological noise masks
    physio_mask = (eroded_wm_mask > 0) | (eroded_csf_mask > 0)
    n_physio_voxels = int(np.count_nonzero(physio_mask))

    if n_physio_voxels < n_components:
        raise ValueError(
            f"Insufficient physiological voxels ({n_physio_voxels}) to extract "
            f"{n_components} aCompCor components. Check anatomical tissue masks."
        )

    # Extract time series of physiological voxels: shape (T, N_voxels)
    flat_data = bold_data[physio_mask].T

    # Standardize time courses per voxel (zero mean, unit variance)
    means = np.mean(flat_data, axis=0)
    stds = np.std(flat_data, axis=0, ddof=0)
    valid_voxels = stds > 1e-6

    if np.count_nonzero(valid_voxels) < n_components:
        raise ValueError(
            f"Fewer than {n_components} physiological voxels have non-zero variance. "
            "Cannot compute aCompCor."
        )

    norm_data = (flat_data[:, valid_voxels] - means[valid_voxels]) / stds[valid_voxels]

    # Run PCA to extract top n_components eigenvectors
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    compcor_components = pca.fit_transform(norm_data)

    return compcor_components.astype(np.float64)


def build_confound_matrix(
    friston_24: np.ndarray,
    acompcor_5: np.ndarray,
) -> pd.DataFrame:
    """Construct the complete 31-column nuisance confound matrix.

    Columns include:
    - 24 Friston motion regressors: motion_friston_01 .. motion_friston_24
    - 5 aCompCor physiological regressors: aCompCor_01 .. aCompCor_05
    - 2 polynomial detrending terms: poly_linear, poly_quadratic

    Enforces historical C-PAC parity:
    - Global Signal Regression: STRICTLY OFF (global0)
    - Mean tissue signal regression: STRICTLY OFF (wm0.gm0.csf0)

    Parameters
    ----------
    friston_24 : np.ndarray
        Array of shape (T, 24) containing Friston-24 motion regressors.
    acompcor_5 : np.ndarray
        Array of shape (T, 5) containing 5 aCompCor physiological regressors.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (T, 31) with strictly formatted column names.
    """
    if friston_24.ndim != 2 or friston_24.shape[1] != 24:
        raise ValueError(f"Expected friston_24 to have shape (T, 24), got {friston_24.shape}")
    if acompcor_5.ndim != 2 or acompcor_5.shape[1] != 5:
        raise ValueError(f"Expected acompcor_5 to have shape (T, 5), got {acompcor_5.shape}")

    t_vol = friston_24.shape[0]
    if acompcor_5.shape[0] != t_vol:
        raise ValueError(
            f"Temporal length mismatch: friston_24 has {t_vol} rows, "
            f"acompcor_5 has {acompcor_5.shape[0]} rows"
        )

    motion_cols = [f"motion_friston_{i:02d}" for i in range(1, 25)]
    compcor_cols = [f"aCompCor_{i:02d}" for i in range(1, 6)]

    df_motion = pd.DataFrame(friston_24, columns=motion_cols)
    df_compcor = pd.DataFrame(acompcor_5, columns=compcor_cols)

    # Linear (t) and Quadratic (t^2) polynomial trends (linear1.quadratic1)
    # Centered on zero to reduce collinearity
    t_steps = np.linspace(-1.0, 1.0, t_vol)
    t_sq = t_steps ** 2

    df_trends = pd.DataFrame(
        {
            "poly_linear": t_steps,
            "poly_quadratic": t_sq,
        }
    )

    confounds_df = pd.concat([df_motion, df_compcor, df_trends], axis=1)

    # Validate finite values
    if confounds_df.isna().any().any():
        raise ValueError("Confound matrix contains NaN values")
    if not np.isfinite(confounds_df.to_numpy()).all():
        raise ValueError("Confound matrix contains non-finite float values")

    # Verify column count is exactly 31
    if confounds_df.shape[1] != 31:
        raise ValueError(f"Expected 31 confound columns, got {confounds_df.shape[1]}")

    # Verify full column rank and report collinearity details if rank-deficient
    confound_arr = confounds_df.to_numpy()
    rank = np.linalg.matrix_rank(confound_arr)
    if rank < confound_arr.shape[1]:
        _, s, _ = np.linalg.svd(confound_arr, full_matrices=False)
        condition_number = float(s[0] / s[-1]) if s[-1] > 1e-12 else float("inf")
        raise ValueError(
            f"Confound matrix is rank-deficient: rank {rank} < {confound_arr.shape[1]} columns. "
            f"Condition number: {condition_number:.1f}. "
            "Check motion parameters for perfect collinearity (e.g., zero-motion volumes, "
            "sinusoidal trajectories producing linearly dependent Friston terms)."
        )

    return confounds_df


def clean_functional_bold(
    bold_img: nib.Nifti1Image,
    confounds_df: pd.DataFrame,
    tr: float,
    low_cutoff_hz: float = 0.01,
    high_cutoff_hz: float = 0.10,
) -> nib.Nifti1Image:
    """Simultaneously perform nuisance regression and 0.01-0.10 Hz bandpass filtering.

    Parameters
    ----------
    bold_img : nib.Nifti1Image
        4D BOLD image in standard space.
    confounds_df : pd.DataFrame
        Confound regressors (Friston 24 + CompCor 5 + polynomial trends).
    tr : float
        Repetition time in seconds (from BIDS JSON sidecar).
    low_cutoff_hz : float
        High-pass frequency cutoff in Hz (default: 0.01).
    high_cutoff_hz : float
        Low-pass frequency cutoff in Hz (default: 0.10).

    Returns
    -------
    nib.Nifti1Image
        Cleaned, bandpass-filtered 4D residual BOLD volume.
    """
    if not (0.5 <= tr <= 5.0):
        raise ValueError(f"Physiologically invalid repetition time TR={tr}s (expected [0.5, 5.0])")

    # Validate bandpass frequencies against Nyquist limit and ordering
    nyquist_hz = 0.5 / tr
    if high_cutoff_hz >= nyquist_hz:
        raise ValueError(
            f"High cutoff frequency {high_cutoff_hz:.4f} Hz meets or exceeds Nyquist frequency "
            f"{nyquist_hz:.4f} Hz for TR={tr:.3f}s. "
            f"Maximum permissible low-pass cutoff for TR={tr:.3f}s is {nyquist_hz * 0.99:.4f} Hz."
        )
    if low_cutoff_hz <= 0.0:
        raise ValueError(f"Low cutoff frequency must be positive, got {low_cutoff_hz} Hz")
    if low_cutoff_hz >= high_cutoff_hz:
        raise ValueError(
            f"Low cutoff {low_cutoff_hz:.4f} Hz must be strictly less than "
            f"high cutoff {high_cutoff_hz:.4f} Hz"
        )

    # Validate BOLD image is finite before cleaning
    bold_arr = np.asanyarray(bold_img.dataobj)
    if not np.isfinite(bold_arr).all():
        n_bad = int(np.count_nonzero(~np.isfinite(bold_arr)))
        raise ValueError(
            f"BOLD image contains {n_bad} non-finite values (NaN or Inf) before cleaning. "
            "Ensure spatial normalization and motion correction produced valid output."
        )

    confound_matrix = confounds_df.to_numpy(dtype=np.float64)

    # Nilearn clean_img executes simultaneous orthogonal regression and filtering
    # detrend=False because polynomial trends are explicitly modeled in confound_matrix
    # standardize=None is strictly enforced so residual amplitudes retain physical scale (None == False in nilearn)
    cleaned_img = clean_img(
        bold_img,
        confounds=confound_matrix,
        detrend=False,
        standardize=None,
        low_pass=high_cutoff_hz,
        high_pass=low_cutoff_hz,
        t_r=tr,
        ensure_finite=True,
    )

    cleaned_data = np.asanyarray(cleaned_img.dataobj)
    if not np.isfinite(cleaned_data).all():
        raise ValueError("Cleaned functional image contains non-finite values (NaN or Inf)")

    return cleaned_img
