"""Craddock 200 (CC200) ROI time series extraction and validation."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker

# Canonical CC200 reference constants
CANONICAL_CC200_SHA256 = "9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650"
CANONICAL_CC200_SHAPE = (63, 75, 61)
CANONICAL_CC200_ZOOMS = (3.0, 3.0, 3.0)
CANONICAL_CC200_AFFINE = np.array(
    [
        [-3.0, 0.0, 0.0, 93.0],
        [0.0, 3.0, 0.0, -129.0],
        [0.0, 0.0, 3.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def verify_canonical_cc200_atlas(
    atlas_path: Union[str, Path],
) -> Dict[str, Any]:
    """Perform rigorous identity and geometric verification of the CC200 atlas file.

    Validates:
    - File existence.
    - Exact SHA-256 cryptographic hash (9467afce23ad7197fbbc8c96aadc8c0c2e36d4938445e3c1570a5deac489c650).
    - Exact 3D grid dimensions (63, 75, 61).
    - Voxel zooms (3.0, 3.0, 3.0) mm.
    - Reference affine matrix.
    - Exactly 200 non-zero contiguous integer labels 1..200.

    Parameters
    ----------
    atlas_path : Union[str, Path]
        Path to the CC200 NIfTI atlas file.

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary confirming atlas verification with details.

    Raises
    ------
    FileNotFoundError
        If atlas file is missing.
    ValueError
        If any checksum, geometry, affine, or label requirement is violated.
    """
    path = Path(atlas_path)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical CC200 atlas file not found at: {path}")

    # 1. Cryptographic SHA-256 Checksum Verification
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    actual_sha256 = hasher.hexdigest()

    if actual_sha256 != CANONICAL_CC200_SHA256:
        raise ValueError(
            f"CC200 atlas SHA-256 mismatch: expected {CANONICAL_CC200_SHA256}, got {actual_sha256}. "
            "Atlas file integrity verification failed; file is corrupt or unauthorized."
        )

    # 2. Geometric Verification
    atlas_img = nib.load(str(path))
    if atlas_img.shape[:3] != CANONICAL_CC200_SHAPE:
        raise ValueError(
            f"CC200 atlas shape mismatch: expected {CANONICAL_CC200_SHAPE}, got {atlas_img.shape[:3]}."
        )

    zooms = tuple(np.round(atlas_img.header.get_zooms()[:3], 1))
    if zooms != CANONICAL_CC200_ZOOMS:
        raise ValueError(
            f"CC200 atlas voxel zoom mismatch: expected {CANONICAL_CC200_ZOOMS}, got {zooms}."
        )

    if not np.allclose(atlas_img.affine, CANONICAL_CC200_AFFINE, atol=1e-3):
        raise ValueError(
            f"CC200 atlas affine mismatch. Expected:\n{CANONICAL_CC200_AFFINE}\nGot:\n{atlas_img.affine}"
        )

    # 3. Label Completeness Verification
    atlas_data = np.asanyarray(atlas_img.dataobj, dtype=np.int32)
    unique_labels = np.unique(atlas_data[atlas_data > 0])
    if len(unique_labels) != 200:
        raise ValueError(
            f"CC200 atlas must contain exactly 200 non-zero ROIs, found {len(unique_labels)}."
        )
    if not np.array_equal(unique_labels, np.arange(1, 201)):
        raise ValueError(
            "CC200 atlas labels are not contiguous integers from 1 to 200."
        )

    return {
        "verified": True,
        "sha256": actual_sha256,
        "shape": atlas_img.shape[:3],
        "zooms": zooms,
        "n_rois": 200,
        "atlas_path": str(path.resolve()),
        "affine": atlas_img.affine,
    }


def extract_cc200_timeseries(
    cleaned_bold_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract mean BOLD time series across 200 CC200 parcels and validate against historical schema.

    Parameters
    ----------
    cleaned_bold_img : nib.Nifti1Image
        Cleaned, bandpass-filtered 4D functional scan in MNI152 space.
    atlas_img : nib.Nifti1Image
        Authentic CC200 atlas image containing integer labels 1..200.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        - Formatted DataFrame matching roi_CC200.csv (200 rows x (T+1) columns).
        - NumPy array of shape (T, 200) containing time series.
    """
    # Validate spatial dimensions and affine alignment between cleaned BOLD and CC200 atlas
    bold_shape = cleaned_bold_img.shape[:3]
    atlas_shape = atlas_img.shape[:3]
    if bold_shape != atlas_shape:
        raise ValueError(
            f"Spatial dimension mismatch between cleaned BOLD {bold_shape} and CC200 atlas {atlas_shape}"
        )
    if not np.allclose(cleaned_bold_img.affine, atlas_img.affine, atol=1e-4):
        raise ValueError(
            "Affine mismatch between cleaned BOLD image and CC200 atlas. "
            "Cleaned BOLD data must be resampled to the exact CC200 reference grid before extraction."
        )

    # Validate CC200 atlas ROI count upfront to fail closed on empty/incomplete atlases
    atlas_data = np.asanyarray(atlas_img.dataobj, dtype=np.int32)
    unique_labels = np.unique(atlas_data[atlas_data > 0])
    if len(unique_labels) != 200:
        raise ValueError(f"Expected exactly 200 ROIs from CC200 atlas, found {len(unique_labels)}")

    # Use nilearn NiftiLabelsMasker for spatial mean extraction per label
    # Setting resampling_target=None ensures unexpected geometry mismatches raise an error
    # instead of silently triggering internal interpolation
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=None,
        detrend=False,
        resampling_target=None,
        verbose=0,
    )

    # signals shape: (T, N_ROIs)
    signals = masker.fit_transform(cleaned_bold_img)
    t_vol, n_rois = signals.shape

    if n_rois != 200:
        raise ValueError(f"Expected exactly 200 ROIs from CC200 atlas, extracted {n_rois}")
    if t_vol < 50:
        raise ValueError(f"Insufficient BOLD timepoints extracted: {t_vol} (minimum required: 50)")

    # Validate signal variance across all 200 ROIs
    stds = np.std(signals, axis=0, ddof=0)
    constant_rois = int(np.count_nonzero(stds == 0.0))
    if constant_rois > 0:
        raise ValueError(
            f"Extraction produced {constant_rois} constant (zero-variance) ROI signals. "
            "Pearson correlations are undefined; scan must be rejected."
        )

    # Format to match exact historical C-PAC roi_CC200.csv layout:
    # 200 rows, column 0 is 'node/volume' with integer labels 1..200
    # remaining columns are timepoints 0, 1, 2, ..., T-1
    roi_by_time = signals.T  # shape: (200, T)
    time_columns = [str(i) for i in range(t_vol)]

    df = pd.DataFrame(roi_by_time, columns=time_columns)
    df.insert(0, "node/volume", np.arange(1, 201))

    return df, signals
