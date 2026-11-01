"""Craddock 200 (CC200) ROI time series extraction and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker


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
    # Use nilearn NiftiLabelsMasker for spatial mean extraction per label
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=None,
        detrend=False,
        resampling_target="labels",
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
