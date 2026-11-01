"""Fisher-z pairwise functional connectivity feature calculation (19,900 edges)."""
from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np

N_ROIS = 200
FISHER_EPSILON = 1e-7


def get_feature_names() -> List[str]:
    """Return the authoritative 19,900 lexicographically ordered edge feature names."""
    pairs = list(itertools.combinations(range(1, N_ROIS + 1), 2))
    return [f"ROI_{i:03d}__ROI_{j:03d}" for i, j in pairs]


def compute_connectivity_vector(time_by_roi: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute 19,900 Fisher-z transformed Pearson correlation edges from (T x 200) time series.

    Parameters
    ----------
    time_by_roi : np.ndarray
        Array of shape (T, 200) where columns represent regions 1..200.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        - 1D float64 array of shape (19,900,) containing Fisher-z connectivity values.
        - List of 19,900 corresponding edge names.
    """
    if time_by_roi.shape[1] != N_ROIS:
        raise ValueError(f"Expected {N_ROIS} ROIs, got {time_by_roi.shape[1]}")

    # Check zero-variance
    stds = np.std(time_by_roi, axis=0, ddof=0)
    if np.any(stds == 0.0):
        raise ValueError("Constant ROI signal detected; Pearson correlation undefined")

    # Pearson correlation matrix (200 x 200)
    corr = np.corrcoef(time_by_roi, rowvar=False)
    if not np.isfinite(corr).all():
        raise ValueError("Non-finite values encountered in Pearson correlation matrix")

    # Enforce symmetry and diagonal = 1.0
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    # Extract upper triangle without diagonal (k=1)
    tri_i, tri_j = np.triu_indices(N_ROIS, k=1)
    edge_corr = corr[tri_i, tri_j]

    # Clip to avoid arctanh singularity at +/- 1.0
    clipped = np.clip(edge_corr, -1.0 + FISHER_EPSILON, 1.0 - FISHER_EPSILON)
    z = np.arctanh(clipped)

    feature_names = get_feature_names()
    return z.astype(np.float64), feature_names
