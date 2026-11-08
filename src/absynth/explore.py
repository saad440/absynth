"""Exploratory statistics for the curve matrix.

Two diagnostics used by the data audit:

- The PCA spectrum: how many directions actually carry the
  population's variability once neighbouring-wavelength correlation
  is accounted for.
- Nearest-neighbour distances: how close every curve is to its
  closest neighbour, a diversity estimate whose lower end doubles as a
  near-duplicate check.

Both require *finite* input: restrict to a complete subset first (for
this archive, the 300–900 nm span is measured for all films). How to
handle unmeasured points is a modelling decision and is not made
here.
"""

from __future__ import annotations

import numpy as np


def _require_finite(values: np.ndarray) -> None:
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D curve matrix, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(
            "curves contain unmeasured points; restrict to a complete subset first"
        )


def pca_explained_variance(
    values: np.ndarray, max_components: int | None = None
) -> np.ndarray:
    """Fraction of variance carried by each principal component.

    Args:
        values: Curve matrix, one curve per row, all finite, in nm⁻¹
            on a common grid.
        max_components: Truncate the spectrum to this many leading
            components; ``None`` returns the full spectrum.

    Returns:
        Explained-variance ratios in descending order, summing to 1
        before truncation.
    """
    _require_finite(values)
    centred = values - values.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    squared = singular_values**2
    ratios = squared / squared.sum()
    return ratios[:max_components]


def nearest_neighbour_distances(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each curve to its closest *other* curve.

    Euclidean distance across the wavelength grid, in the units of
    ``values`` (nm⁻¹ here). Self-matches are excluded, so two
    identical curves report a distance of 0 for both, the
    near-duplicate signal this diagnostic exists for.

    Args:
        values: Finite curve matrix, one curve per row.

    Returns:
        ``(distances, neighbours)``: each curve's nearest-neighbour
        distance, and the row position of that neighbour.
    """
    _require_finite(values)
    # Pairwise squared distances via the Gram matrix: a handful of
    # matrix products instead of an O(n²·d) difference tensor.
    squared_norms = np.einsum("ij,ij->i", values, values)
    pair_squared = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (values @ values.T)
    np.fill_diagonal(pair_squared, np.inf)
    neighbours = pair_squared.argmin(axis=1)
    distances = np.sqrt(np.maximum(pair_squared.min(axis=1), 0.0))
    return distances, neighbours


def complete_prefix_mask(values: np.ndarray) -> np.ndarray:
    """Leading grid columns measured for *every* curve, as a mask.

    Designed for trailing truncation — curves that stop early — where
    the leading run of all-measured columns is the widest span on
    which population statistics can run without dropping films. The
    mask ends at the first column with any unmeasured entry, so an
    interior hole truncates conservatively. If the first column is
    not fully measured, the mask is all-False.
    """
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D curve matrix, got shape {values.shape}")
    measured = np.isfinite(values).all(axis=0)
    prefix = measured.cumprod()  # 1 up to the first unmeasured column
    return prefix.astype(bool)
