"""Exploratory statistics for the curve matrix.

- The PCA spectrum: how many directions actually carry the
  population's variability once neighbouring-wavelength correlation
  is accounted for.
- Nearest-neighbour distances: how close every curve is to its
  closest neighbour, a diversity estimate whose lower end doubles as a
  near-duplicate check.
- Component–descriptor correlations: how the leading variability
  distributes over amplitude, edge and baseline descriptors.
- Thickness stability: how much of each descriptor's spread is the
  thickness factor rather than material variation, the Beer–Lambert
  check that compares raw absorbance with its normalised form.

All require *finite* input: restrict to a complete subset first (for
this archive, the 300–900 nm span is measured for all films). How to
handle unmeasured points is a modelling decision and is not made
here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def pca_scores(values: np.ndarray, max_components: int | None = None) -> np.ndarray:
    """Row coordinates of each curve on the leading principal axes.

    Companion to :func:`pca_explained_variance`: same centring and
    ordering, so ``scores[:, i]`` holds each curve's position on the
    axis whose variance share is ``pca_explained_variance(...)[i]``.
    The sign along each axis is arbitrary (SVD output) and only
    magnitudes and correlations are meaningful.
    """
    _require_finite(values)
    centred = values - values.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    scores = centred @ vt.T
    if max_components is not None:
        scores = scores[:, :max_components]
    return scores


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pairwise-complete Pearson correlation; NaN when undefined."""
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return np.nan
    xs, ys = x[mask], y[mask]
    if xs.std() == 0.0 or ys.std() == 0.0:
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def score_descriptor_correlations(
    scores: np.ndarray, descriptors: pd.DataFrame
) -> pd.DataFrame:
    """Correlate each leading component with each curve descriptor.

    This is the amplitude/edge/baseline attribution: a component that
    tracks the total integral but not the edge position is amplitude
    variation; correlations with edge or baseline descriptors locate
    shape variation instead. Rows are components (``component_1``…),
    columns follow ``descriptors``. Each pair uses only films where
    both values are measured, so grid-limited descriptors (edge
    positions on truncated curves) still participate where they can.
    """
    correlations = {}
    for column in descriptors:
        descriptor = descriptors[column].to_numpy(dtype=float)
        correlations[column] = [
            _pearson(scores[:, i], descriptor) for i in range(scores.shape[1])
        ]
    index = [f"component_{i + 1}" for i in range(scores.shape[1])]
    return pd.DataFrame(correlations, index=index)


def thickness_stability(
    descriptors: pd.DataFrame, thickness_nm: pd.Series | np.ndarray
) -> pd.DataFrame:
    """How much each descriptor moves with film thickness.

    One row per descriptor, two columns:

    - ``cv``: relative spread across films (sample standard deviation
      over mean, unmeasured entries ignored). Under exact Beer–Lambert
      proportionality ``A ∝ t``, raw-absorbance descriptors mix
      material variation with the thickness factor while
      ``q``-descriptors shed it — ``cv(q) < cv(A)`` is the
      normalisation doing its job.
    - ``pearson_r``: pairwise-complete correlation with thickness. The
      Beer–Lambert-exact values are 1 for ``A``-descriptors and 0 for
      ``q``-descriptors. A residual ``q``–``t`` association is partly
      arithmetic — at fixed ``A``, dividing by ``t`` builds in a
      downhill trend — so read its direction cautiously; its
      *magnitude* measures how far the population departs from exact
      proportionality.

    Thickness is consumed by position, like the conversions in
    :mod:`absynth.preprocessing`; ``Dataset.thickness_nm`` is already
    aligned to the curve rows by the loader.
    """
    thickness_nm = np.asarray(thickness_nm, dtype=float)
    if thickness_nm.shape != (len(descriptors),):
        raise ValueError(
            f"thickness has shape {thickness_nm.shape}, expected ({len(descriptors)},)"
        )
    rows = {}
    for column in descriptors:
        values = descriptors[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size >= 2 and finite.mean() != 0.0:
            cv = float(finite.std(ddof=1) / finite.mean())
        else:
            cv = np.nan
        rows[column] = {"cv": cv, "pearson_r": _pearson(values, thickness_nm)}
    return pd.DataFrame.from_dict(rows, orient="index")
