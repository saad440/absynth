"""Evaluation metrics for generated populations of spectra.

A generator is judged on populations, not on single numbers. The
questions this module quantifies:

- **Distribution.** Does the generated population look like measured
  films? Mean curves and quantile bands per wavelength, and the
  deviation between a synthetic and a measured mean curve.
- **Variability.** Does it vary the way films vary? Relative
  differences of the covariance and correlation structure, so a model
  that matches average curves but not their co-variation is caught.
- **Coupling.** Do curves and thicknesses go together in a
  generated population as they do in measured films? The correlation
  between a per-film curve summary and thickness, read against the
  measured value.
- **Thickness resolution.** Where does a model's error live along
  the thickness range? Per-film errors averaged inside thickness
  slices, which the population mean cannot show.
- **Memorization.** Is it new rather than copied? Distances from
  generated curves to the closest *training* curve, read against the
  same distances for held-out real films. The semantics: a copying
  strategy scores 0 by construction (the bootstrap floor is exactly
  that), a perfect generator scores like held-out films, and the
  ratio of the two medians states where a model sits between those
  anchors. A ratio near 0 flags memorization; a ratio near 1 matches
  genuine novelty; a ratio well above 1 means the generator overshoots
  the population's own diversity scale.

Everything returns plain arrays and JSON-serializable dictionaries,
so report numbers and figures come from one code path. Distances are
Euclidean across the wavelength grid, in the units of the input
(nm⁻¹ for optical-density curves), matching
:func:`absynth.explore.nearest_neighbour_distances` so audit and
evaluation numbers stay comparable. Inputs must be finite: fit and
evaluate on the working span (see :mod:`absynth.preprocessing`).
"""

from __future__ import annotations

import numpy as np

DEFAULT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _check_curves(matrix: np.ndarray) -> np.ndarray:
    """Validate a curve matrix and return it as a float array."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("expected a non-empty (n_curves, n_wavelengths) matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(
            "curves must be finite; evaluate on the working span "
            "(see absynth.preprocessing)"
        )
    return matrix


def _check_distances(values: np.ndarray) -> np.ndarray:
    """Validate a 1-D distance sample and return it as a float array."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("expected a non-empty 1-D array of distances")
    if not np.isfinite(values).all():
        raise ValueError("distances must be finite")
    return values


def nearest_training_distances(samples: np.ndarray, training: np.ndarray) -> np.ndarray:
    """Distance from each sample curve to the closest training curve.

    Euclidean across the wavelength grid, in the units of the input.
    Unlike :func:`absynth.explore.nearest_neighbour_distances` there is
    no self-exclusion: a sample that *is* a training curve reports 0 —
    exactly the bootstrap floor's signature. The held-out profile for
    comparison comes from calling this function with held-out films as
    ``samples``; the training-set's own internal diversity is
    :func:`absynth.explore.nearest_neighbour_distances` on the training
    matrix.

    Args:
        samples: Query curve matrix, shape ``(n_samples, n_wavelengths)``.
        training: Reference curve matrix, shape
            ``(n_training, n_wavelengths)``, on the same grid.

    Returns:
        Distances, shape ``(n_samples,)``, in the units of the input.

    Raises:
        ValueError: If either matrix is empty or non-finite, or the
            grids have different lengths.
    """
    samples = _check_curves(samples)
    training = _check_curves(training)
    if samples.shape[1] != training.shape[1]:
        raise ValueError(
            f"sample and training grids differ: {samples.shape[1]} vs "
            f"{training.shape[1]} wavelengths"
        )
    squared_sample = np.einsum("ij,ij->i", samples, samples)
    squared_training = np.einsum("ij,ij->i", training, training)
    pair_squared = (
        squared_sample[:, None] + squared_training[None, :] - 2.0 * (samples @ training.T)
    )
    return np.sqrt(np.maximum(pair_squared.min(axis=1), 0.0))


def memorization_record(
    synthetic_distances: np.ndarray,
    heldout_distances: np.ndarray,
) -> dict[str, float]:
    """Summarize the novelty comparison as JSON-ready numbers.

    Args:
        synthetic_distances: Distances from generated curves to the
            closest training curve (:func:`nearest_training_distances`).
        heldout_distances: The same distances for held-out real films,
            the honest novelty scale of a genuinely new film.

    Returns:
        Median and 10–90 percentile band for each profile, plus
        ``median_ratio`` (synthetic over held-out): near 0 flags
        copying, near 1 matches genuine novelty, well above 1 means
        the generator overshoots the population's diversity.

    Raises:
        ValueError: If either sample is empty or non-finite.
    """
    synthetic = _check_distances(synthetic_distances)
    heldout = _check_distances(heldout_distances)
    synthetic_median = float(np.median(synthetic))
    heldout_median = float(np.median(heldout))
    return {
        "synthetic_median": synthetic_median,
        "synthetic_q10": float(np.quantile(synthetic, 0.1)),
        "synthetic_q90": float(np.quantile(synthetic, 0.9)),
        "heldout_median": heldout_median,
        "heldout_q10": float(np.quantile(heldout, 0.1)),
        "heldout_q90": float(np.quantile(heldout, 0.9)),
        "median_ratio": synthetic_median / heldout_median,
    }


def mean_curve(curves: np.ndarray) -> np.ndarray:
    """Per-wavelength mean of a population of curves."""
    return _check_curves(curves).mean(axis=0)


def quantile_band(
    curves: np.ndarray,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> np.ndarray:
    """Per-wavelength quantiles of a population of curves.

    Returns:
        Array of shape ``(len(quantiles), n_wavelengths)``, one band
        curve per requested quantile.
    """
    curves = _check_curves(curves)
    return np.quantile(curves, quantiles, axis=0)


def mean_curve_deviation(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    """Deviation between two populations' mean curves.

    Args:
        reference: Curve matrix for the reference population.
        other: Curve matrix for the comparison population, on the same
            grid.

    Returns:
        ``max_abs`` and ``mean_abs``: the largest and the average
        per-wavelength absolute difference of the mean curves, in the
        units of the input.

    Raises:
        ValueError: If either matrix is empty or non-finite, or the
            grids have different lengths.
    """
    reference = _check_curves(reference)
    other = _check_curves(other)
    if reference.shape[1] != other.shape[1]:
        raise ValueError(
            f"reference and other grids differ: {reference.shape[1]} vs "
            f"{other.shape[1]} wavelengths"
        )
    difference = np.abs(other.mean(axis=0) - reference.mean(axis=0))
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean())}


def covariance_record(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    """How far two populations' variability structures sit apart.

    Two scales, both relative Frobenius norms of the difference:

    - ``covariance_relative_frobenius``: the full second-order
      structure, spreads and correlations together. 0 means identical
      variability; 1 means the difference is as large as the
      reference's own covariance.
    - ``correlation_relative_frobenius``: the same comparison after
      removing each wavelength's total spread — pure correlation
      structure, insensitive to amplitude. A model can match this and
      still miss overall amplitude, which is why both are reported.

    Args:
        reference: Curve matrix for the reference population (at least
            2 curves).
        other: Curve matrix for the comparison population, same grid,
            at least 2 curves.

    Raises:
        ValueError: If either matrix is empty, non-finite, too small
            for a covariance, mismatched in grid length, or has a
            zero-variance wavelength column.
    """
    reference = _check_curves(reference)
    other = _check_curves(other)
    for name, matrix in (("reference", reference), ("other", other)):
        if matrix.shape[0] < 2:
            raise ValueError(f"{name} population needs at least 2 curves for a covariance")
    if reference.shape[1] != other.shape[1]:
        raise ValueError(
            f"reference and other grids differ: {reference.shape[1]} vs "
            f"{other.shape[1]} wavelengths"
        )
    reference_covariance = np.cov(reference, rowvar=False)
    other_covariance = np.cov(other, rowvar=False)
    scale = np.linalg.norm(reference_covariance)
    covariance_relative = float(np.linalg.norm(other_covariance - reference_covariance) / scale)
    reference_spread = np.sqrt(np.diag(reference_covariance))
    other_spread = np.sqrt(np.diag(other_covariance))
    if (reference_spread == 0).any() or (other_spread == 0).any():
        raise ValueError("zero-variance wavelength column; correlations undefined")
    reference_correlation = reference_covariance / np.outer(reference_spread, reference_spread)
    other_correlation = other_covariance / np.outer(other_spread, other_spread)
    correlation_relative = float(
        np.linalg.norm(other_correlation - reference_correlation)
        / np.linalg.norm(reference_correlation)
    )
    return {
        "covariance_relative_frobenius": covariance_relative,
        "correlation_relative_frobenius": correlation_relative,
    }


def coupling_record(
    curve_summaries: np.ndarray,
    thickness_nm: np.ndarray,
) -> dict[str, float]:
    """The coupling of curve summary and thickness, as a correlation.

    The Pearson correlation between one physical summary per film (an
    integrated attenuation, a plateau height) and that film's
    thickness. A negative value is expected even with no physics
    involved: ``q = A/t`` falls as ``t`` grows at fixed absorbance, a
    coupling built into the definition. The informative comparison is
    therefore not against zero but across populations: measured films
    set the value, resampled measured films reproduce it exactly, and
    curves paired with independently drawn thicknesses collapse it
    toward 0, erasing the coupling.

    Args:
        curve_summaries: One summary per film, shape ``(n_films,)``,
            in physical units.
        thickness_nm: Film thickness in nm, same length and order.

    Returns:
        ``pearson_r``: the correlation, dimensionless, between -1 and
        1.

    Raises:
        ValueError: If the arrays are empty, unequal in length,
            non-finite, or constant, since a constant array has no
            correlation with anything.
    """
    summaries = np.asarray(curve_summaries, dtype=float)
    thickness_nm = np.asarray(thickness_nm, dtype=float)
    if summaries.ndim != 1 or summaries.size == 0:
        raise ValueError("curve summaries must be a non-empty 1-D array")
    if thickness_nm.shape != summaries.shape:
        raise ValueError(
            f"thickness has shape {thickness_nm.shape}, expected {summaries.shape}; "
            "one thickness per film summary"
        )
    if not np.isfinite(summaries).all() or not np.isfinite(thickness_nm).all():
        raise ValueError("summaries and thickness must be finite")
    if summaries.std() == 0.0 or thickness_nm.std() == 0.0:
        raise ValueError("a constant array has no correlation with anything")
    return {"pearson_r": float(np.corrcoef(summaries, thickness_nm)[0, 1])}


def thickness_slice_errors(
    per_film_errors: np.ndarray,
    thickness_nm: np.ndarray,
    slice_edges_nm: np.ndarray,
) -> list[dict[str, object]]:
    """Mean per-film error inside thickness slices.

    Decomposes a per-film error over the thickness range, slices
    bounded by ``slice_edges_nm`` and half-open, lower edge included.
    Where a model's errors sit along the thickness range is a physics
    question, and a population average cannot answer it: a
    reconstruction error growing toward large thickness is invisible
    in the overall mean, which mixes thin films' small errors with
    thick films' large ones.

    Args:
        per_film_errors: One error per film, shape ``(n_films,)``, in
            any physical unit; the slices' means carry the same unit.
        thickness_nm: Film thickness in nm, same length and order,
            all inside the edge range.
        slice_edges_nm: Ascending slice boundaries in nm, shape
            ``(n_slices + 1,)``.

    Returns:
        One record per slice, in slice order: ``slice`` (index),
        ``edges_nm`` (the slice's two boundaries), ``n_films``, and
        ``mean_error`` (the mean per-film error, in the input's
        unit).

    Raises:
        ValueError: On empty or non-finite input, a length mismatch,
            fewer than two edges, non-ascending edges, films outside
            the edge range, or an empty slice.
    """
    per_film_errors = np.asarray(per_film_errors, dtype=float)
    thickness_nm = np.asarray(thickness_nm, dtype=float)
    slice_edges_nm = np.asarray(slice_edges_nm, dtype=float)
    if per_film_errors.ndim != 1 or per_film_errors.size == 0:
        raise ValueError("per-film errors must be a non-empty 1-D array")
    if thickness_nm.shape != per_film_errors.shape:
        raise ValueError(
            f"thickness has shape {thickness_nm.shape}, expected {per_film_errors.shape}; "
            "one thickness per film error"
        )
    if not (
        np.isfinite(per_film_errors).all()
        and np.isfinite(thickness_nm).all()
        and np.isfinite(slice_edges_nm).all()
    ):
        raise ValueError("errors, thickness and edges must be finite")
    if slice_edges_nm.ndim != 1 or slice_edges_nm.size < 2:
        raise ValueError("slice edges must be at least two boundaries")
    if (np.diff(slice_edges_nm) <= 0).any():
        raise ValueError("slice edges must be ascending")
    outside = (thickness_nm < slice_edges_nm[0]) | (thickness_nm > slice_edges_nm[-1])
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} films fall outside the slice edges "
            f"({slice_edges_nm[0]:.6g} to {slice_edges_nm[-1]:.6g} nm)"
        )
    bins = np.searchsorted(slice_edges_nm[1:-1], thickness_nm, side="right")
    records = []
    for bin_index in range(slice_edges_nm.size - 1):
        in_slice = bins == bin_index
        count = int(in_slice.sum())
        if count == 0:
            raise ValueError(f"thickness slice {bin_index} holds no films")
        records.append(
            {
                "slice": bin_index,
                "edges_nm": [float(slice_edges_nm[bin_index]), float(slice_edges_nm[bin_index + 1])],
                "n_films": count,
                "mean_error": float(per_film_errors[in_slice].mean()),
            }
        )
    return records


def component_scores(
    curves: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Project curves onto given axes: ``(curves - mean) @ components.T``.

    With a fitted :class:`~absynth.baselines.PCAGenerator`'s ``mean``
    and ``components`` this yields each curve's coordinates in the
    coordinates the generator draws in; for the generator's own
    samples, the result reproduces the drawn coefficients exactly.
    Comparing real and synthetic score histograms is the test of the
    generator's Gaussian assumption.

    Args:
        curves: Curve matrix, shape ``(n_curves, n_wavelengths)``.
        mean: Centre to subtract, shape ``(n_wavelengths,)``.
        components: Axes as rows, shape
            ``(n_components, n_wavelengths)``.

    Returns:
        Scores, shape ``(n_curves, n_components)``.

    Raises:
        ValueError: On shape mismatches or non-finite input.
    """
    curves = _check_curves(curves)
    mean = np.asarray(mean, dtype=float)
    components = np.asarray(components, dtype=float)
    if mean.shape != (curves.shape[1],):
        raise ValueError(
            f"mean has shape {mean.shape}, expected ({curves.shape[1]},)"
        )
    if components.ndim != 2 or components.shape[1] != curves.shape[1]:
        raise ValueError(
            f"components have shape {components.shape}, expected (n_components, "
            f"{curves.shape[1]})"
        )
    return (curves - mean) @ components.T
