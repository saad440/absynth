"""Thickness-aware generation: films as curve-and-thickness pairs.

The archive stores optical density ``q(λ) = A(λ)/t``, so each measured
film arrives as two aligned quantities: the curve ``q`` in nm⁻¹ and the
thickness ``t`` in nm it was divided by. The pair is the film's
description. Converting a curve back to absorbance, ``A = q·t``, needs
the thickness again, so a generator that emits curves alone still owes
a thickness for every curve, and how those thicknesses are chosen is
part of the generator. Drawing them from a marginal, the distribution
of one quantity on its own, ignores how curve and thickness go
together in the measurements. A curve typical of a 150 nm film can end
up carrying the thickness of a 1900 nm film, and the absorbance
``A = q·t`` of that pairing matches nothing in the archive. The
pairing also erases the coupling between curve and thickness, which
has two sources. One is definitional, the arithmetic of ``q = A/t``
itself. At fixed absorbance ``A``, ``q`` falls as ``t`` grows, so the
two covary even where Beer–Lambert proportionality holds otherwise.
The other is physical. Thickness is set by coating speed and ink
concentration, which also shape the microstructure the curve reflects.

Generators in this module emit the pair together. The joint resampling
floor draws curve and thickness jointly, through one shared film
index. Every drawn pair is a measured film, so the drawn population
reproduces the measured coupling exactly and contains nothing new. It
is the pair-valued counterpart of
:func:`absynth.baselines.bootstrap_resample`, which plays the same
reference role for curves alone. The thickness-slice generator is the
conditional step up: it draws ``t`` from the measured thicknesses and
``q`` from a PCA description of the films in ``t``'s slice, so its
pairs are new curves carrying measured thicknesses.

Curves enter in nm⁻¹, complete on the working span (300–900 nm; see
:mod:`absynth.preprocessing`); thickness in nm, strictly positive.
Nothing is clipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from absynth.baselines import PCAGenerator, fit_pca_generator


def _checked_training_films(
    training_curves: np.ndarray, training_thickness_nm: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a (curves, thickness) table and return both as arrays.

    The input discipline matches :mod:`absynth.baselines` (non-empty,
    finite curve matrix) and :mod:`absynth.preprocessing` (positive,
    finite thickness), with one addition. The thickness must carry one
    entry per curve row, so every curve keeps the thickness it was
    divided by. A thickness Series is consumed by position, its index
    ignored, as everywhere else in this package.
    """
    curves = np.asarray(training_curves, dtype=float)
    if curves.ndim != 2 or curves.shape[0] == 0:
        raise ValueError("training curves must be a non-empty (n_films, n_wavelengths) matrix")
    if not np.isfinite(curves).all():
        raise ValueError(
            "training curves must be finite; resolve unmeasured wavelengths "
            "(the working-span selection) before fitting"
        )
    thickness_nm = np.asarray(training_thickness_nm, dtype=float)
    if thickness_nm.shape != (curves.shape[0],):
        raise ValueError(
            f"thickness has shape {thickness_nm.shape}, expected ({curves.shape[0]},); "
            "one entry per curve row"
        )
    if not np.isfinite(thickness_nm).all():
        raise ValueError("thickness contains non-finite values")
    if (thickness_nm <= 0).any():
        raise ValueError("thickness must be positive to convert between A and q")
    return curves, thickness_nm


def resample_joint(
    training_curves: np.ndarray,
    training_thickness_nm: pd.Series | np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw whole films with replacement: the pair-valued resampling floor.

    :func:`absynth.baselines.bootstrap_resample` draws stored curves
    with replacement, reproducing the curve population exactly while
    producing nothing new. This generator applies the same principle
    to films. One random index selects a stored curve and the
    thickness it was divided by together, so every drawn pair is a
    measured film. The drawn population carries the measured coupling
    between ``q`` and ``t``, definitional and physical alike, and
    introduces no curve or thickness the archive does not contain.
    Like the curve floor, it is the reference the evaluation layer
    reads against, and a learned generator earns its complexity only
    by matching its fidelity while placing its pairs measurably
    farther from the training set.

    Args:
        training_curves: Curve matrix in nm⁻¹, shape
            ``(n_films, n_wavelengths)``, complete on the working span.
            Row order is irrelevant.
        training_thickness_nm: Film thickness in nm, one entry per
            *row* of ``training_curves``, in the same order. A Series
            is consumed by position (its index is ignored);
            ``Dataset.thickness_nm`` is already aligned to the curve
            rows by the loader and is the safe choice.
        n_samples: Number of films to draw.
        rng: Numpy random generator; the draw is reproducible for a
            given generator state, so seeds belong to the experiment
            configuration.

    Returns:
        Tuple ``(curves, thickness_nm)``: the drawn curve matrix in
        nm⁻¹, shape ``(n_samples, n_wavelengths)``, and the drawn
        thicknesses in nm, shape ``(n_samples,)``, aligned row by row.

    Raises:
        ValueError: If the curves are empty or non-finite, the
            thickness does not match the row count, is non-finite or
            non-positive, or the requested count is not positive.
    """
    curves, thickness_nm = _checked_training_films(training_curves, training_thickness_nm)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    indices = rng.integers(0, curves.shape[0], size=n_samples)
    return curves[indices], thickness_nm[indices]


@dataclass(frozen=True)
class ThicknessSliceGenerator:
    """A conditional curve generator: one PCA description per thickness slice.

    Beer–Lambert proportionality makes a film's thickness-normalised
    curve ``q`` independent of the thickness ``t`` it was divided by.
    A measured population can depart from that prediction, through the
    arithmetic of ``q = A/t`` and through coating conditions that shape
    thickness and microstructure together. This generator lets the
    data carry such departures: the training films are sorted into
    ``n_slices`` thickness slices of equal film count, each slice
    receives its own :class:`~absynth.baselines.PCAGenerator`, and a
    draw proceeds in two steps. The thickness is drawn from the
    measured thicknesses themselves, with replacement, so the drawn
    thickness distribution is the measured one; the curve is then drawn
    from the slice that thickness falls in. Conditioning is the only
    difference from the unconditional generator of
    :mod:`absynth.baselines`: the same PCA machinery with the same
    component count, applied slice by slice.

    Assumptions and limitations: within a slice, curves are described
    as Gaussian in principal-component coordinates, the unconditional
    generator's assumption now made per slice. The conditional
    structure is piecewise constant in thickness, so trends finer than
    a slice are not represented. And every slice must hold enough
    films to fit its ``n_components`` directions, which bounds the
    slice count on a small population.

    Attributes:
        thickness_nm: The measured training thicknesses in nm, shape
            ``(n_training,)``; the source of every drawn thickness.
        slice_edges_nm: Slice boundaries in nm, shape
            ``(n_slices + 1,)``, the training thickness quantiles.
            Slices are half-open, lower edge included.
        generators: One fitted
            :class:`~absynth.baselines.PCAGenerator` per slice.
        film_counts: Training films per slice, shape ``(n_slices,)``.
    """

    thickness_nm: np.ndarray
    slice_edges_nm: np.ndarray
    generators: tuple[PCAGenerator, ...]
    film_counts: tuple[int, ...]

    @property
    def n_slices(self) -> int:
        """Number of thickness slices in the model."""
        return len(self.generators)

    @property
    def n_components(self) -> int:
        """PCA directions kept per slice."""
        return self.generators[0].n_components

    def sample(
        self, n_samples: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw films: measured thickness, modelled curve.

        One generator state drives the whole draw, thickness resampling
        first and then curve draws slice by slice, so a given seed
        reproduces the draw exactly.

        Args:
            n_samples: Number of films to draw.
            rng: Numpy random generator; the draw is reproducible for a
                given generator state.

        Returns:
            Tuple ``(curves, thickness_nm)``: the drawn curve matrix in
            nm⁻¹, shape ``(n_samples, n_wavelengths)``, and the drawn
            thicknesses in nm, shape ``(n_samples,)``, aligned row by
            row.

        Raises:
            ValueError: If the requested count is not positive.
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        indices = rng.integers(0, self.thickness_nm.size, size=n_samples)
        thickness_nm = self.thickness_nm[indices]
        bins = np.searchsorted(self.slice_edges_nm[1:-1], thickness_nm, side="right")
        n_wavelengths = self.generators[0].mean.shape[0]
        curves = np.empty((n_samples, n_wavelengths))
        for bin_index, generator in enumerate(self.generators):
            in_slice = bins == bin_index
            if in_slice.any():
                curves[in_slice] = generator.sample(int(in_slice.sum()), rng)
        return curves, thickness_nm


def fit_thickness_slice_generator(
    training_curves: np.ndarray,
    training_thickness_nm: pd.Series | np.ndarray,
    n_slices: int,
    n_components: int,
) -> ThicknessSliceGenerator:
    """Fit one PCA generator per measured thickness slice.

    Args:
        training_curves: Curve matrix in nm⁻¹, shape
            ``(n_films, n_wavelengths)``, complete on the working span.
            Row order is irrelevant.
        training_thickness_nm: Film thickness in nm, one entry per
            *row* of ``training_curves``, in the same order. A Series
            is consumed by position (its index is ignored);
            ``Dataset.thickness_nm`` is already aligned to the curve
            rows by the loader and is the safe choice.
        n_slices: Number of thickness slices, each holding an equal
            share of the training films.
        n_components: PCA directions kept per slice. The same count in
            every slice keeps this generator at the same description
            width as its unconditional counterpart.

    Returns:
        A fitted :class:`ThicknessSliceGenerator`.

    Raises:
        ValueError: If the curves or thickness fail validation, if
            ``n_slices`` is not positive, or if any slice holds fewer
            than ``n_components + 1`` films, the smallest population
            from which ``n_components`` PCA directions can be fitted.
    """
    curves, thickness_nm = _checked_training_films(training_curves, training_thickness_nm)
    if n_slices < 1:
        raise ValueError(f"n_slices must be positive, got {n_slices}")
    slice_edges_nm = np.quantile(thickness_nm, np.linspace(0.0, 1.0, n_slices + 1))
    # side="right" makes each slice half-open, lower edge included; the
    # drawn thicknesses are measured values, so all land inside the range.
    bins = np.searchsorted(slice_edges_nm[1:-1], thickness_nm, side="right")
    counts = np.bincount(bins, minlength=n_slices)
    for bin_index, count in enumerate(counts):
        if count < n_components + 1:
            raise ValueError(
                f"thickness slice {bin_index} "
                f"({slice_edges_nm[bin_index]:.6g} to {slice_edges_nm[bin_index + 1]:.6g} nm) "
                f"holds {count} films; n_components={n_components} needs at least "
                f"{n_components + 1} films in every slice"
            )
    generators = tuple(
        fit_pca_generator(curves[bins == bin_index], n_components) for bin_index in range(n_slices)
    )
    return ThicknessSliceGenerator(
        thickness_nm=thickness_nm,
        slice_edges_nm=slice_edges_nm,
        generators=generators,
        film_counts=tuple(int(count) for count in counts),
    )
