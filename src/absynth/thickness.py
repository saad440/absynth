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
reference role for curves alone.

Curves enter in nm⁻¹, complete on the working span (300–900 nm; see
:mod:`absynth.preprocessing`); thickness in nm, strictly positive.
Nothing is clipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
