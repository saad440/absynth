"""Statistical baselines: reference generators any model must beat.

Before any learned generator, two deliberately simple ones set the
bar. The first does no learning at all: it draws training curves from
a hat (the bootstrap resampling floor). It can only ever reproduce
curves it was shown, so it defines the *fidelity* a copying strategy
achieves for free and the *novelty* floor of zero. The second fits a
Gaussian description of the curve cloud in principal-component
coordinates and draws from it — the strongest purely linear generator
this low-dimensional data admits.

A learned generator earns its complexity only by matching the floor's
fidelity while producing curves measurably farther from the training
set than the floor's are, and by staying close to the held-out
measurements. Comparisons against both references, on films the
generators never saw, are what make that verdict honest.

All generators consume the training-side curve matrix only, as
selected by :mod:`absynth.splits`; fitting never touches held-out
films. Curves enter as loaded: optical density ``OD = A/t`` with
thickness ``t`` in nm, so values are in nm⁻¹, complete on the working
span (300–900 nm; unmeasured wavelengths are resolved by the
working-span selection before fitting).
"""

from __future__ import annotations

import numpy as np


def bootstrap_resample(
    training_curves: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw training curves with replacement: the memorization floor.

    The laziest conceivable generator keeps no model at all: it writes
    the training curves on slips of paper and draws ``n_samples`` of
    them from a hat, returning each slip to the hat before the next
    draw (resampling *with replacement*, the bootstrap). It can only
    ever reproduce curves it was shown, so its generated population has
    zero novelty by construction: every drawn curve is exactly a
    training curve. That is the point. Distributional comparisons
    between this population and real films measure how well a copying
    strategy can look, and any learned generator is required to match
    that fidelity while placing its curves measurably farther from the
    training set.

    Args:
        training_curves: Curve matrix in nm⁻¹, shape
            ``(n_training, n_wavelengths)``, complete on the working
            span. Rows are drawn, so row order is irrelevant.
        n_samples: Number of curves to draw.
        rng: Numpy random generator; the draw is reproducible for a
            given generator state, so seeds belong to the experiment
            configuration.

    Returns:
        Curve matrix in nm⁻¹, shape ``(n_samples, n_wavelengths)``,
        whose rows are training rows.

    Raises:
        ValueError: If the matrix is empty or non-finite, or the
            requested count is not positive.
    """
    training_curves = np.asarray(training_curves, dtype=float)
    if training_curves.ndim != 2 or training_curves.shape[0] == 0:
        raise ValueError("training curves must be a non-empty (n_curves, n_wavelengths) matrix")
    if not np.isfinite(training_curves).all():
        raise ValueError(
            "training curves must be finite; resolve unmeasured wavelengths "
            "(the working-span selection) before fitting"
        )
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    indices = rng.integers(0, training_curves.shape[0], size=n_samples)
    return training_curves[indices]
