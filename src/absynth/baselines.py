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

from dataclasses import dataclass

import numpy as np


def _check_training_curves(training_curves: np.ndarray) -> np.ndarray:
    """Validate a training curve matrix and return it as a float array."""
    training_curves = np.asarray(training_curves, dtype=float)
    if training_curves.ndim != 2 or training_curves.shape[0] == 0:
        raise ValueError("training curves must be a non-empty (n_curves, n_wavelengths) matrix")
    if not np.isfinite(training_curves).all():
        raise ValueError(
            "training curves must be finite; resolve unmeasured wavelengths "
            "(the working-span selection) before fitting"
        )
    return training_curves


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
    training_curves = _check_training_curves(training_curves)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    indices = rng.integers(0, training_curves.shape[0], size=n_samples)
    return training_curves[indices]


@dataclass(frozen=True)
class PCAGenerator:
    """A Gaussian generator in principal-component coordinates.

    A fitted PCA describes the training cloud as: an average curve,
    plus coefficients along a few variability directions. This class
    stores that description — ``mean``, the orthonormal ``components``
    (rows) and each direction's training spread ``scales`` — and
    generates a curve by drawing the coefficients from independent
    Gaussians: ``curve = mean + Σ z_i · component_i`` with
    ``z_i ~ N(0, scale_i²)``.

    Two properties are worth understanding. First, the generated cloud
    reproduces the training cloud's covariance exactly: principal
    component scores are uncorrelated by construction, and each scale
    is the measured standard deviation along its direction, so
    independent Gaussian draws rebuild the measured spreads and the
    wavelength-to-wavelength correlations they carry. Second, the
    generator can only interpolate: it composes variability directions
    it was shown and never leaves their span, so structure the training
    set does not express — non-Gaussian score shapes, several clusters,
    curved structure — is invisible to it. Whether the Gaussian
    assumption holds is a question for the evaluation layer, not an
    assumption to trust here.

    Attributes:
        mean: Average training curve, shape ``(n_wavelengths,)``.
        components: Orthonormal variability directions, shape
            ``(n_components, n_wavelengths)``.
        scales: Standard deviation of the training scores along each
            direction, shape ``(n_components,)``.
        n_training: Number of training curves fitted.
    """

    mean: np.ndarray
    components: np.ndarray
    scales: np.ndarray
    n_training: int

    @property
    def n_components(self) -> int:
        """Number of variability directions in the model."""
        return len(self.scales)

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n_samples`` synthetic curves in nm⁻¹.

        Args:
            n_samples: Number of curves to draw.
            rng: Numpy random generator; reproducible for a given
                generator state.

        Returns:
            Curve matrix, shape ``(n_samples, n_wavelengths)``.

        Raises:
            ValueError: If the requested count is not positive.
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        coefficients = rng.normal(scale=self.scales, size=(n_samples, self.n_components))
        return self.mean + coefficients @ self.components


def fit_pca_generator(
    training_curves: np.ndarray,
    n_components: int,
) -> PCAGenerator:
    """Fit the Gaussian-in-component-coordinates generator.

    PCA is the SVD of the centred curve cloud: the rows of the right
    singular matrix are the cloud's variability directions, and the
    singular values measure how much of the cloud lies along each.
    This fit keeps the ``n_components`` leading directions, stores the
    training average and each kept direction's score spread, and the
    result draws synthetic curves from that description.

    Args:
        training_curves: Curve matrix in nm⁻¹, shape
            ``(n_training, n_wavelengths)``, complete on the working
            span. Row order is irrelevant.
        n_components: Number of variability directions to keep. At
            most ``min(n_training - 1, n_wavelengths)``: ``n`` curves
            have at most ``n - 1`` nonzero spreads, and a kept
            direction needs a measurable spread to draw from.

    Returns:
        A fitted :class:`PCAGenerator`.

    Raises:
        ValueError: If the matrix is empty or non-finite, or the
            requested component count is out of range.
    """
    training_curves = _check_training_curves(training_curves)
    n_training, n_wavelengths = training_curves.shape
    max_components = min(n_training - 1, n_wavelengths)
    if not 1 <= n_components <= max_components:
        raise ValueError(
            f"n_components must be between 1 and {max_components} for "
            f"{n_training} training curves on {n_wavelengths} wavelengths, "
            f"got {n_components}"
        )

    mean = training_curves.mean(axis=0)
    centred = training_curves - mean
    _, singular_values, directions = np.linalg.svd(centred, full_matrices=False)
    components = directions[:n_components]
    # Scores along direction i have sample standard deviation
    # S_i / sqrt(n - 1): the singular value is the total spread, and
    # dividing by sqrt(n - 1) converts it to the ddof=1 standard
    # deviation of the per-curve coefficients.
    scales = singular_values[:n_components] / np.sqrt(n_training - 1)
    return PCAGenerator(mean=mean, components=components, scales=scales, n_training=n_training)
