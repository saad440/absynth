"""Compact learned models for optical-density curves.

The statistical baselines set the standard any learned generator must
clear. The bootstrap floor copies training curves, and the PCA
generator composes linear variability directions. This module adds the
learned counterpart, sized to train on a CPU, with every layer width
an explicit argument that the experiment configuration records.

The autoencoder is the starting point. Its encoder maps each curve, a
vector of ``n_wavelengths`` optical-density values, to a shorter
vector of ``n_latent`` numbers called the latent representation. Its
decoder maps that short vector back to a full-length curve. Training
adjusts both maps until the rebuilt curve matches the input. Because
the latent representation is much shorter than the curve, no encoder
can describe every possible curve exactly. Training therefore settles
on the coordinates that preserve the most of what the training curves
actually do. This short middle of the network is the bottleneck.

The construction is PCA with two differences. PCA also represents a
curve by a few coordinates and rebuilds it from them, but its maps are
linear and follow from the SVD of the centred training curves. The
autoencoder's maps are nonlinear and are found by gradient descent.
Removing the nonlinearities turns the optimal autoencoder back into a
PCA description, so any reconstruction quality beyond PCA comes from
nonlinear structure. A trained autoencoder of this kind represents
curves but does not generate new ones. Every curve maps to one fixed
coordinate vector, and nothing yet specifies which coordinate vectors
correspond to plausible curves.

Curves enter in the units used throughout this package, optical
density ``OD = A/t`` in nm⁻¹, complete on the working span (300–900 nm
at 2 nm). Inputs may be numpy arrays or torch tensors and are
converted to float32. Outputs are torch tensors. Weight initialization
draws from torch's global random generator. Seeding that generator
before construction is therefore a decision for the experiment
configuration, not for the model class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _checked_matrix(
    values: np.ndarray | torch.Tensor, expected_width: int, name: str
) -> torch.Tensor:
    """Validate a curve or representation matrix and convert it to float32.

    The checks match the input discipline of :mod:`absynth.baselines`
    (non-empty, finite, promised width) so that shape and unit mistakes
    surface as exceptions rather than silently wrong training or
    evaluation numbers.
    """
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty matrix with {expected_width} columns")
    if tensor.shape[1] != expected_width:
        raise ValueError(f"{name} has {tensor.shape[1]} columns, expected {expected_width}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


class AutoEncoder(nn.Module):
    """A compact fully connected autoencoder for fixed-length curves.

    The network is a chain of linear maps with nonlinear activations
    between them, one chain in each direction:

    - encoder: ``n_wavelengths → n_hidden → n_latent``
    - decoder: ``n_latent → n_hidden → n_wavelengths``

    The latent width is the number of coordinates available to describe
    each curve, and it is held below the number of wavelengths. This
    short middle is the network's bottleneck. It forces the network to
    describe curves with few coordinates, and training adjusts the maps
    so those coordinates capture the variability the training curves
    actually show.

    The nonlinear activations separate the model from PCA. A purely
    linear autoencoder has the PCA description as its optimum, so any
    reconstruction beyond PCA must pass through the activations. Their
    exact form is not critical at this scale. GELU is used because it
    is smooth everywhere and has no region where its gradient vanishes,
    which keeps small optical-density magnitudes flowing during
    training.

    The output layer carries no activation. Curves are real-valued
    nm⁻¹ numbers, and a bounded output function would clamp precisely
    the amplitudes the model is meant to reproduce.

    Attributes:
        n_wavelengths: Width of the input and output layers, the
            number of grid wavelengths per curve.
        n_hidden: Width of the single hidden layer on each side.
        n_latent: Width of the latent layer, the number of coordinates
            per curve. Held below ``n_wavelengths`` so the network
            cannot pass curves through unchanged. A value near the PCA
            baseline's four components keeps the later comparison at
            equal description width.
    """

    def __init__(self, n_wavelengths: int, n_hidden: int, n_latent: int) -> None:
        """Build the encoder and decoder stacks.

        Raises:
            ValueError: If a width is not a positive integer, or if the
                latent width is not smaller than the grid, since a
                bottleneck as wide as the input compresses nothing.
        """
        super().__init__()
        if n_wavelengths < 2:
            raise ValueError(f"n_wavelengths must be at least 2, got {n_wavelengths}")
        if n_hidden < 1:
            raise ValueError(f"n_hidden must be at least 1, got {n_hidden}")
        if not 1 <= n_latent < n_wavelengths:
            raise ValueError(
                f"n_latent must be between 1 and {n_wavelengths - 1} to be a bottleneck, "
                f"got {n_latent}"
            )
        self.n_wavelengths = n_wavelengths
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.encoder = nn.Sequential(
            nn.Linear(n_wavelengths, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_wavelengths),
        )

    @property
    def parameter_count(self) -> int:
        """Total number of trainable weights in the model.

        The PCA baseline stores roughly
        ``(1 + n_components) · n_wavelengths`` numbers (its mean, its
        components and their scales), so quoting this count keeps the
        size comparison between learned and linear descriptions
        explicit.
        """
        return sum(parameter.numel() for parameter in self.parameters())

    def encode(self, curves: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Map curves to their latent representations.

        Args:
            curves: Curve matrix in nm⁻¹, shape
                ``(n_curves, n_wavelengths)``; numpy array or tensor.

        Returns:
            Latent representation matrix, shape
            ``(n_curves, n_latent)``.

        Raises:
            ValueError: If the matrix is empty, non-finite, or does not
                match the model's input width.
        """
        return self.encoder(_checked_matrix(curves, self.n_wavelengths, "curves"))

    def decode(self, representations: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Map latent representations back to curves in nm⁻¹.

        Args:
            representations: Latent representation matrix, shape
                ``(n_curves, n_latent)``; numpy array or tensor.

        Returns:
            Curve matrix, shape ``(n_curves, n_wavelengths)``.

        Raises:
            ValueError: If the matrix is empty, non-finite, or does not
                match the model's latent width.
        """
        return self.decoder(_checked_matrix(representations, self.n_latent, "representations"))

    def forward(self, curves: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Rebuild curves by encoding and then decoding.

        Computes ``decode(encode(curves))``. Training minimizes the
        distance between the rebuilt curves and their inputs. The same
        quantity on held-out films is the reconstruction diagnostic the
        statistical baselines do not provide.

        Raises:
            ValueError: If the input fails :meth:`encode`'s checks.
        """
        return self.decode(self.encode(curves))


def _checked_numpy_matrix(values: np.ndarray, expected_width: int | None = None) -> np.ndarray:
    """Validate a numpy curve or representation matrix as a float array.

    The numpy counterpart of :func:`_checked_matrix`, for code that
    stays outside torch: data handling and evaluation. Non-empty,
    finite, and (when a width is given) of the promised width.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("expected a non-empty (n_curves, n_wavelengths) matrix")
    if expected_width is not None and matrix.shape[1] != expected_width:
        raise ValueError(f"matrix has {matrix.shape[1]} columns, expected {expected_width}")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite")
    return matrix


@dataclass(frozen=True)
class CurveStandardizer:
    """Per-wavelength shift and rescaling fitted on one population of curves.

    Gradient descent has no sense of units, and these curves carry
    values of order 10⁻³ to 10⁻² nm⁻¹, far below the magnitudes that
    train well. Fitting measures, wavelength by wavelength, the mean
    and the standard deviation of the population. Transform then
    subtracts the mean and divides by the standard deviation, which
    leaves every fitted wavelength at mean 0 with spread 1, expressed
    in dimensionless units. The stored map moves any later curve, real
    or generated, between nm⁻¹ and these standardized units exactly.

    The fitting population is the training films. Fitting on any film
    that later serves as an evaluation target would leak information
    about the evaluation population into the model and bias held-out
    scores optimistic (data leakage).

    Attributes:
        mean: Per-wavelength mean of the fitting curves, nm⁻¹, shape
            ``(n_wavelengths,)``.
        std: Per-wavelength population standard deviation (``ddof=0``)
            of the fitting curves, nm⁻¹, shape ``(n_wavelengths,)``.
    """

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        std = np.asarray(self.std, dtype=float)
        if mean.ndim != 1 or mean.shape != std.shape or mean.size == 0:
            raise ValueError("mean and std must be non-empty, equal-length 1-D arrays")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("mean and std must be finite")
        if (std <= 0).any():
            raise ValueError("std entries must be positive to scale curves")

    @classmethod
    def fit(cls, curves: np.ndarray) -> CurveStandardizer:
        """Measure per-wavelength mean and spread of a curve population.

        Args:
            curves: Curve matrix in nm⁻¹, shape
                ``(n_curves, n_wavelengths)``.

        Returns:
            A fitted standardizer.

        Raises:
            ValueError: If the matrix is empty, non-finite, or contains
                a wavelength column that never varies across the
                fitting curves, which has no spread to scale by.
        """
        matrix = _checked_numpy_matrix(curves)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        if (std == 0).any():
            column = int(np.flatnonzero(std == 0)[0])
            raise ValueError(
                f"wavelength column {column} has zero variance across the "
                "fitting curves; there is no spread to scale by"
            )
        return cls(mean=mean, std=std)

    def transform(self, curves: np.ndarray) -> np.ndarray:
        """Map curves to standardized values, ``(curves - mean) / std``.

        Args:
            curves: Curve matrix in nm⁻¹, shape
                ``(n_curves, n_wavelengths)``.

        Returns:
            Standardized matrix, dimensionless, same shape.

        Raises:
            ValueError: If the matrix is empty, non-finite, or does not
                match the fitted wavelength count.
        """
        matrix = _checked_numpy_matrix(curves, expected_width=len(self.mean))
        return (matrix - self.mean) / self.std

    def inverse_transform(self, standardized: np.ndarray) -> np.ndarray:
        """Map standardized values back to nm⁻¹, ``standardized · std + mean``.

        Inverse of :meth:`transform` up to floating-point rounding.

        Args:
            standardized: Standardized matrix, dimensionless, shape
                ``(n_curves, n_wavelengths)``.

        Returns:
            Curve matrix in nm⁻¹, same shape.

        Raises:
            ValueError: If the matrix is empty, non-finite, or does not
                match the fitted wavelength count.
        """
        matrix = _checked_numpy_matrix(standardized, expected_width=len(self.mean))
        return matrix * self.std + self.mean
