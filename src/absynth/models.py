"""Compact learned models for optical-density curves.

The statistical baselines set the standard any learned generator must
clear. The bootstrap floor copies training curves, and the PCA
generator composes linear variability directions. This module adds the
learned counterpart, sized to train on a CPU, with every layer width
an explicit argument that the experiment configuration records.

The autoencoder is the starting point. Its encoder maps each curve, a
vector of ``n_wavelengths`` optical-density values, to a shorter
vector of ``n_latent`` numbers called the code. Its decoder maps that
short vector back to a full-length curve. Training adjusts both maps
until the rebuilt curve matches the input. Because the code is much
shorter than the curve, no encoder
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

The variational autoencoder makes the step from representation to
generator. Its encoder outputs a mean and a standard deviation per
latent dimension, so a curve's code is a small Gaussian rather than a
coordinate. Training adds a second demand. Over the whole training
population, the codes should look like draws from the standard
normal, the prior. A new curve is then a prior draw passed through
the decoder. The prior-matching demand enters the objective as a KL
term whose weight sets the implied observation noise. train.py
records that implied value alongside every fit.

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
    """Validate a curve or code matrix and convert it to float32.

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


def _decoder_stack(n_latent: int, n_hidden: int, n_wavelengths: int) -> nn.Sequential:
    """The decoder architecture shared by both model classes.

    :class:`AutoEncoder` and :class:`VariationalAutoEncoder` build
    their decoders from this one definition, so their decoders stay
    identical by construction: latent width → hidden → grid, linear
    maps with a GELU activation between, and no output activation.
    """
    return nn.Sequential(
        nn.Linear(n_latent, n_hidden),
        nn.GELU(),
        nn.Linear(n_hidden, n_wavelengths),
    )


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
        self.decoder = _decoder_stack(n_latent, n_hidden, n_wavelengths)

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
        """Map curves to their codes.

        Args:
            curves: Curve matrix in nm⁻¹, shape
                ``(n_curves, n_wavelengths)``; numpy array or tensor.

        Returns:
            Code matrix, shape ``(n_curves, n_latent)``.

        Raises:
            ValueError: If the matrix is empty, non-finite, or does not
                match the model's input width.
        """
        return self.encoder(_checked_matrix(curves, self.n_wavelengths, "curves"))

    def decode(self, codes: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Map codes back to curves in nm⁻¹.

        Args:
            codes: Code matrix, shape ``(n_curves, n_latent)``; numpy
                array or tensor.

        Returns:
            Curve matrix, shape ``(n_curves, n_wavelengths)``.

        Raises:
            ValueError: If the matrix is empty, non-finite, or does not
                match the model's latent width.
        """
        return self.decoder(_checked_matrix(codes, self.n_latent, "codes"))

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
    """Validate a numpy curve or code matrix as a float array.

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


def latent_kl_per_dimension(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Per-dimension KL divergence of latent Gaussians against the prior.

    Each curve's code is one Gaussian per latent dimension, with mean
    ``mean`` and standard deviation ``std``. The
    prior is the standard normal. In closed form, the divergence per
    dimension is ``0.5 · (mean² + std² − log std² − 1)``. It is
    non-negative, and zero exactly when the latent Gaussian is the
    prior itself.

    Averaged over the training films, a dimension's value near zero
    marks a dimension the encoder has stopped using. That state is
    called posterior collapse. The training objective uses the
    per-curve sums of this matrix.

    Args:
        mean: Latent means, shape ``(n_curves, n_latent)``.
        std: Latent standard deviations, same shape, all positive.

    Returns:
        The KL contributions, shape ``(n_curves, n_latent)``.

    Raises:
        ValueError: If the shapes differ, the matrices are not 2-D, or
            an entry of ``std`` is not positive.
    """
    if mean.ndim != 2 or mean.shape != std.shape:
        raise ValueError("mean and std must be 2-D matrices of the same shape")
    if not bool((std > 0).all()):
        raise ValueError("std entries must be positive")
    return 0.5 * (mean * mean + std * std - torch.log(std * std) - 1.0)


class VariationalAutoEncoder(nn.Module):
    """A fully connected variational autoencoder for fixed-length curves.

    The class shares its decoder with :class:`AutoEncoder`. The
    encoder differs in its last layer. Instead of one number per
    latent dimension, it outputs two numbers, a mean and a
    log-variance. Each curve therefore maps to one Gaussian per
    latent dimension, with no correlation between the dimensions.

    Codes are drawn from those Gaussians in :meth:`sample_latent`,
    constructed as ``mean + std · eps`` with ``eps`` standard normal.
    The random step sits outside the differentiable path, so gradients
    reach the mean and the standard deviation.

    Training, in :mod:`absynth.train`, minimizes the reconstruction
    error plus ``kl_weight`` times the KL divergence of the code
    Gaussians against the standard-normal prior, computed in closed
    form by :func:`latent_kl_per_dimension`. The KL term is what makes
    prior draws decodable curves. Without it, the code Gaussians would
    drift wherever reconstruction alone pushes them, and a draw from
    the standard normal would land where no curve's code ever was.

    The weight sets the implied observation noise. Weights far above
    the configured anchor collapse the codes toward the prior, leaving
    the decoder nothing to tell curves apart with. Weights far below
    let the codes drift from the prior, until prior draws decode
    off-distribution.

    Attributes:
        n_wavelengths: Width of the input and output layers, the
            number of grid wavelengths per curve.
        n_hidden: Width of the single hidden layer on each side.
        n_latent: Width of the latent layer, the number of latent
            dimensions per curve.
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
            nn.Linear(n_hidden, 2 * n_latent),
        )
        self.decoder = _decoder_stack(n_latent, n_hidden, n_wavelengths)

    @property
    def parameter_count(self) -> int:
        """Number of trainable weights: the model's honest size.

        Same accounting as :attr:`AutoEncoder.parameter_count`, quoted
        against the PCA baseline's roughly
        ``(1 + n_components) · n_wavelengths`` stored numbers.
        """
        return sum(parameter.numel() for parameter in self.parameters())

    def encode(self, curves: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map curves to per-dimension latent means and standard deviations.

        Args:
            curves: Curve matrix in nm⁻¹, shape
                ``(n_curves, n_wavelengths)``; numpy array or tensor.

        Returns:
            Two matrices of shape ``(n_curves, n_latent)``: the latent
            means and the latent standard deviations. Together they
            define one Gaussian per latent dimension for each curve.

        Raises:
            ValueError: If the matrix is empty, non-finite, or does not
                match the model's input width.
        """
        flat = self.encoder(_checked_matrix(curves, self.n_wavelengths, "curves"))
        mean, log_variance = flat.split(self.n_latent, dim=1)
        return mean, torch.exp(0.5 * log_variance)

    def decode(self, codes: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Map codes back to curves in nm⁻¹.

        Args:
            codes: Code matrix, shape ``(n_curves, n_latent)``; numpy
                array or tensor. Typically a :meth:`sample_latent` draw.

        Returns:
            Curve matrix, shape ``(n_curves, n_wavelengths)``.

        Raises:
            ValueError: If the matrix is empty, non-finite, or does not
                match the model's latent width.
        """
        return self.decoder(_checked_matrix(codes, self.n_latent, "codes"))

    def sample_latent(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw codes from the per-curve Gaussians.

        Each draw is constructed as ``mean + std · eps`` with ``eps``
        standard normal. The noise is generated outside the
        differentiable path, so gradients flow through the mean and
        the standard deviation that produced the draw.

        With ``generator=None`` the noise uses torch's global
        generator, which the training loop seeds. Pass a dedicated
        generator for reproducible draws outside training.

        Args:
            mean: Latent means, shape ``(n_curves, n_latent)``.
            std: Latent standard deviations, same shape.
            generator: Optional dedicated torch generator for the noise.

        Returns:
            The drawn codes, shape ``(n_curves, n_latent)``.

        Raises:
            ValueError: If ``mean`` and ``std`` differ in shape.
        """
        if mean.shape != std.shape:
            raise ValueError("mean and std must have the same shape")
        noise = torch.randn(std.shape, generator=generator, dtype=std.dtype, device=std.device)
        return mean + std * noise

    def forward(
        self, curves: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample codes, rebuild curves, and report the code parameters.

        Returns ``(rebuilt, mean, std)``: the decoded draw together
        with the Gaussian parameters it came from. Training computes
        its reconstruction term from the first element and its KL term
        from the last two.
        """
        mean, std = self.encode(curves)
        return self.decode(self.sample_latent(mean, std)), mean, std
