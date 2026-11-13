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
