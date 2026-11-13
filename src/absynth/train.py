"""Training loop for the compact autoencoder.

Fitting an :class:`~absynth.models.AutoEncoder` means repeatedly
adjusting its weights until the rebuilt curves match their inputs.
This module owns that loop and nothing else: the model class defines
the architecture, the experiment configuration supplies every number
used here, and the evaluation layer judges the result. Curves enter in
nm⁻¹ on the working span, exactly as the statistical baselines
consume them.

Two steps precede the loop. The curves are standardized per
wavelength, with mean and spread measured on the training films alone,
so the network trains on values of order unity (see
:class:`~absynth.models.CurveStandardizer`). The loop then runs
minibatch gradient descent: each step computes the mean squared error
over a small random subset of the training films, differentiates it
with respect to the weights, and moves the weights a short distance
downhill. One pass over all training films is an epoch, and the loss
recorded per epoch makes convergence visible rather than assumed.

Reproducibility: the configuration's seed drives the weight
initialization and the batch shuffling through dedicated torch random
generators, so a given configuration and curve matrix reproduce a
given training run exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from absynth.models import AutoEncoder, CurveStandardizer


@dataclass(frozen=True)
class AutoEncoderConfig:
    """Every number the training loop needs, in one record.

    The experiment configuration file pins one instance per run, so a
    written record can state exactly which architecture and which
    optimization produced which numbers. The widths follow
    :class:`~absynth.models.AutoEncoder`; the remaining fields are the
    optimization recipe.

    Attributes:
        n_hidden: Width of the hidden layer on each side of the
            bottleneck.
        n_latent: Width of the bottleneck, the latent width.
        epochs: Number of passes over the training films.
        learning_rate: Step size of the gradient descent update. The
            Adam optimizer scales it per weight from gradient history.
        batch_size: Films per gradient step. Smaller batches give
            noisier but cheaper gradient estimates.
        seed: Seed for weight initialization and batch shuffling.
    """

    n_hidden: int
    n_latent: int
    epochs: int
    learning_rate: float
    batch_size: int
    seed: int

    def __post_init__(self) -> None:
        if self.n_hidden < 1 or self.n_latent < 1:
            raise ValueError("network widths must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


@dataclass(frozen=True)
class TrainedAutoEncoder:
    """A fitted autoencoder with everything needed to audit the fit.

    Attributes:
        model: The trained network with its final weights, ready for
            encoding, decoding and generation.
        standardizer: The per-wavelength scaling fitted on the training
            films. Every curve crossing between nm⁻¹ and the model's
            standardized units passes through it.
        config: The configuration that produced the fit.
        epoch_losses: Mean squared error per epoch, in standardized
            units squared, averaged over all training films. A falling
            record documents convergence; a flat one documents its
            absence.
    """

    model: AutoEncoder
    standardizer: CurveStandardizer
    config: AutoEncoderConfig
    epoch_losses: tuple[float, ...]


def train_autoencoder(training_curves: np.ndarray, config: AutoEncoderConfig) -> TrainedAutoEncoder:
    """Fit an autoencoder on training curves and return the audited fit.

    The standardizer is fitted on these curves only. Held-out films
    must never reach this function: any influence of the evaluation
    population on the fit would make held-out scores optimistic, the
    leakage failure mode.

    Args:
        training_curves: Curve matrix in nm⁻¹, shape
            ``(n_training, n_wavelengths)``, complete on the working
            span. Row order is irrelevant.
        config: Architecture and optimization numbers.

    Returns:
        A :class:`TrainedAutoEncoder`.

    Raises:
        ValueError: If the curves are empty or non-finite, if a
            wavelength column never varies, or if a configuration
            value fails its checks, including a latent width that is
            not smaller than the grid.
    """
    curves = np.asarray(training_curves, dtype=float)
    if curves.ndim != 2 or curves.shape[0] == 0:
        raise ValueError("training curves must be a non-empty (n_curves, n_wavelengths) matrix")
    standardizer = CurveStandardizer.fit(curves)
    standardized = torch.from_numpy(standardizer.transform(curves).astype(np.float32))

    torch.manual_seed(config.seed)
    model = AutoEncoder(
        n_wavelengths=curves.shape[1],
        n_hidden=config.n_hidden,
        n_latent=config.n_latent,
    )
    batches = DataLoader(
        standardized,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    n_training = standardized.shape[0]
    epoch_losses = []
    for _ in range(config.epochs):
        epoch_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            loss = mse_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * batch.shape[0]
        epoch_losses.append(epoch_loss / n_training)
    return TrainedAutoEncoder(
        model=model,
        standardizer=standardizer,
        config=config,
        epoch_losses=tuple(epoch_losses),
    )
