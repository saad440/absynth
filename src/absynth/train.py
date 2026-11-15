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
The fit also stores the training films' final codes, so a Gaussian
fitted to them can generate curves without any variational machinery:
the KL-free reference the variational model is measured against.

The variational model trains on the same loop with one extra loss
term. Its objective is the reconstruction mean squared error plus
``kl_weight`` times the KL divergence of the code Gaussians against
the standard-normal prior. The weight has a statistical reading:
matching term by term against the exact Gaussian objective shows the
weight implies an observation variance of ``kl_weight · n_wavelengths
/ 2`` in standardized units. Weights far above the anchor collapse
the codes toward the prior. Weights far below let the codes drift
from it, until prior draws decode off-distribution. Setting the
implied variance to the standardized data's own spread, 1, gives the
anchor ``kl_weight = 2 / n_wavelengths`` (about 0.0066 on the
301-point working span), a reasonable centre for a small weight scan.
Every variational fit records the variance its weight implied.

The thickness-conditioned variants run the same loops over (curve,
thickness) pairs. Curves and thicknesses are standardized on
training statistics, the thickness enters the decoder as an input
next to the code, and the objective is unchanged. The trained fits
draw films by resampling measured training thicknesses and decoding
drawn codes at them, returning curve–thickness pairs in the same
contract as the generators of absynth.thickness.

Reproducibility: the configuration's seed drives the weight
initialization and the batch shuffling through dedicated torch random
generators, so a given configuration and curve matrix reproduce a
given training run exactly. A non-finite loss or code Gaussian raises
TrainingDivergenceError at its source instead of propagating NaN into
codes, diagnostics and samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, TensorDataset

from absynth.models import (
    AutoEncoder,
    CurveStandardizer,
    ThicknessConditionedAutoEncoder,
    ThicknessConditionedVariationalAutoEncoder,
    ThicknessStandardizer,
    VariationalAutoEncoder,
    latent_kl_per_dimension,
)


class TrainingDivergenceError(RuntimeError):
    """Raised when gradient descent produces non-finite losses.

    Divergence at a given learning rate is seed-dependent: an
    unlucky initialization can blow the weights up within a few
    epochs while the same recipe converges from other seeds. The
    message names the epoch, the seed and the learning rate; a
    diverged seed is a measured instability of the recipe, reported
    like any other result.
    """


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
        codes: The final codes of the training films, shape
            ``(n_training, n_latent)``, computed after training from
            the fitted weights. This is the fitted representation
            itself, and the population :meth:`sample` draws from.
        epoch_losses: Mean squared error per epoch, in standardized
            units squared, averaged over all training films. A falling
            record documents convergence; a flat one documents its
            absence.
    """

    model: AutoEncoder
    standardizer: CurveStandardizer
    config: AutoEncoderConfig
    codes: np.ndarray
    epoch_losses: tuple[float, ...]

    def sample(self, n_samples: int, seed: int) -> np.ndarray:
        """Draw synthetic curves in nm⁻¹ without variational machinery.

        The plain autoencoder has no prior to draw from: nothing in its
        training says which codes are plausible. This generator
        measures instead of trains. A multivariate Gaussian is fitted
        to the training codes, codes are drawn from it, and the
        decoder turns them into curves. The fit is deterministic given
        the codes; the seed drives the draws. Nothing is clipped.

        The variational model is trained toward a prior. This
        generator accepts whatever code cloud training happened to
        produce. If both look the same under the frozen evaluation,
        the variational machinery added nothing for this dataset.

        Args:
            n_samples: Number of curves to draw.
            seed: Seed for the numpy generator driving the draws.

        Returns:
            Curve matrix in nm⁻¹, shape
            ``(n_samples, n_wavelengths)``.
        """
        rng = np.random.default_rng(seed)
        drawn = rng.multivariate_normal(
            mean=self.codes.mean(axis=0),
            cov=np.cov(self.codes, rowvar=False),
            size=n_samples,
        )
        with torch.no_grad():
            standardized = self.model.decode(torch.from_numpy(drawn.astype(np.float32)))
        return self.standardizer.inverse_transform(standardized.numpy())


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
        TrainingDivergenceError: If the loss becomes non-finite
            during training.
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
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            loss = mse_loss(model(batch), batch)
            if not bool(torch.isfinite(loss)):
                raise TrainingDivergenceError(
                    f"training diverged: the loss is not finite at epoch {epoch} "
                    f"(seed {config.seed}, learning_rate {config.learning_rate}); "
                    "lower the learning rate or report the seed as unstable"
                )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * batch.shape[0]
        epoch_losses.append(epoch_loss / n_training)
    with torch.no_grad():
        codes = model.encode(standardized).numpy()
    return TrainedAutoEncoder(
        model=model,
        standardizer=standardizer,
        config=config,
        codes=codes,
        epoch_losses=tuple(epoch_losses),
    )


def _checked_thickness_pair(
    training_curves: np.ndarray, training_thickness_nm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a (curves, thickness) table for the conditioned loops.

    The curve matrix must be non-empty and finite, as
    :class:`~absynth.models.CurveStandardizer.fit` also requires; the
    thickness must hold one finite, positive entry per curve row, the
    convention shared with :mod:`absynth.thickness`. A Series is
    consumed by position, its index ignored.
    """
    curves = np.asarray(training_curves, dtype=float)
    if curves.ndim != 2 or curves.shape[0] == 0:
        raise ValueError("training curves must be a non-empty (n_films, n_wavelengths) matrix")
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


@dataclass(frozen=True)
class TrainedThicknessConditionedAutoEncoder:
    """A fitted thickness-conditioned autoencoder with everything needed to audit the fit.

    Attributes:
        model: The trained network with its final weights.
        standardizer: The per-wavelength scaling fitted on the
            training curves.
        thickness_standardizer: The scalar scaling fitted on the
            training thicknesses. Every thickness crossing between nm
            and the decoder's input passes through it.
        thickness_nm: The training films' measured thicknesses in nm.
            Generated films' thicknesses are drawn from these measured
            values, with replacement.
        config: The configuration that produced the fit.
        codes: The final codes of the training films, shape
            ``(n_training, n_latent)``, computed after training from
            the fitted weights. The encoder is curve-only, so these
            are the plain autoencoder's codes, and the population
            :meth:`sample` draws from.
        epoch_losses: Mean squared error per epoch, in standardized
            units squared, averaged over all training films.
    """

    model: ThicknessConditionedAutoEncoder
    standardizer: CurveStandardizer
    thickness_standardizer: ThicknessStandardizer
    thickness_nm: np.ndarray
    config: AutoEncoderConfig
    codes: np.ndarray
    epoch_losses: tuple[float, ...]

    def sample(self, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw films: measured thicknesses, code-Gaussian curves decoded at them.

        Each returned pair is a generated film: a thickness the
        campaign produced, and a curve of the kind the decoder builds
        at that thickness. The thicknesses are drawn from the measured
        training thicknesses with replacement, so the drawn thickness
        distribution is the measured one. The codes are drawn from a
        multivariate Gaussian fitted to the training codes, the
        KL-free generator of :class:`TrainedAutoEncoder`, and decoded
        at the drawn thicknesses. One numpy generator, seeded by
        ``seed``, drives both draws in a fixed order, so the draw is
        reproducible.

        Args:
            n_samples: Number of films to draw.
            seed: Seed for the numpy generator driving the thickness
                resampling and the code draws.

        Returns:
            Tuple ``(curves, thickness_nm)``: the drawn curve matrix
            in nm⁻¹, shape ``(n_samples, n_wavelengths)``, and the
            drawn thicknesses in nm, shape ``(n_samples,)``, aligned
            row by row.
        """
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, self.thickness_nm.size, size=n_samples)
        thickness_nm = self.thickness_nm[indices]
        thickness = torch.from_numpy(
            self.thickness_standardizer.transform(thickness_nm).astype(np.float32)
        )
        drawn_codes = rng.multivariate_normal(
            mean=self.codes.mean(axis=0),
            cov=np.cov(self.codes, rowvar=False),
            size=n_samples,
        )
        with torch.no_grad():
            standardized = self.model.decode(
                torch.from_numpy(drawn_codes.astype(np.float32)), thickness
            )
        return self.standardizer.inverse_transform(standardized.numpy()), thickness_nm


def train_thickness_conditioned_autoencoder(
    training_curves: np.ndarray, training_thickness_nm: np.ndarray, config: AutoEncoderConfig
) -> TrainedThicknessConditionedAutoEncoder:
    """Fit a thickness-conditioned autoencoder on training films.

    The same contract as :func:`train_autoencoder`: both standardizers
    are fitted on the training films only, held-out films must never
    reach this function, and a given configuration and (curves,
    thickness) table reproduce the run exactly. The configuration is
    the plain autoencoder's; conditioning lives in the architecture,
    and no configuration entry is added.

    Args:
        training_curves: Curve matrix in nm⁻¹, shape
            ``(n_training, n_wavelengths)``, complete on the working
            span. Row order defines the pairing with the thickness.
        training_thickness_nm: Film thickness in nm, one entry per
            row of ``training_curves``, in the same order. A Series is
            consumed by position (its index is ignored).
        config: Architecture and optimization numbers.

    Returns:
        A :class:`TrainedThicknessConditionedAutoEncoder`.

    Raises:
        ValueError: If the curves or thickness are empty or
            non-finite, the thickness is non-positive or misaligned
            with the curve rows, or a configuration value fails its
            checks, including a latent width that is not smaller than
            the grid.
        TrainingDivergenceError: If the loss becomes non-finite
            during training.
    """
    curves, thickness_nm = _checked_thickness_pair(training_curves, training_thickness_nm)
    standardizer = CurveStandardizer.fit(curves)
    thickness_standardizer = ThicknessStandardizer.fit(thickness_nm)
    standardized = torch.from_numpy(standardizer.transform(curves).astype(np.float32))
    standardized_thickness = torch.from_numpy(
        thickness_standardizer.transform(thickness_nm).astype(np.float32)
    )

    torch.manual_seed(config.seed)
    model = ThicknessConditionedAutoEncoder(
        n_wavelengths=curves.shape[1],
        n_hidden=config.n_hidden,
        n_latent=config.n_latent,
    )
    batches = DataLoader(
        TensorDataset(standardized, standardized_thickness),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    n_training = standardized.shape[0]
    epoch_losses = []
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        for curves_batch, thickness_batch in batches:
            optimizer.zero_grad()
            loss = mse_loss(model(curves_batch, thickness_batch), curves_batch)
            if not bool(torch.isfinite(loss)):
                raise TrainingDivergenceError(
                    f"training diverged: the loss is not finite at epoch {epoch} "
                    f"(seed {config.seed}, learning_rate {config.learning_rate}); "
                    "lower the learning rate or report the seed as unstable"
                )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * curves_batch.shape[0]
        epoch_losses.append(epoch_loss / n_training)
    with torch.no_grad():
        codes = model.encode(standardized).numpy()
    return TrainedThicknessConditionedAutoEncoder(
        model=model,
        standardizer=standardizer,
        thickness_standardizer=thickness_standardizer,
        thickness_nm=thickness_nm,
        config=config,
        codes=codes,
        epoch_losses=tuple(epoch_losses),
    )


@dataclass(frozen=True)
class VariationalAutoEncoderConfig(AutoEncoderConfig):
    """The variational training recipe, adding the KL weight to the autoencoder's.

    Attributes:
        kl_weight: Weight of the KL term in ``MSE + kl_weight · KL``.
            The exact Gaussian objective it approximates implies an
            observation variance of ``kl_weight · n_wavelengths / 2``
            in standardized units. The anchor value 2/n_wavelengths
            sets that implied variance to the standardized data's
            total spread; scan weights around it, and rely on the fit
            record, which states the implied variance either way.
    """

    kl_weight: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kl_weight <= 0:
            raise ValueError("kl_weight must be positive")


@dataclass(frozen=True)
class TrainedVariationalAutoEncoder:
    """A fitted variational autoencoder with its diagnostics.

    Attributes:
        model: The trained network with its final weights.
        standardizer: The per-wavelength scaling fitted on the training
            films.
        config: The configuration that produced the fit.
        epoch_reconstruction: Mean squared error per epoch, in
            standardized units squared, averaged over all training
            films.
        epoch_kl: Mean per-curve KL divergence per epoch, averaged over
            all training films.
        latent_kl: Per-dimension mean KL over the training films at the
            end of training, computed from the posterior parameters.
            Dimensions near zero are collapsed. The encoder no longer
            uses them, and how many dimensions survive is part of the
            reported record.
        implied_noise_variance: The observation variance in standardized
            units that the KL weight implies,
            ``kl_weight · n_wavelengths / 2``. Recorded with the fit
            so the objective's noise assumption is always stated.
    """

    model: VariationalAutoEncoder
    standardizer: CurveStandardizer
    config: VariationalAutoEncoderConfig
    epoch_reconstruction: tuple[float, ...]
    epoch_kl: tuple[float, ...]
    latent_kl: tuple[float, ...]
    implied_noise_variance: float

    def sample(self, n_samples: int, seed: int) -> np.ndarray:
        """Draw synthetic curves in nm⁻¹ from the prior.

        Codes are drawn from the standard normal, decoded, and moved
        back through the standardizer to nm⁻¹. Nothing is clipped.
        Values outside the measured range are model outputs to be
        evaluated, not errors to hide.

        Args:
            n_samples: Number of curves to draw.
            seed: Seed for a dedicated torch generator, so the draw is
                reproducible independently of training.

        Returns:
            Curve matrix in nm⁻¹, shape
            ``(n_samples, n_wavelengths)``.
        """
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            codes = torch.randn(n_samples, self.model.n_latent, generator=generator)
            standardized = self.model.decode(codes).numpy()
        return self.standardizer.inverse_transform(standardized)


def train_variational_autoencoder(
    training_curves: np.ndarray, config: VariationalAutoEncoderConfig
) -> TrainedVariationalAutoEncoder:
    """Fit a variational autoencoder on training curves.

    The same contract as :func:`train_autoencoder`: the standardizer is
    fitted on these curves only, held-out films must never reach this
    function, and a given configuration and curve matrix reproduce the
    run exactly.

    Args:
        training_curves: Curve matrix in nm⁻¹, shape
            ``(n_training, n_wavelengths)``, complete on the working
            span. Row order is irrelevant.
        config: Architecture and optimization numbers, including the
            KL weight.

    Returns:
        A :class:`TrainedVariationalAutoEncoder` with the loss history
        and the collapse diagnostic.

    Raises:
        ValueError: If the curves are empty or non-finite, if a
            wavelength column never varies, or if a configuration
            value fails its checks, including a latent width that is
            not smaller than the grid.
        TrainingDivergenceError: If the loss becomes non-finite or the
            code Gaussians degenerate during training.
    """
    curves = np.asarray(training_curves, dtype=float)
    if curves.ndim != 2 or curves.shape[0] == 0:
        raise ValueError("training curves must be a non-empty (n_curves, n_wavelengths) matrix")
    standardizer = CurveStandardizer.fit(curves)
    standardized = torch.from_numpy(standardizer.transform(curves).astype(np.float32))

    torch.manual_seed(config.seed)
    model = VariationalAutoEncoder(
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
    epoch_reconstruction = []
    epoch_kl = []
    for epoch in range(1, config.epochs + 1):
        epoch_mse = 0.0
        epoch_kl_sum = 0.0
        for batch in batches:
            optimizer.zero_grad()
            mean, std = model.encode(batch)
            if not (
                bool(torch.isfinite(mean).all())
                and bool(torch.isfinite(std).all())
                and bool((std > 0).all())
            ):
                raise TrainingDivergenceError(
                    f"training diverged: the code Gaussians are not finite with "
                    f"positive standard deviations at epoch {epoch} "
                    f"(seed {config.seed}, learning_rate {config.learning_rate}); "
                    "lower the learning rate or report the seed as unstable"
                )
            rebuilt = model.decode(model.sample_latent(mean, std))
            reconstruction = mse_loss(rebuilt, batch)
            kl = latent_kl_per_dimension(mean, std).sum(dim=1).mean()
            loss = reconstruction + config.kl_weight * kl
            loss.backward()
            optimizer.step()
            epoch_mse += float(reconstruction.detach()) * batch.shape[0]
            epoch_kl_sum += float(kl.detach()) * batch.shape[0]
        epoch_reconstruction.append(epoch_mse / n_training)
        epoch_kl.append(epoch_kl_sum / n_training)

    with torch.no_grad():
        mean, std = model.encode(standardized)
        final_kl = latent_kl_per_dimension(mean, std).mean(dim=0)
    return TrainedVariationalAutoEncoder(
        model=model,
        standardizer=standardizer,
        config=config,
        epoch_reconstruction=tuple(epoch_reconstruction),
        epoch_kl=tuple(epoch_kl),
        latent_kl=tuple(float(value) for value in final_kl),
        implied_noise_variance=config.kl_weight * curves.shape[1] / 2.0,
    )


@dataclass(frozen=True)
class TrainedThicknessConditionedVariationalAutoEncoder:
    """A fitted thickness-conditioned variational autoencoder.

    The reconstruction history, KL history, per-dimension KL
    diagnostic and implied noise variance mean here what they mean in
    :class:`TrainedVariationalAutoEncoder`. The thickness enters the
    decoder only, so the codes and their diagnostics are computed
    from the curves alone.

    Attributes:
        model: The trained network with its final weights.
        standardizer: The per-wavelength scaling fitted on the
            training curves.
        thickness_standardizer: The scalar scaling fitted on the
            training thicknesses.
        thickness_nm: The training films' measured thicknesses in nm.
            Generated films' thicknesses are drawn from these measured
            values, with replacement.
        config: The configuration that produced the fit.
        epoch_reconstruction: Mean squared error per epoch, in
            standardized units squared, averaged over all training
            films.
        epoch_kl: Mean per-curve KL divergence per epoch, averaged
            over all training films.
        latent_kl: Per-dimension mean KL over the training films at
            the end of training. Dimensions near zero are collapsed.
        implied_noise_variance: The observation variance in
            standardized units that the KL weight implies,
            ``kl_weight · n_wavelengths / 2``.
    """

    model: ThicknessConditionedVariationalAutoEncoder
    standardizer: CurveStandardizer
    thickness_standardizer: ThicknessStandardizer
    thickness_nm: np.ndarray
    config: VariationalAutoEncoderConfig
    epoch_reconstruction: tuple[float, ...]
    epoch_kl: tuple[float, ...]
    latent_kl: tuple[float, ...]
    implied_noise_variance: float

    def sample(self, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw films: prior codes decoded at measured thicknesses.

        Each returned pair is a generated film: a thickness the
        campaign produced, and the curve the decoder builds at that
        thickness for a code drawn from the standard normal, the
        prior. The codes use a dedicated torch generator; the
        thicknesses are drawn from the measured training thicknesses
        with replacement, using a numpy generator. Both generators are
        seeded from the one ``seed``, so the draw is reproducible.
        Nothing is clipped. Values outside the measured range are
        model outputs to be evaluated, not errors to hide.

        Args:
            n_samples: Number of films to draw.
            seed: Seed driving both the thickness resampling and the
                code draws.

        Returns:
            Tuple ``(curves, thickness_nm)``: the drawn curve matrix
            in nm⁻¹, shape ``(n_samples, n_wavelengths)``, and the
            drawn thicknesses in nm, shape ``(n_samples,)``, aligned
            row by row.
        """
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, self.thickness_nm.size, size=n_samples)
        thickness_nm = self.thickness_nm[indices]
        thickness = torch.from_numpy(
            self.thickness_standardizer.transform(thickness_nm).astype(np.float32)
        )
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            codes = torch.randn(n_samples, self.model.n_latent, generator=generator)
            standardized = self.model.decode(codes, thickness)
        return self.standardizer.inverse_transform(standardized.numpy()), thickness_nm


def train_thickness_conditioned_variational_autoencoder(
    training_curves: np.ndarray,
    training_thickness_nm: np.ndarray,
    config: VariationalAutoEncoderConfig,
) -> TrainedThicknessConditionedVariationalAutoEncoder:
    """Fit a thickness-conditioned variational autoencoder.

    The objective and its reading are those of
    :func:`train_variational_autoencoder`: reconstruction mean squared
    error plus ``kl_weight`` times the KL divergence of the code
    Gaussians against the standard-normal prior. The thickness enters
    the decoder only, so the code Gaussians, and the collapse
    diagnostic with them, are computed from the curves alone. The
    contract is the one this module keeps throughout: training-side
    standardizers only, and a given configuration and (curves,
    thickness) table reproduce the run exactly.

    Args:
        training_curves: Curve matrix in nm⁻¹, shape
            ``(n_training, n_wavelengths)``, complete on the working
            span. Row order defines the pairing with the thickness.
        training_thickness_nm: Film thickness in nm, one entry per
            row of ``training_curves``, in the same order. A Series is
            consumed by position (its index is ignored).
        config: Architecture and optimization numbers, including the
            KL weight.

    Returns:
        A :class:`TrainedThicknessConditionedVariationalAutoEncoder`
        with the loss history and the collapse diagnostic.

    Raises:
        ValueError: If the curves or thickness are empty or
            non-finite, the thickness is non-positive or misaligned
            with the curve rows, or a configuration value fails its
            checks, including a latent width that is not smaller than
            the grid.
        TrainingDivergenceError: If the loss becomes non-finite or the
            code Gaussians degenerate during training.
    """
    curves, thickness_nm = _checked_thickness_pair(training_curves, training_thickness_nm)
    standardizer = CurveStandardizer.fit(curves)
    thickness_standardizer = ThicknessStandardizer.fit(thickness_nm)
    standardized = torch.from_numpy(standardizer.transform(curves).astype(np.float32))
    standardized_thickness = torch.from_numpy(
        thickness_standardizer.transform(thickness_nm).astype(np.float32)
    )

    torch.manual_seed(config.seed)
    model = ThicknessConditionedVariationalAutoEncoder(
        n_wavelengths=curves.shape[1],
        n_hidden=config.n_hidden,
        n_latent=config.n_latent,
    )
    batches = DataLoader(
        TensorDataset(standardized, standardized_thickness),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    n_training = standardized.shape[0]
    epoch_reconstruction = []
    epoch_kl = []
    for epoch in range(1, config.epochs + 1):
        epoch_mse = 0.0
        epoch_kl_sum = 0.0
        for curves_batch, thickness_batch in batches:
            optimizer.zero_grad()
            mean, std = model.encode(curves_batch)
            if not (
                bool(torch.isfinite(mean).all())
                and bool(torch.isfinite(std).all())
                and bool((std > 0).all())
            ):
                raise TrainingDivergenceError(
                    f"training diverged: the code Gaussians are not finite with "
                    f"positive standard deviations at epoch {epoch} "
                    f"(seed {config.seed}, learning_rate {config.learning_rate}); "
                    "lower the learning rate or report the seed as unstable"
                )
            rebuilt = model.decode(model.sample_latent(mean, std), thickness_batch)
            reconstruction = mse_loss(rebuilt, curves_batch)
            kl = latent_kl_per_dimension(mean, std).sum(dim=1).mean()
            loss = reconstruction + config.kl_weight * kl
            loss.backward()
            optimizer.step()
            epoch_mse += float(reconstruction.detach()) * curves_batch.shape[0]
            epoch_kl_sum += float(kl.detach()) * curves_batch.shape[0]
        epoch_reconstruction.append(epoch_mse / n_training)
        epoch_kl.append(epoch_kl_sum / n_training)

    with torch.no_grad():
        mean, std = model.encode(standardized)
        final_kl = latent_kl_per_dimension(mean, std).mean(dim=0)
    return TrainedThicknessConditionedVariationalAutoEncoder(
        model=model,
        standardizer=standardizer,
        thickness_standardizer=thickness_standardizer,
        thickness_nm=thickness_nm,
        config=config,
        epoch_reconstruction=tuple(epoch_reconstruction),
        epoch_kl=tuple(epoch_kl),
        latent_kl=tuple(float(value) for value in final_kl),
        implied_noise_variance=config.kl_weight * curves.shape[1] / 2.0,
    )
