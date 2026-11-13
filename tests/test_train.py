"""Tests for the autoencoder training loop — synthetic curves, tiny runs."""

import numpy as np
import pytest
import torch

from absynth.train import (
    AutoEncoderConfig,
    VariationalAutoEncoderConfig,
    train_autoencoder,
    train_variational_autoencoder,
)

GRID = 16
N_CURVES = 32


def make_curves(n_rows=N_CURVES):
    """Toy curves on a 2-D affine surface (same construction as test_models)."""
    rng = np.random.default_rng(11)
    angle = np.linspace(0.0, np.pi, GRID)
    pattern_a = np.cos(angle)
    pattern_b = np.sin(angle)
    coefficients = rng.uniform(-1.0, 1.0, size=(n_rows, 2))
    return 0.5 + np.outer(coefficients[:, 0], pattern_a) + np.outer(coefficients[:, 1], pattern_b)


def make_config(**overrides):
    """A small default configuration with per-test overrides."""
    values = {
        "n_hidden": 16,
        "n_latent": 2,
        "epochs": 5,
        "learning_rate": 5e-3,
        "batch_size": 8,
        "seed": 0,
    }
    values.update(overrides)
    return AutoEncoderConfig(**values)


# --- reproducibility -----------------------------------------------------


def test_same_config_reproduces_run():
    curves = make_curves()
    first = train_autoencoder(curves, make_config())
    second = train_autoencoder(curves, make_config())
    for weight_first, weight_second in zip(first.model.parameters(), second.model.parameters()):
        assert torch.equal(weight_first, weight_second)
    assert first.epoch_losses == second.epoch_losses


def test_different_seed_changes_weights():
    curves = make_curves()
    first = train_autoencoder(curves, make_config(seed=0))
    second = train_autoencoder(curves, make_config(seed=1))
    assert not torch.equal(first.model.encoder[0].weight, second.model.encoder[0].weight)


# --- fitted quantities ---------------------------------------------------


def test_standardizer_matches_training_statistics():
    curves = make_curves()
    result = train_autoencoder(curves, make_config())
    assert result.standardizer.mean == pytest.approx(curves.mean(axis=0))


def test_model_matches_configured_shape():
    curves = make_curves()
    result = train_autoencoder(curves, make_config(n_latent=3))
    assert result.model.n_wavelengths == GRID
    codes = result.model.encode(torch.from_numpy(curves.astype(np.float32)))
    assert codes.shape == (N_CURVES, 3)


def test_epoch_loss_record_is_complete():
    result = train_autoencoder(make_curves(), make_config(epochs=7))
    assert len(result.epoch_losses) == 7
    assert all(loss > 0 for loss in result.epoch_losses)


def test_loss_decreases_over_training():
    # The toy surface is exactly representable with a latent width of 2,
    # so a converged run leaves the loss far below its starting value.
    curves = make_curves()
    result = train_autoencoder(curves, make_config(epochs=120, learning_rate=5e-3))
    assert result.epoch_losses[-1] < 0.5 * result.epoch_losses[0]


# --- input and configuration discipline ----------------------------------


def test_nonfinite_curves_raise():
    curves = make_curves()
    curves[1, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        train_autoencoder(curves, make_config())


def test_empty_curves_raise():
    with pytest.raises(ValueError, match="non-empty"):
        train_autoencoder(np.zeros((0, GRID)), make_config())


def test_latent_width_must_be_a_bottleneck():
    with pytest.raises(ValueError, match="bottleneck"):
        train_autoencoder(make_curves(), make_config(n_latent=GRID))


@pytest.mark.parametrize(
    "override, message",
    [
        ({"n_hidden": 0}, "widths"),
        ({"n_latent": 0}, "widths"),
        ({"epochs": 0}, "epochs"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_invalid_configurations_raise(override, message):
    with pytest.raises(ValueError, match=message):
        make_config(**override)


# --- variational autoencoder ---------------------------------------------


def make_vae_config(**overrides):
    """A small variational default with per-test overrides.

    The KL weight is the anchor value 2/n_wavelengths.
    """
    values = {
        "n_hidden": 16,
        "n_latent": 2,
        "epochs": 5,
        "learning_rate": 5e-3,
        "batch_size": 8,
        "seed": 0,
        "kl_weight": 2.0 / GRID,
    }
    values.update(overrides)
    return VariationalAutoEncoderConfig(**values)


def test_vae_same_config_reproduces_run():
    curves = make_curves()
    first = train_variational_autoencoder(curves, make_vae_config())
    second = train_variational_autoencoder(curves, make_vae_config())
    for weight_first, weight_second in zip(first.model.parameters(), second.model.parameters()):
        assert torch.equal(weight_first, weight_second)
    assert first.epoch_reconstruction == second.epoch_reconstruction
    assert first.epoch_kl == second.epoch_kl


def test_vae_history_records_both_loss_terms():
    result = train_variational_autoencoder(make_curves(), make_vae_config(epochs=6))
    assert len(result.epoch_reconstruction) == 6
    assert len(result.epoch_kl) == 6
    assert all(value > 0 for value in result.epoch_reconstruction)
    assert all(value >= 0 for value in result.epoch_kl)


def test_implied_noise_variance_matches_formula():
    result = train_variational_autoencoder(make_curves(), make_vae_config(kl_weight=0.5))
    assert result.implied_noise_variance == pytest.approx(0.5 * GRID / 2.0)


def test_latent_kl_diagnostic_is_per_dimension():
    result = train_variational_autoencoder(make_curves(), make_vae_config(n_latent=3))
    assert len(result.latent_kl) == 3
    assert all(value >= 0.0 for value in result.latent_kl)


def test_weaker_kl_weight_lets_codes_drift_from_the_prior():
    # The rate side of the rate-distortion trade-off: a weak KL term
    # leaves the code Gaussians free to carry whatever curve detail
    # reconstruction wants (large KL), while the anchor weight holds
    # them closer to the prior (smaller KL). Fixed seeds make the
    # comparison deterministic. The opposite pressure, collapse, shows
    # up as individual dimensions near zero in latent_kl.
    curves = make_curves()
    anchored = train_variational_autoencoder(curves, make_vae_config(epochs=150))
    weak = train_variational_autoencoder(curves, make_vae_config(epochs=150, kl_weight=1e-4))
    assert sum(weak.latent_kl) > sum(anchored.latent_kl)


def test_vae_training_reduces_reconstruction_error():
    curves = make_curves()
    result = train_variational_autoencoder(curves, make_vae_config(epochs=120, learning_rate=5e-3))
    assert result.epoch_reconstruction[-1] < 0.5 * result.epoch_reconstruction[0]


def test_vae_sample_is_reproducible_and_well_shaped():
    result = train_variational_autoencoder(make_curves(), make_vae_config(epochs=3))
    first = result.sample(10, seed=3)
    again = result.sample(10, seed=3)
    other = result.sample(10, seed=4)
    assert first.shape == (10, GRID)
    assert np.isfinite(first).all()
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_invalid_kl_weight_raises():
    with pytest.raises(ValueError, match="kl_weight"):
        make_vae_config(kl_weight=0.0)
