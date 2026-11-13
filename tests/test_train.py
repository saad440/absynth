"""Tests for the autoencoder training loop — synthetic curves, tiny runs."""

import numpy as np
import pytest
import torch

from absynth.train import AutoEncoderConfig, train_autoencoder

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
