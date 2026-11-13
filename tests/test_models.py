"""Tests for the compact curve models — synthetic curves, tiny grids."""

import numpy as np
import pytest
import torch
from torch.nn.functional import mse_loss

from absynth.models import AutoEncoder

GRID = 16


def make_model(n_latent=3, n_hidden=8, seed=0):
    """A small autoencoder on a 16-wavelength toy grid, freshly seeded."""
    torch.manual_seed(seed)
    return AutoEncoder(n_wavelengths=GRID, n_hidden=n_hidden, n_latent=n_latent)


def make_curves(n_rows=6):
    """Toy curves with the real population's low-dimensional structure.

    Row i = 0.5 + a_i·cos + b_i·sin on the grid, with coefficients a, b
    drawn uniform(−1, 1): all curves lie on a 2-D affine surface, so a
    bottleneck of 3 codes can represent them exactly once trained.
    """
    rng = np.random.default_rng(4)
    angle = np.linspace(0.0, np.pi, GRID)
    pattern_a = np.cos(angle)
    pattern_b = np.sin(angle)
    coefficients = rng.uniform(-1.0, 1.0, size=(n_rows, 2))
    return 0.5 + np.outer(coefficients[:, 0], pattern_a) + np.outer(coefficients[:, 1], pattern_b)


def as_float32(curves):
    return torch.from_numpy(np.asarray(curves, dtype=np.float32))


# --- construction -------------------------------------------------------


def test_same_seed_gives_identical_weights():
    first = make_model(seed=7)
    second = make_model(seed=7)
    for weight_first, weight_second in zip(first.parameters(), second.parameters()):
        assert torch.equal(weight_first, weight_second)


def test_different_seeds_give_different_weights():
    first = make_model(seed=7)
    second = make_model(seed=8)
    assert not torch.equal(first.encoder[0].weight, second.encoder[0].weight)


def test_parameter_count_matches_hand_count():
    # 16→8→3 up and 3→8→16 down: (16·8+8) + (8·3+3) + (3·8+8) + (8·16+16).
    model = make_model()
    assert model.parameter_count == 136 + 27 + 32 + 144


@pytest.mark.parametrize("n_latent", [0, GRID])
def test_latent_width_must_be_a_bottleneck(n_latent):
    with pytest.raises(ValueError, match="bottleneck"):
        AutoEncoder(n_wavelengths=GRID, n_hidden=8, n_latent=n_latent)


def test_degenerate_widths_raise():
    with pytest.raises(ValueError, match="n_wavelengths"):
        AutoEncoder(n_wavelengths=1, n_hidden=8, n_latent=1)
    with pytest.raises(ValueError, match="n_hidden"):
        AutoEncoder(n_wavelengths=GRID, n_hidden=0, n_latent=3)


# --- shapes and plumbing -------------------------------------------------


def test_forward_preserves_curve_shape():
    curves = make_curves(n_rows=5)
    rebuilt = make_model()(as_float32(curves))
    assert rebuilt.shape == (5, GRID)


def test_encode_compresses_to_latent_width():
    codes = make_model().encode(as_float32(make_curves()))
    assert codes.shape == (6, 3)


def test_decode_restores_grid_width():
    model = make_model()
    codes = model.encode(as_float32(make_curves()))
    assert model.decode(codes).shape == (6, GRID)


def test_forward_is_decode_after_encode():
    curves = as_float32(make_curves())
    model = make_model()
    direct = model(curves)
    round_trip = model.decode(model.encode(curves))
    assert torch.allclose(direct, round_trip)


def test_numpy_and_tensor_inputs_agree():
    curves = make_curves()
    model = make_model()
    from_numpy = model.encode(as_float32(curves))
    from_tensor = model.encode(torch.tensor(curves.tolist()))
    assert torch.equal(from_numpy, from_tensor)


# --- input discipline ----------------------------------------------------


def test_wrong_grid_width_raises():
    curves = make_curves()
    with pytest.raises(ValueError, match="columns"):
        make_model().encode(as_float32(np.delete(curves, 0, axis=1)))


def test_wrong_latent_width_raises():
    codes = np.zeros((4, 5))
    with pytest.raises(ValueError, match="columns"):
        make_model().decode(as_float32(codes))


def test_non_2d_input_raises():
    with pytest.raises(ValueError, match="matrix"):
        make_model().encode(as_float32(np.zeros(GRID)))


def test_empty_input_raises():
    with pytest.raises(ValueError, match="non-empty"):
        make_model().encode(as_float32(np.zeros((0, GRID))))


def test_nonfinite_input_raises():
    curves = make_curves()
    curves[2, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        make_model().encode(as_float32(curves))


# --- learnability --------------------------------------------------------


def test_training_reduces_reconstruction_error():
    # A minimal training loop here only to prove gradients flow and the
    # bottleneck can represent the toy surface; the training
    # orchestration itself lives outside this module.
    model = make_model(n_latent=3, n_hidden=8, seed=3)
    curves = as_float32(make_curves())
    initial = float(mse_loss(model(curves), curves).detach())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        optimizer.zero_grad()
        value = mse_loss(model(curves), curves)
        value.backward()
        optimizer.step()
    assert float(value.detach()) < 0.05 * initial
