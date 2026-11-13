"""Tests for the compact curve models — synthetic curves, tiny grids."""

import numpy as np
import pytest
import torch
from torch.nn.functional import mse_loss

from absynth.models import (
    AutoEncoder,
    CurveStandardizer,
    VariationalAutoEncoder,
    latent_kl_per_dimension,
)

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


# --- curve standardizer --------------------------------------------------


def test_transform_gives_zero_mean_unit_spread_columns():
    curves = make_curves()
    standardized = CurveStandardizer.fit(curves).transform(curves)
    assert standardized.mean(axis=0) == pytest.approx(np.zeros(GRID), abs=1e-12)
    assert standardized.std(axis=0) == pytest.approx(np.ones(GRID), abs=1e-12)


def test_inverse_transform_undoes_transform():
    curves = make_curves()
    standardizer = CurveStandardizer.fit(curves)
    round_trip = standardizer.inverse_transform(standardizer.transform(curves))
    assert round_trip == pytest.approx(curves)


def test_manual_standardizer_scales_by_stored_values():
    standardizer = CurveStandardizer(mean=np.array([1.0, 2.0]), std=np.array([2.0, 4.0]))
    standardized = standardizer.transform(np.array([[3.0, 6.0]]))
    assert standardized == pytest.approx(np.array([[1.0, 1.0]]))
    assert standardizer.inverse_transform(standardized) == pytest.approx(np.array([[3.0, 6.0]]))


def test_fit_rejects_zero_variance_column():
    curves = make_curves()
    curves[:, 3] = 7.0
    with pytest.raises(ValueError, match="zero variance"):
        CurveStandardizer.fit(curves)


def test_fit_rejects_nonfinite_or_empty_curves():
    curves = make_curves()
    curves[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        CurveStandardizer.fit(curves)
    with pytest.raises(ValueError, match="non-empty"):
        CurveStandardizer.fit(np.zeros((0, GRID)))


def test_transform_rejects_wrong_width():
    standardizer = CurveStandardizer(mean=np.ones(3), std=np.ones(3))
    with pytest.raises(ValueError, match="columns"):
        standardizer.transform(np.zeros((2, 4)))


def test_invalid_stored_statistics_raise():
    with pytest.raises(ValueError, match="positive"):
        CurveStandardizer(mean=np.ones(3), std=np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="equal-length"):
        CurveStandardizer(mean=np.ones(3), std=np.ones(2))


# --- variational autoencoder ---------------------------------------------


def make_vae(n_latent=3, n_hidden=8, seed=0):
    """A small variational autoencoder on the toy grid, freshly seeded."""
    torch.manual_seed(seed)
    return VariationalAutoEncoder(n_wavelengths=GRID, n_hidden=n_hidden, n_latent=n_latent)


def test_vae_encode_returns_positive_spreads():
    mean, std = make_vae().encode(as_float32(make_curves()))
    assert mean.shape == (6, 3)
    assert std.shape == (6, 3)
    assert bool((std > 0).all())


def test_vae_forward_returns_rebuilt_curves_and_parameters():
    rebuilt, mean, std = make_vae()(as_float32(make_curves()))
    assert rebuilt.shape == (6, GRID)
    assert mean.shape == (6, 3)
    assert std.shape == (6, 3)


def test_vae_same_seed_gives_identical_weights():
    first = make_vae(seed=5)
    second = make_vae(seed=5)
    for weight_first, weight_second in zip(first.parameters(), second.parameters()):
        assert torch.equal(weight_first, weight_second)


def test_vae_has_more_parameters_than_plain_autoencoder():
    # The encoder's last layer outputs twice as many numbers, so the
    # variational model stores more weights at equal widths.
    assert make_vae().parameter_count > make_model().parameter_count


def test_vae_degenerate_widths_raise():
    with pytest.raises(ValueError, match="bottleneck"):
        make_vae(n_latent=GRID)
    with pytest.raises(ValueError, match="n_wavelengths"):
        VariationalAutoEncoder(n_wavelengths=1, n_hidden=8, n_latent=1)


def test_sample_latent_is_reproducible_with_a_dedicated_generator():
    model = make_vae()
    mean, std = model.encode(as_float32(make_curves()))
    first = model.sample_latent(mean, std, generator=torch.Generator().manual_seed(9))
    second = model.sample_latent(mean, std, generator=torch.Generator().manual_seed(9))
    third = model.sample_latent(mean, std, generator=torch.Generator().manual_seed(10))
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_sample_latent_keeps_zero_spread_codes_fixed():
    model = make_vae()
    mean = torch.zeros(4, 3)
    drawn = model.sample_latent(mean, torch.zeros(4, 3), generator=torch.Generator().manual_seed(1))
    assert torch.equal(drawn, mean)


# --- latent KL divergence ------------------------------------------------


def test_latent_kl_is_zero_at_the_prior():
    measured = latent_kl_per_dimension(torch.zeros(5, 3), torch.ones(5, 3))
    assert measured == pytest.approx(torch.zeros(5, 3), abs=1e-12)


def test_latent_kl_matches_hand_values():
    measured = latent_kl_per_dimension(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 2.0, 0.5]]),
    )
    # The models compute in float32, so the anchors use relative tolerance.
    assert measured[0, 0] == pytest.approx(0.5, abs=1e-12)
    assert measured[0, 1] == pytest.approx(0.5 * (3.0 - np.log(4.0)), rel=1e-6)
    assert measured[0, 2] == pytest.approx(0.5 * (0.25 - np.log(0.25) - 1.0), rel=1e-6)


def test_latent_kl_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        latent_kl_per_dimension(torch.zeros(2, 3), torch.zeros(2, 4))
    with pytest.raises(ValueError, match="positive"):
        latent_kl_per_dimension(torch.zeros(2, 3), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="2-D"):
        latent_kl_per_dimension(torch.zeros(3), torch.zeros(3))
