"""Tests for statistical baselines — synthetic curves, exact expectations."""

import numpy as np
import pytest

from absynth.baselines import bootstrap_resample, fit_pca_generator

# Two orthonormal, non-constant patterns on 8 wavelengths: their dot
# product is 0, and neither is proportional to the constant vector a
# centring would remove.
PATTERN_1 = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]) / np.sqrt(8.0)
PATTERN_2 = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]) / np.sqrt(8.0)


def make_training(n_rows=4, n_cols=5):
    """Distinct training rows, exactly representable in binary floats."""
    return np.arange(n_rows * n_cols, dtype=float).reshape(n_rows, n_cols) / 8.0


def make_two_axis_curves(scale_1, scale_2):
    """Training rows with exactly two variability directions.

    Row i = mean + a_i·PATTERN_1 + b_i·PATTERN_2, with the coefficient
    lists a = scale_1·(1,-1,1,-1) and b = scale_2·(1,1,-1,-1): both
    centred, mutually orthogonal, so the centred cloud has exactly two
    nonzero directions with score spreads scale_1·2/√3 and
    scale_2·2/√3 (ddof=1 over 4 rows).
    """
    mean = np.full(8, 0.5)
    a = scale_1 * np.array([1.0, -1.0, 1.0, -1.0])
    b = scale_2 * np.array([1.0, 1.0, -1.0, -1.0])
    return mean + np.outer(a, PATTERN_1) + np.outer(b, PATTERN_2)


# --- bootstrap floor ---------------------------------------------------


def test_drawn_rows_are_training_rows():
    training = make_training()
    drawn = bootstrap_resample(training, n_samples=25, rng=np.random.default_rng(0))
    training_rows = {row.tobytes() for row in training}
    assert len(drawn) == 25
    for row in drawn:
        assert row.tobytes() in training_rows


def test_draws_replace_slips():
    # 50 draws from 2 rows: ending with a single distinct row would
    # need probability 2^-49; both rows appearing is effectively sure.
    training = make_training(n_rows=2)
    drawn = bootstrap_resample(training, n_samples=50, rng=np.random.default_rng(1))
    assert len({row.tobytes() for row in drawn}) == 2


def test_requested_count_is_respected():
    training = make_training(n_rows=7)
    for count in (1, 3, 100):
        drawn = bootstrap_resample(training, n_samples=count, rng=np.random.default_rng(2))
        assert drawn.shape == (count, training.shape[1])


def test_same_seed_same_draw():
    training = make_training()
    first = bootstrap_resample(training, 20, np.random.default_rng(7))
    again = bootstrap_resample(training, 20, np.random.default_rng(7))
    assert np.array_equal(first, again)


def test_nonfinite_training_raises():
    training = make_training()
    training[1, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        bootstrap_resample(training, 5, np.random.default_rng(0))


def test_zero_samples_raises():
    with pytest.raises(ValueError, match="positive"):
        bootstrap_resample(make_training(), 0, np.random.default_rng(0))


# --- PCA generator ------------------------------------------------------


def test_scales_and_directions_match_construction():
    # The analytic expectation: spreads 2/√3 · scale, directions at the
    # two patterns up to the SVD's sign freedom.
    training = make_two_axis_curves(scale_1=0.02, scale_2=0.005)
    generator = fit_pca_generator(training, n_components=2)
    expected_scales = (2.0 / np.sqrt(3.0)) * np.array([0.02, 0.005])
    assert generator.scales == pytest.approx(expected_scales, rel=1e-12)
    for component, pattern in zip(generator.components, (PATTERN_1, PATTERN_2)):
        assert abs(component @ pattern) == pytest.approx(1.0, abs=1e-12)


def test_generator_mean_is_training_mean():
    training = make_two_axis_curves(scale_1=0.02, scale_2=0.005)
    generator = fit_pca_generator(training, n_components=2)
    assert generator.mean == pytest.approx(training.mean(axis=0))


def test_generated_cloud_reproduces_training_covariance():
    # The model's central claim: independent Gaussian draws with the
    # measured per-direction spreads rebuild the training covariance.
    training = make_two_axis_curves(scale_1=0.02, scale_2=0.005)
    generator = fit_pca_generator(training, n_components=2)
    samples = generator.sample(40_000, np.random.default_rng(5))
    expected = np.cov(training, rowvar=False)
    measured = np.cov(samples, rowvar=False)
    relative_error = np.abs(measured - expected).max() / np.abs(expected).max()
    assert relative_error < 0.05


def test_sample_shape_and_component_count():
    training = make_two_axis_curves(scale_1=0.02, scale_2=0.005)
    generator = fit_pca_generator(training, n_components=2)
    samples = generator.sample(11, np.random.default_rng(3))
    assert samples.shape == (11, 8)
    assert generator.n_components == 2


def test_same_seed_same_samples():
    generator = fit_pca_generator(make_two_axis_curves(0.02, 0.005), n_components=2)
    assert np.array_equal(
        generator.sample(10, np.random.default_rng(9)),
        generator.sample(10, np.random.default_rng(9)),
    )


def test_too_many_components_raises():
    training = make_two_axis_curves(scale_1=0.02, scale_2=0.005)  # 4 curves
    with pytest.raises(ValueError, match="n_components"):
        fit_pca_generator(training, n_components=4)  # max is min(3, 8) = 3


def test_nonfinite_training_raises_for_pca():
    training = make_two_axis_curves(scale_1=0.02, scale_2=0.005)
    training[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_pca_generator(training, n_components=2)


def test_zero_samples_raises_for_pca():
    generator = fit_pca_generator(make_two_axis_curves(0.02, 0.005), n_components=2)
    with pytest.raises(ValueError, match="positive"):
        generator.sample(0, np.random.default_rng(0))
