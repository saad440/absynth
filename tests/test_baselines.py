"""Tests for statistical baselines — synthetic curves, exact expectations."""

import numpy as np
import pytest

from absynth.baselines import bootstrap_resample


def make_training(n_rows=4, n_cols=5):
    """Distinct training rows, exactly representable in binary floats."""
    return np.arange(n_rows * n_cols, dtype=float).reshape(n_rows, n_cols) / 8.0


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
