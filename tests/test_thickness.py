"""Tests for thickness-aware generation — synthetic films, exact pairing."""

import numpy as np
import pandas as pd
import pytest

from absynth.thickness import resample_joint


def make_films(n_films=4, n_wavelengths=5):
    """Distinct curves with distinct row-aligned thicknesses."""
    curves = np.arange(n_films * n_wavelengths, dtype=float).reshape(n_films, n_wavelengths) / 8.0
    thickness = 100.0 + 37.0 * np.arange(n_films, dtype=float)
    return curves, thickness


def source_rows(drawn, training):
    """Index of the training row each drawn row equals (rows are distinct)."""
    training_rows = [row.tobytes() for row in training]
    return [training_rows.index(row.tobytes()) for row in drawn]


# --- joint resampling floor ---------------------------------------------


def test_drawn_pairs_are_measured_pairs():
    curves, thickness = make_films()
    drawn_curves, drawn_thickness = resample_joint(
        curves, thickness, 25, np.random.default_rng(0)
    )
    for position, source in enumerate(source_rows(drawn_curves, curves)):
        assert drawn_thickness[position] == thickness[source]


def test_both_films_appear_with_their_own_thickness():
    # 50 draws from 2 films: seeing only one film's curve/thickness
    # combination would need probability 2^-49; both appearing is
    # effectively sure.
    curves, thickness = make_films(n_films=2)
    drawn_curves, drawn_thickness = resample_joint(curves, thickness, 50, np.random.default_rng(1))
    assert len({row.tobytes() for row in drawn_curves}) == 2
    assert len(set(drawn_thickness.tolist())) == 2


def test_series_thickness_is_consumed_by_position():
    curves, thickness = make_films()
    shuffled = pd.Series(thickness, index=[3, 1, 0, 2])
    by_array = resample_joint(curves, thickness, 20, np.random.default_rng(4))
    by_series = resample_joint(curves, shuffled, 20, np.random.default_rng(4))
    assert np.array_equal(by_array[0], by_series[0])
    assert np.array_equal(by_array[1], by_series[1])


def test_shapes_and_count_are_respected():
    curves, thickness = make_films(n_films=7)
    for count in (1, 3, 100):
        drawn_curves, drawn_thickness = resample_joint(
            curves, thickness, count, np.random.default_rng(2)
        )
        assert drawn_curves.shape == (count, curves.shape[1])
        assert drawn_thickness.shape == (count,)


def test_same_seed_same_draw():
    curves, thickness = make_films()
    first = resample_joint(curves, thickness, 20, np.random.default_rng(7))
    again = resample_joint(curves, thickness, 20, np.random.default_rng(7))
    assert np.array_equal(first[0], again[0])
    assert np.array_equal(first[1], again[1])


def test_thickness_length_mismatch_raises():
    curves, thickness = make_films()
    with pytest.raises(ValueError, match="one entry per curve row"):
        resample_joint(curves, thickness[:-1], 5, np.random.default_rng(0))


def test_nonpositive_thickness_raises():
    curves, thickness = make_films()
    thickness[1] = 0.0
    with pytest.raises(ValueError, match="positive"):
        resample_joint(curves, thickness, 5, np.random.default_rng(0))


def test_nonfinite_thickness_raises():
    curves, thickness = make_films()
    thickness[2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        resample_joint(curves, thickness, 5, np.random.default_rng(0))


def test_nonfinite_curves_raise():
    curves, thickness = make_films()
    curves[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        resample_joint(curves, thickness, 5, np.random.default_rng(0))


def test_zero_samples_raises():
    curves, thickness = make_films()
    with pytest.raises(ValueError, match="positive"):
        resample_joint(curves, thickness, 0, np.random.default_rng(0))
