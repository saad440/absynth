"""Tests for thickness-aware generation — synthetic films, exact pairing."""

import numpy as np
import pandas as pd
import pytest

from absynth.thickness import fit_thickness_slice_generator, resample_joint


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


# --- thickness-slice conditional generator -------------------------------


def make_conditional_films():
    """8 films whose curve height falls with thickness, with no shape change.

    Curve i is ``(2.0 / t_i) · PATTERN``: a pure amplitude dependence on
    thickness, so the two slices have different mean heights and the
    conditional structure is known exactly.
    """
    thickness = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
    pattern = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]) / np.sqrt(8.0)
    return np.outer(2.0 / thickness, pattern), thickness


def test_slice_edges_are_training_quantiles():
    curves, thickness = make_conditional_films()
    generator = fit_thickness_slice_generator(curves, thickness, n_slices=2, n_components=1)
    assert generator.slice_edges_nm == pytest.approx(np.quantile(thickness, [0.0, 0.5, 1.0]))


def test_slices_hold_equal_counts_and_films():
    curves, thickness = make_conditional_films()
    generator = fit_thickness_slice_generator(curves, thickness, n_slices=2, n_components=1)
    assert generator.n_slices == 2
    assert generator.film_counts == (4, 4)
    assert generator.n_components == 1


def test_drawn_thicknesses_are_measured_values():
    curves, thickness = make_conditional_films()
    generator = fit_thickness_slice_generator(curves, thickness, n_slices=2, n_components=1)
    _, drawn_thickness = generator.sample(200, np.random.default_rng(0))
    assert set(drawn_thickness.tolist()) <= set(thickness.tolist())


def test_drawn_curve_height_follows_its_slice():
    # The conditional structure, checked end to end: pairs whose drawn
    # thickness falls in a slice have the mean curve that slice was
    # fitted on, and the thin-film slice is the taller one.
    curves, thickness = make_conditional_films()
    generator = fit_thickness_slice_generator(curves, thickness, n_slices=2, n_components=1)
    drawn_curves, drawn_thickness = generator.sample(8_000, np.random.default_rng(1))
    lower = drawn_thickness < generator.slice_edges_nm[1]
    for mask, expected in ((lower, curves[:4].mean(axis=0)), (~lower, curves[4:].mean(axis=0))):
        measured = drawn_curves[mask].mean(axis=0)
        assert measured == pytest.approx(expected, rel=0.05)
    assert np.abs(drawn_curves[lower]).mean() > np.abs(drawn_curves[~lower]).mean()


def test_sample_shapes():
    curves, thickness = make_conditional_films()
    generator = fit_thickness_slice_generator(curves, thickness, n_slices=2, n_components=1)
    drawn_curves, drawn_thickness = generator.sample(10, np.random.default_rng(2))
    assert drawn_curves.shape == (10, curves.shape[1])
    assert drawn_thickness.shape == (10,)


def test_same_seed_same_slice_draw():
    curves, thickness = make_conditional_films()
    generator = fit_thickness_slice_generator(curves, thickness, n_slices=2, n_components=1)
    first = generator.sample(30, np.random.default_rng(5))
    again = generator.sample(30, np.random.default_rng(5))
    assert np.array_equal(first[0], again[0])
    assert np.array_equal(first[1], again[1])


def test_zero_samples_raise_for_slice_generator():
    curves, thickness = make_conditional_films()
    generator = fit_thickness_slice_generator(curves, thickness, n_slices=2, n_components=1)
    with pytest.raises(ValueError, match="positive"):
        generator.sample(0, np.random.default_rng(0))


def test_thin_slice_refuses_to_fit():
    # 4 films per slice cannot support 2 PCA directions, which need
    # n_components + 1 = 3 films; the fit names the offending slice.
    curves, thickness = make_conditional_films()
    with pytest.raises(ValueError, match="holds 2 films"):
        fit_thickness_slice_generator(curves, thickness, n_slices=4, n_components=2)


def test_empty_slice_refuses_to_fit():
    # Repeated thickness values can leave a quantile slice empty; the
    # fit refuses rather than degrade silently.
    curves, _ = make_conditional_films()
    thickness = np.array([100.0, 100.0, 100.0, 100.0, 200.0])
    with pytest.raises(ValueError, match="holds 0 films"):
        fit_thickness_slice_generator(curves[:5], thickness, n_slices=2, n_components=1)


def test_nonpositive_thickness_raises_for_fit():
    curves, thickness = make_conditional_films()
    thickness[0] = -1.0
    with pytest.raises(ValueError, match="positive"):
        fit_thickness_slice_generator(curves, thickness, n_slices=2, n_components=1)
