"""Tests for evaluation metrics — synthetic populations, exact values."""

import numpy as np
import pytest

from absynth.baselines import fit_pca_generator
from absynth.evaluation import (
    component_scores,
    covariance_record,
    mean_curve,
    mean_curve_deviation,
    memorization_record,
    nearest_training_distances,
    quantile_band,
)


def make_training():
    """Distinct training rows, exactly representable in binary floats."""
    return np.arange(10, dtype=float).reshape(2, 5) / 8.0


# --- memorization distances ---------------------------------------------


def test_copy_reports_zero_distance():
    training = make_training()
    samples = np.vstack([training[0], training[1]])
    distances = nearest_training_distances(samples, training)
    assert np.array_equal(distances, [0.0, 0.0])


def test_offset_sample_reports_exact_distance():
    # A single-column shift of one training row: the Euclidean distance
    # is exactly that shift.
    training = make_training()
    sample = training[0].copy()
    sample[2] += 0.5
    distances = nearest_training_distances(sample[None, :], training)
    assert distances[0] == pytest.approx(0.5)


def test_grid_length_mismatch_raises():
    training = make_training()
    with pytest.raises(ValueError, match="grids differ"):
        nearest_training_distances(np.zeros((1, 4)), training)


def test_memorization_record_values_are_exact():
    record = memorization_record(
        synthetic_distances=[0.0, 0.1, 0.2],
        heldout_distances=[0.3, 0.5],
    )
    assert record["synthetic_median"] == pytest.approx(0.1)
    # numpy's linear quantile: position (n - 1)·q into [0.0, 0.1, 0.2].
    assert record["synthetic_q10"] == pytest.approx(0.02)
    assert record["synthetic_q90"] == pytest.approx(0.18)
    assert record["heldout_median"] == pytest.approx(0.4)
    assert record["heldout_q10"] == pytest.approx(0.32)
    assert record["heldout_q90"] == pytest.approx(0.48)
    assert record["median_ratio"] == pytest.approx(0.25)


def test_memorization_record_rejects_nan():
    with pytest.raises(ValueError, match="finite"):
        memorization_record([0.0, np.nan], [0.3, 0.5])


# --- distribution --------------------------------------------------------


def test_mean_curve_of_constant_population():
    curves = np.full((3, 4), 0.02)
    assert np.array_equal(mean_curve(curves), np.full(4, 0.02))


def test_quantile_band_shape_and_flat_case():
    curves = np.full((5, 4), 0.02)
    band = quantile_band(curves, quantiles=(0.1, 0.5, 0.9))
    assert band.shape == (3, 4)
    assert np.array_equal(band, np.full((3, 4), 0.02))


def test_mean_curve_deviation_is_exact():
    reference = np.full((3, 4), 1.0)
    other = np.full((2, 4), 1.25)
    record = mean_curve_deviation(reference, other)
    assert record["max_abs"] == pytest.approx(0.25)
    assert record["mean_abs"] == pytest.approx(0.25)


def test_mean_curve_deviation_tracks_largest_wavelength():
    reference = np.zeros((2, 3))
    other = np.array([[0.1, 0.2, 0.4], [0.1, 0.2, 0.4]])
    record = mean_curve_deviation(reference, other)
    assert record["max_abs"] == pytest.approx(0.4)
    assert record["mean_abs"] == pytest.approx((0.1 + 0.2 + 0.4) / 3.0)


def test_mean_curve_deviation_grid_mismatch_raises():
    with pytest.raises(ValueError, match="grids differ"):
        mean_curve_deviation(np.zeros((2, 3)), np.zeros((2, 4)))


# --- variability ---------------------------------------------------------


def test_identical_populations_show_zero_difference():
    curves = np.arange(8, dtype=float).reshape(4, 2) / 8.0
    record = covariance_record(curves, curves.copy())
    assert record["covariance_relative_frobenius"] == 0.0
    assert record["correlation_relative_frobenius"] == 0.0


def test_amplitude_scaling_moves_covariance_but_not_correlation():
    # Variance scales as amplitude squared: doubling every curve's
    # amplitude quadruples the covariance (|4C - C| / |C| = 3) while
    # leaving correlation structure untouched — exactly why both
    # scales are reported.
    curves = np.arange(8, dtype=float).reshape(4, 2) / 8.0
    record = covariance_record(curves, 2.0 * curves)
    assert record["covariance_relative_frobenius"] == pytest.approx(3.0)
    assert record["correlation_relative_frobenius"] == 0.0


def test_covariance_record_rejects_single_curve():
    with pytest.raises(ValueError, match="at least 2"):
        covariance_record(np.zeros((1, 3)), np.zeros((4, 3)))


# --- generator coordinates ------------------------------------------------


def test_component_scores_reproduce_known_coefficients():
    # Two-axis training cloud (see test_baselines): fitted-generator
    # coordinates of the training rows are the constructed coefficients,
    # up to the SVD's per-axis sign freedom.
    p1 = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]) / np.sqrt(8.0)
    p2 = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]) / np.sqrt(8.0)
    a = 0.02 * np.array([1.0, -1.0, 1.0, -1.0])
    b = 0.005 * np.array([1.0, 1.0, -1.0, -1.0])
    training = np.full((4, 8), 0.5) + np.outer(a, p1) + np.outer(b, p2)
    generator = fit_pca_generator(training, n_components=2)

    scores = component_scores(training, generator.mean, generator.components)
    assert scores.shape == (4, 2)
    assert np.allclose(np.abs(scores[:, 0]), 0.02)
    assert np.allclose(np.abs(scores[:, 1]), 0.005)


def test_component_scores_shape_mismatch_raises():
    curves = np.zeros((3, 4))
    with pytest.raises(ValueError, match="mean has shape"):
        component_scores(curves, mean=np.zeros(3), components=np.zeros((1, 4)))
    with pytest.raises(ValueError, match="components have shape"):
        component_scores(curves, mean=np.zeros(4), components=np.zeros((1, 3)))
