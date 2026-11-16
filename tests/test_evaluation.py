"""Tests for evaluation metrics — synthetic populations, exact values."""

import numpy as np
import pytest

from absynth.baselines import fit_pca_generator
from absynth.evaluation import (
    arithmetic_coupling_pairs,
    component_scores,
    coupling_record,
    covariance_record,
    mean_curve,
    mean_curve_deviation,
    memorization_record,
    nearest_training_distances,
    quantile_band,
    scaling_exponent,
    sensitivity_decomposition,
    thickness_slice_errors,
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


# --- coupling and slice decomposition --------------------------------------


def test_coupling_is_exact_for_a_perfect_line():
    # Pearson correlation is linear: an exact linear relation scores
    # exactly -1, while the curved definitional relation q ∝ 1/t
    # scores strongly negative but short of it.
    thickness = np.array([100.0, 200.0, 300.0, 400.0])
    linear = 800.0 - 2.0 * thickness
    assert coupling_record(linear, thickness)["pearson_r"] == pytest.approx(-1.0)
    hyperbolic = 50_000.0 / thickness
    hyperbolic_r = coupling_record(hyperbolic, thickness)["pearson_r"]
    assert -1.0 < hyperbolic_r < -0.9


def test_independent_pairing_collapses_the_coupling():
    # Erasing the pairing (permutation) while keeping both marginals
    # drives the correlation to 0: the independent-marginal failure,
    # in numbers. The measured value stays strongly negative even
    # though 1/t is curved and Pearson is linear, which attenuates it.
    # 20,000 draws make the permuted spread about 0.007, so the 0.05
    # bound is several sigma wide.
    rng = np.random.default_rng(3)
    thickness = np.linspace(100.0, 1900.0, 20_000)
    summaries = 0.5 / thickness
    measured = coupling_record(summaries, thickness)["pearson_r"]
    erased = coupling_record(rng.permutation(summaries), thickness)["pearson_r"]
    assert measured < -0.7
    assert abs(erased) < 0.05


def test_coupling_rejects_invalid_input():
    with pytest.raises(ValueError, match="one thickness per film summary"):
        coupling_record(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        coupling_record(np.array([1.0, np.nan]), np.array([100.0, 200.0]))
    with pytest.raises(ValueError, match="constant"):
        coupling_record(np.array([0.5, 0.5, 0.5]), np.array([100.0, 200.0, 300.0]))


def test_arithmetic_coupling_of_constant_absorbance_is_the_hyperbolic_coupling():
    # ∫A held fixed while t varies freely reduces the reference
    # population's summary to c/t. Its coupling then reproduces the
    # pure-definition value, strongly negative over a wide range, and
    # an independent recomputation from the same generator state
    # returns it exactly.
    rng = np.random.default_rng(7)
    thickness = np.linspace(100.0, 1900.0, 400)
    reference_q, reference_t = arithmetic_coupling_pairs(
        np.full(400, 0.6), thickness, 20_000, rng
    )
    record = coupling_record(reference_q, reference_t)["pearson_r"]
    assert record == pytest.approx(
        coupling_record(0.6 / reference_t, reference_t)["pearson_r"]
    )
    assert -0.9 < record < -0.6


def test_arithmetic_coupling_turns_exact_proportionality_strongly_negative():
    # Exact proportionality, ∫A ∝ t, gives r = +1 with t; Beer–Lambert
    # predicts exactly this for the integrated summary. Pairing the
    # same summaries with thicknesses independently and converting
    # back through q = A/t produces a strongly negative coupling from
    # arithmetic alone. No physics is involved anywhere.
    rng = np.random.default_rng(11)
    thickness = np.linspace(100.0, 1900.0, 400)
    summaries = 0.6 * thickness
    assert coupling_record(summaries, thickness)["pearson_r"] == pytest.approx(1.0)
    reference_q, reference_t = arithmetic_coupling_pairs(
        summaries, thickness, 20_000, rng
    )
    assert coupling_record(reference_q, reference_t)["pearson_r"] < -0.5


def test_arithmetic_coupling_rejects_invalid_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="one thickness per absorbance summary"):
        arithmetic_coupling_pairs(np.zeros(3), np.zeros(2), 10, rng)
    with pytest.raises(ValueError, match="finite"):
        arithmetic_coupling_pairs(np.array([1.0, np.nan]), np.array([100.0, 200.0]), 10, rng)
    with pytest.raises(ValueError, match="constant"):
        arithmetic_coupling_pairs(np.array([0.5, 0.5]), np.array([100.0, 200.0]), 10, rng)
    with pytest.raises(ValueError, match="positive to convert"):
        arithmetic_coupling_pairs(np.array([0.5, 0.6]), np.array([100.0, 0.0]), 10, rng)
    with pytest.raises(ValueError, match="n_samples must be positive"):
        arithmetic_coupling_pairs(np.array([0.5, 0.6]), np.array([100.0, 200.0]), 0, rng)


def test_scaling_exponent_recovers_an_exact_power_law():
    # ∫A = c·t^β is linear in log-log: the fit reads β back to
    # machine precision, and the bootstrap band of an exact relation
    # contains it.
    thickness = np.array([100.0, 250.0, 500.0, 1000.0, 1900.0])
    summaries = 0.6 * thickness**0.5
    record = scaling_exponent(summaries, thickness, np.random.default_rng(5), n_bootstrap=500)
    assert record["slope"] == pytest.approx(0.5, abs=1e-9)
    low, high = record["slope_ci95"]
    assert low <= 0.5 <= high


def test_scaling_exponent_is_reproducible_and_brackets_its_fit():
    # Multiplicative film-to-film scatter around exact proportionality
    # keeps the fitted exponent near 1; the same generator state
    # reproduces the band exactly, and the band brackets the fit.
    thickness = np.linspace(150.0, 1900.0, 60)
    scatter = np.random.default_rng(2).uniform(0.95, 1.05, 60)
    summaries = 0.6 * thickness * scatter
    first = scaling_exponent(summaries, thickness, np.random.default_rng(5), n_bootstrap=400)
    second = scaling_exponent(summaries, thickness, np.random.default_rng(5), n_bootstrap=400)
    assert first == second
    low, high = first["slope_ci95"]
    assert low < first["slope"] < high


def test_scaling_exponent_rejects_invalid_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="one thickness per absorbance summary"):
        scaling_exponent(np.zeros(3), np.zeros(2), rng)
    with pytest.raises(ValueError, match="strictly positive"):
        scaling_exponent(np.array([0.6, -0.1]), np.array([100.0, 200.0]), rng)
    with pytest.raises(ValueError, match="strictly positive"):
        scaling_exponent(np.array([0.6, 0.7]), np.array([100.0, 0.0]), rng)
    with pytest.raises(ValueError, match="constant"):
        scaling_exponent(np.array([0.6, 0.7]), np.array([100.0, 100.0]), rng)
    with pytest.raises(ValueError, match="n_bootstrap must be positive"):
        scaling_exponent(np.array([0.6, 0.7]), np.array([100.0, 200.0]), rng, n_bootstrap=0)


# --- sensitivity direction ------------------------------------------------


def test_sensitivity_along_the_mean_curve_is_pure_amplitude():
    grid = np.linspace(300.0, 900.0, 31)
    reference = 0.02 + 0.01 * np.cos(np.linspace(0.0, np.pi, 31))
    delta = np.array([3.0 * reference, 0.5 * reference])
    record = sensitivity_decomposition(delta, reference, grid)
    assert record["amplitude_fraction"] == pytest.approx(1.0)
    assert record["amplitude_fraction_per_code_mean"] == pytest.approx(1.0)
    assert record["residual_wavelength_pearson_r"] == 0.0


def test_sensitivity_residual_reads_its_wavelength_direction():
    reference = np.full(31, 0.02)
    trend = np.linspace(1.0, -1.0, 31)
    delta = np.array([2.0 * reference, 0.001 * trend + 0.002 * reference])
    record = sensitivity_decomposition(delta, reference, np.linspace(300.0, 900.0, 31))
    # The first row is pure amplitude; the second mixes amplitude with
    # a residual that shrinks along the wavelength grid, the
    # scattering-like direction.
    assert 0.0 < record["amplitude_fraction"] < 1.0
    assert 0.0 < record["amplitude_fraction_per_code_mean"] < 1.0
    assert record["residual_wavelength_pearson_r"] == pytest.approx(-1.0)


def test_sensitivity_decomposition_rejects_invalid_input():
    grid = np.linspace(300.0, 900.0, 4)
    with pytest.raises(ValueError, match="reference curve has shape"):
        sensitivity_decomposition(np.zeros((2, 4)), np.zeros(3), grid)
    with pytest.raises(ValueError, match="wavelengths have shape"):
        sensitivity_decomposition(np.zeros((2, 4)), np.zeros(4), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        sensitivity_decomposition(np.array([[0.0, np.nan, 0.0, 0.0]]), np.full(4, 0.02), grid)
    with pytest.raises(ValueError, match="no direction"):
        sensitivity_decomposition(np.zeros((2, 4)), np.zeros(4), grid)


def test_slice_errors_are_exact_slice_means():
    errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    thickness = np.array([110.0, 130.0, 150.0, 170.0, 190.0, 210.0])
    records = thickness_slice_errors(errors, thickness, [100.0, 160.0, 220.0])
    assert [record["slice"] for record in records] == [0, 1]
    assert records[0]["edges_nm"] == [100.0, 160.0]
    assert records[0]["n_films"] == 3
    assert records[0]["mean_error"] == pytest.approx(2.0)
    assert records[1]["edges_nm"] == [160.0, 220.0]
    assert records[1]["n_films"] == 3
    assert records[1]["mean_error"] == pytest.approx(5.0)


def test_slice_edge_film_joins_the_upper_slice():
    # Slices are half-open, lower edge included: the film sitting on
    # the boundary belongs above it.
    errors = np.array([1.0, 2.0])
    thickness = np.array([149.0, 150.0])
    records = thickness_slice_errors(errors, thickness, [100.0, 150.0, 200.0])
    assert records[0]["n_films"] == 1
    assert records[1]["n_films"] == 1
    assert records[1]["mean_error"] == pytest.approx(2.0)


def test_slice_errors_reject_films_outside_the_edges():
    with pytest.raises(ValueError, match="outside the slice edges"):
        thickness_slice_errors(np.array([1.0, 2.0]), np.array([90.0, 150.0]), [100.0, 200.0])


def test_slice_errors_reject_an_empty_slice():
    # Two films at the extremes leave the middle slice unfilled.
    errors = np.array([1.0, 2.0])
    thickness = np.array([110.0, 190.0])
    with pytest.raises(ValueError, match="holds no films"):
        thickness_slice_errors(errors, thickness, [100.0, 140.0, 180.0, 220.0])


def test_slice_errors_reject_non_ascending_edges():
    errors = np.array([1.0, 2.0])
    thickness = np.array([110.0, 190.0])
    with pytest.raises(ValueError, match="ascending"):
        thickness_slice_errors(errors, thickness, [100.0, 200.0, 150.0])


def test_slice_errors_reject_mismatched_or_nonfinite_input():
    with pytest.raises(ValueError, match="one thickness per film error"):
        thickness_slice_errors(np.zeros(3), np.zeros(2), [100.0, 200.0])
    with pytest.raises(ValueError, match="finite"):
        thickness_slice_errors(np.array([1.0, np.nan]), np.array([110.0, 190.0]), [100.0, 200.0])
