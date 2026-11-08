"""Tests for exploratory curve statistics — synthetic, hand-checkable."""

import numpy as np
import pandas as pd
import pytest

from absynth.explore import (
    complete_prefix_mask,
    nearest_neighbour_distances,
    pca_explained_variance,
    pca_scores,
    score_descriptor_correlations,
    thickness_stability,
)


def test_pca_spectrum_collapses_a_line():
    # Twelve points along one line, with tiny off-line jitter: one
    # direction carries (almost) all the variance.
    line = np.linspace(0.0, 1.0, 12)[:, None] * np.array([[1.0, 2.0, 3.0, 4.0]])
    values = line + np.array([0.001, -0.001, 0.002, -0.002])
    ratios = pca_explained_variance(values)

    assert ratios[0] > 0.999
    assert (ratios[1:] < 0.001).all()
    assert ratios.sum() == pytest.approx(1.0)


def test_pca_truncates_the_spectrum():
    values = np.linspace(0.0, 1.0, 6)[:, None] * np.array([[1.0, 2.0]])
    full = pca_explained_variance(values)
    # the spectrum has min(n_samples, n_features) entries, here 2
    assert len(full) == 2
    assert len(pca_explained_variance(values, max_components=1)) == 1
    # descending order
    assert (np.diff(full) <= 0).all()


def test_pca_rejects_unmeasured_points():
    values = np.ones((3, 4))
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete subset"):
        pca_explained_variance(values)


def test_nearest_neighbour_excludes_self():
    # Three collinear points: distances to the closest *other* point.
    values = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    distances, neighbours = nearest_neighbour_distances(values)
    assert distances == pytest.approx([1.0, 1.0, 2.0])
    assert neighbours.tolist() == [1, 0, 1]


def test_nearest_neighbour_flags_duplicates():
    values = np.array([[1.0, 2.0], [1.0, 2.0], [5.0, 5.0]])
    distances, neighbours = nearest_neighbour_distances(values)
    assert distances[0] == 0.0 and distances[1] == 0.0
    # 'c' is equidistant from the two duplicates; argmin takes the first
    assert neighbours.tolist() == [1, 0, 0]
    assert distances[2] == pytest.approx(5.0)  # sqrt(4² + 3²)


def test_nearest_neighbour_rejects_unmeasured_points():
    values = np.ones((3, 4))
    values[2, 3] = np.nan
    with pytest.raises(ValueError, match="complete subset"):
        nearest_neighbour_distances(values)


def test_complete_prefix_mask_stops_at_first_hole():
    values = np.array([
        [1.0, 2.0, np.nan, 4.0],
        [1.0, 2.0, 3.0, np.nan],
    ])
    assert complete_prefix_mask(values).tolist() == [True, True, False, False]


def test_complete_prefix_mask_all_or_nothing():
    assert complete_prefix_mask(np.ones((3, 4))).all()
    values = np.ones((3, 4))
    values[0, 0] = np.nan
    assert not complete_prefix_mask(values).any()


# --- component scores and descriptor attribution --------------------------


def test_pca_scores_match_the_explained_variance_ordering():
    # Two independent directions with clearly different spreads: the
    # leading axis carries the x-direction. Per-axis sign is
    # arbitrary (SVD), magnitudes are not.
    values = np.array(
        [
            [3.0, 0.0],
            [-3.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    scores = pca_scores(values)
    ratios = pca_explained_variance(values)
    assert np.allclose(np.abs(scores[:, 0]), [3.0, 3.0, 0.0, 0.0])
    assert np.allclose(np.abs(scores[:, 1]), [0.0, 0.0, 1.0, 1.0])
    assert ratios[0] > ratios[1]


def test_pca_scores_truncate_like_the_variance_spectrum():
    values = np.linspace(0.0, 1.0, 6)[:, None] * np.array([[1.0, 2.0]])
    assert pca_scores(values, max_components=1).shape == (6, 1)
    assert pca_scores(values).shape == (6, 2)


def test_score_descriptor_correlations_are_pairwise_complete():
    scores = np.array([[1.0], [2.0], [3.0], [4.0]])
    descriptors = pd.DataFrame(
        {
            "exact_positive": [2.0, 4.0, 6.0, 8.0],
            "exact_negative": [-1.0, -2.0, -3.0, -4.0],
            # Only pairs (1,4), (3,2), (4,1) survive the gap; they lie
            # on a line of slope -1, so the correlation stays exact.
            "with_a_gap": [4.0, np.nan, 2.0, 1.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
        }
    )
    table = score_descriptor_correlations(scores, descriptors)
    assert table.index.tolist() == ["component_1"]
    assert table.columns.tolist() == list(descriptors.columns)
    assert table.loc["component_1", "exact_positive"] == pytest.approx(1.0)
    assert table.loc["component_1", "exact_negative"] == pytest.approx(-1.0)
    assert table.loc["component_1", "with_a_gap"] == pytest.approx(-1.0)
    assert np.isnan(table.loc["component_1", "constant"])


def test_thickness_stability_separates_q_from_its_thickness_factor():
    # q varies with deviations [1,-1,-1,1], exactly orthogonal to the
    # thickness deviations, while A = t·q inherits the thickness
    # trend. Beer–Lambert-exact behaviour: cv(A) > cv(q), r(A,t) > 0,
    # r(q,t) = 0.
    thickness_nm = pd.Series([100.0, 200.0, 300.0, 400.0])
    q = np.array([0.022, 0.018, 0.018, 0.022])
    descriptors = pd.DataFrame(
        {"q_integral": q, "a_integral": thickness_nm.to_numpy() * q}
    )
    table = thickness_stability(descriptors, thickness_nm)
    assert table.loc["q_integral", "pearson_r"] == pytest.approx(0.0, abs=1e-12)
    assert table.loc["a_integral", "pearson_r"] == pytest.approx(0.978, abs=5e-3)
    assert table.loc["a_integral", "cv"] > table.loc["q_integral", "cv"]
    # Sample-std CV of q: (0.002 · 2/√3) / 0.02
    assert table.loc["q_integral", "cv"] == pytest.approx(0.1155, rel=1e-3)


def test_thickness_stability_guards_for_gaps_and_singletons():
    thickness_nm = pd.Series([100.0, 200.0, 300.0, 400.0])
    descriptors = pd.DataFrame(
        {
            "constant": [0.05, 0.05, 0.05, 0.05],
            "one_gap": [0.03, np.nan, 0.02, 0.03],
            "one_value": [0.02, np.nan, np.nan, np.nan],
        }
    )
    table = thickness_stability(descriptors, thickness_nm)
    assert table.loc["constant", "cv"] == pytest.approx(0.0)
    assert np.isnan(table.loc["constant", "pearson_r"])  # zero variance
    assert np.isfinite(table.loc["one_gap", "cv"])  # three films suffice
    assert np.isfinite(table.loc["one_gap", "pearson_r"])
    # A single finite film supports neither spread nor correlation.
    assert np.isnan(table.loc["one_value", "cv"])
    assert np.isnan(table.loc["one_value", "pearson_r"])


def test_thickness_stability_rejects_misaligned_thickness():
    descriptors = pd.DataFrame({"d": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="shape"):
        thickness_stability(descriptors, pd.Series([100.0, 200.0]))
