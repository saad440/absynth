"""Tests for exploratory curve statistics — synthetic, hand-checkable."""

import numpy as np
import pytest

from absynth.explore import nearest_neighbour_distances, pca_explained_variance


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
