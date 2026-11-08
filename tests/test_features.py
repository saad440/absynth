"""Tests for spectral descriptors — synthetic curves, analytic values."""

import numpy as np
import pandas as pd
import pytest

from absynth.features import (
    BASELINE_WINDOW_NM,
    INTEGRATION_WINDOWS_NM,
    baseline_level,
    edge_descriptors,
    window_integrals,
)
from absynth.loader import CurveSet

SAMPLE_IDS = ["BPD_sample_001", "BPD_sample_002"]
# The working span on the storage grid: 300-900 nm at 2 nm, 301 points.
GRID_NM = np.arange(300.0, 900.0 + 1.0, 2.0)


def make_curves(values, wavelengths_nm=GRID_NM):
    """Build a CurveSet directly; ``values`` is (n_samples, n_wavelengths)."""
    return CurveSet(
        wavelengths_nm=np.asarray(wavelengths_nm, dtype=float),
        sample_ids=pd.Index(SAMPLE_IDS[: len(values)], name="Sample"),
        optical_density=np.asarray(values, dtype=float),
    )


def test_constant_curve_integral_is_exact_area():
    # Trapezoid over a constant is exact: area = level × span.
    q = np.full((2, GRID_NM.size), 0.01)
    q[1] = 0.02
    integrals = window_integrals(make_curves(q))
    assert integrals.loc["BPD_sample_001", "q_int_300_900"] == pytest.approx(0.01 * 600)
    assert integrals.loc["BPD_sample_002", "q_int_300_900"] == pytest.approx(0.02 * 600)
    assert integrals.loc["BPD_sample_001", "q_int_300_600"] == pytest.approx(0.01 * 300)
    assert integrals.loc["BPD_sample_001", "q_int_600_900"] == pytest.approx(0.01 * 300)


def test_linear_ramp_integral_matches_analytic_value():
    # q(λ) = 1e-5 · λ nm⁻¹ ⇒ ∫q dλ = 1e-5/2 · (900² - 300²) = 3.6,
    # and the trapezoidal rule is exact for a linear integrand.
    q = np.tile(1e-5 * GRID_NM, (2, 1))
    integrals = window_integrals(make_curves(q))
    assert integrals["q_int_300_900"].to_numpy() == pytest.approx([3.6, 3.6])


def test_window_bounds_are_inclusive_over_grid_points():
    # A window boundary between grid points keeps only points inside;
    # on the 2 nm grid, [304, 306] holds exactly those two columns.
    q = np.full((1, GRID_NM.size), 0.01)
    integrals = window_integrals(make_curves(q), windows_nm=[(304.0, 306.0)])
    assert integrals["q_int_304_306"].iloc[0] == pytest.approx(0.01 * 2.0)


def test_integrals_are_indexed_and_named_per_window():
    curves = make_curves(np.full((2, GRID_NM.size), 0.01))
    integrals = window_integrals(curves)
    assert integrals.index.tolist() == SAMPLE_IDS
    expected = {f"q_int_{lo:g}_{hi:g}" for lo, hi in INTEGRATION_WINDOWS_NM}
    assert set(integrals.columns) == expected


def test_nan_inside_a_window_raises_but_outside_all_windows_is_fine():
    # Full-grid curves with one truncated tail: integrals up to 900 nm
    # must work, a gap inside any window must be refused. Fresh arrays
    # per case: CurveSet stores the array it is given, and in-place
    # edits on one fixture would leak into the next.
    grid_full = np.arange(300.0, 1000.0 + 1.0, 2.0)
    with_gap_inside = make_curves(np.full((2, 351), 0.01), wavelengths_nm=grid_full)
    with_gap_inside.optical_density[0, 100] = np.nan  # 500 nm
    with pytest.raises(ValueError, match="unmeasured"):
        window_integrals(with_gap_inside)

    truncated = make_curves(np.full((2, 351), 0.01), wavelengths_nm=grid_full)
    truncated.optical_density[0, grid_full > 900.0] = np.nan
    integrals = window_integrals(truncated)
    assert integrals["q_int_300_900"].notna().all()


def test_window_off_the_grid_raises():
    curves = make_curves(np.full((1, GRID_NM.size), 0.01))
    with pytest.raises(ValueError, match="need at least 2"):
        window_integrals(curves, windows_nm=[(309.0, 310.0)])


def test_baseline_default_window_is_pinned_policy():
    # Golden value: 850-900 nm, sub-edge for the MAPbI3 prior (~800 nm).
    assert BASELINE_WINDOW_NM == (850.0, 900.0)


def test_baseline_equals_constant_level_with_name_and_index():
    q = np.full((2, GRID_NM.size), 0.004)
    baseline = baseline_level(make_curves(q))
    assert baseline.name == "q_baseline_850_900"
    assert baseline.index.tolist() == SAMPLE_IDS
    assert baseline.to_numpy() == pytest.approx([0.004, 0.004])


def test_baseline_rejects_unmeasured_points_in_its_window():
    q = np.full((1, GRID_NM.size), 0.01)
    curves = make_curves(q)
    curves.optical_density[0, GRID_NM > 860.0] = np.nan
    with pytest.raises(ValueError, match="unmeasured"):
        baseline_level(curves)


# --- edge descriptors ------------------------------------------------------


def descent_values(edge_start, edge_end, plateau=0.02, tail=0.001):
    """One curve as a 1-D array: flat plateau, linear descent, flat tail."""
    return np.where(
        GRID_NM <= edge_start,
        plateau,
        np.where(
            GRID_NM >= edge_end,
            tail,
            plateau + (GRID_NM - edge_start) * (tail - plateau) / (edge_end - edge_start),
        ),
    )


def test_edge_position_and_width_match_analytic_descent():
    # Plateau 0.02 to 700 nm, descent to 0.001 at 780 nm; slope is
    # 2.375e-4 per nm, so on the even 2 nm grid a level counts at the
    # last point still >= level:
    #   90% 0.018: q(708)=0.01810 >=, q(710)=0.01763 <  -> 708
    #   50% 0.010: q(742)=0.01003 >=, q(744)=0.00955 <  -> 742
    #   10% 0.002: q(774)=0.00243 >=, q(776)=0.00195 <  -> 774
    curves = make_curves(np.stack([descent_values(700.0, 780.0)]))
    edges = edge_descriptors(curves)
    assert edges["edge_50_nm"].tolist() == [742.0]
    assert edges["edge_width_10_90_nm"].tolist() == [774.0 - 708.0]


def test_edge_tracks_each_films_descent_independently():
    # Same descent shape, the second film shifted 20 nm to the red:
    # crossings move by exactly 20, the width is unchanged.
    curves = make_curves(
        np.stack([descent_values(700.0, 780.0), descent_values(720.0, 800.0)])
    )
    edges = edge_descriptors(curves)
    assert edges["edge_50_nm"].tolist() == [742.0, 762.0]
    assert edges["edge_width_10_90_nm"].tolist() == [66.0, 66.0]


def test_width_is_nan_when_the_descent_stops_above_the_lowest_level():
    # Tail at 25% of the plateau: the 10% crossing never happens, so
    # the width is undefined while the position still is.
    # 50% level 0.010: q(752)=0.01025 >=, q(754)=0.00988 <  -> 752
    curves = make_curves(np.stack([descent_values(700.0, 780.0, tail=0.005)]))
    edges = edge_descriptors(curves)
    assert edges["edge_50_nm"].tolist() == [752.0]
    assert np.isnan(edges["edge_width_10_90_nm"].iloc[0])


def test_truncation_does_not_fake_an_edge():
    # The curve stays above half-plateau through its measured end at
    # 900 nm and is NaN beyond: reporting the truncation point as the
    # edge would be exactly the false crossing the rule forbids.
    grid_full = np.arange(300.0, 1000.0 + 1.0, 2.0)
    values = np.where(
        grid_full <= 700.0,
        0.02,
        np.where(grid_full >= 780.0, 0.012, 0.02 - (grid_full - 700.0) * 0.0001),
    )
    values[grid_full > 900.0] = np.nan
    curves = make_curves(values[np.newaxis, :], wavelengths_nm=grid_full)
    edges = edge_descriptors(curves)
    assert np.isnan(edges["edge_50_nm"].iloc[0])
    assert np.isnan(edges["edge_width_10_90_nm"].iloc[0])


def test_flat_curve_has_no_edge_at_all():
    curves = make_curves(np.full((1, GRID_NM.size), 0.02))
    assert edge_descriptors(curves).isna().all().all()


def test_edge_requires_a_measured_plateau():
    curves = make_curves(np.full((1, GRID_NM.size), 0.02))
    curves.optical_density[0, GRID_NM <= 500.0] = np.nan
    with pytest.raises(ValueError, match="unmeasured"):
        edge_descriptors(curves)
