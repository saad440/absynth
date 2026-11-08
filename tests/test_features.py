"""Tests for spectral descriptors — synthetic curves, analytic values."""

import numpy as np
import pandas as pd
import pytest

from absynth.features import (
    BASELINE_WINDOW_NM,
    INTEGRATION_WINDOWS_NM,
    baseline_level,
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
