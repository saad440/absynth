"""Tests for integrity diagnostics — synthetic datasets, no real data."""

import json

import numpy as np
import pandas as pd
import pytest

from absynth.audit import (
    OD_MEASURED_COLUMN,
    ODINT_COLUMNS,
    ODINT_PRIMARY,
    audit_dataset,
    curve_diagnostics,
    parameter_diagnostics,
    summarize_curve_diagnostics,
)
from absynth.loader import THICKNESS_COLUMN, CurveSet, Dataset

WAVELENGTHS = np.array([300.0, 400.0, 500.0, 600.0, 700.0])
STEP = 100.0

# Eight films covering every diagnostic category. NaN marks an
# unmeasured wavelength, mirroring the shipped archive's convention.
VALUES = {
    # a healthy, fully measured curve
    "normal": [0.001, 0.010, 0.020, 0.010, 0.001],
    # measured data stops two grid points before the end
    "truncated": [0.002, 0.020, 0.030, np.nan, np.nan],
    # nothing measured at the first grid point
    "late_start": [np.nan, 0.002, 0.004, 0.002, 0.001],
    # a hole between measured points
    "interior_gap": [0.001, 0.010, np.nan, 0.010, 0.001],
    # fully measured, with one slightly negative baseline point
    "negative": [0.001, -0.002, 0.005, 0.004, 0.002],
    # constant nonzero curve
    "flat": [0.02, 0.02, 0.02, 0.02, 0.02],
    # constant zero curve
    "zero": [0.0, 0.0, 0.0, 0.0, 0.0],
    # nothing measured at all
    "empty": [np.nan, np.nan, np.nan, np.nan, np.nan],
}
SAMPLE_IDS = list(VALUES)


def make_curves() -> CurveSet:
    values = np.array([VALUES[sample_id] for sample_id in SAMPLE_IDS])
    return CurveSet(
        wavelengths_nm=WAVELENGTHS,
        sample_ids=pd.Index(SAMPLE_IDS, name="Sample"),
        optical_density=values,
    )


def make_parameters() -> pd.DataFrame:
    # One film per anomaly: 'empty' lacks the primary ODint,
    # 'truncated' lacks a secondary window, 'zero' has no thickness,
    # 'interior_gap' has nonpositive thickness.
    # All five ODint windows exist in the real database; the fixture
    # mirrors that schema.
    frame = pd.DataFrame(
        {
            THICKNESS_COLUMN: [300.0, 600.0, 450.0, -5.0, 900.0, 1200.0, 0.0, np.nan],
            ODINT_PRIMARY: [1.09, 1.5, 0.8, 1.2, 2.0, 0.5, 1.0, np.nan],
            ODINT_COLUMNS[1]: [0.75, np.nan, 0.7, 0.9, 1.1, 0.3, 0.6, 0.4],
            ODINT_COLUMNS[2]: [0.70, 1.10, 0.6, 0.8, 1.0, 0.2, 0.5, 0.3],
            ODINT_COLUMNS[3]: [0.55, 0.90, 0.5, 0.7, 0.9, 0.2, 0.4, 0.3],
            ODINT_COLUMNS[4]: [0.34, 0.70, 0.4, 0.5, 0.7, 0.1, 0.3, 0.2],
            OD_MEASURED_COLUMN: [True, True, True, True, True, True, True, True],
        },
        index=pd.Index(SAMPLE_IDS, name="Sample"),
    )
    # The flag-vs-curve mismatches: 'normal' is measured but flagged
    # False; 'empty' is empty but flagged True.
    frame.loc["normal", OD_MEASURED_COLUMN] = False
    return frame


def make_dataset() -> Dataset:
    return Dataset(curves=make_curves(), parameters=make_parameters())


def diag_for(curves, sample_id):
    return curve_diagnostics(curves).loc[sample_id]


def test_per_film_missingness_facts():
    curves = make_curves()

    healthy = diag_for(curves, "normal")
    assert healthy["n_measured"] == 5 and healthy["n_missing"] == 0
    assert healthy["first_measured_nm"] == 300.0
    assert healthy["last_measured_nm"] == 700.0
    assert not healthy["starts_late"] and not healthy["ends_early"]

    truncated = diag_for(curves, "truncated")
    assert truncated["n_measured"] == 3 and truncated["n_missing"] == 2
    assert truncated["last_measured_nm"] == 500.0
    assert truncated["ends_early"] and not truncated["starts_late"]
    assert not truncated["has_interior_gap"]

    late = diag_for(curves, "late_start")
    assert late["starts_late"] and late["first_measured_nm"] == 400.0
    assert not late["ends_early"] and not late["has_interior_gap"]

    holed = diag_for(curves, "interior_gap")
    assert holed["has_interior_gap"]
    assert not holed["starts_late"] and not holed["ends_early"]

    empty = diag_for(curves, "empty")
    assert empty["is_empty"] and empty["n_measured"] == 0
    assert np.isnan(empty["first_measured_nm"]) and np.isnan(empty["od_max"])


def test_per_film_value_facts():
    curves = make_curves()

    negative = diag_for(curves, "negative")
    assert negative["n_negative"] == 1
    assert negative["od_min"] == pytest.approx(-0.002)
    assert not negative["is_flat"] and not negative["is_zero"]

    flat = diag_for(curves, "flat")
    assert flat["is_flat"] and not flat["is_zero"]

    zero = diag_for(curves, "zero")
    assert zero["is_flat"] and zero["is_zero"]

    assert curve_diagnostics(curves)["od_min"].min() == pytest.approx(-0.002)


def test_population_summary_counts():
    curves = make_curves()
    summary = summarize_curve_diagnostics(curve_diagnostics(curves), curves)

    assert summary["n_samples"] == 8
    assert summary["n_wavelengths"] == 5
    assert summary["grid_start_nm"] == 300.0
    assert summary["grid_end_nm"] == 700.0
    assert summary["grid_step_nm"] == STEP
    # 'normal', 'negative', 'flat' and 'zero' have no unmeasured point;
    # the rest miss at least one ('empty' misses all five).
    assert summary["n_curves_complete"] == 4
    assert summary["n_curves_with_missing"] == 4
    assert summary["n_starts_late"] == 1
    assert summary["n_ends_early"] == 1
    assert summary["n_with_interior_gap"] == 1
    assert summary["n_empty_films"] == 1
    # 'truncated' ends at 500 nm; 'late_start' starts at 400 nm
    assert summary["earliest_last_measured_nm"] == 500.0
    assert summary["latest_first_measured_nm"] == 400.0
    assert summary["n_negative_films"] == 1
    assert summary["n_negative_values"] == 1
    assert summary["n_flat_films"] == 2  # 'flat' and 'zero'
    assert summary["n_zero_films"] == 1


def test_parameter_summary_counts():
    summary = parameter_diagnostics(make_parameters())

    assert summary["n_rows"] == 8
    assert summary["missing_odint_by_column"][ODINT_PRIMARY] == 1  # 'empty'
    assert summary["missing_odint_by_column"][ODINT_COLUMNS[1]] == 1  # 'truncated'
    assert all(
        summary["missing_odint_by_column"][column] == 0
        for column in ODINT_COLUMNS[2:]
    )
    assert summary["rows_missing_primary_odint"] == 1
    assert summary["od_measured_true"] == 7
    assert summary["od_measured_false"] == 1
    assert summary["od_measured_missing"] == 0
    assert summary["thickness_missing"] == 1
    assert summary["thickness_le_zero"] == 2  # -5.0 and 0.0
    assert summary["thickness_min_nm"] == -5.0
    assert summary["thickness_max_nm"] == 1200.0


def test_dataset_audit_cross_checks_the_flag():
    report = audit_dataset(make_dataset())

    # 'normal' measured but flagged False; 'empty' flagged True but
    # has no measured points.
    assert report.summary["flag_curve_disagreements"] == 2
    assert report.summary["flag_false_curve_measured"] == 1
    assert report.summary["flag_true_curve_empty"] == 1
    assert report.curve_diagnostics.index.tolist() == SAMPLE_IDS


def test_summary_is_json_serializable():
    report = audit_dataset(make_dataset())
    # Numbers must be plain Python types, not numpy scalars, or the
    # reports/ artifacts would leak non-JSON values.
    assert isinstance(json.dumps(report.summary), str)
