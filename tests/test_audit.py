"""Tests for integrity diagnostics — synthetic datasets, no real data."""

import json

import numpy as np
import pytest
from synth import SAMPLE_IDS, STEP, make_curves, make_dataset, make_parameters

from absynth.audit import (
    ODINT_COLUMNS,
    ODINT_PRIMARY,
    audit_dataset,
    curve_diagnostics,
    parameter_diagnostics,
    summarize_curve_diagnostics,
)


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
        summary["missing_odint_by_column"][column] == 0 for column in ODINT_COLUMNS[2:]
    )
    assert summary["rows_missing_primary_odint"] == 1
    assert summary["od_measured_true"] == 7
    assert summary["od_measured_false"] == 1
    assert summary["od_measured_missing"] == 0
    assert summary["thickness_missing"] == 1
    assert summary["thickness_le_zero"] == 2  # -5.0 and 0.0
    assert summary["thickness_min_nm"] == -5.0
    assert summary["thickness_max_nm"] == 1200.0
    assert summary["n_substrate_stacks"] == 2
    assert summary["substrate_missing"] == 0
    # 'not a date' on 'flat' is counted, not silently dropped
    assert summary["date_min"] == "2023-02-01"
    assert summary["date_max"] == "2023-09-03"
    assert summary["dates_unparseable"] == 1


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
