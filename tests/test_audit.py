"""Tests for integrity diagnostics — synthetic datasets, no real data."""

import json

import numpy as np
import pandas as pd
import pytest
from synth import SAMPLE_IDS, STEP, make_curves, make_dataset, make_parameters

from absynth.audit import (
    OD_MEASURED_COLUMN,
    ODINT_COLUMNS,
    ODINT_PRIMARY,
    audit_dataset,
    curve_diagnostics,
    parameter_diagnostics,
    subset_min_observations,
    summarize_curve_diagnostics,
    variability_facts,
)
from absynth.loader import (
    DATE_COLUMN,
    SUBSTRATE_STACK_COLUMN,
    THICKNESS_COLUMN,
    CurveSet,
    Dataset,
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
    assert summary["composition_distinct"] == 1
    assert summary["solvents_distinct"] == 1
    assert summary["additive_arm_counts"] == {"NaLES/DDABr": 3, "none": 3, "NaLES": 2}
    assert summary["substrate_stack_counts"] == {"corning": 6, "glass": 2}
    # below the 2.0 a.u. marker: 1.09, 1.5, 0.8, 1.2, 0.5, 1.0 —
    # not 2.0 itself (strict comparison), not NaN
    assert summary["films_below_odint2_label"] == 6
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


def test_subset_min_observations_keeps_populated_arms():
    frame = pd.DataFrame(
        {"arm": ["a", "a", "a", "b", "b", "c"]},
        index=pd.Index([f"film_{i}" for i in range(6)], name="Sample"),
    )
    kept = subset_min_observations(frame, "arm", min_count=2)
    # 'c' has a single observation and is dropped with its rows
    assert kept.tolist() == ["film_0", "film_1", "film_2", "film_3", "film_4"]
    assert len(subset_min_observations(frame, "arm", min_count=1)) == 6
    assert len(subset_min_observations(frame, "arm", min_count=4)) == 0


def test_variability_facts_without_common_span_are_null():
    # The synthetic fixture has one film with nothing measured, so no
    # wavelength is measured for every film: nulls, not an error.
    # Statistics on a partial population would silently change the sample.
    report = audit_dataset(make_dataset())
    assert report.summary["span_start_nm"] is None
    assert report.summary["closest_pair"] is None


def test_variability_facts_on_a_complete_dataset():
    curves = CurveSet(
        wavelengths_nm=np.array([300.0, 400.0, 500.0]),
        sample_ids=pd.Index(["a", "b", "c"], name="Sample"),
        optical_density=np.array([
            [0.01, 0.02, 0.01],
            [0.02, 0.04, 0.02],
            [0.03, 0.06, 0.03],
        ]),
    )
    parameters = pd.DataFrame(
        {
            THICKNESS_COLUMN: [1.0, 2.0, 3.0],
            ODINT_PRIMARY: [1.0, 1.0, 1.0],
            OD_MEASURED_COLUMN: [True, True, True],
            DATE_COLUMN: ["01/01/2024", "01/01/2024", "01/01/2024"],
            SUBSTRATE_STACK_COLUMN: ["s", "s", "s"],
        },
        index=pd.Index(["a", "b", "c"], name="Sample"),
    )
    facts = variability_facts(Dataset(curves=curves, parameters=parameters))

    assert facts["span_start_nm"] == 300.0 and facts["span_end_nm"] == 500.0
    # three rows on one line through the origin: a single component
    assert facts["pca_components_for_50pct_variance"] == 1
    assert facts["pca_components_for_99pct_variance"] == 1
    # a-b and b-c are equally close; argmin ties resolve to the first row
    assert facts["closest_pair"] == ["a", "b"]
    assert facts["nn_distance_min_nm_inv"] == pytest.approx(0.01 * np.sqrt(6))
