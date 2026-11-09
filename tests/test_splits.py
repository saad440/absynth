"""Tests for date-block splits — synthetic parameter tables, no data."""

import json

import pandas as pd
import pytest

from absynth.loader import DATE_COLUMN, SUBSTRATE_STACK_COLUMN, THICKNESS_COLUMN
from absynth.splits import (
    DEFAULT_TEST_FRACTION,
    RARE_STACK_MAX_FILMS,
    SplitError,
    date_block_holdout,
)

# Two substrate families, several dates each, uneven sizes: the greedy
# selection has real choices, and every admissible outcome satisfies
# the assertions below regardless of which seed shuffles the dates.
GLASS_DAYS = [("05/01/2024", 12), ("01/02/2024", 10), ("20/02/2024", 9), ("08/03/2024", 11)]
CORNING_DAYS = [("02/04/2024", 8), ("25/04/2024", 6), ("15/05/2024", 7)]
BASE = [(date, "glass", n) for date, n in GLASS_DAYS]
BASE += [(date, "corning", n) for date, n in CORNING_DAYS]


def make_parameters(date_stacks, extra_rows=()):
    """Parameter table from ``(date_iso, stack, n_films)`` triples.

    Dates use the archive's ``DD/MM/YYYY`` convention. Row order and
    identifiers are deterministic; thickness values are distinct so
    per-film rows never collapse. ``extra_rows`` appends raw dicts for
    fault-injection tests.
    """
    rows = []
    sample = 0
    for date, stack, n in date_stacks:
        for _ in range(n):
            sample += 1
            rows.append(
                {
                    DATE_COLUMN: date,
                    SUBSTRATE_STACK_COLUMN: stack,
                    THICKNESS_COLUMN: 100.0 + sample,
                }
            )
    rows.extend(extra_rows)
    identifiers = pd.Index(
        [f"BPD_sample_{i:03d}" for i in range(1, len(rows) + 1)], name="Sample"
    )
    return pd.DataFrame(rows, index=identifiers)


def test_dates_never_span_both_splits():
    params = make_parameters(BASE)
    result = date_block_holdout(params, seed=3)
    train_days = set(params.loc[result.train_ids, DATE_COLUMN])
    test_days = set(params.loc[result.test_ids, DATE_COLUMN])
    assert not train_days & test_days


def test_both_families_on_both_sides():
    # Stratification is the point: each substrate family contributes
    # held-out films, so no metric silently measures one stack only.
    params = make_parameters(BASE)
    result = date_block_holdout(params, seed=3)
    train_stacks = set(params.loc[result.train_ids, SUBSTRATE_STACK_COLUMN])
    test_stacks = set(params.loc[result.test_ids, SUBSTRATE_STACK_COLUMN])
    assert train_stacks == {"glass", "corning"}
    assert test_stacks == {"glass", "corning"}


def test_held_out_fraction_lands_near_target():
    params = make_parameters(BASE)
    result = date_block_holdout(params, seed=3)
    glass = result.provenance["families"]["glass"]
    assert glass["eligible_films"] == 42
    assert glass["target_test_films"] == 8
    # Greedy guarantee: within one date's size of the target.
    assert abs(glass["test_films"] - glass["target_test_films"]) < 12


def test_same_seed_same_split():
    params = make_parameters(BASE)
    first = date_block_holdout(params, seed=11)
    again = date_block_holdout(params, seed=11)
    assert list(first.test_ids) == list(again.test_ids)
    assert first.provenance == again.provenance


def test_masks_partition_every_film():
    params = make_parameters(BASE)
    result = date_block_holdout(params, seed=3)
    train, test = result.train_mask, result.test_mask
    assert train.dtype == test.dtype == bool
    assert not (train & test).any()
    assert (train | test).all()
    assert train.sum() == len(result.train_ids)
    assert test.sum() == len(result.test_ids)


def test_rare_stack_dates_locked_to_training():
    # Three FTO/TiO2 films: too few for any held-out claim, so their
    # whole day stays in fitting and the scope is stated in the record.
    params = make_parameters(BASE + [("30/12/2023", "FTO / TiO2-c", 3)])
    result = date_block_holdout(params, seed=3, rare_stack_max_films=5)
    assert result.provenance["rare_stacks"] == ["FTO / TiO2-c"]
    assert "2023-12-30" in result.provenance["train_locked_dates"]
    assert "2023-12-30" in result.provenance["train"]["dates"]
    assert "2023-12-30" not in result.provenance["test"]["dates"]


def test_provenance_record_is_json_serializable():
    params = make_parameters(BASE)
    record = date_block_holdout(params, seed=3).provenance
    assert json.loads(json.dumps(record))["seed"] == 3


def test_split_summaries_report_thickness_reach():
    params = make_parameters(BASE)
    result = date_block_holdout(params, seed=3)
    for side in ("train", "test"):
        reach = result.provenance[side]["thickness_nm"]
        assert set(reach) == {"min", "median", "max"}
        assert reach["min"] <= reach["median"] <= reach["max"]


def test_defaults_are_pinned_policy():
    assert DEFAULT_TEST_FRACTION == 0.2
    assert RARE_STACK_MAX_FILMS == 20


def test_missing_date_column_raises():
    params = make_parameters(BASE).drop(columns=[DATE_COLUMN])
    with pytest.raises(SplitError, match="lack"):
        date_block_holdout(params)


def test_unparseable_date_raises():
    params = make_parameters(
        BASE,
        extra_rows=[
            {DATE_COLUMN: "not a date", SUBSTRATE_STACK_COLUMN: "glass", THICKNESS_COLUMN: 500.0}
        ],
    )
    with pytest.raises(SplitError, match="unparseable"):
        date_block_holdout(params)


def test_missing_stack_raises():
    params = make_parameters(
        BASE,
        extra_rows=[
            {DATE_COLUMN: "10/07/2024", SUBSTRATE_STACK_COLUMN: None, THICKNESS_COLUMN: 500.0}
        ],
    )
    with pytest.raises(SplitError, match="no 'Substrate_stack_sequence' entry"):
        date_block_holdout(params)


def test_fraction_out_of_range_raises():
    params = make_parameters(BASE)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(SplitError, match="strictly between"):
            date_block_holdout(params, test_fraction=bad)


def test_all_dates_locked_raises():
    params = make_parameters(BASE)
    with pytest.raises(SplitError, match="no date can be held out"):
        date_block_holdout(params, rare_stack_max_films=10_000)


def test_selection_that_holds_out_nothing_raises():
    # A two-film family can be eligible (not rare) yet fall below any
    # sensible held-out target; the guard must say so.
    params = make_parameters([("05/01/2024", "glass", 2)])
    with pytest.raises(SplitError, match="held out no films"):
        date_block_holdout(params, rare_stack_max_films=1)
