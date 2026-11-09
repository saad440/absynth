"""Held-out film selection: date-block splits stratified by substrate.

A generative model is judged on films it has never seen. Which films
may move together is dictated by how the campaign was fabricated, not
by preference: films coated on the same day share the precursor ink
batch, ambient conditions, instrument state and operator, so same-day
films are correlated samples. Treating them as independent would let
a model score well on held-out films by having absorbed their
day-mates' quirks — remembering, not generalizing. The fabrication
date is therefore the unit of independence here, and whole date blocks
move as one.

Substrate stack is a different kind of quantity: a physical property
of a film, not a sampling unit. The held-out side should contain the
same substrate populations as the fitting side; otherwise comparisons
between generated films and held-out measurements would pick up the
deliberate optical-context shift (substrate stacks move the spectral
baseline strongly) instead of the variability the model was asked to
learn. Dates are therefore selected for the held-out side within each
substrate family separately, where a date's family is its most common
substrate stack — in this campaign every date is stack-pure apart from
two.

Stacks with very few films support no held-out claim either way: a
statistic computed on eight films is an anecdote. Dates containing
rare-stack films are locked to fitting, so held-out numbers describe
the common stacks only; the provenance record states this scope.

The split records everything it decided: the random seed that pins the
draw, the held-out fraction, the rare-stack threshold, the selected
dates per family, and per-split summaries of stacks, dates and
thickness, so any shift between fitting and held-out populations is
visible rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from absynth.loader import DATE_COLUMN, DATE_FORMAT, SUBSTRATE_STACK_COLUMN, THICKNESS_COLUMN

DEFAULT_TEST_FRACTION = 0.2
DEFAULT_SEED = 0
RARE_STACK_MAX_FILMS = 20


class SplitError(RuntimeError):
    """Raised when a split request cannot be honoured as specified."""


@dataclass(frozen=True)
class SplitResult:
    """A fitting/held-out assignment of films with its provenance.

    Attributes:
        train_ids: Sample identifiers used for fitting.
        test_ids: Sample identifiers held out from all fitting.
        row_index: The full sample-identifier index in dataset row
            order, as loaded; the masks index into this order.
        provenance: JSON-serializable record of the split decisions:
            the random seed, the held-out fraction, the rare-stack
            policy, selected dates per family, and per-split
            stack/date/thickness summaries.
    """

    train_ids: pd.Index
    test_ids: pd.Index
    row_index: pd.Index
    provenance: dict[str, object]

    def __post_init__(self) -> None:
        overlap = self.train_ids.intersection(self.test_ids)
        if len(overlap):
            raise SplitError(
                f"{len(overlap)} films assigned to both splits, e.g. {sorted(overlap)[:5]}"
            )
        missing = self.row_index.difference(self.train_ids.union(self.test_ids))
        if len(missing):
            raise SplitError(f"{len(missing)} films left unassigned, e.g. {sorted(missing)[:5]}")
        unknown = self.train_ids.union(self.test_ids).difference(self.row_index)
        if len(unknown):
            raise SplitError(
                f"{len(unknown)} assigned films are not in the dataset, e.g. {sorted(unknown)[:5]}"
            )

    @property
    def train_mask(self) -> np.ndarray:
        """Boolean mask over dataset row order selecting fitting films."""
        return self.row_index.isin(self.train_ids)

    @property
    def test_mask(self) -> np.ndarray:
        """Boolean mask over dataset row order selecting held-out films."""
        return self.row_index.isin(self.test_ids)


def _parse_dates(parameters: pd.DataFrame) -> pd.Series:
    """Parse the fabrication-date column; refuse unusable entries."""
    if DATE_COLUMN not in parameters.columns:
        raise SplitError(f"parameters lack a {DATE_COLUMN!r} column")
    dates = pd.to_datetime(parameters[DATE_COLUMN], format=DATE_FORMAT, errors="coerce")
    unusable = int(dates.isna().sum())
    if unusable:
        raise SplitError(f"{unusable} films have missing or unparseable {DATE_COLUMN!r} entries")
    return dates


def _stack_families(parameters: pd.DataFrame, dates: pd.Series) -> pd.Series:
    """Assign each date its family: the modal substrate stack of that day.

    ``Series.mode`` sorts its values, so a tie would resolve
    deterministically to the alphabetically first stack; the shipped
    campaign has no ties.
    """
    if SUBSTRATE_STACK_COLUMN not in parameters.columns:
        raise SplitError(f"parameters lack a {SUBSTRATE_STACK_COLUMN!r} column")
    stacks = parameters[SUBSTRATE_STACK_COLUMN]
    unstacked = int(stacks.isna().sum())
    if unstacked:
        raise SplitError(f"{unstacked} films have no {SUBSTRATE_STACK_COLUMN!r} entry")
    return stacks.groupby(dates).agg(lambda day: day.mode().iloc[0])


def _select_test_dates(
    eligible_dates: list[pd.Timestamp],
    family_of_date: pd.Series,
    films_per_date: pd.Series,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[pd.Timestamp], dict[str, dict[str, object]]]:
    """Pick held-out dates within each family, one seeded greedy pass.

    Within a family the eligible dates are shuffled, then a date joins
    the held-out side whenever that moves the family's held-out film
    count closer to ``test_fraction`` of the family's eligible films.
    The outcome is reproducible for a given seed and lands within one
    date's size of the target, which the per-family record states.

    Returns:
        The selected dates, plus a JSON-serializable per-family record
        of targets and outcomes.
    """
    selected: list[pd.Timestamp] = []
    records: dict[str, dict[str, object]] = {}
    for family in sorted({str(family_of_date[date]) for date in eligible_dates}):
        in_family = sorted(date for date in eligible_dates if str(family_of_date[date]) == family)
        eligible_films = int(films_per_date[in_family].sum())
        target = int(np.floor(test_fraction * eligible_films + 0.5))
        held = 0
        chosen: list[pd.Timestamp] = []
        for position in rng.permutation(len(in_family)):
            date = in_family[int(position)]
            candidate = held + int(films_per_date[date])
            if abs(candidate - target) < abs(held - target):
                chosen.append(date)
                held = candidate
        selected.extend(chosen)
        records[family] = {
            "eligible_films": eligible_films,
            "target_test_films": target,
            "test_films": held,
            "test_dates": [date.date().isoformat() for date in chosen],
        }
    return selected, records


def _split_summary(
    parameters: pd.DataFrame, dates: pd.Series, ids: pd.Index
) -> dict[str, object]:
    """Summarize one side of the split: dates, stacks, thickness reach."""
    films = parameters.loc[ids]
    side_dates = dates.loc[ids]
    stacks = films[SUBSTRATE_STACK_COLUMN].value_counts()
    thickness = films[THICKNESS_COLUMN]
    return {
        "n_films": len(ids),
        "n_dates": int(side_dates.nunique()),
        "dates": [day.date().isoformat() for day in sorted(pd.DatetimeIndex(side_dates.unique()))],
        "stacks": {str(stack): int(count) for stack, count in stacks.items()},
        "thickness_nm": {
            "min": float(thickness.min()),
            "median": float(thickness.median()),
            "max": float(thickness.max()),
        },
    }


def date_block_holdout(
    parameters: pd.DataFrame,
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
    rare_stack_max_films: int = RARE_STACK_MAX_FILMS,
) -> SplitResult:
    """Split films into fitting and held-out sets by fabrication date.

    Whole days move together (same-day films are correlated samples),
    held-out days are chosen within each substrate family so both
    families appear on both sides, and days containing rare-stack films
    (at most ``rare_stack_max_films`` films in total) are locked to
    fitting, since a few films support no held-out claim.

    Args:
        parameters: Film parameter table in dataset row order, as
            produced by :func:`absynth.loader.load_dataset`; needs the
            date, substrate-stack and thickness columns.
        test_fraction: Target share of each family's eligible films
            for the held-out side.
        seed: Seed for the date selection; same seed, same split.
        rare_stack_max_films: Stacks with at most this many films in
            total count as rare; their dates are locked to fitting, so
            held-out metrics describe the common stacks only.

    Returns:
        A :class:`SplitResult` whose masks index the parameter-table
        row order (the curve rows) and whose provenance record states
        every choice the split made.

    Raises:
        SplitError: If dates or stacks are unusable, the fraction is
            out of range, or no date can be held out under the policy.
    """
    if not 0.0 < test_fraction < 1.0:
        raise SplitError(f"test_fraction must be strictly between 0 and 1, got {test_fraction}")

    dates = _parse_dates(parameters)
    family_of_date = _stack_families(parameters, dates)
    films_per_date = dates.groupby(dates).size()

    stack_totals = parameters[SUBSTRATE_STACK_COLUMN].value_counts()
    rare_stacks = sorted(stack_totals.index[stack_totals < rare_stack_max_films].tolist())
    is_rare = parameters[SUBSTRATE_STACK_COLUMN].isin(rare_stacks)
    locked_dates = sorted(pd.DatetimeIndex(dates[is_rare].unique()))
    all_dates = sorted(pd.DatetimeIndex(dates.unique()))
    eligible_dates = [date for date in all_dates if date not in set(locked_dates)]
    if not eligible_dates:
        raise SplitError("every date contains a rare-stack film; no date can be held out")

    test_dates, family_records = _select_test_dates(
        eligible_dates, family_of_date, films_per_date, test_fraction, np.random.default_rng(seed)
    )
    if not test_dates:
        raise SplitError("the seeded selection held out no films; raise test_fraction")

    train_dates = [date for date in all_dates if date not in set(test_dates)]
    train_ids = parameters.index[dates.isin(train_dates)]
    test_ids = parameters.index[dates.isin(test_dates)]

    provenance: dict[str, object] = {
        "method": "date_block_holdout",
        "seed": int(seed),
        "test_fraction": float(test_fraction),
        "rare_stack_max_films": int(rare_stack_max_films),
        "rare_stacks": rare_stacks,
        "train_locked_dates": [date.date().isoformat() for date in locked_dates],
        "families": family_records,
        "train": _split_summary(parameters, dates, train_ids),
        "test": _split_summary(parameters, dates, test_ids),
    }
    return SplitResult(
        train_ids=train_ids, test_ids=test_ids, row_index=parameters.index, provenance=provenance
    )
