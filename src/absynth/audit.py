"""Integrity diagnostics for the loaded dataset.

The loader (:mod:`absynth.loader`) enforces the archive's structural
contract at read time. This module measures content quality on top of
a loaded :class:`~absynth.loader.Dataset` — missingness, out-of-range
values, degenerate curves, and cross-table consistency — and returns
plain pandas/numpy structures. It deliberately reports without fixing.
What to do about a finding is then a deliberate decision, not a silent
side effect.

Two views are produced:

- A per-film diagnostics table (one row per film) with missingness and
  value facts for each curve, suitable for a CSV quality-control table.
- A JSON-serializable summary aggregating those facts over the
  population and adding parameter-table facts (missing ``ODint``
  entries, thickness sanity, and the measured-flag cross-check).

Conventions carried over from the loader:

- Curve values are optical density ``OD(λ) = A(λ)/t`` in nm⁻¹, with
  film thickness ``t`` in nm.
- ``NaN`` means "wavelength not measured"; missing values are counted
  and classified (late start / early end / interior gap), never filled.
- Slightly negative values are possible after instrument baseline
  corrections; they are counted, not clipped.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from absynth.loader import THICKNESS_COLUMN, CurveSet, Dataset

ODINT_PRIMARY = "Perovskite_ODint_300_900_nm"
ODINT_COLUMNS = (
    ODINT_PRIMARY,
    "Perovskite_ODint_300_600_nm",
    "Perovskite_ODint_400_700_nm",
    "Perovskite_ODint_500_800_nm",
    "Perovskite_ODint_600_900_nm",
)
OD_MEASURED_COLUMN = "Perovskite_OD_measured"
DATE_COLUMN = "Date"
SUBSTRATE_STACK_COLUMN = "Substrate_stack_sequence"
ADDITIVES_COLUMN = "Perovskite_synthesis_additives_surfactants"
DATE_FORMAT = "%d/%m/%Y"  # the archive's fabrication-date convention


@dataclass(frozen=True)
class AuditReport:
    """Per-film curve diagnostics plus a population-level summary.

    Attributes:
        curve_diagnostics: One row per film (indexed by sample
            identifier) with the columns produced by
            :func:`curve_diagnostics`.
        summary: JSON-serializable dictionary of dataset-level counts
            and aggregates, combining curve and parameter findings.
    """

    curve_diagnostics: pd.DataFrame
    summary: dict[str, object]


def curve_diagnostics(curves: CurveSet) -> pd.DataFrame:
    """Compute one row of missingness/value facts per film.

    A wavelength counts as *measured* when its value is finite; the
    archive marks unmeasured points with ``NaN``. Flatness is defined
    as ``max == min`` over the measured values (a constant curve);
    a curve that is zero everywhere is a special case of flat, so
    ``n_flat_films >= n_zero_films`` always holds.

    Returns:
        DataFrame indexed by sample identifier with columns:

        - ``n_measured``, ``n_missing``: grid points measured / not.
        - ``first_measured_nm``, ``last_measured_nm``: wavelength of
          the first/last measured point (NaN for an empty curve).
        - ``starts_late``, ``ends_early``: measured data does not
          reach the start/end of the grid.
        - ``has_interior_gap``: unmeasured points between two measured
          ones (a hole, as opposed to a clean truncation).
        - ``od_min``, ``od_max``: extremes over measured values, in
          nm⁻¹ (NaN for an empty curve).
        - ``n_negative``: measured values below zero.
        - ``is_empty`` (nothing measured), ``is_flat``, ``is_zero``.
    """
    values = curves.optical_density
    finite = np.isfinite(values)
    n_points = values.shape[1]

    any_measured = finite.any(axis=1)
    first_index = np.where(any_measured, finite.argmax(axis=1), -1)
    # argmax on the reversed rows finds the first measured point from
    # the end; converting back gives the last measured index.
    last_index = np.where(any_measured, n_points - 1 - finite[:, ::-1].argmax(axis=1), -1)

    n_missing = (~finite).sum(axis=1)
    # Missing points before the first and after the last measured one
    # are truncation; the remainder, if any, are interior holes.
    leading_missing = np.where(any_measured, first_index, 0)
    trailing_missing = np.where(any_measured, n_points - 1 - last_index, 0)
    interior_missing = n_missing - leading_missing - trailing_missing

    # Extremes over measured points only: unmeasured slots are replaced
    # by neutral infinities, and fully empty rows end up as NaN.
    padded_low = np.where(finite, values, np.inf)
    padded_high = np.where(finite, values, -np.inf)
    od_min = np.where(any_measured, padded_low.min(axis=1), np.nan)
    od_max = np.where(any_measured, padded_high.max(axis=1), np.nan)

    is_flat = any_measured & (od_max == od_min)
    is_zero = any_measured & (od_max == 0.0) & (od_min == 0.0)

    wavelengths = curves.wavelengths_nm
    return pd.DataFrame(
        {
            "n_measured": finite.sum(axis=1),
            "n_missing": n_missing,
            "first_measured_nm": np.where(any_measured, wavelengths[first_index], np.nan),
            "last_measured_nm": np.where(any_measured, wavelengths[last_index], np.nan),
            "starts_late": any_measured & (first_index > 0),
            "ends_early": any_measured & (last_index < n_points - 1),
            # An empty curve has no span between measured points, so it
            # is not an interior gap — guard explicitly.
            "has_interior_gap": any_measured & (interior_missing > 0),
            "od_min": od_min,
            "od_max": od_max,
            "n_negative": ((values < 0) & finite).sum(axis=1),
            "is_empty": ~any_measured,
            "is_flat": is_flat,
            "is_zero": is_zero,
        },
        index=curves.sample_ids,
    )


def summarize_curve_diagnostics(diag: pd.DataFrame, curves: CurveSet) -> dict[str, object]:
    """Aggregate the per-film table into a population-level summary.

    Empty films carry no wavelength information, so the ``min``/
    ``max``-measured-wavelength aggregates below are computed over the
    films with at least one measured point (their count is reported as
    ``n_empty_films``).

    Returns:
        Dictionary with scalar counts and grid facts. All values are
        plain Python types so the result survives ``json.dumps``.
    """
    wavelengths = curves.wavelengths_nm
    steps = np.diff(wavelengths)
    measured = diag.loc[~diag["is_empty"], "last_measured_nm"]
    return {
        "n_samples": len(diag),
        "n_wavelengths": int(curves.n_wavelengths),
        "grid_start_nm": float(wavelengths[0]),
        "grid_end_nm": float(wavelengths[-1]),
        "grid_step_nm": float(steps[0]),
        "grid_points_measured": int(diag["n_measured"].sum()),
        "grid_points_missing": int(diag["n_missing"].sum()),
        "n_curves_complete": int((diag["n_missing"] == 0).sum()),
        "n_curves_with_missing": int((diag["n_missing"] > 0).sum()),
        "n_starts_late": int(diag["starts_late"].sum()),
        "n_ends_early": int(diag["ends_early"].sum()),
        "n_with_interior_gap": int(diag["has_interior_gap"].sum()),
        "n_empty_films": int(diag["is_empty"].sum()),
        "earliest_last_measured_nm": float(measured.min()),
        "latest_first_measured_nm": float(
            diag.loc[~diag["is_empty"], "first_measured_nm"].max()
        ),
        "n_negative_films": int((diag["n_negative"] > 0).sum()),
        "n_negative_values": int(diag["n_negative"].sum()),
        "od_min": float(diag["od_min"].min()),
        "n_flat_films": int(diag["is_flat"].sum()),
        "n_zero_films": int(diag["is_zero"].sum()),
    }


def parameter_diagnostics(parameters: pd.DataFrame) -> dict[str, object]:
    """Summarize parameter-table completeness and value sanity.

    Checks the quantities the audit and the paper's subset definition
    depend on: the five integrated-density columns (missing values
    here are what the paper's first 245-film filter drops), the
    boolean "curve evaluated" flag, and film thickness.

    Returns:
        Dictionary with plain Python values; see the keys below.

        - ``n_rows``: number of films in the table.
        - ``missing_odint_by_column``: missing-value count per
          ``ODint`` window (units a.u.·nm, as shipped).
        - ``rows_missing_primary_odint``: films with no
          300–900 nm ``ODint`` (the paper's filter target).
        - ``od_measured_true`` / ``od_measured_false`` /
          ``od_measured_missing``: flag tallies.
        - ``thickness_missing``, ``thickness_le_zero``: thickness
          sanity, in nm.
        - ``thickness_min_nm``, ``thickness_max_nm``.
        - ``n_substrate_stacks``, ``substrate_missing``: how many
          distinct substrate stack sequences, and films without one.
        - ``date_min`` / ``date_max`` (ISO dates) and
          ``dates_unparseable``: the fabrication-date range under
          ``DATE_FORMAT``, plus the count of unusable entries.
    """
    thickness = parameters[THICKNESS_COLUMN]
    flag = parameters[OD_MEASURED_COLUMN]
    flag_true = int((flag == True).sum())  # flag may be object-typed
    flag_false = int((flag == False).sum())
    return {
        "n_rows": len(parameters),
        "missing_odint_by_column": {
            column: int(parameters[column].isna().sum()) for column in ODINT_COLUMNS
        },
        "rows_missing_primary_odint": int(parameters[ODINT_PRIMARY].isna().sum()),
        "od_measured_true": flag_true,
        "od_measured_false": flag_false,
        "od_measured_missing": int(len(flag) - flag_true - flag_false),
        "thickness_missing": int(thickness.isna().sum()),
        "thickness_le_zero": int((thickness <= 0).sum()),
        "thickness_min_nm": float(thickness.min()),
        "thickness_max_nm": float(thickness.max()),
        "n_substrate_stacks": int(parameters[SUBSTRATE_STACK_COLUMN].nunique()),
        "substrate_missing": int(parameters[SUBSTRATE_STACK_COLUMN].isna().sum()),
        **_date_facts(parameters[DATE_COLUMN]),
    }


def subset_min_observations(
    parameters: pd.DataFrame, column: str, min_count: int
) -> pd.Index:
    """Rows whose category in *column* has at least *min_count* observations.

    The source paper builds its modelling subset by excluding additive
    categories "with too few observations" (threshold unspecified,
    its §2.3). This implements that stated step so the audit can test
    which thresholds reproduce the published subset size. The other
    stated step — dropping films with a missing target ODint —
    removes no rows from the shipped archive and is left to the
    caller to compose.

    Returns:
        Indices of the kept rows, in table order.
    """
    counts = parameters[column].value_counts()
    kept_arms = counts.index[counts >= min_count]
    return parameters.index[parameters[column].isin(kept_arms)]


def _date_facts(dates: pd.Series) -> dict[str, object]:
    """Parse fabrication dates and reduce them to a JSON-friendly range.

    The archive stores ``DD/MM/YYYY`` strings. Anything that does not
    parse under ``DATE_FORMAT`` becomes ``NaT`` and is counted, never
    dropped silently — the range then describes the parseable films.
    """
    parsed = pd.to_datetime(dates, format=DATE_FORMAT, errors="coerce")
    valid = parsed.dropna()
    return {
        "dates_unparseable": int(parsed.isna().sum()),
        "date_min": valid.min().date().isoformat() if len(valid) else None,
        "date_max": valid.max().date().isoformat() if len(valid) else None,
    }


def audit_dataset(dataset: Dataset) -> AuditReport:
    """Run all integrity diagnostics on a loaded dataset.

    Also cross-checks the two views: the ``Perovskite_OD_measured``
    flag against whether each curve actually has measured points.
    Agreement is expected; disagreements are reported, not resolved —
    deciding which view to trust belongs to the audit report.
    """
    diag = curve_diagnostics(dataset.curves)
    summary: dict[str, object] = summarize_curve_diagnostics(diag, dataset.curves)
    summary.update(parameter_diagnostics(dataset.parameters))

    # True = curve says measured, flag says not (or vice versa).
    curve_says_measured = ~diag["is_empty"]
    flag_says_measured = dataset.parameters[OD_MEASURED_COLUMN] == True
    mismatch = curve_says_measured != flag_says_measured
    summary["flag_curve_disagreements"] = int(mismatch.sum())
    summary["flag_false_curve_measured"] = int(
        (mismatch & curve_says_measured).sum()
    )
    summary["flag_true_curve_empty"] = int((mismatch & ~curve_says_measured).sum())

    return AuditReport(curve_diagnostics=diag, summary=summary)
