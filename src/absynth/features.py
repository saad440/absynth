"""Physically interpretable per-film spectral descriptors.

Descriptors compress each curve into few numbers with explicit
physical meaning: integrated attenuation over wavelength windows (the
quantity the source paper ships as ``ODint`` columns), the baseline
level below the absorption edge, and the position and width of the
edge descent itself. They serve physical interpretation and the
amplitude/edge/baseline decomposition of curve variability.

All descriptors consume ``q`` curves in nm⁻¹ (the archive's
``OD = A/t`` convention; :mod:`absynth.preprocessing`) and index
results by sample identifier. Windows are stated in nanometres;
integrals use the trapezoidal rule over the storage grid — no
smoothing, no resampling (grid contract in the preprocessing module).
A window only requires the curves to be complete *inside* it, so
descriptors also work on the full grid where only the truncated tails
are unmeasured.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from absynth.loader import CurveSet

INTEGRATION_WINDOWS_NM: tuple[tuple[float, float], ...] = (
    (300.0, 900.0),
    (300.0, 600.0),
    (400.0, 700.0),
    (500.0, 800.0),
    (600.0, 900.0),
)
"""The five windows of the source paper's shipped ``ODint`` columns;
computing them from curves lets the notebook cross-check against the
database values."""

BASELINE_WINDOW_NM = (850.0, 900.0)
"""Sub-edge window for the baseline: comfortably below the ~775–800 nm
MAPbI₃ absorption edge (literature prior, not measurement), so
residual ``q`` there reflects scattering and other non-absorbing
losses rather than band-to-band absorption."""


def _window_values(curves: CurveSet, lo: float, hi: float) -> np.ndarray:
    """Curve columns inside ``[lo, hi]`` nm, guarding completeness there."""
    mask = (curves.wavelengths_nm >= lo) & (curves.wavelengths_nm <= hi)
    n_inside = int(mask.sum())
    if n_inside < 2:
        raise ValueError(
            f"window {lo}-{hi} nm contains {n_inside} grid points, need at least 2"
        )
    values = curves.optical_density[:, mask]
    if np.isnan(values).any():
        raise ValueError(
            f"window {lo}-{hi} nm contains unmeasured points; "
            "restrict to curves that are complete there first"
        )
    return values


def window_integrals(
    curves: CurveSet,
    windows_nm: Sequence[tuple[float, float]] = INTEGRATION_WINDOWS_NM,
) -> pd.DataFrame:
    """Integrate each curve over wavelength windows, one row per film.

    Args:
        curves: Curves in nm⁻¹ on the uniform storage grid.
        windows_nm: Inclusive ``(lo, hi)`` wavelength windows in nm; by
            default the five windows the source paper ships as
            ``ODint`` columns.

    Returns:
        One row per film (indexed by sample identifier), one column
        per window named ``q_int_<lo>_<hi>``. Values carry units of
        nm⁻¹·nm: the same integral computed on absorbance ``A``
        instead of ``q`` would differ by the thickness factor, which
        is why the units stay explicit rather than collapsing to 1.
    """
    columns = {}
    for lo, hi in windows_nm:
        values = _window_values(curves, lo, hi)
        mask = (curves.wavelengths_nm >= lo) & (curves.wavelengths_nm <= hi)
        columns[f"q_int_{lo:g}_{hi:g}"] = np.trapezoid(
            values, curves.wavelengths_nm[mask], axis=1
        )
    return pd.DataFrame(columns, index=curves.sample_ids)


def baseline_level(
    curves: CurveSet, window_nm: tuple[float, float] = BASELINE_WINDOW_NM
) -> pd.Series:
    """Mean ``q`` over the stated window — the non-absorbing floor.

    Defaults to 850–900 nm (see :data:`BASELINE_WINDOW_NM`). An
    elevated baseline flags scattering, substrate effects or a
    red-shifted edge; it is a diagnostic, not a value to subtract
    away. Returns nm⁻¹, one entry per film.
    """
    lo, hi = window_nm
    values = _window_values(curves, lo, hi)
    name = f"q_baseline_{lo:g}_{hi:g}"
    return pd.Series(values.mean(axis=1), index=curves.sample_ids, name=name)


PLATEAU_WINDOW_NM = (300.0, 500.0)
"""Above-edge window defining a curve's plateau: absorption is
saturated there, so the window median measures the curve's full
height. TiO₂-bearing substrate stacks absorb extra UV light below
~390 nm; the median keeps that from moving the plateau."""

EDGE_LEVELS = (0.9, 0.5, 0.1)
"""Descent levels, as fractions of the plateau, that define the edge
position (the 50% crossing) and the 10–90 width."""


def _descending_crossings(
    values: np.ndarray, wavelengths_nm: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Largest wavelength where each curve still meets its threshold.

    One threshold per row. A crossing only counts when a *measured*
    point below the threshold follows it, so a curve that stays high
    up to its truncation yields NaN instead of an edge at the
    truncation point.
    """
    _, n_cols = values.shape
    finite = np.isfinite(values)
    columns = np.arange(n_cols)
    above = finite & (values >= thresholds[:, np.newaxis])
    last_above = np.where(above, columns, -1).max(axis=1)
    below_after = (
        finite
        & (values < thresholds[:, np.newaxis])
        & (columns > last_above[:, np.newaxis])
    )
    crossed = (last_above >= 0) & below_after.any(axis=1)
    return np.where(crossed, wavelengths_nm[last_above.clip(min=0)], np.nan)


def edge_descriptors(
    curves: CurveSet, plateau_window_nm: tuple[float, float] = PLATEAU_WINDOW_NM
) -> pd.DataFrame:
    """Locate each curve's absorption edge on the storage grid.

    The plateau is the median ``q`` over ``plateau_window_nm``, the
    above-edge region where absorption saturates. Scanning down the
    descent, the edge position ``edge_50_nm`` is where the curve last
    meets half the plateau, and ``edge_width_10_90_nm`` spans the 90%
    and 10% crossings of the same descent.

    Grid caveat: crossings are grid points, so positions carry the
    2 nm quantisation of the storage grid — this is not an
    optical-resolution claim, and the grid's provenance is open (see
    ``reports/data_audit.md``). No smoothing is applied before the
    search. A curve whose descent never reaches a level inside the
    grid gets NaN for that descriptor; report NaN counts, never drop
    films silently.
    """
    plateau = np.median(_window_values(curves, *plateau_window_nm), axis=1)
    crossings = {
        level: _descending_crossings(
            curves.optical_density, curves.wavelengths_nm, level * plateau
        )
        for level in EDGE_LEVELS
    }
    return pd.DataFrame(
        {
            "edge_50_nm": crossings[0.5],
            "edge_width_10_90_nm": crossings[0.1] - crossings[0.9],
        },
        index=curves.sample_ids,
    )
