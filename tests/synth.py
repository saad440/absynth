"""Shared synthetic dataset builders for the test suite.

Mirrors the shipped archive's schema (see ``data/README.md``): a
sample-by-wavelength optical-density matrix with ``NaN`` for unmeasured
points, and a parameter table with the columns the audit reads. No
real data — every value below is invented.

The eight synthetic films each exercise a different diagnostic:

- ``normal`` — healthy, fully measured curve;
- ``truncated`` — measured data stops two grid points early;
- ``late_start`` — nothing measured at the first grid point;
- ``interior_gap`` — a hole between measured points;
- ``negative`` — fully measured, one slightly negative baseline point;
- ``flat`` — constant nonzero curve, unparseable fabrication date;
- ``zero`` — constant zero curve, zero thickness;
- ``empty`` — nothing measured at all, missing primary ``ODint``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from absynth.audit import OD_MEASURED_COLUMN, ODINT_COLUMNS, ODINT_PRIMARY
from absynth.loader import THICKNESS_COLUMN, CurveSet, Dataset

WAVELENGTHS = np.array([300.0, 400.0, 500.0, 600.0, 700.0])
STEP = 100.0  # uniform grid step, nm

# NaN marks an unmeasured wavelength, the archive's convention.
VALUES = {
    "normal": [0.001, 0.010, 0.020, 0.010, 0.001],
    "truncated": [0.002, 0.020, 0.030, np.nan, np.nan],
    "late_start": [np.nan, 0.002, 0.004, 0.002, 0.001],
    "interior_gap": [0.001, 0.010, np.nan, 0.010, 0.001],
    "negative": [0.001, -0.002, 0.005, 0.004, 0.002],
    "flat": [0.02, 0.02, 0.02, 0.02, 0.02],
    "zero": [0.0, 0.0, 0.0, 0.0, 0.0],
    "empty": [np.nan, np.nan, np.nan, np.nan, np.nan],
}
SAMPLE_IDS = list(VALUES)


def make_curves() -> CurveSet:
    """A CurveSet with one synthetic film per diagnostic category."""
    values = np.array([VALUES[sample_id] for sample_id in SAMPLE_IDS])
    return CurveSet(
        wavelengths_nm=WAVELENGTHS,
        sample_ids=pd.Index(SAMPLE_IDS, name="Sample"),
        optical_density=values,
    )


def make_parameters() -> pd.DataFrame:
    """A parameter table mirroring the real database's audit columns.

    Dates are ``DD/MM/YYYY`` (the archive's convention); one is
    deliberately garbage. Substrate stacks come in two flavours.
    """
    frame = pd.DataFrame(
        {
            "Date": [
                "01/02/2023",
                "05/03/2023",
                "11/04/2023",
                "02/05/2023",
                "20/06/2023",
                "not a date",
                "15/08/2023",
                "03/09/2023",
            ],
            "Substrate_stack_sequence": [
                "corning",
                "corning",
                "glass",
                "corning",
                "glass",
                "corning",
                "corning",
                "corning",
            ],
            "Perovskite_synthesis_composition_short": ["MAPbICl"] * 8,
            "Perovskite_synthesis_solvents": ["ME/ACN"] * 8,
            "Perovskite_synthesis_additives_surfactants": [
                "NaLES/DDABr",
                "none",
                "NaLES",
                "NaLES/DDABr",
                "none",
                "NaLES",
                "NaLES/DDABr",
                "none",
            ],
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
    """The full synthetic dataset: curves joined with parameters."""
    return Dataset(curves=make_curves(), parameters=make_parameters())
