"""Load the published archive into analysis-ready structures.

The acquisition module (:mod:`absynth.data`) guarantees that the local
bytes match the published archive. This module turns those bytes into
pandas/numpy structures with a checked structural contract:

- curves arrive as a ``(sample_id x wavelength)`` matrix on a uniform
  nanometre grid;
- film parameters arrive indexed by the same sample identifiers;
- the two views are joined on *identical* identifier sets, so
  downstream code never re-checks alignment.

Conventions of the source archive (provenance in ``data/README.md``):

- Curve values are optical density, ``OD(λ) = A(λ) / t`` with
  thickness ``t`` in nm — units of ``nm⁻¹``, *not* absorbance itself.
- ``NaN`` entries mean "wavelength not measured", not zero; NaNs are
  passed through untouched, and imputation is a preprocessing
  decision, never a loader side effect.
- The variable-description CSV is Windows-1252 encoded (it contains
  characters such as ``²``); the database and curve CSVs are UTF-8.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATABASE_CSV = "blade_perovskite_database.csv"
CURVES_CSV = "curvedata_opticaldensity.csv"
METADATA_CSV = "blade_perovskite_metadata.csv"

SAMPLE_COLUMN = "Sample"
WAVELENGTH_COLUMN = "Wavelength_nm"
THICKNESS_COLUMN = "Perovskite_thickness_nm"
DATE_COLUMN = "Date"
SUBSTRATE_STACK_COLUMN = "Substrate_stack_sequence"

METADATA_ENCODING = "cp1252"
DATE_FORMAT = "%d/%m/%Y"  # the archive's fabrication-date convention


class DataContractError(RuntimeError):
    """Raised when archive files violate the structural contract."""


@dataclass(frozen=True)
class CurveSet:
    """Optical-density curves as a sample-by-wavelength matrix.

    Attributes:
        wavelengths_nm: Ascending uniform wavelength grid in nm,
            shape ``(n_wavelengths,)``.
        sample_ids: Film identifier for each row of ``optical_density``.
        optical_density: Curve matrix in nm⁻¹, shape
            ``(n_samples, n_wavelengths)``. ``NaN`` marks wavelengths
            that were not measured for that film.
    """

    wavelengths_nm: np.ndarray
    sample_ids: pd.Index
    optical_density: np.ndarray

    @property
    def n_samples(self) -> int:
        """Number of films (matrix rows)."""
        return len(self.sample_ids)

    @property
    def n_wavelengths(self) -> int:
        """Number of grid points (matrix columns)."""
        return len(self.wavelengths_nm)

    def to_frame(self) -> pd.DataFrame:
        """Return the tidy long form: one row per (film, wavelength).

        Columns: ``sample_id``, ``wavelength_nm`` (nm), and
        ``optical_density`` (nm⁻¹, with NaN where unmeasured). The
        long form is convenient for per-wavelength grouping and for
        plots; the matrix form is what models consume.
        """
        return pd.DataFrame(
            {
                "sample_id": np.repeat(np.asarray(self.sample_ids), self.n_wavelengths),
                "wavelength_nm": np.tile(self.wavelengths_nm, self.n_samples),
                "optical_density": self.optical_density.ravel(),
            }
        )


@dataclass(frozen=True)
class Dataset:
    """A curve matrix joined with its per-film parameter table.

    Attributes:
        curves: The optical-density curves.
        parameters: Film parameters indexed by sample identifier, with
            the row order matching ``curves.sample_ids``.
    """

    curves: CurveSet
    parameters: pd.DataFrame

    @property
    def thickness_nm(self) -> pd.Series:
        """Film thickness in nm, aligned with the curve rows."""
        return self.parameters[THICKNESS_COLUMN]


def _check_uniform_grid(wavelengths_nm: np.ndarray) -> None:
    """Raise unless the grid is finite, strictly increasing and uniform."""
    if wavelengths_nm.size < 2:
        raise DataContractError(
            f"wavelength grid needs at least 2 points, got {wavelengths_nm.size}"
        )
    if not np.isfinite(wavelengths_nm).all():
        raise DataContractError("wavelength grid contains non-finite values")
    steps = np.diff(wavelengths_nm)
    if np.any(steps <= 0):
        raise DataContractError("wavelength grid must be strictly increasing")
    if not np.isclose(steps, steps[0]).all():
        raise DataContractError(
            f"wavelength grid must be uniform, found steps from {steps.min()} to {steps.max()} nm"
        )


def load_curve_set(path: str | Path) -> CurveSet:
    """Read the curve CSV into a :class:`CurveSet`.

    The file has one ``Wavelength_nm`` column plus one column per film
    (``BPD_sample_001``…). Column order defines row order; the header
    names define the sample identifiers verbatim.
    """
    frame = pd.read_csv(path)
    if WAVELENGTH_COLUMN not in frame.columns:
        raise DataContractError(f"curve file lacks a {WAVELENGTH_COLUMN!r} column")
    wavelengths_nm = frame[WAVELENGTH_COLUMN].to_numpy(dtype=float)
    _check_uniform_grid(wavelengths_nm)
    sample_ids = frame.columns.drop(WAVELENGTH_COLUMN)
    if sample_ids.empty:
        raise DataContractError("curve file has no sample columns")
    # The CSV stores one row per wavelength and one column per film, so
    # the raw block is wavelength-major; CurveSet is sample-major.
    optical_density = frame[sample_ids].to_numpy(dtype=float).T
    return CurveSet(
        wavelengths_nm=wavelengths_nm,
        sample_ids=pd.Index(sample_ids, name=SAMPLE_COLUMN),
        optical_density=optical_density,
    )


def load_parameters(path: str | Path) -> pd.DataFrame:
    """Read the film database, indexed by unique sample identifier."""
    frame = pd.read_csv(path)
    if SAMPLE_COLUMN not in frame.columns:
        raise DataContractError(f"database lacks a {SAMPLE_COLUMN!r} column")
    duplicates = frame.index[frame[SAMPLE_COLUMN].duplicated()].tolist()
    if duplicates:
        raise DataContractError(
            f"database has duplicated {SAMPLE_COLUMN!r} values, e.g. "
            f"{sorted(frame.loc[duplicates, SAMPLE_COLUMN])[:5]}"
        )
    return frame.set_index(SAMPLE_COLUMN)


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Read the variable descriptions (``Variable``, ``Description``, ``Units``)."""
    return pd.read_csv(path, encoding=METADATA_ENCODING)


def load_dataset(root: str | Path) -> Dataset:
    """Load curves and parameters from an extracted archive directory.

    Raises :class:`DataContractError` if the curve columns and the
    database identifiers are not exactly the same set, or if the
    thickness column needed to interpret the OD convention is absent.
    """
    root = Path(root)
    curves = load_curve_set(root / CURVES_CSV)
    parameters = load_parameters(root / DATABASE_CSV)

    missing = parameters.index.difference(curves.sample_ids)
    extra = curves.sample_ids.difference(parameters.index)
    if len(missing) or len(extra):
        raise DataContractError(
            f"curve/database identifier mismatch: {len(missing)} in database only"
            f" (e.g. {sorted(missing)[:5]}), {len(extra)} in curves only"
            f" (e.g. {sorted(extra)[:5]})"
        )
    if THICKNESS_COLUMN not in parameters.columns:
        raise DataContractError(f"database lacks a {THICKNESS_COLUMN!r} column")
    parameters = parameters.loc[curves.sample_ids]
    return Dataset(curves=curves, parameters=parameters)
