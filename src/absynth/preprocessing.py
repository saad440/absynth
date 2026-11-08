"""Unit-aware preprocessing of optical-density curves.

The archive ships optical density as the source paper defines it:
``OD(λ) = A(λ)/t`` with thickness ``t`` in nanometres
The thickness-normalised quantity called ``q`` elsewhere in this
package, in units of nm⁻¹ rather than absorbance itself.

Wavelength-grid contract
------------------------

Curves arrive on a uniform 2 nm grid from 300 to 1000 nm, while the
source paper reports a 10 nm scanning step over 300–900 nm. We treat
the 2 nm spacing as a *storage grid*, not verified optical resolution.
Two policies follow:

- No smoothing, resampling or interpolation is applied here: grids
  pass through verbatim, and any future edge-position analysis must
  restate this policy next to its numbers.
- Analysis defaults to the *working span*, 300–900 nm, the window over
  which every shipped curve is measured. A missing point inside the
  working span means the data no longer match the audited archive, so
  an exception is raised.

Units and thickness
-------------------

Curves live in nm⁻¹ (``q``), absorbance ``A`` is dimensionless, and
the conversions below multiply or divide by thickness in nm. A
forgotten or doubled thickness factor therefore surfaces as an
order-of-magnitude error, not a shape error.
The audited archive contains no negative values, and nothing here
clips or shifts them. Small negatives in a future dataset would need
diagnosis before any positivity transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from absynth.loader import CurveSet, DataContractError

WORKING_SPAN_NM = (300.0, 900.0)


def restrict_to_span(
    curves: CurveSet, span_nm: tuple[float, float] = WORKING_SPAN_NM
) -> CurveSet:
    """Slice curves to the wavelengths inside ``span_nm``, bounds inclusive.

    The grid itself is never altered: slicing a uniform grid keeps it
    uniform, and no smoothing or resampling happens here.
    Raises :class:`DataContractError` if fewer than two wavelengths
    fall inside the span.
    """
    lo, hi = span_nm
    if lo > hi:
        raise DataContractError(f"span bounds must be ascending, got {lo} to {hi}")
    mask = (curves.wavelengths_nm >= lo) & (curves.wavelengths_nm <= hi)
    n_inside = int(mask.sum())
    if n_inside < 2:
        raise DataContractError(
            f"span {lo}-{hi} nm contains {n_inside} wavelengths, need at least 2"
        )
    return CurveSet(
        wavelengths_nm=curves.wavelengths_nm[mask],
        sample_ids=curves.sample_ids,
        optical_density=curves.optical_density[:, mask],
    )


def require_complete(curves: CurveSet) -> None:
    """Raise unless every film is measured at every grid wavelength.

    The loader passes NaN through as "wavelength not measured". Over
    the working span the audited archive has none, so a NaN signals
    that the data no longer match the audit.
    """
    incomplete = np.isnan(curves.optical_density).any(axis=1)
    if incomplete.any():
        offenders = curves.sample_ids[incomplete]
        raise DataContractError(
            f"{len(offenders)} curves have unmeasured wavelengths, "
            f"e.g. {sorted(offenders)[:5]}"
        )


def _checked_thickness(
    thickness_nm: pd.Series | np.ndarray, n_samples: int
) -> np.ndarray:
    """Validate a per-film thickness vector. Return it as a plain array."""
    thickness_nm = np.asarray(thickness_nm, dtype=float)
    if thickness_nm.shape != (n_samples,):
        raise ValueError(
            f"thickness has shape {thickness_nm.shape}, expected ({n_samples},)"
        )
    if not np.isfinite(thickness_nm).all():
        raise ValueError("thickness contains non-finite values")
    if (thickness_nm <= 0).any():
        raise ValueError("thickness must be positive to convert between A and q")
    return thickness_nm


def absorbance_from_q(
    q: np.ndarray, thickness_nm: pd.Series | np.ndarray
) -> np.ndarray:
    """Recover absorbance from thickness-normalised curves, ``A = q·t``.

    Args:
        q: Curve matrix in nm⁻¹, shape ``(n_samples, n_wavelengths)``.
        thickness_nm: Film thickness in nm, one entry per *row* of
            ``q``, in the same order. A Series is consumed by position
            (its index is ignored); ``Dataset.thickness_nm`` is already
            aligned to the curve rows by the loader and is the safe
            choice.

    Returns:
        Absorbance matrix, dimensionless (base-10 optical depth).
    """
    if q.ndim != 2:
        raise ValueError(f"expected a 2-D curve matrix, got shape {q.shape}")
    thickness_nm = _checked_thickness(thickness_nm, q.shape[0])
    return q * thickness_nm[:, np.newaxis]


def q_from_absorbance(
    absorbance: np.ndarray, thickness_nm: pd.Series | np.ndarray
) -> np.ndarray:
    """Convert absorbance to thickness-normalised curves, ``q = A/t``.

    Inverse of :func:`absorbance_from_q`; the result is in nm⁻¹ when
    the thickness is in nm. A zero or negative thickness cannot
    normalise a curve and is raised, never divided through silently.
    """
    if absorbance.ndim != 2:
        raise ValueError(f"expected a 2-D curve matrix, got shape {absorbance.shape}")
    thickness_nm = _checked_thickness(thickness_nm, absorbance.shape[0])
    return absorbance / thickness_nm[:, np.newaxis]
