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
"""

from __future__ import annotations

import numpy as np

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
