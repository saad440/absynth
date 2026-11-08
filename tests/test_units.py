"""Tests for unit-aware preprocessing — synthetic curves, no real data."""

import numpy as np
import pandas as pd
import pytest

from absynth.loader import CurveSet, DataContractError
from absynth.preprocessing import WORKING_SPAN_NM, require_complete, restrict_to_span

SAMPLE_IDS = ["BPD_sample_001", "BPD_sample_002"]


def make_curves(values, wavelengths_nm):
    """Build a CurveSet directly; ``values`` is (n_samples, n_wavelengths)."""
    return CurveSet(
        wavelengths_nm=np.asarray(wavelengths_nm, dtype=float),
        sample_ids=pd.Index(SAMPLE_IDS[: len(values)], name="Sample"),
        optical_density=np.asarray(values, dtype=float),
    )


def test_default_span_is_the_audited_working_span():
    # Pin the policy: all 517 films are complete over exactly this window.
    assert WORKING_SPAN_NM == (300.0, 900.0)


def test_span_slicing_is_inclusive_and_preserves_rows():
    curves = make_curves(
        [[0.001, 0.010, 0.020, 0.010, 0.001], [0.002, 0.020, 0.030, 0.020, 0.002]],
        wavelengths_nm=[300.0, 302.0, 304.0, 306.0, 308.0],
    )
    sliced = restrict_to_span(curves, span_nm=(302.0, 306.0))
    assert sliced.wavelengths_nm.tolist() == [302.0, 304.0, 306.0]
    assert sliced.sample_ids.tolist() == SAMPLE_IDS
    assert sliced.optical_density.shape == (2, 3)
    assert sliced.optical_density[1, 1] == pytest.approx(0.030)


def test_too_narrow_or_inverted_span_raises():
    curves = make_curves([[0.0] * 6], wavelengths_nm=np.arange(300.0, 311.0, 2.0))
    with pytest.raises(DataContractError, match="at least 2"):
        restrict_to_span(curves, span_nm=(303.0, 304.0))
    with pytest.raises(DataContractError, match="ascending"):
        restrict_to_span(curves, span_nm=(306.0, 302.0))


def test_require_complete_passes_without_nan():
    curves = make_curves([[0.01, 0.02], [0.03, 0.04]], wavelengths_nm=[300.0, 302.0])
    require_complete(curves)  # must not raise


def test_require_complete_names_offending_films():
    values = [[0.01, np.nan, 0.02], [0.03, 0.04, 0.05]]
    curves = make_curves(values, wavelengths_nm=[300.0, 302.0, 304.0])
    with pytest.raises(DataContractError) as excinfo:
        require_complete(curves)
    assert "1 curves" in str(excinfo.value)
    assert "BPD_sample_001" in str(excinfo.value)


def test_working_span_recipe_on_a_truncated_archive_replica():
    # Mirror the shipped archive: 300-1000 nm at 2 nm, one film whose
    # curve stops at 900 nm. The working-span recipe keeps it; the
    # completeness check alone refuses the full grid.
    wavelengths_nm = np.arange(300.0, 1000.0 + 1.0, 2.0)
    assert len(wavelengths_nm) == 351
    complete = np.linspace(0.001, 0.02, wavelengths_nm.size)
    truncated = np.where(wavelengths_nm > 900.0, np.nan, complete)
    curves = make_curves([complete, truncated], wavelengths_nm=wavelengths_nm)

    with pytest.raises(DataContractError, match="unmeasured"):
        require_complete(curves)

    working = restrict_to_span(curves)
    assert working.wavelengths_nm[0] == 300.0
    assert working.wavelengths_nm[-1] == pytest.approx(900.0)
    require_complete(working)  # the truncated tail is gone; must not raise
