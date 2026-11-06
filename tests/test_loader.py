"""Tests for archive loading — tiny synthetic CSVs, no real data."""

import numpy as np
import pandas as pd
import pytest

from absynth.loader import (
    CURVES_CSV,
    DATABASE_CSV,
    METADATA_CSV,
    WAVELENGTH_COLUMN,
    DataContractError,
    load_curve_set,
    load_dataset,
    load_metadata,
    load_parameters,
)

WAVELENGTHS = [300.0, 302.0, 304.0, 306.0, 308.0]
SAMPLE_IDS = ["BPD_sample_001", "BPD_sample_002", "BPD_sample_003"]
# The third film is "truncated": its first two wavelengths are unmeasured,
# mirroring the shipped archive where some curves stop early.
CURVE_VALUES = np.array(
    [
        [0.001, 0.010, 0.020, 0.010, 0.001],
        [0.002, 0.020, 0.030, 0.020, 0.002],
        [np.nan, np.nan, 0.005, 0.010, 0.005],
    ]
)


def write_curves(path, wavelengths, sample_ids, values):
    # The CSV stores one row per wavelength; ``values`` is sample-major
    # (n_samples, n_wavelengths), hence the transpose on the way in.
    frame = pd.DataFrame(values.T, columns=sample_ids)
    frame.insert(0, WAVELENGTH_COLUMN, wavelengths)
    frame.to_csv(path, index=False)


def write_database(path, sample_ids, thicknesses):
    frame = pd.DataFrame(
        {
            "Sample": sample_ids,
            "Perovskite_thickness_nm": thicknesses,
            "Perovskite_deposition_blade_speed_mm_s": [10.0] * len(sample_ids),
        }
    )
    frame.to_csv(path, index=False)


@pytest.fixture
def archive(tmp_path):
    """A miniature archive: 3 films, 5 wavelengths, one truncated curve."""
    write_curves(tmp_path / CURVES_CSV, WAVELENGTHS, SAMPLE_IDS, CURVE_VALUES)
    write_database(tmp_path / DATABASE_CSV, SAMPLE_IDS, [300.0, 600.0, 1200.0])
    return tmp_path


def test_curve_set_preserves_grid_ids_and_nans(archive):
    curves = load_curve_set(archive / CURVES_CSV)
    assert curves.n_samples == 3
    assert curves.n_wavelengths == 5
    assert curves.wavelengths_nm.tolist() == WAVELENGTHS
    assert curves.sample_ids.tolist() == SAMPLE_IDS
    # NaNs survive the round trip exactly where they were written:
    assert np.isnan(curves.optical_density[2, :2]).all()
    assert curves.optical_density[2, 2] == pytest.approx(0.005)


def test_curve_set_tidy_frame_matches_matrix(archive):
    tidy = load_curve_set(archive / CURVES_CSV).to_frame()
    assert len(tidy) == 3 * 5
    per_film = tidy.groupby("sample_id")["optical_density"].apply(lambda s: s.to_numpy())
    for i, sample_id in enumerate(SAMPLE_IDS):
        assert np.array_equal(per_film[sample_id], CURVE_VALUES[i], equal_nan=True)


def test_curve_set_rejects_broken_grids(tmp_path):
    write_curves(tmp_path / CURVES_CSV, [300.0, 302.0, 303.0], SAMPLE_IDS[:2], np.zeros((2, 3)))
    with pytest.raises(DataContractError, match="uniform"):
        load_curve_set(tmp_path / CURVES_CSV)

    write_curves(tmp_path / CURVES_CSV, [304.0, 302.0], SAMPLE_IDS[:2], np.zeros((2, 2)))
    with pytest.raises(DataContractError, match="strictly increasing"):
        load_curve_set(tmp_path / CURVES_CSV)


def test_parameters_are_indexed_by_unique_sample(archive):
    parameters = load_parameters(archive / DATABASE_CSV)
    assert parameters.index.tolist() == SAMPLE_IDS
    assert parameters.index.name == "Sample"

    frame = pd.read_csv(archive / DATABASE_CSV)
    duplicated = pd.concat([frame, frame.iloc[[0]]])
    duplicated.to_csv(archive / DATABASE_CSV, index=False)
    with pytest.raises(DataContractError, match="duplicated"):
        load_parameters(archive / DATABASE_CSV)


def test_dataset_joins_and_orders_rows(archive):
    dataset = load_dataset(archive)
    assert dataset.curves.sample_ids.tolist() == dataset.parameters.index.tolist()
    assert dataset.thickness_nm.tolist() == [300.0, 600.0, 1200.0]


def test_dataset_rejects_identifier_mismatch(tmp_path):
    write_curves(tmp_path / CURVES_CSV, WAVELENGTHS, SAMPLE_IDS[:2], CURVE_VALUES[:2])
    write_database(tmp_path / DATABASE_CSV, SAMPLE_IDS, [300.0, 600.0, 1200.0])
    with pytest.raises(DataContractError) as excinfo:
        load_dataset(tmp_path)
    message = str(excinfo.value)
    assert "1 in database only" in message and "BPD_sample_003" in message


def test_dataset_requires_thickness(tmp_path):
    write_curves(tmp_path / CURVES_CSV, WAVELENGTHS, SAMPLE_IDS[:1], CURVE_VALUES[:1])
    frame = pd.DataFrame({"Sample": SAMPLE_IDS[:1], "Date": ["01/02/2023"]})
    frame.to_csv(tmp_path / DATABASE_CSV, index=False)
    with pytest.raises(DataContractError, match="thickness"):
        load_dataset(tmp_path)


def test_metadata_reads_windows_1252(tmp_path):
    frame = pd.DataFrame(
        {"Variable": ["Substrate_area_total_cm2"], "Description": ["area"], "Units": ["cm²"]}
    )
    frame.to_csv(tmp_path / METADATA_CSV, index=False, encoding="cp1252")
    loaded = load_metadata(tmp_path / METADATA_CSV)
    assert loaded.loc[0, "Units"] == "cm²"
