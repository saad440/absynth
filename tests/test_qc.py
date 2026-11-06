"""Tests for the QC report writer — synthetic dataset, temporary output."""

import json

import pandas as pd
from synth import SAMPLE_IDS, make_dataset

from absynth.qc import CURVE_DIAGNOSTICS_CSV, QC_SUMMARY_JSON, write_qc_report


def test_writes_table_and_summary(tmp_path):
    summary = write_qc_report(make_dataset(), tmp_path)

    # The JSON file must round-trip to exactly the returned dict —
    # this is the "same code path" guarantee for report numbers.
    saved = json.loads((tmp_path / QC_SUMMARY_JSON).read_text())
    assert saved == summary

    table = pd.read_csv(tmp_path / CURVE_DIAGNOSTICS_CSV, index_col=0)
    assert table.index.tolist() == SAMPLE_IDS
    assert int(table["n_measured"].sum()) == summary["grid_points_measured"]


def test_creates_missing_output_directory(tmp_path):
    out = tmp_path / "reports" / "qc"
    write_qc_report(make_dataset(), out)
    assert (out / QC_SUMMARY_JSON).exists()
