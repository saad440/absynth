"""Write the machine-readable quality-control artifacts.

Runs the audit (:mod:`absynth.audit`) on a loaded dataset and writes
two files into a reports directory:

- ``curve_diagnostics.csv`` — the per-film diagnostics table;
- ``qc_summary.json`` — the population-level summary.

Every number quoted in ``reports/data_audit.md`` comes from these
artifacts, so tables, JSON and prose cannot drift apart. Raw data is
not committed to the repository (see ``data/README.md``); these small
derived artifacts are.
"""

from __future__ import annotations

import json
from pathlib import Path

from absynth.audit import AuditReport, audit_dataset
from absynth.loader import Dataset

CURVE_DIAGNOSTICS_CSV = "curve_diagnostics.csv"
QC_SUMMARY_JSON = "qc_summary.json"


def write_qc_report(dataset: Dataset, out_dir: str | Path) -> dict[str, object]:
    """Audit ``dataset`` and write the QC table and summary.

    Args:
        dataset: A loaded dataset; the loader's structural contract is
            assumed to have been enforced at read time.
        out_dir: Directory for the artifacts; created if missing.

    Returns:
        The summary dictionary — identical to the JSON file written.
    """
    report: AuditReport = audit_dataset(dataset)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.curve_diagnostics.to_csv(out_dir / CURVE_DIAGNOSTICS_CSV)
    (out_dir / QC_SUMMARY_JSON).write_text(json.dumps(report.summary, indent=2) + "\n")
    return report.summary
