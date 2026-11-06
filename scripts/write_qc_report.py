"""Write the QC table and summary from a local archive copy.

Usage (from the repository root, after ``pip install -e ".[dev]"``):

    python scripts/write_qc_report.py [--data-dir DIR] [--out-dir DIR]

Defaults: ``--data-dir data/raw/Dataset``, ``--out-dir reports``.
The script is orchestration only — all logic lives in ``absynth``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from absynth.loader import load_dataset
from absynth.qc import write_qc_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the QC table and summary for a local archive copy."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/Dataset"),
        help="extracted archive directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="directory for the CSV table and JSON summary",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    summary = write_qc_report(dataset, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
