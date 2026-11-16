"""Provenance helpers shared by the experiment scripts.

Every experiment record carries the facts that make it reproducible
and auditable: the configuration path, the git state of the working
tree, and the versions of everything the numbers depend on. These
helpers collect those facts, so all scripts write records of the same
shape and no script can quietly omit them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import sklearn
import torch

import absynth
from absynth import __version__


def _run_git(*arguments: str) -> str | None:
    """Trailing newline removed; ``None`` if git is unavailable.

    Only the trailing newline is removed: ``git status --porcelain``
    marks staged/unstaged state with *leading* spaces, which stripping
    would corrupt.
    """
    try:
        result = subprocess.run(["git", *arguments], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip("\n")


def _path_is_untracked(relative: str, untracked_directories: set[str]) -> bool:
    """True if the path itself, or any directory holding it, is untracked."""
    parts = relative.split("/")
    return any(
        "/".join(parts[:count]) in untracked_directories for count in range(1, len(parts) + 1)
    )


def git_provenance(config_path: Path, script_path: Path) -> dict[str, object] | None:
    """Revision, tracked source changes, and untracked files that shape the run.

    Three facts, each stated with its own precision. The revision is
    the exact code state when tracked files are unmodified.
    ``tracked_changes`` names tracked source files that differ from
    it. Generated artifacts are excluded on both sides: records and
    figures never shape a run — the numbers depend on sources,
    configs and the data checksum, and each artifact carries its own
    provenance. On the untracked side, only files that actually shape
    the numbers are reported: modules the interpreter imported from
    the package tree, the calling script, and the configuration file.

    Args:
        config_path: Path of the configuration file the run reads.
        script_path: Path of the calling script, typically
            ``Path(__file__)``.
    """
    revision = _run_git("rev-parse", "HEAD")
    toplevel = _run_git("rev-parse", "--show-toplevel")
    status = _run_git("status", "--porcelain")
    if revision is None or toplevel is None or status is None:
        return None
    toplevel_path = Path(toplevel).resolve()
    # A wholly untracked directory collapses to one "?? directory/"
    # entry, so membership has to ask about ancestor directories too.
    untracked = {line[3:].rstrip("/") for line in status.splitlines() if line[:2] == "??"}
    # Generated artifacts never shape a run, so modified records and
    # figures stay out of the tracked changes — the tracked-side
    # counterpart of the untracked_source rule below.
    generated = ("reports/", "figures/")
    tracked_changes = [
        line[3:]
        for line in status.splitlines()
        if line[:2] != "??" and not line[3:].startswith(generated)
    ]

    package_root = Path(absynth.__file__).resolve().parent
    shaped: set[Path] = {script_path.resolve(), config_path.resolve()}
    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            module_path = Path(module_file).resolve()
            if package_root in module_path.parents:
                shaped.add(module_path)

    untracked_source = []
    for path in sorted(shaped):
        try:
            relative = path.relative_to(toplevel_path).as_posix()
        except ValueError:  # a file outside the repository cannot be untracked in it
            continue
        if _path_is_untracked(relative, untracked):
            untracked_source.append(relative)
    return {
        "revision": revision,
        "tracked_changes": tracked_changes,
        "untracked_source": untracked_source,
    }


def package_versions() -> dict[str, str]:
    """Versions of everything the record's numbers depend on."""
    return {
        "absynth": __version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit-learn": sklearn.__version__,
        "torch": torch.__version__,
    }
