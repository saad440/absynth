"""Smoke test: the package installs and reports its version."""

import spectralab


def test_version():
    assert spectralab.__version__ == "0.1.0"
