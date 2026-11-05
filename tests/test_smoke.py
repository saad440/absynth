"""Smoke test: the package installs and reports its version."""

import absynth


def test_version():
    assert absynth.__version__ == "0.1.0"
