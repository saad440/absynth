"""Tests for dataset download primitives — synthetic bytes, no network."""

import hashlib
import urllib.request

import pytest

from absynth.data import ChecksumError, download, file_digest, parse_checksum, verify

PAYLOAD = b"absynth test payload\n" * 1000
PAYLOAD_MD5 = hashlib.md5(PAYLOAD).hexdigest()
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse:
    """Minimal stand-in for the streaming response used by download()."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self, size: int = -1) -> bytes:
        if not self._payload:
            return b""
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def fake_network(monkeypatch):
    """Replace urlopen with a fake serving ``payload``; count invocations."""
    calls = {"n": 0}

    def serve(request, timeout=None):
        calls["n"] += 1
        return FakeResponse(PAYLOAD)

    monkeypatch.setattr(urllib.request, "urlopen", serve)
    return calls


def test_file_digest_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(PAYLOAD)
    assert file_digest(path, "md5") == PAYLOAD_MD5
    assert file_digest(path, "sha256") == PAYLOAD_SHA256


def test_parse_checksum():
    assert parse_checksum("md5:abc123") == ("md5", "abc123")
    assert parse_checksum("SHA256:ABC") == ("sha256", "abc")
    with pytest.raises(ValueError):
        parse_checksum("abc123")  # no algorithm prefix


def test_download_writes_file_and_cleans_up(fake_network, tmp_path):
    destination = tmp_path / "raw" / "Dataset.zip"
    result = download("https://example.invalid/Dataset.zip", destination, f"md5:{PAYLOAD_MD5}")
    assert result.read_bytes() == PAYLOAD
    assert fake_network["n"] == 1
    assert not destination.with_name(destination.name + ".part").exists()


def test_download_rejects_corrupt_payload(fake_network, tmp_path):
    destination = tmp_path / "Dataset.zip"
    with pytest.raises(ChecksumError):
        download("https://example.invalid/Dataset.zip", destination, f"md5:{'0' * 32}")
    assert not destination.exists()  # bad bytes never reach the target path
    assert not destination.with_name(destination.name + ".part").exists()


def test_cached_file_is_reused_but_never_trusted(fake_network, tmp_path):
    destination = tmp_path / "Dataset.zip"
    destination.write_bytes(PAYLOAD)
    result = download("https://example.invalid/Dataset.zip", destination, f"md5:{PAYLOAD_MD5}")
    assert result == destination
    assert fake_network["n"] == 0  # cache hit, no network at all

    destination.write_bytes(PAYLOAD + b"corrupted")
    with pytest.raises(ChecksumError):
        download("https://example.invalid/Dataset.zip", destination, f"md5:{PAYLOAD_MD5}")


def test_verify_raises_on_mismatch(tmp_path):
    path = tmp_path / "Dataset.zip"
    path.write_bytes(PAYLOAD)
    assert verify(path, f"sha256:{PAYLOAD_SHA256}") == path
    with pytest.raises(ChecksumError):
        verify(path, f"sha256:{'f' * 64}")
