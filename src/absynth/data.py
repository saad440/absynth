"""Dataset acquisition with atomic downloads and checksum verification.

Absynth uses published experimental spectra from independent sources.
This module fetches dataset archives locally with three guarantees:

- **Atomicity**: A download streams to a ``.part`` file that is renamed
  into place only after success, so an interrupted transfer never
  leaves a corrupt file at the target path.
- **Caching**: An existing destination is reused, but never trusted
  blindly. When a publisher checksum is given, the cached file is
  re-verified instead of silently re-downloaded.
- **Verification**: Every file is checked against the publisher's
  checksum (MD5 for Zenodo records, SHA-256 for Mendeley Data files),
  so later analyses can trust that the local bytes are the archived
  bytes of record.

"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

CHUNK_SIZE = 1 << 20  # stream in 1 MiB chunks
DOWNLOAD_TIMEOUT_S = 120
USER_AGENT = "absynth/0.1"

# Published experimental datasets used by Absynth. Raw files stay out of
# git; provenance is mirrored in the tracked data/README.md. The Mendeley
# collection (CC BY 4.0, doi:10.17632/zt2j5j4v44.3) holds many per-sample
# files and is fetched selectively rather than wholesale.
DATASETS = {
    "unam_blade_coated": {
        "url": "https://zenodo.org/api/records/19237616/files/Dataset.zip/content",
        "doi": "10.5281/zenodo.19237616",
        "licence": "CC BY 4.0",
        "checksum": "md5:6599f206e94c527a8a4a6b534e99967a",
    },
}


class ChecksumError(RuntimeError):
    """Raised when a file does not match its published checksum."""


def parse_checksum(checksum: str) -> tuple[str, str]:
    """Split a publisher checksum like ``"md5:<hex>"`` into parts.

    Zenodo publishes ``"md5:..."`` strings and Mendeley Data publishes
    SHA-256 hashes, so the algorithm is always stated explicitly rather
    than guessed from the hex length.
    """
    algorithm, separator, value = checksum.partition(":")
    if not separator or not value:
        raise ValueError(f"checksum must look like '<algorithm>:<hex>', got {checksum!r}")
    return algorithm.lower(), value.lower()


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    """Return the hex digest of a file, hashing in fixed-size chunks.

    Streaming matters for habit: archives can grow, and a one-shot
    ``hashlib.file_digest``-style read would hold every byte in memory.
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify(path: str | Path, expected_checksum: str) -> Path:
    """Raise :class:`ChecksumError` unless ``path`` matches the publisher checksum."""
    algorithm, expected = parse_checksum(expected_checksum)
    actual = file_digest(path, algorithm)
    if actual != expected:
        raise ChecksumError(f"{path}: {algorithm} {actual} != expected {expected}")
    return Path(path)


def download(
    url: str,
    destination: str | Path,
    expected_checksum: str | None = None,
    force: bool = False,
) -> Path:
    """Fetch ``url`` to ``destination`` atomically, cached and verified.

    Existing files are reused; with ``expected_checksum`` the cached
    bytes are re-verified. New downloads stream to ``<destination>.part``,
    hash the payload while it arrives (one pass, no second read), and
    rename only if the checksum matches. A failed verification deletes the
    partial file and raises before the destination is ever touched.
    """
    destination = Path(destination)
    if destination.is_file() and not force:
        if expected_checksum is not None:
            verify(destination, expected_checksum)
        return destination

    algorithm = expected = None
    if expected_checksum is not None:
        algorithm, expected = parse_checksum(expected_checksum)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response, partial.open("wb") as out:
            hasher = hashlib.new(algorithm) if algorithm is not None else None
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                if hasher is not None:
                    hasher.update(chunk)
        if hasher is not None and hasher.hexdigest() != expected:
            raise ChecksumError(
                f"{url}: downloaded {algorithm} {hasher.hexdigest()} != expected {expected}"
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination
