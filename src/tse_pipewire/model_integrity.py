"""Model integrity verification using SHA-256 checksums."""

import hashlib
from pathlib import Path

CHUNK_SIZE = 8192


class ModelIntegrityError(Exception):
    """Raised when a model file fails integrity verification."""


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file using chunked reads."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_checksums(checksums_path: Path) -> dict[str, str]:
    """Parse a sha256sum-format file into {filename: hash} dict."""
    checksums = {}
    with open(checksums_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hash_hex, filename = line.split(None, 1)
            # Strip path prefix, keep only basename
            filename = Path(filename).name
            checksums[filename] = hash_hex
    return checksums


def verify_model_integrity(model_path: Path, checksums_path: Path) -> None:
    """Verify model file against checksums. No-op if checksums missing or model not listed."""
    model_path = Path(model_path)
    checksums_path = Path(checksums_path)

    if not checksums_path.exists():
        return

    checksums = load_checksums(checksums_path)
    model_name = model_path.name

    if model_name not in checksums:
        return

    actual_hash = compute_sha256(model_path)
    expected_hash = checksums[model_name]

    if actual_hash != expected_hash:
        raise ModelIntegrityError(
            f"Integrity check failed for {model_name}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
