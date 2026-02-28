"""Tests for model integrity verification."""

import hashlib
from pathlib import Path

import pytest

from tse_pipewire.model_integrity import (
    ModelIntegrityError,
    compute_sha256,
    load_checksums,
    verify_model_integrity,
)


def test_compute_sha256(tmp_path):
    test_file = tmp_path / "test.bin"
    content = b"hello world"
    test_file.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert compute_sha256(test_file) == expected


def test_load_checksums(tmp_path):
    checksums_file = tmp_path / "checksums.sha256"
    checksums_file.write_text(
        "abc123  speaker_encoder.onnx\n"
        "def456  tse_model.onnx\n"
    )

    result = load_checksums(checksums_file)
    assert result == {
        "speaker_encoder.onnx": "abc123",
        "tse_model.onnx": "def456",
    }


def test_load_checksums_strips_path_prefix(tmp_path):
    checksums_file = tmp_path / "checksums.sha256"
    checksums_file.write_text("abc123  models/foo.onnx\n")

    result = load_checksums(checksums_file)
    assert result == {"foo.onnx": "abc123"}


def test_verify_passes_valid(tmp_path):
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"model data")
    expected_hash = hashlib.sha256(b"model data").hexdigest()

    checksums_file = tmp_path / "checksums.sha256"
    checksums_file.write_text(f"{expected_hash}  model.onnx\n")

    verify_model_integrity(model_file, checksums_file)


def test_verify_raises_on_mismatch(tmp_path):
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"model data")

    checksums_file = tmp_path / "checksums.sha256"
    checksums_file.write_text("badhash  model.onnx\n")

    with pytest.raises(ModelIntegrityError, match="model.onnx"):
        verify_model_integrity(model_file, checksums_file)


def test_verify_skips_missing_checksums_file(tmp_path):
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"model data")

    missing_checksums = tmp_path / "checksums.sha256"
    verify_model_integrity(model_file, missing_checksums)


def test_verify_skips_unknown_model(tmp_path):
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"model data")

    checksums_file = tmp_path / "checksums.sha256"
    checksums_file.write_text("abc123  other_model.onnx\n")

    verify_model_integrity(model_file, checksums_file)
