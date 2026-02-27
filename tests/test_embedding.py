"""Tests for speaker embedding computation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tse_pipewire.embedding import (
    EmbeddingExtractor,
    load_embedding,
    save_embedding,
)


def test_save_and_load_embedding(tmp_path):
    embedding = np.random.randn(192).astype(np.float32)
    path = tmp_path / "test.npy"
    save_embedding(embedding, path)
    loaded = load_embedding(path)
    np.testing.assert_array_almost_equal(embedding, loaded)


def test_load_embedding_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_embedding(tmp_path / "nonexistent.npy")


def test_save_embedding_creates_parent_dirs(tmp_path):
    embedding = np.random.randn(192).astype(np.float32)
    path = tmp_path / "subdir" / "deep" / "test.npy"
    save_embedding(embedding, path)
    assert path.exists()
    loaded = load_embedding(path)
    np.testing.assert_array_almost_equal(embedding, loaded)


class TestEmbeddingExtractor:
    def test_init_stores_model_path(self, tmp_path):
        extractor = EmbeddingExtractor.__new__(EmbeddingExtractor)
        extractor._model_path = str(tmp_path / "model.onnx")
        extractor._session = None
        assert extractor._model_path == str(tmp_path / "model.onnx")

    @patch("tse_pipewire.embedding.ort.InferenceSession")
    def test_compute_embedding_returns_numpy_array(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="audio", shape=[1, -1]),
        ]
        fake_embedding = np.random.randn(1, 192).astype(np.float32)
        mock_session.run.return_value = [fake_embedding]
        mock_session_cls.return_value = mock_session

        extractor = EmbeddingExtractor(model_path="fake.onnx")
        audio = np.random.randn(16000).astype(np.float32)
        result = extractor.compute_embedding(audio, sr=16000)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    @patch("tse_pipewire.embedding.ort.InferenceSession")
    def test_compute_embedding_resamples_if_needed(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="audio", shape=[1, -1]),
        ]
        fake_embedding = np.random.randn(1, 192).astype(np.float32)
        mock_session.run.return_value = [fake_embedding]
        mock_session_cls.return_value = mock_session

        extractor = EmbeddingExtractor(model_path="fake.onnx", target_sr=16000)
        audio_48k = np.random.randn(48000).astype(np.float32)
        result = extractor.compute_embedding(audio_48k, sr=48000)

        assert isinstance(result, np.ndarray)

    @patch("tse_pipewire.embedding.ort.InferenceSession")
    def test_extract_and_save(self, mock_session_cls, tmp_path):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="audio", shape=[1, -1]),
        ]
        fake_embedding = np.random.randn(1, 192).astype(np.float32)
        mock_session.run.return_value = [fake_embedding]
        mock_session_cls.return_value = mock_session

        extractor = EmbeddingExtractor(model_path="fake.onnx")
        audio = np.random.randn(16000).astype(np.float32)
        path = tmp_path / "speaker.npy"
        extractor.extract_and_save(audio, sr=16000, output_path=path)

        assert path.exists()
        loaded = load_embedding(path)
        assert loaded.shape == (192,)
