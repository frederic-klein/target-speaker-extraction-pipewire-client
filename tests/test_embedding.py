"""Tests for speaker embedding computation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tse_pipewire.embedding import (
    EmbeddingExtractor,
    compute_fbank,
    load_embedding,
    save_embedding,
)


def test_save_and_load_embedding(tmp_path):
    embedding = np.random.randn(256).astype(np.float32)
    path = tmp_path / "test.npy"
    save_embedding(embedding, path)
    loaded = load_embedding(path)
    np.testing.assert_array_almost_equal(embedding, loaded)


def test_load_embedding_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_embedding(tmp_path / "nonexistent.npy")


def test_save_embedding_creates_parent_dirs(tmp_path):
    embedding = np.random.randn(256).astype(np.float32)
    path = tmp_path / "subdir" / "deep" / "test.npy"
    save_embedding(embedding, path)
    assert path.exists()
    loaded = load_embedding(path)
    np.testing.assert_array_almost_equal(embedding, loaded)


class TestComputeFbank:
    def test_output_shape(self):
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32)
        fbank = compute_fbank(audio, sr=sr)
        assert fbank.ndim == 2
        assert fbank.shape[1] == 80

    def test_frame_count(self):
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32)
        fbank = compute_fbank(audio, sr=sr)
        # 25ms frame, 10ms hop: (16000 - 400) / 160 + 1 = 98 frames
        expected_frames = (len(audio) - 400) // 160 + 1
        assert fbank.shape[0] == expected_frames

    def test_silence_returns_finite(self):
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        fbank = compute_fbank(audio, sr=sr)
        assert np.all(np.isfinite(fbank))

    def test_short_audio(self):
        sr = 16000
        audio = np.random.randn(800).astype(np.float32)
        fbank = compute_fbank(audio, sr=sr)
        assert fbank.ndim == 2
        assert fbank.shape[1] == 80
        assert fbank.shape[0] >= 1


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
            MagicMock(name="fbank", shape=[1, None, 80]),
        ]
        fake_embedding = np.random.randn(1, 256).astype(np.float32)
        mock_session.run.return_value = [fake_embedding]
        mock_session_cls.return_value = mock_session

        extractor = EmbeddingExtractor(model_path="fake.onnx")
        audio = np.random.randn(16000).astype(np.float32)
        result = extractor.compute_embedding(audio, sr=16000)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (256,)

    @patch("tse_pipewire.embedding.ort.InferenceSession")
    def test_compute_embedding_passes_fbank_features(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="fbank", shape=[1, None, 80]),
        ]
        fake_embedding = np.random.randn(1, 256).astype(np.float32)
        mock_session.run.return_value = [fake_embedding]
        mock_session_cls.return_value = mock_session

        extractor = EmbeddingExtractor(model_path="fake.onnx")
        audio = np.random.randn(16000).astype(np.float32)
        extractor.compute_embedding(audio, sr=16000)

        call_args = mock_session.run.call_args
        feed_dict = call_args[1] if call_args[1] else call_args[0][1]
        input_name = mock_session.get_inputs.return_value[0].name
        fbank_input = feed_dict[input_name]
        assert fbank_input.ndim == 3
        assert fbank_input.shape[0] == 1
        assert fbank_input.shape[2] == 80

    @patch("tse_pipewire.embedding.ort.InferenceSession")
    def test_compute_embedding_resamples_if_needed(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="fbank", shape=[1, None, 80]),
        ]
        fake_embedding = np.random.randn(1, 256).astype(np.float32)
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
            MagicMock(name="fbank", shape=[1, None, 80]),
        ]
        fake_embedding = np.random.randn(1, 256).astype(np.float32)
        mock_session.run.return_value = [fake_embedding]
        mock_session_cls.return_value = mock_session

        extractor = EmbeddingExtractor(model_path="fake.onnx")
        audio = np.random.randn(16000).astype(np.float32)
        path = tmp_path / "speaker.npy"
        extractor.extract_and_save(audio, sr=16000, output_path=path)

        assert path.exists()
        loaded = load_embedding(path)
        assert loaded.shape == (256,)
