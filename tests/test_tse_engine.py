"""Tests for TSE ONNX inference engine."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tse_pipewire.tse_engine import TSEEngine


class TestTSEEngine:
    def _make_mock_session(self, chunk_size=320, embedding_dim=192):
        mock_session = MagicMock()
        # Input: audio chunk + speaker embedding
        mock_input_audio = MagicMock()
        mock_input_audio.name = "audio"
        mock_input_audio.shape = [1, chunk_size]

        mock_input_emb = MagicMock()
        mock_input_emb.name = "embedding"
        mock_input_emb.shape = [1, embedding_dim]

        mock_session.get_inputs.return_value = [mock_input_audio, mock_input_emb]

        mock_output = MagicMock()
        mock_output.name = "output"
        mock_output.shape = [1, chunk_size]
        mock_session.get_outputs.return_value = [mock_output]

        return mock_session

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_init_loads_model(self, mock_session_cls):
        mock_session_cls.return_value = self._make_mock_session()
        embedding = np.random.randn(192).astype(np.float32)

        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        mock_session_cls.assert_called_once()
        assert engine.chunk_size == 320

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_process_chunk_returns_same_length(self, mock_session_cls):
        chunk_size = 320
        mock_session = self._make_mock_session(chunk_size=chunk_size)
        output_chunk = np.random.randn(1, chunk_size).astype(np.float32)
        mock_session.run.return_value = [output_chunk]
        mock_session_cls.return_value = mock_session

        embedding = np.random.randn(192).astype(np.float32)
        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        input_chunk = np.random.randn(chunk_size).astype(np.float32)
        result = engine.process_chunk(input_chunk)

        assert isinstance(result, np.ndarray)
        assert result.shape == (chunk_size,)

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_process_chunk_passes_embedding(self, mock_session_cls):
        chunk_size = 320
        mock_session = self._make_mock_session(chunk_size=chunk_size)
        output_chunk = np.random.randn(1, chunk_size).astype(np.float32)
        mock_session.run.return_value = [output_chunk]
        mock_session_cls.return_value = mock_session

        embedding = np.random.randn(192).astype(np.float32)
        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        input_chunk = np.random.randn(chunk_size).astype(np.float32)
        engine.process_chunk(input_chunk)

        call_args = mock_session.run.call_args
        feed_dict = call_args[1] if call_args[1] else call_args[0][1]
        assert "embedding" in feed_dict
        np.testing.assert_array_equal(feed_dict["embedding"].squeeze(), embedding)

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_reset_clears_state(self, mock_session_cls):
        mock_session = self._make_mock_session()
        mock_session.run.return_value = [
            np.zeros((1, 320), dtype=np.float32),
        ]
        mock_session_cls.return_value = mock_session

        embedding = np.random.randn(192).astype(np.float32)
        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        engine.process_chunk(np.random.randn(320).astype(np.float32))
        engine.reset()

        # After reset, internal buffer should be cleared
        assert engine._buffer is None or len(engine._buffer) == 0

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_process_chunk_wrong_size_raises(self, mock_session_cls):
        mock_session = self._make_mock_session(chunk_size=320)
        mock_session_cls.return_value = mock_session

        embedding = np.random.randn(192).astype(np.float32)
        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        with pytest.raises(ValueError, match="chunk size"):
            engine.process_chunk(np.random.randn(100).astype(np.float32))

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_process_file_returns_full_output(self, mock_session_cls):
        chunk_size = 320
        mock_session = self._make_mock_session(chunk_size=chunk_size)

        def fake_run(output_names, feed_dict):
            audio_in = feed_dict["audio"]
            return [audio_in * 0.5]

        mock_session.run.side_effect = fake_run
        mock_session_cls.return_value = mock_session

        embedding = np.random.randn(192).astype(np.float32)
        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        # Process 3 chunks worth of audio
        audio = np.random.randn(chunk_size * 3).astype(np.float32)
        result = engine.process_file(audio)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)
