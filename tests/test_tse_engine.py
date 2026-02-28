"""Tests for TSE ONNX inference engine."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tse_pipewire.model_integrity import ModelIntegrityError
from tse_pipewire.tse_engine import TSEEngine


class TestTSEEngine:
    def _make_mock_session(self, embedding_dim=256):
        mock_session = MagicMock()
        mock_input_audio = MagicMock()
        mock_input_audio.name = "audio"
        mock_input_audio.shape = [1, None]

        mock_input_emb = MagicMock()
        mock_input_emb.name = "embedding"
        mock_input_emb.shape = [1, embedding_dim]

        mock_session.get_inputs.return_value = [mock_input_audio, mock_input_emb]

        mock_output = MagicMock()
        mock_output.name = "output"
        mock_output.shape = [1, None]
        mock_session.get_outputs.return_value = [mock_output]

        return mock_session

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_init_loads_model(self, mock_session_cls):
        mock_session_cls.return_value = self._make_mock_session()
        embedding = np.random.randn(256).astype(np.float32)

        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        mock_session_cls.assert_called_once()
        assert engine.segment_samples == 2560

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_init_custom_segment_samples(self, mock_session_cls):
        mock_session_cls.return_value = self._make_mock_session()
        embedding = np.random.randn(256).astype(np.float32)

        engine = TSEEngine(
            model_path="fake.onnx", embedding=embedding, segment_samples=4000
        )

        assert engine.segment_samples == 4000

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_process_segment_returns_same_length(self, mock_session_cls):
        segment_size = 2560
        mock_session = self._make_mock_session()

        def fake_run(output_names, feed_dict):
            audio_in = feed_dict["audio"]
            return [audio_in * 0.5]

        mock_session.run.side_effect = fake_run
        mock_session_cls.return_value = mock_session

        embedding = np.random.randn(256).astype(np.float32)
        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        input_segment = np.random.randn(segment_size).astype(np.float32)
        result = engine.process_segment(input_segment)

        assert isinstance(result, np.ndarray)
        assert result.shape == (segment_size,)

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_process_segment_variable_length(self, mock_session_cls):
        mock_session = self._make_mock_session()

        def fake_run(output_names, feed_dict):
            audio_in = feed_dict["audio"]
            return [audio_in * 0.5]

        mock_session.run.side_effect = fake_run
        mock_session_cls.return_value = mock_session

        embedding = np.random.randn(256).astype(np.float32)
        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        for length in [1000, 2560, 5000]:
            input_segment = np.random.randn(length).astype(np.float32)
            result = engine.process_segment(input_segment)
            assert result.shape == (length,)

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_process_segment_passes_embedding(self, mock_session_cls):
        mock_session = self._make_mock_session()

        def fake_run(output_names, feed_dict):
            audio_in = feed_dict["audio"]
            return [audio_in * 0.5]

        mock_session.run.side_effect = fake_run
        mock_session_cls.return_value = mock_session

        embedding = np.random.randn(256).astype(np.float32)
        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        input_segment = np.random.randn(2560).astype(np.float32)
        engine.process_segment(input_segment)

        call_args = mock_session.run.call_args
        feed_dict = call_args[1] if call_args[1] else call_args[0][1]
        assert "embedding" in feed_dict
        np.testing.assert_array_equal(feed_dict["embedding"].squeeze(), embedding)

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_reset_clears_state(self, mock_session_cls):
        mock_session = self._make_mock_session()
        mock_session.run.return_value = [
            np.zeros((1, 2560), dtype=np.float32),
        ]
        mock_session_cls.return_value = mock_session

        embedding = np.random.randn(256).astype(np.float32)
        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        engine.process_segment(np.random.randn(2560).astype(np.float32))
        engine.reset()

        assert engine._buffer is None or len(engine._buffer) == 0

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_process_file_returns_full_output(self, mock_session_cls):
        mock_session = self._make_mock_session()

        def fake_run(output_names, feed_dict):
            audio_in = feed_dict["audio"]
            return [audio_in * 0.5]

        mock_session.run.side_effect = fake_run
        mock_session_cls.return_value = mock_session

        embedding = np.random.randn(256).astype(np.float32)
        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        audio = np.random.randn(2560 * 3).astype(np.float32)
        result = engine.process_file(audio)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_process_file_handles_remainder(self, mock_session_cls):
        mock_session = self._make_mock_session()

        def fake_run(output_names, feed_dict):
            audio_in = feed_dict["audio"]
            return [audio_in * 0.5]

        mock_session.run.side_effect = fake_run
        mock_session_cls.return_value = mock_session

        embedding = np.random.randn(256).astype(np.float32)
        engine = TSEEngine(model_path="fake.onnx", embedding=embedding)

        audio = np.random.randn(2560 * 2 + 500).astype(np.float32)
        result = engine.process_file(audio)

        assert len(result) == len(audio)

    @patch("tse_pipewire.tse_engine.ort.InferenceSession")
    def test_raises_on_checksum_mismatch(self, mock_session_cls, tmp_path):
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"model data")

        checksums_file = tmp_path / "checksums.sha256"
        checksums_file.write_text("badhash  model.onnx\n")

        embedding = np.random.randn(256).astype(np.float32)

        with pytest.raises(ModelIntegrityError, match="model.onnx"):
            TSEEngine(model_path=str(model_file), embedding=embedding)
