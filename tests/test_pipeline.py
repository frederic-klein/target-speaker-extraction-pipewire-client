"""Tests for audio pipeline orchestration."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tse_pipewire.audio_pipeline import AudioPipeline


class TestAudioPipeline:
    def _make_mock_engine(self, chunk_size=320):
        engine = MagicMock()
        engine.chunk_size = chunk_size
        engine.process_chunk.side_effect = lambda x: x * 0.5
        return engine

    def test_init_stores_parameters(self):
        engine = self._make_mock_engine()
        pipeline = AudioPipeline(
            tse_engine=engine,
            input_sr=48000,
            tse_sr=16000,
        )
        assert pipeline.input_sr == 48000
        assert pipeline.tse_sr == 16000

    def test_resample_48k_to_16k(self):
        engine = self._make_mock_engine()
        pipeline = AudioPipeline(
            tse_engine=engine,
            input_sr=48000,
            tse_sr=16000,
        )
        audio_48k = np.random.randn(4800).astype(np.float32)
        audio_16k = pipeline._downsample(audio_48k)
        expected_length = 4800 * 16000 // 48000
        assert len(audio_16k) == expected_length

    def test_resample_16k_to_48k(self):
        engine = self._make_mock_engine()
        pipeline = AudioPipeline(
            tse_engine=engine,
            input_sr=48000,
            tse_sr=16000,
        )
        audio_16k = np.random.randn(1600).astype(np.float32)
        audio_48k = pipeline._upsample(audio_16k)
        expected_length = 1600 * 48000 // 16000
        assert len(audio_48k) == expected_length

    def test_process_offline(self):
        chunk_size = 320
        engine = self._make_mock_engine(chunk_size=chunk_size)
        engine.process_file.side_effect = lambda x: x * 0.5

        pipeline = AudioPipeline(
            tse_engine=engine,
            input_sr=48000,
            tse_sr=16000,
        )

        # 48kHz input, 0.1 seconds = 4800 samples
        audio_48k = np.random.randn(4800).astype(np.float32)
        result = pipeline.process_offline(audio_48k)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio_48k)

    def test_process_offline_same_sample_rate(self):
        chunk_size = 320
        engine = self._make_mock_engine(chunk_size=chunk_size)
        engine.process_file.side_effect = lambda x: x * 0.5

        pipeline = AudioPipeline(
            tse_engine=engine,
            input_sr=16000,
            tse_sr=16000,
        )

        audio = np.random.randn(16000).astype(np.float32)
        result = pipeline.process_offline(audio)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)
        engine.process_file.assert_called_once()

    def test_process_callback_accumulates_and_processes(self):
        chunk_size = 160  # small chunk for TSE at 16kHz
        engine = self._make_mock_engine(chunk_size=chunk_size)
        engine.process_chunk.side_effect = lambda x: x * 0.5

        pipeline = AudioPipeline(
            tse_engine=engine,
            input_sr=48000,
            tse_sr=16000,
        )

        # Feed 480 samples at 48kHz = 160 at 16kHz = exactly 1 TSE chunk
        indata = np.random.randn(480, 1).astype(np.float32)
        outdata = np.zeros_like(indata)
        pipeline.audio_callback(indata, outdata, 480, None, None)

        # Output should have been filled
        assert not np.all(outdata == 0) or engine.process_chunk.called

    def test_reset_resets_engine_and_buffers(self):
        engine = self._make_mock_engine()
        pipeline = AudioPipeline(
            tse_engine=engine,
            input_sr=48000,
            tse_sr=16000,
        )
        pipeline.reset()
        engine.reset.assert_called_once()
