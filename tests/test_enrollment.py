"""Tests for enrollment process."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from tse_pipewire.enrollment import (
    EnrollmentResult,
    compute_snr,
    enroll_speaker,
    save_enrollment_wav,
)


def test_compute_snr_clean_signal():
    """Speech-like signal: loud bursts with quiet gaps -> high SNR."""
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    signal = np.zeros(sr, dtype=np.float32)
    # Simulate speech bursts with silent gaps
    signal[: sr // 3] = np.sin(2 * np.pi * 440 * t[: sr // 3]) * 0.8
    signal[sr // 2 : sr * 5 // 6] = np.sin(2 * np.pi * 300 * t[: sr * 5 // 6 - sr // 2]) * 0.7
    snr = compute_snr(signal, sr)
    assert snr > 20.0


def test_compute_snr_noisy_signal():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noise = np.random.randn(sr).astype(np.float32) * 0.5
    noisy = signal + noise
    snr = compute_snr(noisy, sr)
    assert snr < 20.0


def test_compute_snr_silence():
    audio = np.zeros(16000, dtype=np.float32)
    snr = compute_snr(audio, 16000)
    assert snr < 0


def test_save_enrollment_wav(tmp_path):
    sr = 16000
    audio = np.random.randn(sr * 2).astype(np.float32) * 0.5
    path = tmp_path / "enrollment.wav"
    save_enrollment_wav(audio, sr, path)

    assert path.exists()
    loaded, loaded_sr = sf.read(path)
    assert loaded_sr == sr
    assert len(loaded) == len(audio)


def test_save_enrollment_wav_creates_parent_dirs(tmp_path):
    sr = 16000
    audio = np.random.randn(sr).astype(np.float32)
    path = tmp_path / "sub" / "dir" / "enrollment.wav"
    save_enrollment_wav(audio, sr, path)
    assert path.exists()


class TestEnrollSpeaker:
    @patch("tse_pipewire.enrollment.sd.rec")
    @patch("tse_pipewire.enrollment.sd.wait")
    @patch("tse_pipewire.enrollment.EmbeddingExtractor")
    def test_enroll_speaker_success(
        self, mock_extractor_cls, mock_wait, mock_rec, tmp_path
    ):
        sr = 16000
        duration = 5
        fake_audio = np.random.randn(sr * duration).astype(np.float32)
        # sounddevice.rec returns a 2D array (samples, channels)
        mock_rec.return_value = fake_audio.reshape(-1, 1)

        mock_extractor = MagicMock()
        fake_embedding = np.random.randn(256).astype(np.float32)
        mock_extractor.compute_embedding.return_value = fake_embedding
        mock_extractor_cls.return_value = mock_extractor

        result = enroll_speaker(
            name="testuser",
            duration=duration,
            sample_rate=sr,
            profiles_dir=tmp_path,
            model_path="fake.onnx",
        )

        assert isinstance(result, EnrollmentResult)
        assert result.name == "testuser"
        assert result.embedding_path == tmp_path / "testuser.npy"
        assert result.wav_path == tmp_path / "testuser_enrollment.wav"
        assert isinstance(result.snr, float)
        assert result.embedding_path.exists() or mock_extractor.extract_and_save.called

    @patch("tse_pipewire.enrollment.sd.rec")
    @patch("tse_pipewire.enrollment.sd.wait")
    @patch("tse_pipewire.enrollment.EmbeddingExtractor")
    def test_enroll_speaker_low_snr_warns(
        self, mock_extractor_cls, mock_wait, mock_rec, tmp_path
    ):
        sr = 16000
        # Nearly silent audio -> low SNR
        fake_audio = np.random.randn(sr * 5).astype(np.float32) * 0.001
        mock_rec.return_value = fake_audio.reshape(-1, 1)

        mock_extractor = MagicMock()
        fake_embedding = np.random.randn(256).astype(np.float32)
        mock_extractor.compute_embedding.return_value = fake_embedding
        mock_extractor_cls.return_value = mock_extractor

        result = enroll_speaker(
            name="quiet",
            duration=5,
            sample_rate=sr,
            profiles_dir=tmp_path,
            model_path="fake.onnx",
        )

        assert result.snr_warning is True
