"""Tests for CLI entry point."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from click.testing import CliRunner

from tse_pipewire.cli import main


def test_main_group_shows_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "TSE-PipeWire" in result.output


def test_main_group_shows_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_enroll_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["enroll", "--help"])
    assert result.exit_code == 0
    assert "--name" in result.output
    assert "--duration" in result.output


def test_start_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["start", "--help"])
    assert result.exit_code == 0
    assert "--profile" in result.output
    assert "--input-device" in result.output


def test_stop_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["stop", "--help"])
    assert result.exit_code == 0


def test_devices_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["devices", "--help"])
    assert result.exit_code == 0


def test_test_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["test", "--help"])
    assert result.exit_code == 0
    assert "--profile" in result.output
    assert "--input" in result.output
    assert "--output" in result.output


def test_status_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--help"])
    assert result.exit_code == 0


class TestDevicesCommand:
    @patch("tse_pipewire.cli.list_audio_devices")
    def test_devices_lists_sources(self, mock_list):
        mock_list.return_value = [
            {"name": "alsa_input.usb", "description": "USB Mic", "media_class": "Audio/Source"},
        ]
        runner = CliRunner()
        result = runner.invoke(main, ["devices"])
        assert result.exit_code == 0
        assert "USB Mic" in result.output

    @patch("tse_pipewire.cli.list_audio_devices")
    def test_devices_empty(self, mock_list):
        mock_list.return_value = []
        runner = CliRunner()
        result = runner.invoke(main, ["devices"])
        assert result.exit_code == 0
        assert "No audio" in result.output


class TestEnrollCommand:
    @patch("tse_pipewire.cli.enroll_speaker")
    @patch("tse_pipewire.cli.Config")
    def test_enroll_calls_enroll_speaker(self, mock_config_cls, mock_enroll):
        mock_config = MagicMock()
        mock_config.profiles_dir = Path("/tmp/test-profiles")
        mock_config.models_dir = Path("/tmp/test-models")
        mock_config.get.return_value = 16000
        mock_config_cls.return_value = mock_config

        mock_enroll.return_value = MagicMock(
            name="fred",
            embedding_path=Path("/tmp/test-profiles/fred.npy"),
            wav_path=Path("/tmp/test-profiles/fred_enrollment.wav"),
            snr=25.0,
            snr_warning=False,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["enroll", "--name", "fred"])
        assert result.exit_code == 0
        mock_enroll.assert_called_once()

    @patch("tse_pipewire.cli.enroll_speaker")
    @patch("tse_pipewire.cli.Config")
    def test_enroll_warns_on_low_snr(self, mock_config_cls, mock_enroll):
        mock_config = MagicMock()
        mock_config.profiles_dir = Path("/tmp/test-profiles")
        mock_config.models_dir = Path("/tmp/test-models")
        mock_config.get.return_value = 16000
        mock_config_cls.return_value = mock_config

        mock_enroll.return_value = MagicMock(
            name="fred",
            embedding_path=Path("/tmp/test-profiles/fred.npy"),
            wav_path=Path("/tmp/test-profiles/fred_enrollment.wav"),
            snr=5.0,
            snr_warning=True,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["enroll", "--name", "fred"])
        assert result.exit_code == 0
        assert "Warning" in result.output or "warning" in result.output.lower()


class TestTestCommand:
    @patch("tse_pipewire.cli.sf.write")
    @patch("tse_pipewire.cli.sf.read")
    @patch("tse_pipewire.cli.AudioPipeline")
    @patch("tse_pipewire.cli.TSEEngine")
    @patch("tse_pipewire.cli.load_embedding")
    @patch("tse_pipewire.cli.Config")
    def test_test_processes_wav(
        self, mock_config_cls, mock_load_emb, mock_engine_cls, mock_pipeline_cls,
        mock_sf_read, mock_sf_write
    ):
        mock_config = MagicMock()
        mock_config.profile_path.return_value = Path("/tmp/profiles/fred.npy")
        mock_config.models_dir = Path("/tmp/models")
        mock_config.get.side_effect = lambda key, **kw: {
            "audio.sample_rate": 48000,
            "audio.tse_sample_rate": 16000,
            "audio.segment_ms": 160,
        }.get(key, kw.get("default"))
        mock_config_cls.return_value = mock_config

        mock_load_emb.return_value = np.random.randn(256).astype(np.float32)
        mock_sf_read.return_value = (np.random.randn(48000).astype(np.float32), 48000)

        mock_pipeline = MagicMock()
        mock_pipeline.process_offline.return_value = np.random.randn(48000).astype(np.float32)
        mock_pipeline_cls.return_value = mock_pipeline

        runner = CliRunner()
        result = runner.invoke(main, ["test", "--profile", "fred", "--input", "in.wav", "--output", "out.wav"])
        assert result.exit_code == 0
        mock_sf_write.assert_called_once()

    @patch("tse_pipewire.cli.sf.write")
    @patch("tse_pipewire.cli.sf.read")
    @patch("tse_pipewire.cli.AudioPipeline")
    @patch("tse_pipewire.cli.TSEEngine")
    @patch("tse_pipewire.cli.load_embedding")
    @patch("tse_pipewire.cli.Config")
    def test_test_passes_segment_samples(
        self, mock_config_cls, mock_load_emb, mock_engine_cls, mock_pipeline_cls,
        mock_sf_read, mock_sf_write
    ):
        mock_config = MagicMock()
        mock_config.profile_path.return_value = Path("/tmp/profiles/fred.npy")
        mock_config.models_dir = Path("/tmp/models")
        mock_config.get.side_effect = lambda key, **kw: {
            "audio.sample_rate": 48000,
            "audio.tse_sample_rate": 16000,
            "audio.segment_ms": 160,
        }.get(key, kw.get("default"))
        mock_config_cls.return_value = mock_config

        mock_load_emb.return_value = np.random.randn(256).astype(np.float32)
        mock_sf_read.return_value = (np.random.randn(48000).astype(np.float32), 48000)

        mock_pipeline = MagicMock()
        mock_pipeline.process_offline.return_value = np.random.randn(48000).astype(np.float32)
        mock_pipeline_cls.return_value = mock_pipeline

        runner = CliRunner()
        runner.invoke(main, ["test", "--profile", "fred", "--input", "in.wav", "--output", "out.wav"])

        # segment_samples = 160ms * 16000 / 1000 = 2560
        mock_engine_cls.assert_called_once()
        call_kwargs = mock_engine_cls.call_args
        assert call_kwargs[1]["segment_samples"] == 2560


def test_delete_profile_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["delete-profile", "--help"])
    assert result.exit_code == 0
    assert "--name" in result.output


class TestDeleteProfileCommand:
    @patch("tse_pipewire.cli.Config")
    def test_delete_profile_removes_files(self, mock_config_cls, tmp_path):
        mock_config = MagicMock()
        mock_config.profiles_dir = tmp_path
        mock_config_cls.return_value = mock_config

        npy_file = tmp_path / "fred.npy"
        wav_file = tmp_path / "fred_enrollment.wav"
        npy_file.write_bytes(b"fake embedding")
        wav_file.write_bytes(b"fake wav")

        runner = CliRunner()
        result = runner.invoke(main, ["delete-profile", "--name", "fred", "--yes"])
        assert result.exit_code == 0
        assert not npy_file.exists()
        assert not wav_file.exists()
        assert "fred.npy" in result.output
        assert "fred_enrollment.wav" in result.output

    @patch("tse_pipewire.cli.Config")
    def test_delete_profile_no_profile_found(self, mock_config_cls, tmp_path):
        mock_config = MagicMock()
        mock_config.profiles_dir = tmp_path
        mock_config_cls.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(main, ["delete-profile", "--name", "nonexistent", "--yes"])
        assert result.exit_code == 0
        assert "No profile found" in result.output

    @patch("tse_pipewire.cli.Config")
    def test_delete_profile_partial_files(self, mock_config_cls, tmp_path):
        mock_config = MagicMock()
        mock_config.profiles_dir = tmp_path
        mock_config_cls.return_value = mock_config

        npy_file = tmp_path / "fred.npy"
        npy_file.write_bytes(b"fake embedding")

        runner = CliRunner()
        result = runner.invoke(main, ["delete-profile", "--name", "fred", "--yes"])
        assert result.exit_code == 0
        assert not npy_file.exists()
        assert "fred.npy" in result.output
