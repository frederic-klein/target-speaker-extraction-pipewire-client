"""Tests for PipeWire client integration."""

import json
import subprocess
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from tse_pipewire.pipewire_client import PipeWireClient, list_audio_devices


class TestListAudioDevices:
    @patch("tse_pipewire.pipewire_client.subprocess.run")
    def test_returns_device_list(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "id": 42,
                        "type": "PipeWire:Interface:Node",
                        "info": {
                            "props": {
                                "node.name": "alsa_input.usb",
                                "node.description": "USB Microphone",
                                "media.class": "Audio/Source",
                            }
                        },
                    },
                    {
                        "id": 43,
                        "type": "PipeWire:Interface:Node",
                        "info": {
                            "props": {
                                "node.name": "alsa_output.pci",
                                "node.description": "Speakers",
                                "media.class": "Audio/Sink",
                            }
                        },
                    },
                ]
            ),
        )
        devices = list_audio_devices()
        assert len(devices) == 1
        assert devices[0]["name"] == "alsa_input.usb"
        assert devices[0]["description"] == "USB Microphone"

    @patch("tse_pipewire.pipewire_client.subprocess.run")
    def test_returns_empty_on_failure(self, mock_run):
        mock_run.side_effect = FileNotFoundError("pw-dump not found")
        devices = list_audio_devices()
        assert devices == []


class TestPipeWireClient:
    @patch("tse_pipewire.pipewire_client.subprocess.Popen")
    def test_create_virtual_mic(self, mock_popen):
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        client = PipeWireClient(sample_rate=48000)
        client.create_virtual_mic("TSE Filtered Mic")

        assert client._virtual_mic_name == "TSE Filtered Mic"
        mock_popen.assert_called_once()

    @patch("tse_pipewire.pipewire_client.subprocess.Popen")
    def test_destroy_virtual_mic(self, mock_popen):
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        client = PipeWireClient(sample_rate=48000)
        client.create_virtual_mic("TSE Filtered Mic")
        client.destroy_virtual_mic()

        mock_process.terminate.assert_called_once()

    @patch("tse_pipewire.pipewire_client.subprocess.Popen")
    def test_destroy_without_create_is_noop(self, mock_popen):
        client = PipeWireClient(sample_rate=48000)
        client.destroy_virtual_mic()  # Should not raise

    def test_default_sample_rate(self):
        client = PipeWireClient()
        assert client.sample_rate == 48000

    def test_custom_sample_rate(self):
        client = PipeWireClient(sample_rate=44100)
        assert client.sample_rate == 44100

    @patch("tse_pipewire.pipewire_client.subprocess.Popen")
    def test_context_manager(self, mock_popen):
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        with PipeWireClient(sample_rate=48000) as client:
            client.create_virtual_mic("TSE Filtered Mic")

        mock_process.terminate.assert_called_once()
