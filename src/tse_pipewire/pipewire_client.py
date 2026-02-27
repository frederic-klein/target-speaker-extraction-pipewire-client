"""PipeWire client for virtual microphone creation and audio routing."""

import json
import subprocess


def list_audio_devices() -> list[dict]:
    """List available PipeWire audio input devices using pw-dump."""
    try:
        result = subprocess.run(
            ["pw-dump"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    try:
        nodes = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    devices = []
    for node in nodes:
        if node.get("type") != "PipeWire:Interface:Node":
            continue
        props = node.get("info", {}).get("props", {})
        media_class = props.get("media.class", "")
        if "Source" not in media_class:
            continue
        devices.append(
            {
                "id": node.get("id"),
                "name": props.get("node.name", ""),
                "description": props.get("node.description", ""),
                "media_class": media_class,
            }
        )

    return devices


class PipeWireClient:
    """Manages a virtual PipeWire microphone using pw-loopback."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self._virtual_mic_name: str | None = None
        self._process: subprocess.Popen | None = None

    def create_virtual_mic(self, name: str = "TSE Filtered Mic"):
        self._virtual_mic_name = name
        self._process = subprocess.Popen(
            [
                "pw-loopback",
                "--capture-props",
                f"node.name=capture.tse media.class=Audio/Sink audio.rate={self.sample_rate}",
                "--playback-props",
                f"node.name={name} node.description=\"{name}\" media.class=Audio/Source audio.rate={self.sample_rate}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def destroy_virtual_mic(self):
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None
            self._virtual_mic_name = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy_virtual_mic()
        return False
