"""CLI entry point for tse-pipewire."""

from pathlib import Path

import click
import soundfile as sf

from tse_pipewire.audio_pipeline import AudioPipeline
from tse_pipewire.config import Config
from tse_pipewire.embedding import load_embedding
from tse_pipewire.enrollment import enroll_speaker
from tse_pipewire.pipewire_client import PipeWireClient, list_audio_devices
from tse_pipewire.tse_engine import TSEEngine


@click.group()
@click.version_option()
def main():
    """TSE-PipeWire: Real-time voice isolation for Linux."""


@main.command()
@click.option("--name", required=True, help="Profile name for the speaker.")
@click.option("--duration", default=15, help="Recording duration in seconds.")
def enroll(name: str, duration: int):
    """Create a speaker profile by recording a voice sample."""
    config = Config()
    config.ensure_dirs()

    model_path = str(config.models_dir / "speaker_encoder.onnx")
    sample_rate = config.get("audio.tse_sample_rate")

    click.echo(f"Recording {duration}s of audio for profile '{name}'...")
    result = enroll_speaker(
        name=name,
        duration=duration,
        sample_rate=sample_rate,
        profiles_dir=config.profiles_dir,
        model_path=model_path,
    )

    if result.snr_warning:
        click.echo(f"Warning: Low SNR ({result.snr:.1f} dB). Consider re-recording in a quieter environment.")

    click.echo(f"Enrollment complete. Profile saved to {result.embedding_path}")


@main.command()
@click.option("--profile", required=True, help="Speaker profile to use.")
@click.option("--input-device", default=None, help="PipeWire input device name.")
def start(profile: str, input_device: str | None):
    """Start the TSE filter with a speaker profile."""
    config = Config()

    embedding_path = config.profile_path(profile)
    embedding = load_embedding(embedding_path)

    tse_model_path = str(config.models_dir / "tse_model.onnx")
    tse_sr = config.get("audio.tse_sample_rate")
    segment_ms = config.get("audio.segment_ms")
    segment_samples = int(segment_ms * tse_sr / 1000)
    engine = TSEEngine(
        model_path=tse_model_path, embedding=embedding,
        segment_samples=segment_samples,
    )

    input_sr = config.get("audio.sample_rate")
    pipeline = AudioPipeline(tse_engine=engine, input_sr=input_sr, tse_sr=tse_sr)

    with PipeWireClient(sample_rate=input_sr) as pw_client:
        pw_client.create_virtual_mic("TSE Filtered Mic")
        click.echo("TSE filter started. Press Ctrl+C to stop.")
        try:
            import sounddevice as sd

            with sd.Stream(
                samplerate=input_sr,
                channels=1,
                dtype="float32",
                callback=pipeline.audio_callback,
                device=input_device,
            ):
                while True:
                    sd.sleep(1000)
        except KeyboardInterrupt:
            click.echo("\nStopping TSE filter...")


@main.command()
def stop():
    """Stop the running TSE filter."""
    click.echo("Stopping TSE filter - send SIGTERM to the running process.")


@main.command()
def devices():
    """List available audio input devices."""
    devs = list_audio_devices()
    if not devs:
        click.echo("No audio input devices found.")
        return

    for dev in devs:
        click.echo(f"  {dev['name']}: {dev['description']}")


@main.command()
@click.option("--profile", required=True, help="Speaker profile to use.")
@click.option("--input", "input_file", required=True, help="Input WAV file.")
@click.option("--output", "output_file", required=True, help="Output WAV file.")
def test(profile: str, input_file: str, output_file: str):
    """Test a profile by processing a WAV file offline."""
    config = Config()

    embedding = load_embedding(config.profile_path(profile))

    tse_model_path = str(config.models_dir / "tse_model.onnx")
    tse_sr = config.get("audio.tse_sample_rate")
    segment_ms = config.get("audio.segment_ms")
    segment_samples = int(segment_ms * tse_sr / 1000)
    engine = TSEEngine(
        model_path=tse_model_path, embedding=embedding,
        segment_samples=segment_samples,
    )

    input_sr = config.get("audio.sample_rate")
    pipeline = AudioPipeline(tse_engine=engine, input_sr=input_sr, tse_sr=tse_sr)

    audio, sr = sf.read(input_file, dtype="float32")
    result = pipeline.process_offline(audio)
    sf.write(output_file, result, sr)

    click.echo(f"Processed {input_file} -> {output_file}")


@main.command()
def status():
    """Show current TSE filter status."""
    config = Config()

    profiles = list(config.profiles_dir.glob("*.npy")) if config.profiles_dir.exists() else []
    models_exist = (config.models_dir / "tse_model.onnx").exists()

    click.echo(f"Config dir: {config.config_dir}")
    click.echo(f"Models dir: {config.models_dir}")
    click.echo(f"TSE model: {'found' if models_exist else 'not found'}")
    click.echo(f"Profiles: {len(profiles)}")
    for p in profiles:
        click.echo(f"  - {p.stem}")
