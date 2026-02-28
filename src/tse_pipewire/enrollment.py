"""Speaker enrollment process - recording, quality check, and embedding extraction."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from tse_pipewire.embedding import EmbeddingExtractor, save_embedding

SNR_WARNING_THRESHOLD = 10.0


@dataclass
class EnrollmentResult:
    name: str
    embedding_path: Path
    wav_path: Path
    snr: float
    snr_warning: bool


def compute_snr(audio: np.ndarray, sr: int) -> float:
    """Estimate SNR by comparing signal energy to noise floor energy.

    Uses a simple energy-based approach: signal frames above threshold
    vs. frames below threshold (assumed noise).
    """
    frame_length = int(0.025 * sr)
    hop = int(0.010 * sr)

    frames = []
    for start in range(0, len(audio) - frame_length, hop):
        frame = audio[start : start + frame_length]
        frames.append(np.sum(frame**2) / frame_length)

    if not frames:
        return -np.inf

    energies = np.array(frames)
    sorted_energies = np.sort(energies)

    # Bottom 10% as noise floor estimate, top 50% as signal
    n = len(sorted_energies)
    noise_energy = np.mean(sorted_energies[: max(1, n // 10)])
    signal_energy = np.mean(sorted_energies[n // 2 :])

    if noise_energy <= 0:
        noise_energy = 1e-10
    if signal_energy <= 0:
        return -np.inf

    return float(10 * np.log10(signal_energy / noise_energy))


def save_enrollment_wav(audio: np.ndarray, sr: int, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    sf.write(str(path), audio, sr)
    os.chmod(path, 0o600)


def enroll_speaker(
    name: str,
    duration: int,
    sample_rate: int,
    profiles_dir: Path,
    model_path: str,
) -> EnrollmentResult:
    """Record audio, check quality, extract and save speaker embedding."""
    profiles_dir = Path(profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(profiles_dir, 0o700)

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    audio = audio.squeeze()

    wav_path = profiles_dir / f"{name}_enrollment.wav"
    save_enrollment_wav(audio, sample_rate, wav_path)

    snr = compute_snr(audio, sample_rate)
    snr_warning = snr < SNR_WARNING_THRESHOLD

    extractor = EmbeddingExtractor(model_path=model_path, target_sr=sample_rate)
    embedding_path = profiles_dir / f"{name}.npy"
    extractor.extract_and_save(audio, sr=sample_rate, output_path=embedding_path)

    return EnrollmentResult(
        name=name,
        embedding_path=embedding_path,
        wav_path=wav_path,
        snr=snr,
        snr_warning=snr_warning,
    )
