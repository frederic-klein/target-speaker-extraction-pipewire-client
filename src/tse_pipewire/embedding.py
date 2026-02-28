"""Speaker embedding extraction using ONNX Runtime."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 80) -> np.ndarray:
    """Create a mel-scale filterbank matrix."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


def compute_fbank(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """Compute log mel-filterbank features from audio.

    Returns array of shape (num_frames, n_mels).
    """
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    n_fft = frame_length

    num_frames = (len(audio) - frame_length) // hop_length + 1
    if num_frames < 1:
        num_frames = 1

    frames = np.zeros((num_frames, frame_length), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        if end <= len(audio):
            frames[i] = audio[start:end]
        else:
            frames[i, : len(audio) - start] = audio[start:]

    window = np.hanning(frame_length).astype(np.float32)
    frames *= window

    spectrum = np.fft.rfft(frames, n=n_fft)
    power_spectrum = np.abs(spectrum) ** 2

    mel_filter = _mel_filterbank(sr, n_fft, n_mels)
    mel_spec = power_spectrum @ mel_filter.T

    # Avoid log(0)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)

    return log_mel.astype(np.float32)


def save_embedding(embedding: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embedding)


def load_embedding(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding not found: {path}")
    return np.load(path)


class EmbeddingExtractor:
    """Extracts speaker embeddings using an ONNX speaker encoder model."""

    def __init__(self, model_path: str, target_sr: int = 16000):
        self._model_path = model_path
        self._target_sr = target_sr
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def compute_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr != self._target_sr:
            audio = resample_poly(audio, self._target_sr, sr).astype(np.float32)

        fbank = compute_fbank(audio, sr=self._target_sr)
        # Shape: (1, num_frames, 80)
        fbank = fbank[np.newaxis, :, :]

        result = self._session.run(None, {self._input_name: fbank})
        embedding = result[0]

        if embedding.ndim > 1:
            embedding = embedding.squeeze()

        return embedding

    def extract_and_save(
        self,
        audio: np.ndarray,
        sr: int,
        output_path: Path,
    ) -> np.ndarray:
        embedding = self.compute_embedding(audio, sr)
        save_embedding(embedding, output_path)
        return embedding
