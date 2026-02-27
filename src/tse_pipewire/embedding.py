"""Speaker embedding extraction using ONNX Runtime."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly


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

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        result = self._session.run(None, {self._input_name: audio})
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
