"""ONNX Runtime inference engine for Target Speaker Extraction."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from tse_pipewire.model_integrity import verify_model_integrity


class TSEEngine:
    """Streaming TSE inference using an ONNX model conditioned on speaker embedding."""

    def __init__(
        self, model_path: str, embedding: np.ndarray, segment_samples: int = 2560
    ):
        checksums_path = Path(model_path).parent / "checksums.sha256"
        verify_model_integrity(Path(model_path), checksums_path)
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)

        self.segment_samples = segment_samples

        if embedding.ndim == 1:
            embedding = embedding[np.newaxis, :]
        self._embedding = embedding.astype(np.float32)

        self._buffer: np.ndarray = np.array([], dtype=np.float32)

    def process_segment(self, audio_segment: np.ndarray) -> np.ndarray:
        audio = audio_segment.astype(np.float32)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        feed_dict = {
            "audio": audio,
            "embedding": self._embedding,
        }

        result = self._session.run(None, feed_dict)
        output = result[0]

        if output.ndim > 1:
            output = output.squeeze()

        return output

    def process_file(self, audio: np.ndarray) -> np.ndarray:
        """Process a full audio array by splitting into segments."""
        original_length = len(audio)
        n_segments = len(audio) // self.segment_samples
        remainder = len(audio) % self.segment_samples

        if remainder > 0:
            padding = np.zeros(
                self.segment_samples - remainder, dtype=np.float32
            )
            audio = np.concatenate([audio, padding])
            n_segments += 1

        outputs = []
        for i in range(n_segments):
            start = i * self.segment_samples
            segment = audio[start : start + self.segment_samples]
            outputs.append(self.process_segment(segment))

        result = np.concatenate(outputs)
        return result[:original_length]

    def reset(self):
        self._buffer = np.array([], dtype=np.float32)
