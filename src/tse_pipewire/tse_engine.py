"""ONNX Runtime inference engine for Target Speaker Extraction."""

import numpy as np
import onnxruntime as ort


class TSEEngine:
    """Streaming TSE inference using an ONNX model conditioned on speaker embedding."""

    def __init__(self, model_path: str, embedding: np.ndarray):
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)

        inputs = self._session.get_inputs()
        self._input_names = {inp.name: inp for inp in inputs}

        audio_input = inputs[0]
        self.chunk_size = audio_input.shape[-1]

        if embedding.ndim == 1:
            embedding = embedding[np.newaxis, :]
        self._embedding = embedding.astype(np.float32)

        self._buffer: np.ndarray = np.array([], dtype=np.float32)

    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        if len(audio_chunk) != self.chunk_size:
            raise ValueError(
                f"Expected chunk size {self.chunk_size}, got {len(audio_chunk)}"
            )

        audio = audio_chunk.astype(np.float32)
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
        """Process a full audio array by splitting into chunks."""
        n_chunks = len(audio) // self.chunk_size
        remainder = len(audio) % self.chunk_size

        if remainder > 0:
            padding = np.zeros(self.chunk_size - remainder, dtype=np.float32)
            audio = np.concatenate([audio, padding])
            n_chunks += 1

        outputs = []
        for i in range(n_chunks):
            start = i * self.chunk_size
            chunk = audio[start : start + self.chunk_size]
            outputs.append(self.process_chunk(chunk))

        result = np.concatenate(outputs)

        original_length = len(audio) - (self.chunk_size - remainder if remainder > 0 else 0)
        return result[:original_length]

    def reset(self):
        self._buffer = np.array([], dtype=np.float32)
