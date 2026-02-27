"""Audio pipeline orchestration: capture -> resample -> TSE -> resample -> output."""

import numpy as np
from scipy.signal import resample_poly

from tse_pipewire.tse_engine import TSEEngine


class AudioPipeline:
    """Orchestrates the full audio processing chain."""

    def __init__(
        self,
        tse_engine: TSEEngine,
        input_sr: int = 48000,
        tse_sr: int = 16000,
    ):
        self.input_sr = input_sr
        self.tse_sr = tse_sr
        self._engine = tse_engine
        self._input_buffer = np.array([], dtype=np.float32)
        self._output_buffer = np.array([], dtype=np.float32)

    def _downsample(self, audio: np.ndarray) -> np.ndarray:
        if self.input_sr == self.tse_sr:
            return audio
        return resample_poly(audio, self.tse_sr, self.input_sr).astype(np.float32)

    def _upsample(self, audio: np.ndarray) -> np.ndarray:
        if self.input_sr == self.tse_sr:
            return audio
        return resample_poly(audio, self.input_sr, self.tse_sr).astype(np.float32)

    def process_offline(self, audio: np.ndarray) -> np.ndarray:
        """Process a full audio array offline (for testing/file processing)."""
        original_length = len(audio)

        downsampled = self._downsample(audio)
        processed = self._engine.process_file(downsampled)
        upsampled = self._upsample(processed)

        # Match original length
        if len(upsampled) > original_length:
            upsampled = upsampled[:original_length]
        elif len(upsampled) < original_length:
            padding = np.zeros(original_length - len(upsampled), dtype=np.float32)
            upsampled = np.concatenate([upsampled, padding])

        return upsampled

    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Sounddevice-compatible audio callback for real-time processing."""
        audio_in = indata[:, 0] if indata.ndim > 1 else indata
        audio_in = audio_in.astype(np.float32)

        # Downsample input to TSE rate
        downsampled = self._downsample(audio_in)

        # Accumulate in input buffer
        self._input_buffer = np.concatenate([self._input_buffer, downsampled])

        chunk_size = self._engine.chunk_size

        # Process all complete chunks
        while len(self._input_buffer) >= chunk_size:
            chunk = self._input_buffer[:chunk_size]
            self._input_buffer = self._input_buffer[chunk_size:]
            processed = self._engine.process_chunk(chunk)
            upsampled = self._upsample(processed)
            self._output_buffer = np.concatenate([self._output_buffer, upsampled])

        # Fill output
        n_out = len(outdata)
        if len(self._output_buffer) >= n_out:
            outdata[:, 0] = self._output_buffer[:n_out]
            self._output_buffer = self._output_buffer[n_out:]
        else:
            available = len(self._output_buffer)
            if available > 0:
                outdata[:available, 0] = self._output_buffer
                self._output_buffer = np.array([], dtype=np.float32)
            outdata[available:, 0] = 0.0

    def reset(self):
        self._engine.reset()
        self._input_buffer = np.array([], dtype=np.float32)
        self._output_buffer = np.array([], dtype=np.float32)
