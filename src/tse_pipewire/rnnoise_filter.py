"""RNNoise integration via PipeWire filter-chain configuration."""

from dataclasses import dataclass, field


@dataclass
class RNNoiseConfig:
    vad_threshold: float = 50.0
    vad_grace_period_ms: int = 200
    sample_rate: int = 48000
    ladspa_path: str = "librnnoise_ladspa.so"


def generate_filter_chain_config(config: RNNoiseConfig) -> str:
    """Generate a PipeWire filter-chain config for RNNoise denoising."""
    return f"""\
context.modules = [{{
    name = libpipewire-module-filter-chain
    args = {{
        node.description = "RNNoise Denoiser"
        filter.graph = {{
            nodes = [{{
                type = ladspa
                name = rnnoise
                plugin = {config.ladspa_path}
                label = noise_suppressor_mono
                control = {{
                    "VAD Threshold (%)" = {config.vad_threshold}
                    "VAD Grace Period (ms)" = {config.vad_grace_period_ms}
                }}
            }}]
        }}
        capture.props = {{
            node.name = "capture.rnnoise"
            node.passive = true
            audio.rate = {config.sample_rate}
        }}
        playback.props = {{
            node.name = "rnnoise_source"
            media.class = Audio/Source
            audio.rate = {config.sample_rate}
        }}
    }}
}}]
"""
