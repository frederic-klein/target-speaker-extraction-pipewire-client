"""Tests for RNNoise filter integration."""

from pathlib import Path

import pytest

from tse_pipewire.rnnoise_filter import RNNoiseConfig, generate_filter_chain_config


class TestRNNoiseConfig:
    def test_default_values(self):
        config = RNNoiseConfig()
        assert config.vad_threshold == 50.0
        assert config.vad_grace_period_ms == 200
        assert config.sample_rate == 48000

    def test_custom_values(self):
        config = RNNoiseConfig(vad_threshold=75.0, vad_grace_period_ms=300)
        assert config.vad_threshold == 75.0
        assert config.vad_grace_period_ms == 300


class TestGenerateFilterChainConfig:
    def test_generates_valid_config(self):
        config = RNNoiseConfig()
        result = generate_filter_chain_config(config)
        assert "libpipewire-module-filter-chain" in result
        assert "librnnoise_ladspa" in result
        assert "noise_suppressor_mono" in result

    def test_contains_vad_threshold(self):
        config = RNNoiseConfig(vad_threshold=75.0)
        result = generate_filter_chain_config(config)
        assert "75.0" in result

    def test_contains_grace_period(self):
        config = RNNoiseConfig(vad_grace_period_ms=300)
        result = generate_filter_chain_config(config)
        assert "300" in result

    def test_contains_sample_rate(self):
        config = RNNoiseConfig(sample_rate=48000)
        result = generate_filter_chain_config(config)
        assert "48000" in result

    def test_contains_node_names(self):
        config = RNNoiseConfig()
        result = generate_filter_chain_config(config)
        assert "capture.rnnoise" in result
        assert "rnnoise_source" in result

    def test_write_config_to_file(self, tmp_path):
        config = RNNoiseConfig()
        result = generate_filter_chain_config(config)
        path = tmp_path / "tse-rnnoise.conf"
        path.write_text(result)
        assert path.exists()
        content = path.read_text()
        assert "libpipewire-module-filter-chain" in content

    def test_custom_ladspa_path(self):
        config = RNNoiseConfig(ladspa_path="/usr/lib/ladspa/librnnoise_ladspa.so")
        result = generate_filter_chain_config(config)
        assert "/usr/lib/ladspa/librnnoise_ladspa.so" in result
