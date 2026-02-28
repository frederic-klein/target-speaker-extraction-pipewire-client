#!/usr/bin/env python3
"""Export WeSep BSRNN speaker encoder and separator to ONNX.

One-time setup tool. Requires PyTorch + WeSep (not needed at runtime).

Usage:
    pip install torch wesep modelscope
    python scripts/export_bsrnn_onnx.py [--output-dir DIR]
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


def download_model(cache_dir: Path) -> Path:
    """Download bsrnn_ecapa_vox1 from ModelScope."""
    from modelscope.hub.snapshot_download import snapshot_download

    model_id = "damo/speech_bsrnn_ecapa-tdnn_16k_vox1"
    local_dir = snapshot_download(model_id, cache_dir=str(cache_dir))
    return Path(local_dir)


def load_wesep_model(model_dir: Path):
    """Load the BSRNN model using WeSep utilities."""
    sys.path.insert(0, str(model_dir))

    from wesep.models import get_model
    from wesep.utils.checkpoint import load_pretrained_model

    # Load config
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", config)
    model = get_model(model_config["model_name"])(model_config)

    checkpoint = model_dir / "model.pt"
    if not checkpoint.exists():
        # Try alternative names
        for name in ["avg_model.pt", "best_model.pt"]:
            alt = model_dir / name
            if alt.exists():
                checkpoint = alt
                break

    load_pretrained_model(model, str(checkpoint))
    model.eval()
    return model, model_config


class SpeakerEncoderWrapper(nn.Module):
    """Wraps the BSRNN speaker encoder (ResNet34) for ONNX export.

    Input: fbank features (1, num_frames, 80)
    Output: speaker embedding (1, 256)
    """

    def __init__(self, speaker_model):
        super().__init__()
        self.speaker_model = speaker_model

    def forward(self, fbank):
        return self.speaker_model(fbank)


class BSRNNSeparatorWrapper(nn.Module):
    """Wraps the BSRNN separator for ONNX export, bypassing internal speaker model.

    Input: audio (1, T), embedding (1, 256)
    Output: separated audio (1, T)
    """

    def __init__(self, full_model):
        super().__init__()
        self.full_model = full_model

    def forward(self, audio, embedding):
        return self.full_model.separate(audio, embedding)


def export_speaker_encoder(model, output_path: Path):
    """Export speaker encoder to ONNX."""
    wrapper = SpeakerEncoderWrapper(model.spk_model)
    wrapper.eval()

    dummy_fbank = torch.randn(1, 200, 80)

    torch.onnx.export(
        wrapper,
        (dummy_fbank,),
        str(output_path),
        input_names=["fbank"],
        output_names=["embedding"],
        dynamic_axes={
            "fbank": {1: "num_frames"},
        },
        opset_version=17,
    )
    print(f"Speaker encoder exported to {output_path}")


def export_separator(model, output_path: Path):
    """Export BSRNN separator to ONNX."""
    wrapper = BSRNNSeparatorWrapper(model)
    wrapper.eval()

    dummy_audio = torch.randn(1, 16000)
    dummy_embedding = torch.randn(1, 256)

    torch.onnx.export(
        wrapper,
        (dummy_audio, dummy_embedding),
        str(output_path),
        input_names=["audio", "embedding"],
        output_names=["output"],
        dynamic_axes={
            "audio": {1: "num_samples"},
            "output": {1: "num_samples"},
        },
        opset_version=17,
    )
    print(f"BSRNN separator exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export WeSep BSRNN to ONNX")
    default_dir = os.path.join(
        os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
        "tse-pipewire",
        "models",
    )
    parser.add_argument(
        "--output-dir",
        default=default_dir,
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for downloaded model",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / ".cache"

    print("Downloading BSRNN model from ModelScope...")
    model_dir = download_model(cache_dir)

    print("Loading WeSep BSRNN model...")
    model, config = load_wesep_model(model_dir)

    export_speaker_encoder(model, output_dir / "speaker_encoder.onnx")
    export_separator(model, output_dir / "tse_model.onnx")

    # Generate checksums for integrity verification
    checksums_path = output_dir / "checksums.sha256"
    with open(checksums_path, "w") as f:
        for model_name in ["speaker_encoder.onnx", "tse_model.onnx"]:
            model_file = output_dir / model_name
            sha256 = hashlib.sha256()
            with open(model_file, "rb") as mf:
                while chunk := mf.read(8192):
                    sha256.update(chunk)
            f.write(f"{sha256.hexdigest()}  {model_name}\n")
    print(f"Checksums written to {checksums_path}")

    print(f"\nDone! Models saved to {output_dir}")
    print("  - speaker_encoder.onnx (ResNet34, fbank → 256-dim embedding)")
    print("  - tse_model.onnx (BSRNN separator, audio + embedding → separated audio)")


if __name__ == "__main__":
    main()
