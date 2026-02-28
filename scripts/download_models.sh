#!/usr/bin/env bash
set -euo pipefail

echo "=== Exporting WeSep BSRNN models to ONNX ==="

MODELS_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/tse-pipewire/models"
mkdir -p "${MODELS_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install export dependencies (not needed at runtime)
pip install torch wesep modelscope pyyaml

# Run the export script
python3 "${SCRIPT_DIR}/export_bsrnn_onnx.py" --output-dir "${MODELS_DIR}"

echo ""
echo "=== Models exported successfully ==="
echo "  speaker_encoder.onnx - ResNet34 speaker encoder (fbank -> 256-dim)"
echo "  tse_model.onnx       - BSRNN separator (audio + embedding -> separated audio)"
echo ""
echo "Models saved to: ${MODELS_DIR}"
