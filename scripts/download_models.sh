#!/usr/bin/env bash
set -euo pipefail

echo "=== Downloading WeSep & WeSpeaker models ==="

MODELS_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/tse-pipewire/models"
mkdir -p "${MODELS_DIR}"

# Install WeSpeaker for speaker encoder
pip install wespeaker

# Download WeSpeaker ECAPA-TDNN model and export to ONNX
python3 -c "
import wespeaker
model = wespeaker.load_model('english')
model.export_onnx('${MODELS_DIR}/speaker_encoder.onnx')
print('Speaker encoder exported to ${MODELS_DIR}/speaker_encoder.onnx')
"

echo ""
echo "=== WeSpeaker model downloaded ==="
echo ""
echo "NOTE: TSE model (tse_model.onnx) requires manual setup:"
echo "  1. Clone WeSep: git clone https://github.com/wenet-e2e/wesep.git"
echo "  2. Download a pretrained causal model (Conv-TasNet or similar)"
echo "  3. Export to ONNX and place at: ${MODELS_DIR}/tse_model.onnx"
echo ""
echo "See https://github.com/wenet-e2e/wesep for available pretrained models."
