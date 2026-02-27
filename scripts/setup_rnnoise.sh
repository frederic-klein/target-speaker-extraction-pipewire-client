#!/usr/bin/env bash
set -euo pipefail

echo "=== Building RNNoise LADSPA Plugin ==="

BUILD_DIR="/tmp/rnnoise-build"
INSTALL_DIR="${HOME}/.local/lib/ladspa"

rm -rf "${BUILD_DIR}"
git clone https://github.com/werman/noise-suppression-for-voice.git "${BUILD_DIR}"

cd "${BUILD_DIR}"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel "$(nproc)"

mkdir -p "${INSTALL_DIR}"
cp build/bin/ladspa/librnnoise_ladspa.so "${INSTALL_DIR}/"

echo "=== RNNoise LADSPA plugin installed to ${INSTALL_DIR}/librnnoise_ladspa.so ==="
echo "Set LADSPA_PATH=${INSTALL_DIR} or copy to /usr/lib/ladspa/"
