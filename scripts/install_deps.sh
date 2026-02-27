#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing system dependencies for tse-pipewire ==="

if command -v apt-get &>/dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        pipewire \
        pipewire-jack \
        libpipewire-0.3-dev \
        python3-dev \
        libsndfile1-dev \
        portaudio19-dev \
        ladspa-sdk
elif command -v pacman &>/dev/null; then
    sudo pacman -S --noconfirm \
        pipewire \
        pipewire-jack \
        python \
        libsndfile \
        portaudio \
        ladspa
else
    echo "Unsupported package manager. Install manually:"
    echo "  pipewire, pipewire-jack, python3-dev, libsndfile1-dev, portaudio19-dev, ladspa-sdk"
    exit 1
fi

echo "=== System dependencies installed ==="
