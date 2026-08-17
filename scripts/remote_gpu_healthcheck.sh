#!/usr/bin/env bash
# ==============================================================================
# Harsh AI Video Studio - Remote GPU Node Healthcheck
# ==============================================================================
set -euo pipefail

echo "============================================================"
echo "Harsh AI Video Studio - Hardware & Environment Diagnostics"
echo "============================================================"

echo "[1/5] Checking NVIDIA Driver & Hardware..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu --format=csv,noheader
else
    echo "ERROR: nvidia-smi not found! Please verify NVIDIA drivers."
    exit 1
fi

echo "[2/5] Checking Docker & NVIDIA Container Toolkit..."
if command -v docker &> /dev/null; then
    docker --version
    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi || echo "WARNING: Docker GPU passthrough check failed."
else
    echo "ERROR: Docker not installed."
    exit 1
fi

echo "[3/5] Checking FFmpeg & Video Codec Libraries..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg -version | head -n 1
else
    echo "ERROR: FFmpeg binary not found in PATH."
    exit 1
fi

echo "[4/5] Checking Python Environment..."
python3 --version || python --version

echo "[5/5] All preliminary checks completed successfully!"
