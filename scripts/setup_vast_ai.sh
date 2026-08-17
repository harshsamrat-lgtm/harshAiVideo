#!/usr/bin/env bash
# ==============================================================================
# Harsh AI Video Studio - Vast.ai / Cloud GPU Fast Setup Script
# ==============================================================================
set -euo pipefail

echo "============================================================"
echo "Harsh AI Video Studio - Fast Setup for Vast.ai / Cloud GPU"
echo "============================================================"

# 1. Create workspace directories
mkdir -p models projects outputs cache

# 2. Setup .env
if [ ! -f .env ]; then
    cp .env.example .env
    sed -i 's/APP_ENV=development/APP_ENV=production/g' .env
    sed -i 's/GPU_MODE=remote/GPU_MODE=local/g' .env
    sed -i 's/ENGINE=lightx2v/ENGINE=lightx2v/g' .env
fi

# 3. Install Python dependencies
echo "[1/3] Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt

# 4. Verify GPU & CUDA
echo "[2/3] Verifying NVIDIA GPU & CUDA..."
nvidia-smi

# 5. Run GPU Benchmark & Health Check
echo "[3/3] Running GPU VRAM & CUDA Benchmark..."
python scripts/benchmark_gpu.py || true

echo "============================================================"
echo "Setup Complete! You can now start the studio using:"
echo "python backend/app/main.py"
echo "============================================================"
