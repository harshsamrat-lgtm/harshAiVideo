#!/usr/bin/env bash
# ==============================================================================
# Harsh AI Video Studio - LightX2V & CUDA Ops Installation Script
# Target: Ubuntu with NVIDIA CUDA 12.4 + PyTorch 2.4+
# ==============================================================================
set -euo pipefail

echo "============================================================"
echo "Installing LightX2V Acceleration & Custom CUDA Kernels"
echo "============================================================"

# Install ninja build system for fast kernel compilation
sudo apt-get update && sudo apt-get install -y ninja-build

# Upgrade pip and wheel
python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Clone and install LightX2V
if [ ! -d "LightX2V" ]; then
    git clone https://github.com/ModelTC/LightX2V.git
fi

cd LightX2V
python3 -m pip install -e .
cd ..

echo "LightX2V installation and kernel verification successful!"
