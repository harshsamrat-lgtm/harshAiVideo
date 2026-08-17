#!/usr/bin/env bash
# ==============================================================================
# Harsh AI Video Studio - Remote GPU Node Deployment Bootstrap Script
# Target Node: Ubuntu 22.04 / 24.04 LTS (Rented GPU Server & RTX 5090 Node)
# ==============================================================================
set -euo pipefail

echo "============================================================"
echo "Harsh AI Video Studio - Remote GPU Deployment Bootstrap"
echo "============================================================"

# 1. Update and install base packages
echo "[1/6] Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    ca-certificates \
    gnupg \
    lsb-release

# 2. Verify / Install NVIDIA Container Toolkit (Only on bare-metal OS with systemd)
if command -v systemctl &> /dev/null && [ -d /run/systemd/system ]; then
    echo "[2/6] Verifying NVIDIA Container Toolkit on host..."
    if ! command -v nvidia-ctk &> /dev/null; then
        echo "Configuring NVIDIA Container Toolkit repository..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
          sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
          sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker || true
    fi
else
    echo "[2/6] Running inside Container environment (Vast.ai/RunPod) - Using direct GPU mapping."
fi

# 3. Create required directories
echo "[3/6] Initializing storage mounts..."
mkdir -p models projects outputs cache

# 4. Copy environment configuration
echo "[4/6] Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    # Configure for remote GPU server
    sed -i 's/APP_ENV=development/APP_ENV=production/g' .env
    sed -i 's/GPU_MODE=remote/GPU_MODE=local/g' .env
    sed -i 's/ENGINE=lightx2v/ENGINE=lightx2v/g' .env
    echo ".env created with GPU_MODE=local."
fi

# 5. Build and launch Docker containers with remote-gpu profile
echo "[5/6] Building and starting Docker containers (Profile: remote-gpu)..."
docker compose --profile remote-gpu up -d --build

# 6. Verify health
echo "[6/6] Verifying cluster health..."
sleep 5
curl -s http://localhost:8000/health || echo "Waiting for backend to initialize..."

echo "============================================================"
echo "Deployment Complete! Server is active on port 8000."
echo "Access Swagger UI: http://<YOUR_SERVER_IP>:8000/docs"
echo "============================================================"
