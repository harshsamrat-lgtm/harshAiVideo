#!/usr/bin/env bash
# ==============================================================================
# Harsh AI Video Studio - 1-Click Public Web Studio Launcher (Cloudflare Tunnel)
# ==============================================================================
set -euo pipefail

echo "============================================================"
echo "Cleaning up old background processes..."
echo "============================================================"

# Kill any old running server on port 8000
pkill -f "uvicorn" || true
pkill -f "start_studio" || true
pkill -f "cloudflared" || true
sleep 1

echo "============================================================"
echo "Starting Harsh AI Video Studio on Port 8000..."
echo "============================================================"

# 1. Install missing dependencies if needed
echo "Verifying Python dependencies & FFmpeg..."
which ffmpeg >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq ffmpeg)
pip install -q -r backend/requirements.txt git+https://github.com/huggingface/diffusers || true

# 2. Start Studio Backend on Port 8000 in background
cd /workspace/harshAiVideo || cd /root/harshAiVideo || cd "$(pwd)"
PYTHONPATH=backend python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > studio_backend.log 2>&1 &
BACKEND_PID=$!

echo "Backend started successfully (PID: ${BACKEND_PID})"
sleep 2

# 2. Download Cloudflare Tunnel if not present
if [ ! -f "cloudflared" ]; then
    echo "Downloading secure HTTPS tunnel client..."
    curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
    chmod +x cloudflared
fi

echo "============================================================"
echo "YOUR PUBLIC WEB STUDIO LINK WILL APPEAR BELOW:"
echo "============================================================"

# 3. Launch tunnel and output public link
./cloudflared tunnel --url http://localhost:8000
