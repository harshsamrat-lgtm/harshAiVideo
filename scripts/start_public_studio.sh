#!/usr/bin/env bash
# ==============================================================================
# Harsh AI Video Studio - 1-Click Public Web Studio Launcher (Cloudflare Tunnel)
# ==============================================================================
set -euo pipefail

echo "============================================================"
echo "Starting Harsh AI Video Studio with Public Web UI Link..."
echo "============================================================"

# 1. Start Studio Backend on Port 8000 in background
cd /workspace/harshAiVideo
PYTHONPATH=backend python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > studio_backend.log 2>&1 &
BACKEND_PID=$!

echo "Backend started on localhost:8000 (PID: ${BACKEND_PID})"
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
