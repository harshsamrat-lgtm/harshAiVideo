#!/usr/bin/env bash
# ==============================================================================
# Harsh AI Video Studio - 1-Click Public Web Studio Launcher (Cloudflare Tunnel)
# ==============================================================================
set -e

# Detect repository root dynamically (works in /root/harshAiVideo, /workspace/harshAiVideo, etc.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🚀 Harsh AI Video Studio: Starting Launcher..."
echo "📂 Project Directory: $SCRIPT_DIR"
echo "============================================================"

# 1. Clean up old running processes
pkill -f "uvicorn app.main:app" || true
pkill -f "cloudflared" || true
sleep 1

# 2. Ensure all required Python backend dependencies are installed
echo "📦 Checking and installing web backend dependencies..."
pip install -q fastapi uvicorn[standard] pydantic pydantic-settings python-multipart aiosqlite sqlalchemy psutil requests httpx diffusers transformers accelerate sentencepiece protobuf edge-tts gTTS imageio imageio-ffmpeg 2>/dev/null || true

# 3. Start Studio Backend on Port 8000
echo "⚡ Starting FastAPI Backend on Port 8000..."
PYTHONPATH=backend python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > studio_backend.log 2>&1 &
BACKEND_PID=$!

echo "Backend starting (PID: ${BACKEND_PID}). Waiting for server to initialize..."

# 4. Wait until FastAPI server is responding on port 8000
for i in {1..15}; do
    if curl -s http://127.0.0.1:8000/docs >/dev/null 2>&1; then
        echo "✅ FastAPI Backend is LIVE and healthy on http://127.0.0.1:8000!"
        break
    fi
    sleep 1
done

if ! curl -s http://127.0.0.1:8000/docs >/dev/null 2>&1; then
    echo "⚠️ Warning: Backend didn't respond in time. Checking logs:"
    cat studio_backend.log | tail -n 25
fi

# 5. Download Cloudflare Tunnel if not present
if [ ! -f "cloudflared" ]; then
    echo "📥 Downloading secure HTTPS tunnel client (cloudflared)..."
    curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
    chmod +x cloudflared
fi

echo "============================================================"
echo "🎉 YOUR SECURE PUBLIC WEB STUDIO LINK WILL APPEAR BELOW:"
echo "============================================================"

# 6. Launch tunnel and output public link
./cloudflared tunnel --url http://127.0.0.1:8000
