#!/usr/bin/env bash
# ==============================================================================
# Harsh AI Video Studio - Self-Healing 1-Click Public Web Studio Launcher
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

echo "============================================================"
echo "🚀 Initializing Harsh AI Video Studio on GPU Server..."
echo "============================================================"

# 1. Clean up any previous processes
echo "🧹 Cleaning up existing processes on port 8000..."
pkill -f "uvicorn" || true
pkill -f "cloudflared" || true
sleep 1

# 2. Ensure python3 & pip3 are available
PY_BIN="python3"
if ! command -v python3 &>/dev/null; then
    PY_BIN="python"
fi

# 3. Ensure core dependencies and FFmpeg are present
echo "📦 Verifying core dependencies (FastAPI, Uvicorn, Diffusers, FFmpeg)..."
which ffmpeg >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq ffmpeg)
${PY_BIN} -m pip install -q -r backend/requirements.txt || true
${PY_BIN} -m pip install -q git+https://github.com/huggingface/diffusers transformers accelerate torch torchvision timm || true

# 4. Start backend with explicit PYTHONPATH
echo "⚙️ Starting FastAPI server on port 8000..."
export PYTHONPATH="${ROOT_DIR}/backend:${PYTHONPATH:-}"

${PY_BIN} -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > "${ROOT_DIR}/studio_backend.log" 2>&1 &
BACKEND_PID=$!
echo "Backend process started (PID: ${BACKEND_PID})"

# 5. Wait for backend to be fully healthy on port 8000
echo "⏳ Waiting for Studio Backend to respond on port 8000..."
READY=false
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1 || curl -s http://localhost:8000/ >/dev/null 2>&1; then
        READY=true
        echo "✅ Studio Backend is ONLINE and responding on http://localhost:8000!"
        break
    fi
    sleep 1
done

if [ "$READY" = false ]; then
    echo "❌ ERROR: Backend server failed to start on port 8000!"
    echo "📜 Showing last 30 lines of studio_backend.log:"
    echo "------------------------------------------------------------"
    tail -n 30 "${ROOT_DIR}/studio_backend.log" || true
    echo "------------------------------------------------------------"
    exit 1
fi

# 6. Download and start Cloudflare Tunnel
if [ ! -f "${ROOT_DIR}/cloudflared" ]; then
    echo "📥 Downloading Cloudflare Tunnel client..."
    curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o "${ROOT_DIR}/cloudflared"
    chmod +x "${ROOT_DIR}/cloudflared"
fi

echo "============================================================"
echo "🎉 SUCCESS! YOUR PUBLIC HTTPS STUDIO LINK IS GENERATING BELOW:"
echo "   (Open the .trycloudflare.com URL in your browser)"
echo "============================================================"

"${ROOT_DIR}/cloudflared" tunnel --url http://127.0.0.1:8000
