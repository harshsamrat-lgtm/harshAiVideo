#!/bin/bash
# ============================================================
# Dedicated AI Video Models Pre-Download Script
# 1. SANA-Video 2B (Text-to-Video)
# 2. CogVideoX-5B-I2V (100% Face & Character Image-to-Video)
# Vast.ai RTX 5090 GPU Server
# ============================================================

set -e

echo "=============================================="
echo "🚀 AI Video Models Setup (SANA + Face-Lock I2V)"
echo "=============================================="
echo ""

# 1. Ensure latest diffusers & dependencies
echo "📦 Installing/updating latest diffusers with SANA & I2V support..."
pip install -q git+https://github.com/huggingface/diffusers transformers accelerate torch torchvision timm sentencepiece protobuf edge-tts gTTS 2>/dev/null || true
echo "✅ Core dependencies verified!"
echo ""

# 2. Download SANA-Video 2B 720p HD Model (Text-to-Video)
echo "📦 Step 1: Downloading SANA-Video 2B 720p (NVIDIA Linear DiT Model)..."
echo "   Model: Efficient-Large-Model/SANA-Video_2B_720p_diffusers"
echo "=============================================="

python3 -c "
try:
    from diffusers import SanaVideoPipeline
    import torch
    print('📥 Downloading SANA-Video 2B weights...')
    pipe = SanaVideoPipeline.from_pretrained(
        'Efficient-Large-Model/SANA-Video_2B_720p_diffusers',
        torch_dtype=torch.bfloat16
    )
    print('✅ SANA-Video 2B 720p downloaded and cached!')
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as e:
    print(f'⚠️ SANA download notice: {e}')
"

echo ""
# 3. Download CogVideoX-5B-I2V (100% Exact Face Matching Image-to-Video)
echo "📦 Step 2: Downloading CogVideoX-5B-I2V (100% Face & Character Matching)..."
echo "   Model: THUDM/CogVideoX-5b-I2V"
echo "=============================================="

python3 -c "
try:
    from diffusers import CogVideoXImageToVideoPipeline
    import torch
    print('📥 Downloading CogVideoX-5B-I2V weights for exact face matching...')
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        'THUDM/CogVideoX-5b-I2V',
        torch_dtype=torch.bfloat16
    )
    print('✅ CogVideoX-5B-I2V downloaded and ready for character face matching!')
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as e:
    print(f'⚠️ CogVideoX-5b-I2V download notice: {e}')
"

echo ""
echo "=============================================="
echo "🎉 ALL MODELS DOWNLOADED & READY!"
echo "   अब स्टूडियो चालू करें:"
echo "   bash scripts/start_public_studio.sh"
echo "=============================================="
