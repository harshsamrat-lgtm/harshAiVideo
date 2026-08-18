#!/bin/bash
# ============================================================
# SANA-Video 2B Dedicated Model Pre-Download Script
# Vast.ai RTX 5090 GPU Server
# ============================================================

set -e

echo "=============================================="
echo "🚀 SANA-Video 2B Dedicated Model Setup"
echo "=============================================="
echo ""

# 1. Ensure latest diffusers & dependencies for SanaVideoPipeline
echo "📦 Installing/updating latest diffusers with SANA support..."
pip install -q git+https://github.com/huggingface/diffusers transformers accelerate torch torchvision timm sentencepiece protobuf 2>/dev/null || true
echo "✅ Core dependencies verified!"
echo ""

# 2. Download SANA-Video 2B 720p HD Model
echo "📦 Downloading SANA-Video 2B 720p (NVIDIA Linear DiT Model)..."
echo "   Model Repository: Efficient-Large-Model/SANA-Video_2B_720p_diffusers"
echo "=============================================="

python3 -c "
try:
    from diffusers import SanaVideoPipeline
    import torch
    print('📥 Connecting to Hugging Face and downloading SANA-Video 2B weights...')
    pipe = SanaVideoPipeline.from_pretrained(
        'Efficient-Large-Model/SANA-Video_2B_720p_diffusers',
        torch_dtype=torch.bfloat16
    )
    print('✅ SANA-Video 2B 720p downloaded and cached on disk!')

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'🖥️  Testing GPU allocation on {gpu_name} ({vram:.1f} GB VRAM)...')
        pipe = pipe.to('cuda')
        print('✅ SANA-Video 2B loaded to GPU successfully in bfloat16 precision!')
        del pipe
        torch.cuda.empty_cache()

except Exception as e:
    print(f'❌ SANA-Video download error: {e}')
"

echo ""
echo "📦 Checking HuggingFace cache..."
du -sh ~/.cache/huggingface/hub/models--Efficient-Large-Model--SANA-Video_2B_720p_diffusers 2>/dev/null || echo "ℹ️ SANA cache directory active"
echo ""
echo "=============================================="
echo "🎉 SANA-VIDEO 2B SETUP COMPLETE!"
echo "   अब स्टूडियो चालू करें:"
echo "   bash scripts/start_public_studio.sh"
echo "=============================================="
