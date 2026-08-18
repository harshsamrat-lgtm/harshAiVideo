#!/bin/bash
# ============================================================
# AI Video Models Pre-Download Script
# Vast.ai GPU Server पर सभी मॉडल पहले से डाउनलोड करें
# ============================================================

echo "=============================================="
echo "🚀 AI Video Models Download Script"
echo "=============================================="
echo ""

# Ensure pip dependencies (latest diffusers from git for SanaVideoPipeline support)
echo "📦 Installing/updating diffusers (latest git version for SANA-Video support)..."
pip install -q git+https://github.com/huggingface/diffusers transformers accelerate torch sentencepiece protobuf 2>/dev/null
echo ""

echo "📦 Step 1: Downloading CogVideoX-5B (5B Parameter HD Video Model)..."
echo "=============================================="

python3 -c "
try:
    from diffusers import CogVideoXPipeline
    import torch
    print('📥 Downloading CogVideoX-5B from HuggingFace...')
    print('   Model: THUDM/CogVideoX-5b (~10GB)')
    pipe = CogVideoXPipeline.from_pretrained(
        'THUDM/CogVideoX-5b',
        torch_dtype=torch.float16
    )
    print('✅ CogVideoX-5B downloaded and cached!')
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as e:
    print(f'❌ CogVideoX-5B download error: {e}')
"

echo ""
echo "📦 Step 2: Downloading SANA-Video 2B 720p (NVIDIA Linear DiT Video Model)..."
echo "=============================================="

python3 -c "
try:
    from diffusers import SanaVideoPipeline
    import torch
    print('📥 Downloading SANA-Video 2B 720p from HuggingFace...')
    print('   Model: Efficient-Large-Model/SANA-Video_2B_720p_diffusers')
    pipe = SanaVideoPipeline.from_pretrained(
        'Efficient-Large-Model/SANA-Video_2B_720p_diffusers',
        torch_dtype=torch.bfloat16
    )
    print('✅ SANA-Video 2B 720p downloaded and cached!')

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'🖥️  GPU: {gpu_name} ({vram:.1f} GB VRAM)')
        pipe = pipe.to('cuda')
        print('✅ SANA-Video 2B loaded to GPU successfully!')
        del pipe
        torch.cuda.empty_cache()

except ImportError as e:
    print(f'⚠️ SanaVideoPipeline not available in current diffusers. Updating...')
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'git+https://github.com/huggingface/diffusers'], check=True)
    print('🔄 diffusers updated. Please re-run this script.')
except Exception as e:
    print(f'❌ SANA-Video download error: {e}')
"

echo ""
echo "📦 Step 3: Checking HuggingFace cache sizes..."
echo "=============================================="
echo ""
du -sh ~/.cache/huggingface/hub/models--THUDM--CogVideoX-5b 2>/dev/null || echo "⚠️ CogVideoX-5B not cached"
du -sh ~/.cache/huggingface/hub/models--Efficient-Large-Model--SANA-Video_2B_720p_diffusers 2>/dev/null || echo "⚠️ SANA-Video 2B not cached"
du -sh ~/.cache/huggingface/hub/models--damo-vilab--text-to-video-ms-1.7b 2>/dev/null || echo "ℹ️ ModelScope-1.7B not cached"
echo ""
echo "=============================================="
echo "🎉 ALL DOWNLOADS COMPLETE!"
echo "   अब server restart करें:"
echo "   bash scripts/start_public_studio.sh"
echo "=============================================="
