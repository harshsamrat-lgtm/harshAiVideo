#!/bin/bash
# ============================================================
# AI Video Models Pre-Download Script
# Vast.ai GPU Server पर सभी मॉडल पहले से डाउनलोड करें
# ============================================================

echo "=============================================="
echo "🚀 AI Video Models Download Script"
echo "=============================================="
echo ""

# Ensure pip dependencies
pip install -q diffusers transformers accelerate torch sentencepiece protobuf 2>/dev/null

echo "📦 Step 1: Downloading CogVideoX-5B (5B Parameter HD Video Model)..."
echo "=============================================="

python3 -c "
import sys
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
echo "📦 Step 2: Downloading SANA-1600M (High-Quality Image+Video Model)..."
echo "=============================================="

python3 -c "
import sys
try:
    from diffusers import SanaPipeline
    import torch
    print('📥 Downloading SANA 1600M from HuggingFace...')
    print('   Model: Efficient-Large-Model/Sana_1600M_1024px')
    pipe = SanaPipeline.from_pretrained(
        'Efficient-Large-Model/Sana_1600M_1024px',
        torch_dtype=torch.bfloat16
    )
    print('✅ SANA 1600M downloaded and cached!')
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'🖥️  GPU: {gpu_name} ({vram:.1f} GB VRAM)')
        pipe = pipe.to('cuda')
        print('✅ SANA loaded to GPU successfully!')
        del pipe
        torch.cuda.empty_cache()
    
except Exception as e:
    print(f'⚠️ SanaPipeline not found, trying DiffusionPipeline...')
    try:
        from diffusers import DiffusionPipeline
        import torch
        pipe = DiffusionPipeline.from_pretrained(
            'Efficient-Large-Model/Sana_1600M_1024px',
            torch_dtype=torch.bfloat16
        )
        print('✅ SANA 1600M downloaded and cached (via DiffusionPipeline)!')
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e2:
        print(f'❌ SANA download error: {e2}')
"

echo ""
echo "📦 Step 3: Checking HuggingFace cache sizes..."
echo "=============================================="
echo ""
du -sh ~/.cache/huggingface/hub/models--THUDM--CogVideoX-5b 2>/dev/null || echo "⚠️ CogVideoX-5B not cached"
du -sh ~/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_1600M_1024px 2>/dev/null || echo "⚠️ SANA not cached"
du -sh ~/.cache/huggingface/hub/models--damo-vilab--text-to-video-ms-1.7b 2>/dev/null || echo "ℹ️ ModelScope-1.7B not cached"
echo ""
echo "=============================================="
echo "🎉 ALL DOWNLOADS COMPLETE!"
echo "   अब server restart करें:"
echo "   bash scripts/start_public_studio.sh"
echo "=============================================="
