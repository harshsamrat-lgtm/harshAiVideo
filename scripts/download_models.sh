#!/bin/bash
# ============================================================
# CogVideoX-5B Model Pre-Download Script
# Vast.ai GPU Server पर मॉडल पहले से डाउनलोड करें
# ============================================================

echo "=============================================="
echo "🚀 CogVideoX-5B Model Download Script"
echo "=============================================="

# Ensure pip dependencies
pip install -q diffusers transformers accelerate torch sentencepiece protobuf 2>/dev/null

echo ""
echo "📦 Step 1: Downloading CogVideoX-5B (5 Billion Parameter HD Model)..."
echo "⏳ This will take 5-15 minutes depending on internet speed..."
echo ""

python3 -c "
import sys
print('🔄 Importing diffusers library...')
try:
    from diffusers import CogVideoXPipeline
    import torch
    print('✅ Diffusers imported successfully')
    
    print('')
    print('📥 Downloading CogVideoX-5B from HuggingFace...')
    print('   Model: THUDM/CogVideoX-5b (~10GB)')
    print('   This download happens only ONCE and gets cached.')
    print('')
    
    pipe = CogVideoXPipeline.from_pretrained(
        'THUDM/CogVideoX-5b',
        torch_dtype=torch.float16
    )
    print('')
    print('✅ CogVideoX-5B downloaded and cached successfully!')
    print('')
    
    # Verify it can load to GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'🖥️  GPU: {gpu_name} ({vram:.1f} GB VRAM)')
        pipe = pipe.to('cuda')
        print('✅ CogVideoX-5B loaded to GPU successfully!')
        del pipe
        torch.cuda.empty_cache()
        print('✅ GPU memory cleared.')
    else:
        print('⚠️  No CUDA GPU detected. Model downloaded but not tested on GPU.')
    
    print('')
    print('============================================')
    print('🎉 DOWNLOAD COMPLETE!')  
    print('   अब server restart करें:')
    print('   bash scripts/start_public_studio.sh')
    print('============================================')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo ""
echo "📦 Step 2: Checking HuggingFace cache..."
du -sh ~/.cache/huggingface/hub/models--THUDM--CogVideoX-5b 2>/dev/null || echo "⚠️ CogVideoX-5B cache not found"
du -sh ~/.cache/huggingface/hub/models--damo-vilab--text-to-video-ms-1.7b 2>/dev/null || echo "ℹ️ ModelScope-1.7B cache not found"
echo ""
echo "Done!"
