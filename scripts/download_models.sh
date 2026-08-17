#!/usr/bin/env bash
# ==============================================================================
# Harsh AI Video Studio - Wan 2.2 I2V A14B Model Weights Downloader
# Executed ON THE GPU SERVER ONLY (Do not execute on dev laptop)
# ==============================================================================
set -euo pipefail

MODEL_DIR="${1:-./models/Wan2.2-I2V-14B-720P}"
echo "============================================================"
echo "Downloading Wan 2.2 I2V 14B Checkpoint to ${MODEL_DIR}"
echo "============================================================"

mkdir -p "${MODEL_DIR}"

# Install huggingface_hub cli if not present
python3 -m pip install -q huggingface_hub

# Download Wan 2.2 I2V 14B 720P weights
python3 -c "
from huggingface_hub import snapshot_download
print('Downloading Wan2.1/Wan2.2 14B weights...')
snapshot_download(
    repo_id='Wan-Video/Wan2.1-I2V-14B-720P',
    local_dir='${MODEL_DIR}',
    local_dir_use_symlinks=False
)
print('Download completed successfully!')
"

echo "Model checkpoints ready in ${MODEL_DIR}"
