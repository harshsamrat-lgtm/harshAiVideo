# Harsh AI Video Studio - 17-Step Remote GPU Test Plan

When the rented GPU server is connected in Phase 13, execute testing in this **exact sequential order**. Do not skip or jump ahead.

| Test # | Test Name | Objective | Pass Criteria |
|---|---|---|---|
| **TEST 1** | GPU Detection | Verify NVIDIA hardware visibility via host OS | `nvidia-smi` detects GPU name and PCI bus |
| **TEST 2** | CUDA Verification | Test CUDA runtime and kernel compilation | `torch.cuda.is_available() == True` |
| **TEST 3** | Docker GPU Access | Confirm NVIDIA Container Toolkit passes GPU into container | `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` succeeds |
| **TEST 4** | LightX2V Installation | Verify LightX2V package and compiled custom CUDA ops | `import lightx2v` without symbol errors |
| **TEST 5** | Wan 2.2 Model Loading | Load Wan 2.2 I2V A14B weights into VRAM | Weights load within VRAM threshold; no OOM |
| **TEST 6** | One Image → 5s Video | Basic 5-second image-to-video inference | Valid MP4 output created with smooth motion |
| **TEST 7** | 720p Generation | Render at full $1280 \times 720$ resolution | Accurate resolution confirmed via ffprobe |
| **TEST 8** | Generation Speed | Measure seconds-per-frame (SPF) & throughput | Meets latency targets with NVFP4 |
| **TEST 9** | VRAM Usage Tracking | Record baseline and peak VRAM allocation | Peak VRAM $< 28 \text{ GB}$ (within 32GB budget) |
| **TEST 10** | Character Consistency | Single character across multiple consecutive shots | Face similarity score $\ge 0.75$ |
| **TEST 11** | Multi-Character Test | Scene containing 2 to 3 distinct characters | Independent facial identity preserved |
| **TEST 12** | Multi-Location Test | Same character transitioning across 3+ environments | Background architecture consistent per scene |
| **TEST 13** | Voice Consistency | Character dialogue speech synthesis | Constant timbre and acoustic profile per character |
| **TEST 14** | Lip-Sync Test | Align character dialogue audio with video mouth frames | Visual phoneme alignment with $< 50\text{ms}$ drift |
| **TEST 15** | 30-Second Video Test | Assemble 6 consecutive 5-second shots | Continuous narrative and smooth cut transitions |
| **TEST 16** | 1-Minute Video Test | Multi-scene assembly with background audio & mixing | Seamless multi-scene continuity |
| **TEST 17** | 5-Minute Master Test | Full 60-shot end-to-end studio render (1080p output) | Complete 5-minute video output passing automated QC |

---

## 📊 Performance Logging Matrix

For each test execution, record:
- GPU Model
- VRAM Allocated / Peak
- Resolution
- Diffusion Steps
- Seed
- Inference Time (seconds)
- Video Duration (seconds)
- GPU Utilization %
- Temperature (°C)
- Power Draw (W)
- Output File Size (MB)
