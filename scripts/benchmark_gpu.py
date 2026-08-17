"""
Utility script to record GPU performance, VRAM, and generation latency during Remote GPU testing.
"""
import time
import json
import sys


def log_benchmark_result(
    test_id: str,
    gpu_name: str,
    vram_peak_gb: float,
    resolution: str,
    duration_s: float,
    inference_time_s: float,
    fps: int = 24
):
    fps_generated = (duration_s * fps) / max(inference_time_s, 0.001)
    record = {
        "test_id": test_id,
        "gpu_name": gpu_name,
        "vram_peak_gb": vram_peak_gb,
        "resolution": resolution,
        "video_duration_s": duration_s,
        "inference_time_s": inference_time_s,
        "effective_fps_speed": round(fps_generated, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    print("Harsh AI Video Studio - Performance Benchmark Utility")
    log_benchmark_result(
        test_id="TEST_06_FIRST_5S_CLIP",
        gpu_name="NVIDIA RTX 5090",
        vram_peak_gb=22.4,
        resolution="1280x720",
        duration_s=5.0,
        inference_time_s=14.2
    )
