"""
Harsh AI Video Studio — Real Neural AI Video Diffusion Engine.

Model Cascade (Best → Fallback):
  1. CogVideoX-5B (THUDM/CogVideoX-5b) — 5 Billion parameter, 720×480 native HD, bfloat16
  2. CogVideoX-2B (THUDM/CogVideoX-2b) — Lighter 2B fallback if VRAM < 20GB
  3. ModelScope 1.7B (damo-vilab/text-to-video-ms-1.7b) — Fast legacy fallback

All models run on NVIDIA GPU with CUDA.
Includes Dual-Track Hindi Voice-over, Semantic Prompt Enhancement, and HD Post-Processing.
"""
from typing import Dict, Any, Optional
import os
import time
import shutil
import subprocess
from pathlib import Path
import numpy as np

from app.engines.base_engine import BaseVideoEngine
from app.services.prompt_service import parse_prompt_and_voiceover, translate_and_enhance_hindi_prompt
from app.services.audio_service import audio_service
from app.core.logging import logger
from app.core.config import settings

# Global pipeline caches
_COGVIDEO_5B_PIPE = None
_COGVIDEO_2B_PIPE = None
_MODELSCOPE_PIPE = None
_ACTIVE_MODEL_NAME = None


def _get_vram_gb() -> float:
    """Returns available VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    return 0.0


class LightX2VEngine(BaseVideoEngine):
    """
    Multi-Model Neural Video Diffusion Engine with Automatic Quality Cascade.
    Tries CogVideoX-5B first (best quality), falls back gracefully.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="LightX2V-CogVideoX-5B", config=config)
        self.precision = self.config.get("precision", settings.LIGHTX2V_PRECISION)
        self.sparse_attention = self.config.get("sparse_attention", settings.LIGHTX2V_SPARSE_ATTENTION)
        self.is_loaded = False

    async def load_model(self) -> bool:
        global _COGVIDEO_5B_PIPE, _COGVIDEO_2B_PIPE, _MODELSCOPE_PIPE, _ACTIVE_MODEL_NAME

        if _ACTIVE_MODEL_NAME is not None and (_COGVIDEO_5B_PIPE or _COGVIDEO_2B_PIPE or _MODELSCOPE_PIPE):
            self.is_loaded = True
            return True

        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("No CUDA GPU detected. Video generation requires CUDA.")
                self.is_loaded = True
                return True

            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🖥️ Detected GPU: {gpu_name} ({vram:.1f} GB VRAM)")

            # ── TRY 1: CogVideoX-5B (Best Quality, needs ~18GB VRAM) ──
            if vram >= 16.0:
                try:
                    from diffusers import CogVideoXPipeline
                    logger.info("🚀 Loading CogVideoX-5B (5 Billion parameter HD model)...")
                    _COGVIDEO_5B_PIPE = CogVideoXPipeline.from_pretrained(
                        "THUDM/CogVideoX-5b",
                        torch_dtype=torch.bfloat16
                    )
                    _COGVIDEO_5B_PIPE.vae.enable_tiling()
                    _COGVIDEO_5B_PIPE.vae.enable_slicing()
                    _COGVIDEO_5B_PIPE = _COGVIDEO_5B_PIPE.to("cuda")
                    _ACTIVE_MODEL_NAME = "CogVideoX-5B"
                    self.name = "CogVideoX-5B-HD"
                    logger.info("✅ CogVideoX-5B loaded successfully on GPU!")
                    self.is_loaded = True
                    return True
                except Exception as e:
                    logger.warning(f"CogVideoX-5B load notice ({e}). Trying CogVideoX-2B...")

            # ── TRY 2: CogVideoX-2B (Good Quality, needs ~10GB VRAM) ──
            if vram >= 8.0:
                try:
                    from diffusers import CogVideoXPipeline
                    logger.info("🚀 Loading CogVideoX-2B (2 Billion parameter model)...")
                    _COGVIDEO_2B_PIPE = CogVideoXPipeline.from_pretrained(
                        "THUDM/CogVideoX-2b",
                        torch_dtype=torch.float16
                    )
                    _COGVIDEO_2B_PIPE.vae.enable_tiling()
                    _COGVIDEO_2B_PIPE.vae.enable_slicing()
                    _COGVIDEO_2B_PIPE = _COGVIDEO_2B_PIPE.to("cuda")
                    _ACTIVE_MODEL_NAME = "CogVideoX-2B"
                    self.name = "CogVideoX-2B"
                    logger.info("✅ CogVideoX-2B loaded successfully on GPU!")
                    self.is_loaded = True
                    return True
                except Exception as e:
                    logger.warning(f"CogVideoX-2B load notice ({e}). Trying ModelScope 1.7B...")

            # ── TRY 3: ModelScope 1.7B (Legacy Fallback) ──
            try:
                from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
                logger.info("🚀 Loading ModelScope text-to-video-ms-1.7b (fallback)...")
                _MODELSCOPE_PIPE = DiffusionPipeline.from_pretrained(
                    "damo-vilab/text-to-video-ms-1.7b",
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
                _MODELSCOPE_PIPE.scheduler = DPMSolverMultistepScheduler.from_config(
                    _MODELSCOPE_PIPE.scheduler.config
                )
                _MODELSCOPE_PIPE.enable_vae_slicing()
                _MODELSCOPE_PIPE = _MODELSCOPE_PIPE.to("cuda")
                _ACTIVE_MODEL_NAME = "ModelScope-1.7B"
                self.name = "ModelScope-1.7B"
                logger.info("✅ ModelScope 1.7B loaded on GPU.")
                self.is_loaded = True
                return True
            except Exception as e:
                logger.warning(f"ModelScope load notice ({e}).")

        except Exception as e:
            logger.error(f"Engine initialization error: {e}", exc_info=True)

        _ACTIVE_MODEL_NAME = "none"
        self.is_loaded = True
        return True

    async def unload_model(self) -> bool:
        global _COGVIDEO_5B_PIPE, _COGVIDEO_2B_PIPE, _MODELSCOPE_PIPE, _ACTIVE_MODEL_NAME
        _COGVIDEO_5B_PIPE = None
        _COGVIDEO_2B_PIPE = None
        _MODELSCOPE_PIPE = None
        _ACTIVE_MODEL_NAME = None
        self.is_loaded = False
        return True

    async def generate_image_to_video(
        self,
        prompt: str,
        reference_image_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        duration_seconds: float = 8.0,
        resolution: str = "1280x720",
        seed: int = -1,
        steps: int = 50,
        guidance_scale: float = 6.0,
        output_path: Optional[str] = None,
        callback: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        global _COGVIDEO_5B_PIPE, _COGVIDEO_2B_PIPE, _MODELSCOPE_PIPE, _ACTIVE_MODEL_NAME

        start_time = time.time()
        actual_seed = seed if seed != -1 else int(time.time() * 1000) % 1000000
        target_duration = max(4.0, float(duration_seconds or 8.0))

        # Ensure pipeline is loaded
        if _COGVIDEO_5B_PIPE is None and _COGVIDEO_2B_PIPE is None and _MODELSCOPE_PIPE is None:
            logger.info("Pipeline not loaded yet, loading model now...")
            await self.load_model()

        # ── 1. DUAL-TRACK PARSE: VISUAL SCENE vs SPOKEN VOICEOVER ──
        visual_raw, voiceover_dialogue = parse_prompt_and_voiceover(prompt)
        clean_english_prompt = translate_and_enhance_hindi_prompt(visual_raw)

        logger.info(f"🎬 Active Model: {_ACTIVE_MODEL_NAME}")
        logger.info(f"🎬 Enriched Prompt: '{clean_english_prompt[:100]}...' (Duration: {target_duration}s)")
        if voiceover_dialogue:
            logger.info(f"🎙️ Hindi Voice-over: '{voiceover_dialogue}'")

        out_dir = Path(settings.OUTPUT_ROOT)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_path or str(out_dir / f"cogvideo_{actual_seed}.mp4")
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        raw_temp_video = out_dir / f"raw_{actual_seed}.mp4"
        voice_speech_file = out_dir / f"speech_{actual_seed}.mp3"
        ambient_music_file = out_dir / f"music_{actual_seed}.aac"
        final_mixed_audio = out_dir / f"final_audio_{actual_seed}.aac"

        neg_prompt = (
            negative_prompt or
            "blurry, low quality, distorted, deformed, watermark, text, lowres, "
            "bad anatomy, bad proportions, extra limbs, ugly, duplicate, jpeg artifacts"
        )

        # ── 2. SYNTHESIZE NEURAL HINDI VOICEOVER & AMBIENT MUSIC ──
        has_voiceover = False
        if voiceover_dialogue:
            has_voiceover = await audio_service.generate_hindi_voiceover_speech(
                text=voiceover_dialogue,
                output_speech_path=voice_speech_file,
                voice="hi-IN-MadhurNeural"
            )

        audio_service.generate_ambient_music_for_prompt(
            prompt=clean_english_prompt,
            duration_seconds=target_duration,
            output_music_path=ambient_music_file
        )

        if has_voiceover and voice_speech_file.exists():
            audio_service.mix_voiceover_with_music(
                voice_path=voice_speech_file,
                music_path=ambient_music_file,
                final_audio_path=final_mixed_audio
            )
        elif ambient_music_file.exists():
            shutil.copy(str(ambient_music_file), str(final_mixed_audio))

        # ── 3. GENERATE VIDEO WITH ACTIVE MODEL ──
        generated = False

        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA GPU available")

            generator = torch.Generator("cuda").manual_seed(actual_seed)

            # ═══ CogVideoX-5B (PRIMARY — BEST QUALITY) ═══
            if _COGVIDEO_5B_PIPE is not None:
                logger.info("🎬 Generating with CogVideoX-5B (5B params, bfloat16, HD)...")
                video_output = _COGVIDEO_5B_PIPE(
                    prompt=clean_english_prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=50,
                    guidance_scale=6.0,
                    num_frames=49,
                    generator=generator,
                )
                frames = video_output.frames[0]

                from diffusers.utils import export_to_video
                export_to_video(frames, str(raw_temp_video), fps=8)
                generated = True
                logger.info(f"✅ CogVideoX-5B generated {len(frames)} frames")

            # ═══ CogVideoX-2B (FALLBACK) ═══
            elif _COGVIDEO_2B_PIPE is not None:
                logger.info("🎬 Generating with CogVideoX-2B (2B params, fp16)...")
                video_output = _COGVIDEO_2B_PIPE(
                    prompt=clean_english_prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=50,
                    guidance_scale=6.0,
                    num_frames=49,
                    generator=generator,
                )
                frames = video_output.frames[0]

                from diffusers.utils import export_to_video
                export_to_video(frames, str(raw_temp_video), fps=8)
                generated = True
                logger.info(f"✅ CogVideoX-2B generated {len(frames)} frames")

            # ═══ ModelScope 1.7B (LEGACY LAST RESORT) ═══
            elif _MODELSCOPE_PIPE is not None:
                logger.info("🎬 Generating with ModelScope 1.7B (fallback mode)...")
                video_output = _MODELSCOPE_PIPE(
                    prompt=clean_english_prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=35,
                    guidance_scale=9.0,
                    num_frames=24,
                    generator=generator,
                )
                frames = video_output.frames[0]

                import imageio
                imageio.mimwrite(
                    str(raw_temp_video), frames, fps=8,
                    codec="libx264", quality=9, pixelformat="yuv420p"
                )
                generated = True
                logger.info(f"✅ ModelScope 1.7B generated {len(frames)} frames")

        except Exception as e:
            logger.error(f"❌ Video generation inference error: {e}", exc_info=True)

        if not generated or not raw_temp_video.exists():
            logger.error("⚠️ Video generation failed to create frames.")
            return {
                "engine": _ACTIVE_MODEL_NAME or self.name,
                "status": "FAILED",
                "error": "AI video diffusion could not generate frames. Check CUDA GPU VRAM and connection.",
                "output_path": "",
                "seed": actual_seed,
                "duration": target_duration,
                "resolution": resolution,
                "has_voiceover": has_voiceover,
                "generation_time_seconds": round(time.time() - start_time, 2)
            }

        # ── 4. HD POST-PROCESSING: UPSCALE, SHARPEN, 24FPS INTERPOLATION ──
        target_w, target_h = (1280, 720) if "720" in resolution else (1920, 1080)
        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"

        raw_fps = 8
        if _ACTIVE_MODEL_NAME and "CogVideo" in _ACTIVE_MODEL_NAME:
            raw_duration = 49.0 / raw_fps  # ~6.125s
        else:
            raw_duration = 24.0 / raw_fps  # 3.0s

        time_stretch = target_duration / raw_duration

        enhance_cmd = [
            ffmpeg_cmd, "-y",
            "-i", str(raw_temp_video),
            "-vf", (
                f"setpts={time_stretch}*PTS,"
                f"minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir,"
                f"scale={target_w}:{target_h}:flags=lanczos,"
                f"unsharp=5:5:0.8:5:5:0.4"
            ),
            "-t", str(target_duration),
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_file)
        ]

        try:
            result = subprocess.run(
                enhance_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300
            )
            if result.returncode != 0:
                logger.warning(f"FFmpeg enhance notice: {result.stderr[:200]}")
                shutil.copy(str(raw_temp_video), str(out_file))
        except Exception as e:
            logger.warning(f"FFmpeg notice ({e}), using raw video")
            shutil.copy(str(raw_temp_video), str(out_file))

        if raw_temp_video.exists():
            try:
                raw_temp_video.unlink()
            except Exception:
                pass

        # ── 5. MUX FINAL AUDIO (VOICEOVER + MUSIC) INTO MP4 ──
        if final_mixed_audio.exists() and out_file.exists():
            audio_service.mux_audio_into_video(
                video_path=out_file,
                audio_path=final_mixed_audio,
                final_output_path=out_file
            )
            for f in [voice_speech_file, ambient_music_file, final_mixed_audio]:
                if f.exists():
                    try:
                        f.unlink()
                    except Exception:
                        pass

        gen_time = round(time.time() - start_time, 2)
        return {
            "engine": _ACTIVE_MODEL_NAME or self.name,
            "status": "COMPLETED",
            "output_path": str(out_file),
            "seed": actual_seed,
            "duration": target_duration,
            "resolution": resolution,
            "has_voiceover": has_voiceover,
            "generation_time_seconds": gen_time
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "active_model": _ACTIVE_MODEL_NAME or "not_loaded",
            "is_loaded": self.is_loaded,
            "precision": self.precision,
            "target_hardware": "NVIDIA RTX 5090 (32GB VRAM)",
            "model_cascade": ["CogVideoX-5B", "CogVideoX-2B", "ModelScope-1.7B"],
            "supports_voiceover": True
        }

    async def cancel(self, job_id: str) -> bool:
        return True

    def validate_environment(self) -> Dict[str, Any]:
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            gpu = torch.cuda.get_device_name(0) if cuda_ok else "none"
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_ok else 0
        except Exception:
            cuda_ok, gpu, vram = False, "none", 0

        return {
            "engine": self.name,
            "cuda_available": cuda_ok,
            "gpu": gpu,
            "vram_gb": round(vram, 1),
            "active_model": _ACTIVE_MODEL_NAME
        }

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supported_resolutions": ["1280x720", "1920x1080"],
            "max_duration_seconds": 12.0,
            "supports_text_to_video": True,
            "supports_voiceover_tts": True,
            "model_cascade": ["CogVideoX-5B (primary)", "CogVideoX-2B (fallback)", "ModelScope-1.7B (legacy)"]
        }

    def estimate_vram_requirement(self, resolution: str = "1280x720", duration_seconds: float = 8.0) -> float:
        return 18.5
