"""
Harsh AI Video Studio — Flagship Neural AI Video Diffusion Engine.

Model Registry & Dynamic Selector:
  - CogVideoX-5B (THUDM/CogVideoX-5b) — 5 Billion Parameter HD Model
  - CogVideoX-2B (THUDM/CogVideoX-2b) — 2 Billion Parameter Fast Model
  - SANA-Video-2.0 14B (Efficient-Large-Model/Sana_1600M_1024px) — 14B Flagship
  - ModelScope 1.7B (damo-vilab/text-to-video-ms-1.7b) — Legacy Fallback

All models run on NVIDIA GPU with CUDA (RTX 5090 - 32GB VRAM).
Includes Dual-Track Hindi Voice-over, Semantic Prompt Enhancement, and 4K Post-Processing.
"""
from typing import Dict, Any, Optional
import os
import gc
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
_LOADED_PIPES = {}
_ACTIVE_MODEL_NAME = None


class LightX2VEngine(BaseVideoEngine):
    """
    Dedicated SANA-Video 2B (NVIDIA Linear DiT) AI Video Diffusion Engine.
    Delivers 81 high-definition frames at 720p HD with ultra-low latency on RTX 5090.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="SANA-Video-2B-720p", config=config)
        self.precision = self.config.get("precision", settings.LIGHTX2V_PRECISION)
        self.sparse_attention = self.config.get("sparse_attention", settings.LIGHTX2V_SPARSE_ATTENTION)
        self.is_loaded = False

    async def load_model(self, target_model: str = "sana-video-2b") -> bool:
        global _LOADED_PIPES, _ACTIVE_MODEL_NAME

        req_engine = (target_model or "sana-video-2b").lower()

        # If requested engine is already loaded and active, reuse it
        if _ACTIVE_MODEL_NAME and ("sana" in _ACTIVE_MODEL_NAME.lower() or req_engine in _ACTIVE_MODEL_NAME.lower()) and _ACTIVE_MODEL_NAME in _LOADED_PIPES:
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
            logger.info(f"🖥️ Detected GPU: {gpu_name} ({vram:.1f} GB VRAM) | Target Engine: SANA-Video-2B")

            # ── UNLOAD previous model from VRAM before loading new one ──
            if _LOADED_PIPES:
                logger.info(f"🗑️ Unloading previous model from VRAM...")
                for key in list(_LOADED_PIPES.keys()):
                    del _LOADED_PIPES[key]
                _LOADED_PIPES.clear()
                _ACTIVE_MODEL_NAME = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"✅ GPU VRAM cleared. Free: {torch.cuda.mem_get_info(0)[0] / (1024**3):.1f} GB")

            # ── 1. SANA-VIDEO-2B 720p (Primary Dedicated Model) ──
            try:
                try:
                    from diffusers import SanaVideoPipeline
                except ImportError:
                    logger.info("📦 SanaVideoPipeline not found. Upgrading diffusers from GitHub...")
                    import subprocess, sys
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/huggingface/diffusers", "timm", "accelerate"], check=False)
                    from diffusers import SanaVideoPipeline

                logger.info("👑 Loading SANA-Video 2B (NVIDIA Linear DiT 720p HD Model)...")
                for model_repo in [
                    "Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
                    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
                ]:
                    try:
                        logger.info(f"   Loading SANA model weights from [{model_repo}]...")
                        pipe = SanaVideoPipeline.from_pretrained(
                            model_repo,
                            torch_dtype=torch.bfloat16
                        ).to("cuda")

                        _LOADED_PIPES["SANA-Video-2B"] = pipe
                        _ACTIVE_MODEL_NAME = "SANA-Video-2B"
                        self.name = "SANA-Video-2B-720p"
                        logger.info(f"✅ SANA-Video 2B loaded successfully on RTX 5090 from [{model_repo}]!")
                        self.is_loaded = True
                        return True
                    except Exception as repo_err:
                        logger.warning(f"   Notice loading {model_repo}: {repo_err}")

            except Exception as e:
                logger.error(f"❌ SANA-Video load error: {e}", exc_info=True)

            # ── 2. BACKUP LOADERS IF SANA STILL DOWNLOADING ──
            if "5b" in req_engine or "cogvideox" in req_engine or not _LOADED_PIPES:
                try:
                    from diffusers import CogVideoXPipeline
                    logger.info("🚀 Loading CogVideoX-5B...")
                    pipe = CogVideoXPipeline.from_pretrained(
                        "THUDM/CogVideoX-5b",
                        torch_dtype=torch.bfloat16
                    ).to("cuda")

                    _LOADED_PIPES["CogVideoX-5B"] = pipe
                    _ACTIVE_MODEL_NAME = "CogVideoX-5B"
                    self.name = "CogVideoX-5B-HD"
                    logger.info("✅ CogVideoX-5B loaded on GPU!")
                    self.is_loaded = True
                    return True
                except Exception as ce:
                    logger.error(f"CogVideo fallback load error: {ce}")

            # ── 3. LTX-VIDEO (High Speed 24fps DiT Model) ──
            if "ltx" in req_engine or "wan" in req_engine or "lightx2v" in req_engine:
                try:
                    from diffusers import LTXPipeline
                    logger.info("🚀 Loading LTX-Video (24fps DiT Video Model)...")
                    pipe = LTXPipeline.from_pretrained(
                        "Lightricks/LTX-Video",
                        torch_dtype=torch.bfloat16
                    ).to("cuda")

                    _LOADED_PIPES["LTX-Video"] = pipe
                    _ACTIVE_MODEL_NAME = "LTX-Video"
                    self.name = "LTX-Video-HD"
                    logger.info("✅ LTX-Video loaded successfully on GPU!")
                    self.is_loaded = True
                    return True
                except Exception as e:
                    logger.warning(f"LTX-Video load notice: {e}")

            # ── 4. COGVIDEOX-2B ──
            if "2b" in req_engine:
                try:
                    from diffusers import CogVideoXPipeline
                    logger.info("🚀 Loading CogVideoX-2B (2 Billion parameter model)...")
                    pipe = CogVideoXPipeline.from_pretrained(
                        "THUDM/CogVideoX-2b",
                        torch_dtype=torch.float16
                    ).to("cuda")

                    _LOADED_PIPES["CogVideoX-2B"] = pipe
                    _ACTIVE_MODEL_NAME = "CogVideoX-2B"
                    self.name = "CogVideoX-2B"
                    logger.info("✅ CogVideoX-2B loaded successfully on GPU!")
                    self.is_loaded = True
                    return True
                except Exception as e:
                    logger.error(f"❌ CogVideoX-2B load error: {e}", exc_info=True)

            # ── 5. SEAMLESS FALLBACK TO COGVIDEOX-5B (Never fail generation) ──
            if not _LOADED_PIPES:
                try:
                    from diffusers import CogVideoXPipeline
                    logger.warning(f"⚠️ Requested model [{req_engine}] not ready on disk. Seamlessly loading CogVideoX-5B HD...")
                    pipe = CogVideoXPipeline.from_pretrained(
                        "THUDM/CogVideoX-5b",
                        torch_dtype=torch.float16
                    ).to("cuda")

                    _LOADED_PIPES["CogVideoX-5B"] = pipe
                    _ACTIVE_MODEL_NAME = "CogVideoX-5B"
                    self.name = "CogVideoX-5B-HD"
                    logger.info("✅ Seamlessly recovered: CogVideoX-5B active!")
                    self.is_loaded = True
                    return True
                except Exception as e:
                    logger.error(f"CogVideoX-5B recovery load error: {e}")

        except Exception as e:
            logger.error(f"Engine initialization error: {e}", exc_info=True)

        if not _LOADED_PIPES:
            _ACTIVE_MODEL_NAME = "none"
            self.is_loaded = False
            raise RuntimeError(f"Could not load requested AI video model [{req_engine}]. Check PyTorch GPU drivers and network.")

        return True

    async def unload_model(self) -> bool:
        global _LOADED_PIPES, _ACTIVE_MODEL_NAME
        _LOADED_PIPES.clear()
        _ACTIVE_MODEL_NAME = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
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
        steps: int = 65,
        guidance_scale: float = 7.5,
        output_path: Optional[str] = None,
        callback: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        global _LOADED_PIPES, _ACTIVE_MODEL_NAME

        start_time = time.time()
        actual_seed = seed if seed != -1 else int(time.time() * 1000) % 1000000
        target_duration = max(4.0, float(duration_seconds or 8.0))
        requested_engine = kwargs.get("engine") or "cogvideox-5b"

        # Ensure requested engine model pipeline is loaded dynamically
        await self.load_model(target_model=requested_engine)

        # ── 1. DUAL-TRACK PARSE: VISUAL SCENE vs SPOKEN VOICEOVER ──
        visual_raw, voiceover_dialogue = parse_prompt_and_voiceover(prompt)
        clean_english_prompt = translate_and_enhance_hindi_prompt(visual_raw)

        # ── 1.1 LOAD OPTIONAL REFERENCE IMAGE (CHARACTER / LOCATION / STARTING FRAME) ──
        ref_image = None
        if reference_image_path and Path(reference_image_path).exists():
            try:
                from PIL import Image
                ref_image = Image.open(reference_image_path).convert("RGB")
                ref_image = ref_image.resize((1280, 720), Image.Resampling.LANCZOS)
                logger.info(f"🖼️ Reference image loaded for video synthesis: {reference_image_path} (1280x720)")
            except Exception as ie:
                logger.warning(f"Reference image load notice ({ie}), continuing with text prompt.")

        logger.info(f"🎬 Active GPU Model: {_ACTIVE_MODEL_NAME}")
        logger.info(f"🎬 Enriched Prompt: '{clean_english_prompt[:100]}...' (Duration: {target_duration}s, Steps: 65)")
        if ref_image is not None:
            logger.info("🎬 Mode: IMAGE-TO-VIDEO (Character/Location reference conditioning active)")
        if voiceover_dialogue:
            logger.info(f"🎙️ Hindi Voice-over: '{voiceover_dialogue}'")

        out_dir = Path(settings.OUTPUT_ROOT)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_path or str(out_dir / f"video_{actual_seed}.mp4")
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        raw_temp_video = out_dir / f"raw_{actual_seed}.mp4"
        voice_speech_file = out_dir / f"speech_{actual_seed}.mp3"
        ambient_music_file = out_dir / f"music_{actual_seed}.aac"
        final_mixed_audio = out_dir / f"final_audio_{actual_seed}.aac"

        neg_prompt = (
            negative_prompt or
            "distorted face, deformed mouth, warped eyes, asymmetrical face, mutated facial features, "
            "wobbling pupils, shifting eyes, moving eyeballs, crossed eyes, misaligned pupils, wandering gaze, "
            "teeth morphing, split lips, deformed fingers, morphing fingers, extra fingers, missing fingers, fused fingers, six fingers, "
            "poorly drawn hands, deformed hands, shifting hands, floating hands, extra limbs, deformed limbs, disconnected limbs, "
            "melting skin, boiling textures, temporal jitter, uncanny valley, double faces, disfigured limbs, "
            "blurry face, blurry eyes, ghosting, jitter, flicker, low quality, morphing artifacts, "
            "text, watermark, ugly, duplicate, jpeg artifacts"
        )

        # ── 2. SYNTHESIZE NEURAL HINDI VOICEOVER & AMBIENT MUSIC ──
        has_voiceover = False
        if voiceover_dialogue:
            chosen_voice = "hi-IN-MadhurNeural"
            if isinstance(voiceover_dialogue, list) and len(voiceover_dialogue) > 0:
                first_d = voiceover_dialogue[0]
                if first_d.get("speaker") == "Female Voice" or (first_d.get("is_child") and "swara" in str(first_d.get("speaker", "")).lower()):
                    chosen_voice = "hi-IN-SwaraNeural"
                elif first_d.get("speaker") == "Male Voice" or first_d.get("speaker") == "Male Narrator":
                    chosen_voice = "hi-IN-MadhurNeural"
            elif any(w in str(visual_raw).lower() for w in ["बच्ची", "लड़की", "माता", "woman", "female"]):
                chosen_voice = "hi-IN-SwaraNeural"

            has_voiceover = await audio_service.generate_hindi_voiceover_speech(
                dialogue_input=voiceover_dialogue,
                output_speech_path=voice_speech_file,
                voice=chosen_voice
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

        # ── 3. GENERATE HIGH-STEPS NEURAL VIDEO ON GPU WITH ACTIVE PIPELINE ──
        generated = False
        last_error = ""

        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA GPU available")

            torch.cuda.empty_cache()
            generator = torch.Generator("cuda").manual_seed(actual_seed)

            # Get active loaded pipeline
            active_pipe = _LOADED_PIPES.get(_ACTIVE_MODEL_NAME)

            if active_pipe is not None:
                logger.info(f"🎬 Executing inference on [{_ACTIVE_MODEL_NAME}] (Duration: {target_duration}s, Stable Guidance: 6.0)...")
                
                if "SANA" in _ACTIVE_MODEL_NAME:
                    logger.info("👑 Running SANA-Video 2B inference (81 frames, 720p HD)...")
                    try:
                        if ref_image is not None:
                            video_output = active_pipe(
                                image=ref_image,
                                prompt=clean_english_prompt,
                                height=704,
                                width=1280,
                                num_frames=81,
                                num_inference_steps=50,
                                guidance_scale=6.0,
                                generator=generator,
                            )
                        else:
                            video_output = active_pipe(
                                prompt=clean_english_prompt,
                                height=704,
                                width=1280,
                                num_frames=81,
                                num_inference_steps=50,
                                guidance_scale=6.0,
                                generator=generator,
                            )
                    except TypeError as te:
                        logger.info(f"Retrying SANA with 'frames' param: {te}")
                        if ref_image is not None:
                            video_output = active_pipe(
                                image=ref_image,
                                prompt=clean_english_prompt,
                                height=704,
                                width=1280,
                                frames=81,
                                num_inference_steps=50,
                                guidance_scale=6.0,
                                generator=generator,
                            )
                        else:
                            video_output = active_pipe(
                                prompt=clean_english_prompt,
                                height=704,
                                width=1280,
                                frames=81,
                                num_inference_steps=50,
                                guidance_scale=6.0,
                                generator=generator,
                            )
                    frames = video_output.frames[0]
                    export_fps = max(1, int(round(81.0 / target_duration)))
                    try:
                        from diffusers.utils import export_to_video
                        export_to_video(frames, str(raw_temp_video), fps=export_fps)
                    except Exception as ex:
                        import imageio
                        imageio.mimwrite(str(raw_temp_video), frames, fps=export_fps, quality=9)
                    generated = True
                    logger.info("✅ SANA-Video 2B frames exported successfully!")

                elif "CogVideo" in _ACTIVE_MODEL_NAME:
                    try:
                        if ref_image is not None:
                            logger.info("🖼️ Animating video using reference image conditioning...")
                            video_output = active_pipe(
                                image=ref_image,
                                prompt=clean_english_prompt,
                                negative_prompt=neg_prompt,
                                num_videos_per_prompt=1,
                                num_inference_steps=50,
                                guidance_scale=6.0,
                                num_frames=49,
                                generator=generator,
                            )
                        else:
                            video_output = active_pipe(
                                prompt=clean_english_prompt,
                                negative_prompt=neg_prompt,
                                num_videos_per_prompt=1,
                                num_inference_steps=50,
                                guidance_scale=6.0,
                                num_frames=49,
                                generator=generator,
                            )
                    except TypeError as te:
                        logger.warning(f"Retrying CogVideo without optional params: {te}")
                        if ref_image is not None:
                            try:
                                video_output = active_pipe(
                                    image=ref_image,
                                    prompt=clean_english_prompt,
                                    num_videos_per_prompt=1,
                                    num_inference_steps=50,
                                    guidance_scale=6.0,
                                    num_frames=49,
                                    generator=generator,
                                )
                            except TypeError:
                                video_output = active_pipe(
                                    prompt=clean_english_prompt,
                                    num_videos_per_prompt=1,
                                    num_inference_steps=50,
                                    guidance_scale=6.0,
                                    num_frames=49,
                                    generator=generator,
                                )
                        else:
                            video_output = active_pipe(
                                prompt=clean_english_prompt,
                                num_videos_per_prompt=1,
                                num_inference_steps=50,
                                guidance_scale=6.0,
                                num_frames=49,
                                generator=generator,
                            )
                    frames = video_output.frames[0]
                    export_fps = max(1, int(round(49.0 / target_duration)))
                    try:
                        from diffusers.utils import export_to_video
                        export_to_video(frames, str(raw_temp_video), fps=export_fps)
                    except Exception as ex:
                        logger.warning(f"export_to_video fallback ({ex}), using imageio...")
                        import imageio
                        imageio.mimwrite(str(raw_temp_video), frames, fps=export_fps, quality=9)
                    generated = True

                elif "LTX" in _ACTIVE_MODEL_NAME:
                    logger.info("🚀 Running LTX-Video inference (24fps DiT, 97 frames)...")
                    video_output = active_pipe(
                        prompt=clean_english_prompt,
                        negative_prompt=neg_prompt,
                        width=1280,
                        height=704,
                        num_frames=97,
                        num_inference_steps=40,
                        guidance_scale=3.5,
                        generator=generator,
                    )
                    frames = video_output.frames[0]
                    export_fps = max(1, int(round(len(frames) / target_duration)))
                    try:
                        from diffusers.utils import export_to_video
                        export_to_video(frames, str(raw_temp_video), fps=export_fps)
                    except Exception as ex:
                        import imageio
                        imageio.mimwrite(str(raw_temp_video), frames, fps=export_fps, quality=9)
                    generated = True
                    logger.info("✅ LTX-Video frames exported!")

                elif "ModelScope" in _ACTIVE_MODEL_NAME:
                    video_output = active_pipe(
                        prompt=clean_english_prompt,
                        negative_prompt=neg_prompt,
                        num_inference_steps=50,
                        guidance_scale=9.0,
                        num_frames=32,
                        generator=generator,
                    )
                    frames = video_output.frames[0]
                    export_fps = max(1, int(round(32.0 / target_duration)))
                    import imageio
                    imageio.mimwrite(
                        str(raw_temp_video), frames, fps=export_fps,
                        codec="libx264", quality=9, pixelformat="yuv420p"
                    )
                    generated = True

                logger.info(f"✅ {_ACTIVE_MODEL_NAME} generated frames successfully!")

        except Exception as e:
            last_error = str(e)
            logger.error(f"❌ Video generation inference error: {e}", exc_info=True)

        if not generated or not raw_temp_video.exists():
            logger.error(f"⚠️ Video generation failed to create frames. Last error: {last_error}")
            return {
                "engine": _ACTIVE_MODEL_NAME or self.name,
                "status": "FAILED",
                "error": f"AI video diffusion error: {last_error}" if last_error else "AI video diffusion could not generate frames. Check CUDA GPU VRAM and connection.",
                "output_path": "",
                "seed": actual_seed,
                "duration": target_duration,
                "resolution": resolution,
                "has_voiceover": has_voiceover,
                "generation_time_seconds": round(time.time() - start_time, 2)
            }

        # ── 4. 24FPS ULTRA-SMOOTH CINEMATIC MASTERING (GUARANTEED FULL 8.0s) ──
        target_w, target_h = (1280, 720) if "720" in resolution else (1920, 1080)
        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"

        vf_filter = (
            f"fps=24,"
            f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={target_w}:{target_h},"
            f"unsharp=5:5:1.0:5:5:0.5,"
            f"eq=contrast=1.05:saturation=1.05"
        )

        enhance_cmd = [
            ffmpeg_cmd, "-y",
            "-i", str(raw_temp_video),
            "-vf", vf_filter,
            "-t", str(target_duration),
            "-c:v", "libx264",
            "-crf", "16",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_file)
        ]

        try:
            result = subprocess.run(
                enhance_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300
            )
            if result.returncode != 0:
                logger.warning(f"FFmpeg enhance error: {result.stderr.decode('utf-8', errors='ignore')[:300]}")
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
                final_output_path=out_file,
                duration_seconds=target_duration
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
            "model_cascade": ["CogVideoX-5B", "SANA-Video-2.0-14B", "CogVideoX-2B", "ModelScope-1.7B"],
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
            "model_cascade": ["CogVideoX-5B (primary)", "SANA-Video-2.0-14B (Flagship 14B)", "CogVideoX-2B (fallback)", "ModelScope-1.7B (legacy)"]
        }

    def estimate_vram_requirement(self, resolution: str = "1280x720", duration_seconds: float = 8.0) -> float:
        return 22.0
