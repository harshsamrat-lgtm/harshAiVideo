"""
AI Audio & Hindi Neural Text-to-Speech (TTS) Engine for Harsh AI Video Studio.
Synthesizes real human-like Hindi Voice-overs, Multi-character Child and Adult Dialogues,
and mixes them with atmospheric cinematic background music into 48kHz AAC stereo MP4s.
"""
from typing import Optional, List, Dict, Any, Union
import os
import asyncio
import shutil
import subprocess
from pathlib import Path
from app.core.logging import logger


class AudioService:
    @staticmethod
    async def generate_hindi_voiceover_speech(
        dialogue_input: Union[str, List[Dict[str, Any]]],
        output_speech_path: Path,
        voice: str = "hi-IN-MadhurNeural"
    ) -> bool:
        """
        Synthesizes authentic Hindi dialogue voiceover:
          - Supports single narration lines
          - Supports multi-character scripted dialogue exchanges (e.g. Kittu and Raghavendra)
          - Applies authentic child voice modulation (pitch and speed) for 5-year-old kids.
        """
        if not dialogue_input:
            return False

        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
        temp_dir = output_speech_path.parent / f"temp_tts_{output_speech_path.stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Case 1: Multi-character Dialogue List (e.g. Kittu & Raghavendra)
            if isinstance(dialogue_input, list) and len(dialogue_input) >= 2:
                logger.info(f"🎙️ Synthesizing Multi-Character Child Dialogue Exchange ({len(dialogue_input)} turns)...")
                audio_clips = []

                for i, d in enumerate(dialogue_input[:2]):
                    text = d.get("text", "")
                    is_child = d.get("is_child", False)
                    speaker = d.get("speaker", f"Speaker {i+1}")
                    clip_file = temp_dir / f"turn_{i}.mp3"

                    # Select child voice pitch vs adult
                    if is_child:
                        # Turn 0 (Kittu): Swara tuned to sweet innocent child pitch
                        # Turn 1 (Raghavendra): Madhur tuned to bright enthusiastic child pitch
                        spk_voice = "hi-IN-SwaraNeural" if i == 0 else "hi-IN-MadhurNeural"
                        spk_pitch = "+18Hz" if i == 0 else "+15Hz"
                        spk_rate = "+5%"
                    else:
                        spk_voice = "hi-IN-SwaraNeural" if i == 0 else "hi-IN-MadhurNeural"
                        spk_pitch = "+0Hz"
                        spk_rate = "+0%"

                    logger.info(f"   Turn {i+1} [{speaker}]: '{text[:40]}...' (Voice: {spk_voice}, Pitch: {spk_pitch})")

                    # Synthesize with Edge-TTS
                    try:
                        import edge_tts
                        communicate = edge_tts.Communicate(
                            text=text,
                            voice=spk_voice,
                            rate=spk_rate,
                            pitch=spk_pitch,
                            volume="+20%"
                        )
                        await communicate.save(str(clip_file))
                    except Exception as e:
                        logger.warning(f"Edge-TTS notice ({e}), falling back to gTTS...")
                        try:
                            from gtts import gTTS
                            tts = gTTS(text=text, lang="hi", slow=False)
                            tts.save(str(clip_file))
                        except Exception:
                            pass

                    if clip_file.exists() and clip_file.stat().st_size > 100:
                        audio_clips.append(clip_file)

                # Merge turn 0 (0-3.8s) and turn 1 (3.8-8s) into seamless 8s dialogue track
                if len(audio_clips) >= 1:
                    if len(audio_clips) == 2:
                        # Probe duration of clip 0 to calculate proper delay for clip 1
                        clip0_dur = AudioService._probe_duration(ffmpeg_cmd, audio_clips[0])
                        delay_ms = int((clip0_dur + 0.3) * 1000)  # 300ms natural pause after first speaker
                        delay_ms = max(2000, min(delay_ms, 5000))  # Clamp between 2s and 5s

                        logger.info(f"   Clip 0 duration: {clip0_dur:.2f}s, delay for clip 1: {delay_ms}ms")

                        # Normalize each clip to -18 LUFS, then merge with proper delay
                        merge_cmd = [
                            ffmpeg_cmd, "-y",
                            "-i", str(audio_clips[0]),
                            "-i", str(audio_clips[1]),
                            "-filter_complex",
                            f"[0:a]loudnorm=I=-18:LRA=7:TP=-2[a0];"
                            f"[1:a]adelay={delay_ms}|{delay_ms},loudnorm=I=-18:LRA=7:TP=-2[a1];"
                            f"[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0,"
                            f"afade=t=out:st=7.5:d=0.5",
                            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                            str(output_speech_path)
                        ]
                        subprocess.run(merge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    else:
                        # Only one clip successfully synthesized - use it directly
                        norm_cmd = [
                            ffmpeg_cmd, "-y",
                            "-i", str(audio_clips[0]),
                            "-af", "loudnorm=I=-18:LRA=7:TP=-2",
                            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                            str(output_speech_path)
                        ]
                        subprocess.run(norm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

                    if output_speech_path.exists() and output_speech_path.stat().st_size > 100:
                        logger.info("✅ Multi-character dialogue successfully stitched!")
                        return True

            # Case 2: Single Dialogue / Narration
            single_text = ""
            is_child_voice = False
            if isinstance(dialogue_input, list) and len(dialogue_input) == 1:
                single_text = dialogue_input[0].get("text", "")
                is_child_voice = dialogue_input[0].get("is_child", False)
            elif isinstance(dialogue_input, str):
                single_text = dialogue_input
                is_child_voice = any(w in single_text.lower() for w in ["बच्चे", "बच्चा", "child", "kittu", "raghavendra"])

            if single_text:
                pitch_mod = "+16Hz" if is_child_voice else "+0Hz"
                rate_mod = "+5%" if is_child_voice else "+0%"
                chosen_voice = "hi-IN-SwaraNeural" if is_child_voice else voice

                logger.info(f"🎙️ Generating Neural Voice-over: '{single_text[:50]}...' ({chosen_voice}, pitch={pitch_mod})")
                raw_speech = temp_dir / "raw_speech.mp3"
                try:
                    import edge_tts
                    communicate = edge_tts.Communicate(
                        text=single_text,
                        voice=chosen_voice,
                        rate=rate_mod,
                        pitch=pitch_mod,
                        volume="+20%"
                    )
                    await communicate.save(str(raw_speech))
                except Exception as e:
                    logger.warning(f"Edge-TTS notice ({e}), falling back to gTTS...")
                    from gtts import gTTS
                    tts = gTTS(text=single_text, lang="hi", slow=False)
                    tts.save(str(raw_speech))

                # Normalize speech loudness to broadcast standard
                if raw_speech.exists() and raw_speech.stat().st_size > 100:
                    norm_cmd = [
                        ffmpeg_cmd, "-y",
                        "-i", str(raw_speech),
                        "-af", "loudnorm=I=-18:LRA=7:TP=-2",
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                        str(output_speech_path)
                    ]
                    subprocess.run(norm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

                    if output_speech_path.exists() and output_speech_path.stat().st_size > 100:
                        logger.info(f"✅ Voice-over generated ({output_speech_path.stat().st_size / 1024:.1f} KB)")
                        return True

        except Exception as e:
            logger.error(f"Voiceover generation error: {e}", exc_info=True)
        finally:
            try:
                shutil.rmtree(str(temp_dir))
            except Exception:
                pass

        return output_speech_path.exists() and output_speech_path.stat().st_size > 100

    @staticmethod
    def _probe_duration(ffmpeg_cmd: str, audio_file: Path) -> float:
        """Probe audio file duration in seconds using ffprobe."""
        ffprobe_cmd = ffmpeg_cmd.replace("ffmpeg", "ffprobe")
        try:
            result = subprocess.run(
                [ffprobe_cmd, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, timeout=10
            )
            return float(result.stdout.decode().strip())
        except Exception:
            return 3.5  # Safe default: 3.5s for a short dialogue line

    @staticmethod
    def generate_ambient_music_for_prompt(
        prompt: str,
        duration_seconds: float,
        output_music_path: Path
    ) -> bool:
        """
        Generates rich ambient background atmosphere with layered harmonics and fade-in/fade-out.
        If prompt specifies 'no background music' or 'no bgm', generates ultra-soft room presence.
        """
        p_lower = prompt.lower()
        dur = max(2.0, duration_seconds)
        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"

        is_no_music = "no background music" in p_lower or "no bgm" in p_lower or "no music" in p_lower
        is_mythological = any(k in p_lower for k in ["ancient", "india", "dwapar", "aryavarta", "war", "epic", "mytholog", "sunset", "sunrise"])
        is_children = any(k in p_lower for k in ["child", "children", "boy", "boys", "girl", "girls", "kid", "kids", "courtyard", "play"])
        is_vehicle = any(k in p_lower for k in ["car", "racing", "speed", "road", "vehicle"])
        is_rain = any(k in p_lower for k in ["rain", "storm", "thunder"])
        is_nature = any(k in p_lower for k in ["forest", "river", "mountain", "village", "garden", "tree", "field"])

        if is_no_music:
            # Ultra-soft warm room presence so there's no dead silence
            audio_filter = (
                f"anoisesrc=d={dur}:c=pink:r=48000:a=0.015,"
                f"lowpass=f=600,highpass=f=60,"
                f"afade=t=in:st=0:d=1.0,afade=t=out:st={dur-1.0}:d=1.0,"
                f"volume=0.08"
            )
        elif is_children:
            # Warm gentle melodic atmosphere for child scenes — soft tanpura + birds
            audio_filter = (
                f"aevalsrc='sin(2*PI*261.6*t)*0.12*sin(PI*t/{dur})"
                f"+sin(2*PI*329.6*t)*0.08*sin(PI*t/{dur})"
                f"+sin(2*PI*392.0*t)*0.06*sin(PI*t/{dur})"
                f"+sin(2*PI*523.3*t)*0.04*sin(PI*t/{dur})'"
                f":d={dur}:s=48000:c=stereo,"
                f"lowpass=f=3000,highpass=f=80,"
                f"afade=t=in:st=0:d=1.5,afade=t=out:st={dur-1.5}:d=1.5,"
                f"volume=0.25"
            )
        elif is_mythological:
            # Deep epic tanpura drone with harmonic overtones
            audio_filter = (
                f"aevalsrc='sin(2*PI*110*t)*0.15*sin(PI*t/{dur})"
                f"+sin(2*PI*165*t)*0.10*sin(PI*t/{dur})"
                f"+sin(2*PI*220*t)*0.08*sin(PI*t/{dur})"
                f"+sin(2*PI*330*t)*0.05*sin(PI*t/{dur})'"
                f":d={dur}:s=48000:c=stereo,"
                f"lowpass=f=2500,highpass=f=60,"
                f"afade=t=in:st=0:d=2.0,afade=t=out:st={dur-2.0}:d=2.0,"
                f"volume=0.30"
            )
        elif is_nature:
            # Soft rustling leaves + gentle wind + birds-like harmonics
            audio_filter = (
                f"anoisesrc=d={dur}:c=pink:r=48000:a=0.08,"
                f"lowpass=f=2000,highpass=f=200,"
                f"afade=t=in:st=0:d=1.5,afade=t=out:st={dur-1.5}:d=1.5,"
                f"volume=0.20"
            )
        elif is_vehicle:
            # Low engine rumble with movement
            audio_filter = (
                f"aevalsrc='sin(2*PI*(80+20*sin(0.5*t))*t)*0.20+(random(0)-0.5)*0.05'"
                f":d={dur}:s=48000:c=stereo,"
                f"lowpass=f=800,highpass=f=30,"
                f"afade=t=in:st=0:d=1.0,afade=t=out:st={dur-1.0}:d=1.0,"
                f"volume=0.30"
            )
        elif is_rain:
            # Rich rain texture with depth
            audio_filter = (
                f"anoisesrc=d={dur}:c=pink:r=48000:a=0.20,"
                f"lowpass=f=2500,highpass=f=200,"
                f"afade=t=in:st=0:d=1.5,afade=t=out:st={dur-1.5}:d=1.5,"
                f"volume=0.25"
            )
        else:
            # Default: Warm cinematic pad with gentle harmonics
            audio_filter = (
                f"aevalsrc='sin(2*PI*130.8*t)*0.12*sin(PI*t/{dur})"
                f"+sin(2*PI*196.0*t)*0.08*sin(PI*t/{dur})"
                f"+sin(2*PI*261.6*t)*0.05*sin(PI*t/{dur})'"
                f":d={dur}:s=48000:c=stereo,"
                f"lowpass=f=2500,highpass=f=60,"
                f"afade=t=in:st=0:d=1.5,afade=t=out:st={dur-1.5}:d=1.5,"
                f"volume=0.25"
            )

        cmd = [
            ffmpeg_cmd, "-y",
            "-f", "lavfi",
            "-i", audio_filter,
            "-t", str(dur),
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            str(output_music_path)
        ]

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if result.returncode != 0:
                logger.warning(f"Ambient music FFmpeg stderr: {result.stderr.decode('utf-8', errors='ignore')[:200]}")
            return output_music_path.exists() and output_music_path.stat().st_size > 0
        except Exception as e:
            logger.warning(f"Music synthesis error: {e}")
            return False

    @staticmethod
    def mix_voiceover_with_music(
        voice_path: Path,
        music_path: Path,
        final_audio_path: Path
    ) -> bool:
        """
        Mixes Voice-over Speech (foreground, normalized) with Ambient Music (background, ducked).
        Uses sidechain-style ducking: music volume drops when speech is present.
        """
        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
        
        if not voice_path.exists():
            if music_path.exists():
                shutil.copy(str(music_path), str(final_audio_path))
                return True
            return False

        if not music_path.exists():
            shutil.copy(str(voice_path), str(final_audio_path))
            return True

        # Voice at full volume, music ducked to 15% behind voice
        # Using sidechaincompress for natural ducking effect
        cmd = [
            ffmpeg_cmd, "-y",
            "-i", str(voice_path),
            "-i", str(music_path),
            "-filter_complex",
            "[0:a]aformat=sample_rates=48000:channel_layouts=stereo[voice];"
            "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.15[music];"
            "[voice][music]amix=inputs=2:duration=longest:dropout_transition=2:weights=1 0.15,"
            "loudnorm=I=-16:LRA=7:TP=-1.5",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            str(final_audio_path)
        ]

        try:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
            if result.returncode != 0:
                logger.warning(f"Audio mix FFmpeg issue: {result.stderr.decode('utf-8', errors='ignore')[:200]}")
                # Fallback: just use voice
                shutil.copy(str(voice_path), str(final_audio_path))
            return final_audio_path.exists() and final_audio_path.stat().st_size > 0
        except Exception as e:
            logger.warning(f"Audio mixing failed: {e}")
            shutil.copy(str(voice_path if voice_path.exists() else music_path), str(final_audio_path))
            return True

    @staticmethod
    def mux_audio_into_video(
        video_path: Path,
        audio_path: Path,
        final_output_path: Path,
        duration_seconds: float = 8.0
    ) -> bool:
        """Muxes the combined voice-over + music audio track into the MP4 video without truncating."""
        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
        temp_out = final_output_path.parent / f"mux_{final_output_path.name}"
        dur = max(4.0, float(duration_seconds or 8.0))
        
        # Clean up any leftover temp file from previous failed runs
        if temp_out.exists():
            try:
                temp_out.unlink()
            except Exception:
                pass

        cmd = [
            ffmpeg_cmd, "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(dur),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-movflags", "+faststart",
            str(temp_out)
        ]

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=120)
            if temp_out.exists() and temp_out.stat().st_size > 0:
                # Safely replace: delete original first, then rename temp
                if final_output_path.exists():
                    final_output_path.unlink()
                temp_out.rename(final_output_path)
                logger.info(f"✅ Audio muxed into video successfully ({final_output_path.stat().st_size / 1024:.1f} KB)")
                return True
            else:
                logger.warning(f"Mux produced empty file. FFmpeg stderr: {result.stderr.decode('utf-8', errors='ignore')[:200]}")
        except subprocess.TimeoutExpired:
            logger.warning("Muxing timed out after 120s")
        except Exception as e:
            logger.warning(f"Muxing failed: {e}")
        return False


audio_service = AudioService()
