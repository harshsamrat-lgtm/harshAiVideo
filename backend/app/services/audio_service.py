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
                            volume="+35%"
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

                # Merge turn 0 (0-4s) and turn 1 (4-8s) into seamless 8s dialogue track
                if len(audio_clips) == 2:
                    # Delay clip 1 by 3800ms so Raghavendra speaks right after Kittu
                    merge_cmd = [
                        ffmpeg_cmd, "-y",
                        "-i", str(audio_clips[0]),
                        "-i", str(audio_clips[1]),
                        "-filter_complex",
                        "[0:a]volume=1.6[a0];"
                        "[1:a]adelay=3800|3800,volume=1.6[a1];"
                        "[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0",
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                        str(output_speech_path)
                    ]
                    subprocess.run(merge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    if output_speech_path.exists() and output_speech_path.stat().st_size > 100:
                        logger.info("✅ Multi-character child dialogue successfully stitched into 8s track!")
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
                try:
                    import edge_tts
                    communicate = edge_tts.Communicate(
                        text=single_text,
                        voice=chosen_voice,
                        rate=rate_mod,
                        pitch=pitch_mod,
                        volume="+35%"
                    )
                    await communicate.save(str(output_speech_path))
                    if output_speech_path.exists() and output_speech_path.stat().st_size > 100:
                        logger.info(f"✅ Voice-over generated ({output_speech_path.stat().st_size / 1024:.1f} KB)")
                        return True
                except Exception as e:
                    logger.warning(f"Edge-TTS notice ({e}), falling back to gTTS...")
                    from gtts import gTTS
                    tts = gTTS(text=single_text, lang="hi", slow=False)
                    tts.save(str(output_speech_path))
                    return output_speech_path.exists() and output_speech_path.stat().st_size > 100

        except Exception as e:
            logger.error(f"Voiceover generation error: {e}", exc_info=True)
        finally:
            try:
                shutil.rmtree(str(temp_dir))
            except Exception:
                pass

        return output_speech_path.exists() and output_speech_path.stat().st_size > 100

    @staticmethod
    def generate_ambient_music_for_prompt(
        prompt: str,
        duration_seconds: float,
        output_music_path: Path
    ) -> bool:
        """
        Generates ambient music or soft room atmosphere.
        If prompt specifies 'no background music' or 'no bgm', generates soft gentle presence.
        """
        p_lower = prompt.lower()
        dur = max(2.0, duration_seconds)
        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"

        is_no_music = "no background music" in p_lower or "no bgm" in p_lower or "no music" in p_lower
        is_mythological = any(k in p_lower for k in ["ancient", "india", "dwapar", "aryavarta", "war", "epic", "mytholog", "sunset", "sunrise"])
        is_vehicle = any(k in p_lower for k in ["car", "racing", "speed", "road", "vehicle"])
        is_rain = any(k in p_lower for k in ["rain", "storm", "thunder"])

        if is_no_music:
            # Soft warm ambient air presence so video is not dead silent between words
            audio_filter = f"anoisesrc=d={dur}:c=pink:r=48000:a=0.03,lowpass=f=800,volume=0.1"
        elif is_mythological:
            audio_filter = (
                f"aevalsrc='sin(2*PI*110*t)*0.2+sin(2*PI*165*t)*0.15+sin(2*PI*220*t)*0.1':d={dur}:s=48000:c=stereo,"
                "lowpass=f=2200,volume=0.4"
            )
        elif is_vehicle:
            audio_filter = (
                f"aevalsrc='sin(2*PI*(90+30*t)*t)*0.3+(random(0)-0.5)*0.1':d={dur}:s=48000:c=stereo,"
                "lowpass=f=1200,volume=0.4"
            )
        elif is_rain:
            audio_filter = f"anoisesrc=d={dur}:c=pink:r=48000:a=0.25,lowpass=f=2200,highpass=f=300,volume=0.3"
        else:
            audio_filter = (
                f"aevalsrc='sin(2*PI*130*t)*0.2+sin(2*PI*195*t)*0.15':d={dur}:s=48000:c=stereo,"
                "lowpass=f=2200,volume=0.3"
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
        Mixes Voice-over Speech (Volume 1.6, prominent) with Ambient Music (Volume 0.2, subtle back).
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

        cmd = [
            ffmpeg_cmd, "-y",
            "-i", str(voice_path),
            "-i", str(music_path),
            "-filter_complex", "[0:a]volume=1.6[a1];[1:a]volume=0.2[a2];[a1][a2]amix=inputs=2:duration=longest:dropout_transition=1",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            str(final_audio_path)
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
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
        """Muxes the combined voice-over + music audio track into the MP4 video without truncating video."""
        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
        temp_out = final_output_path.parent / f"mux_{final_output_path.name}"
        dur = max(4.0, float(duration_seconds or 8.0))
        
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
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if temp_out.exists() and temp_out.stat().st_size > 0:
                if final_output_path.exists():
                    final_output_path.unlink()
                temp_out.rename(final_output_path)
                return True
        except Exception as e:
            logger.warning(f"Muxing failed: {e}")
        return False


audio_service = AudioService()
