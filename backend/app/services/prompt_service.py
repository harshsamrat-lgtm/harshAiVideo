"""
Prompt Synthesizer Engine for Harsh AI Video Studio.
Composes augmented prompts merging Character Bibles (multi-character),
Location Bibles (architecture/lighting), Scene narrative, and LoRA triggers.
"""
from typing import List, Optional, Dict, Any
from app.models.schemas import CharacterResponse, LocationResponse


class PromptService:
    @staticmethod
    def synthesize_shot_prompt(
        action_prompt: str,
        characters: Optional[List[CharacterResponse]] = None,
        location: Optional[LocationResponse] = None,
        camera_motion: str = "cinematic tracking",
        lens_style: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Synthesizes an optimal prompt for Wan 2.2 I2V / LightX2V NVFP4 diffusion.
        Combines character identity anchors, location atmosphere, action, and negatives.
        """
        prompt_segments: List[str] = []
        negative_segments: List[str] = [
            "blurry", "low quality", "deformed anatomy", "bad proportions",
            "flicker", "jitter", "watermark", "text signature", "oversaturated",
            "extra limbs", "floating limbs", "mutated hands"
        ]

        # 1. Location Environment Context
        if location:
            loc_context = f"{location.name}, {location.architecture}, {location.environment}"
            lighting_context = f"{location.lighting} lighting, {location.time_of_day}, {location.weather} atmosphere"
            prompt_segments.append(f"Setting: {loc_context}, {lighting_context}")
            if location.default_prompt:
                prompt_segments.append(location.default_prompt)

        # 2. Multi-Character Descriptions & Identity Anchors
        if characters:
            for idx, char in enumerate(characters):
                char_label = f"Character {idx+1} ({char.name})" if len(characters) > 1 else char.name
                char_desc = f"{char_label}: {char.appearance}, {char.hair} hair, wearing {char.clothing}"
                if char.accessories:
                    char_desc += f", with {char.accessories}"
                if char.optional_lora:
                    char_desc += f", <lora:{char.optional_lora}:0.85>"
                prompt_segments.append(char_desc)
                if char.default_prompt:
                    prompt_segments.append(char.default_prompt)
                if char.negative_prompt:
                    negative_segments.append(char.negative_prompt)

        # 3. Scene Action & Narrative Focus
        prompt_segments.append(f"Action: {action_prompt}")

        # 4. Camera Framing & Motion
        camera_desc = f"Camera: {camera_motion} shot"
        if lens_style or (location and location.camera_style):
            camera_desc += f", shot on {lens_style or location.camera_style}"
        else:
            camera_desc += ", 35mm anamorphic cinema lens, photorealistic 8k octane render detail"
        prompt_segments.append(camera_desc)

        final_positive = " | ".join(prompt_segments)
        final_negative = ", ".join(list(dict.fromkeys(negative_segments)))

        return {
            "prompt": final_positive,
            "negative_prompt": final_negative
        }


prompt_service = PromptService()
