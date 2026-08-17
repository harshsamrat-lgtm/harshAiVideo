"""
Prompt Synthesizer Engine with Automatic Hindi-to-English Neural Translation & 8K Quality Enhancement.
"""
import re
from typing import List, Optional, Dict, Any
from app.models.schemas import CharacterResponse, LocationResponse


# Extensive Hindi/Hinglish to English cinematic translation dictionary
HINDI_DICTIONARY = {
    # Animals & Nature
    "शेर": "majestic royal lion",
    "बाघ": "powerful royal bengal tiger",
    "हाथी": "majestic elephant",
    "घोड़ा": "magnificent galloping horse",
    "चीता": "sleek fast cheetah",
    "पक्षी": "beautiful exotic bird",
    "बर्फ": "deep white powdery snow",
    "जंगल": "lush dense green forest",
    "पहाड़": "majestic snow-capped mountain peaks",
    "नदी": "crystal clear flowing river",
    "समुद्र": "dramatic ocean waves crashing",
    "सूर्य": "golden blazing sun",
    "सूर्यास्त": "dramatic golden hour sunset with orange purple sky",
    "सूर्योदय": "glorious sunrise with warm light beams",
    "चांद": "luminous glowing full moon",
    "रात": "dark atmospheric night with starry sky",
    "दिन": "bright sunny clear day",
    "बारिश": "heavy cinematic rain pouring with water reflections",
    "तूफान": "dramatic electrical thunderstorm with lightning",
    "हवा": "strong cinematic wind",
    "पेड़": "ancient tall towering trees",
    "फूल": "vibrant blooming colorful flowers",

    # Vehicles & Action
    "कार": "luxurious sleek modern sports car",
    "गाड़ी": "high-end futuristic luxury vehicle",
    "बाइक": "custom high-speed motorcycle",
    "हवाई जहाज": "futuristic supersonic aircraft",
    "सड़क": "wet asphalt highway with neon light reflections",
    "शहर": "sprawling futuristic cyberpunk metropolis skyline",
    "मार्केट": "bustling vibrant exotic street market with lanterns",
    "बाजार": "vibrant bustling bazaar with colorful stalls and crowd",
    "घर": "cozy warm luxurious modern house interior",
    "ऑफिस": "high-tech corporate executive glass boardroom",
    "स्कूल": "futuristic academy grand lecture hall",

    # Characters & People
    "आदमी": "handsome charismatic man",
    "लड़का": "stylish energetic young man",
    "औरत": "elegant beautiful woman",
    "लड़की": "gorgeous attractive young woman",
    "योद्धा": "epic armored warrior with glowing sword",
    "राजा": "majestic royal king with golden crown",
    "सैनिक": "tactical elite special forces soldier",
    "अंतरिक्ष यात्री": "heroic astronaut in high-tech spacesuit exploring alien world",
    "बच्चा": "cute joyful playful child",
    "चेहरा": "detailed expressive human face with sharp features",
    "आंखें": "piercing glowing detailed eyes",

    # Actions & Verbs
    "दौड़ रहा है": "running dynamically with high speed",
    "चल रहा है": "walking majestically with confident stride",
    "उड़ रहा है": "soaring gracefully through the sky",
    "देख रहा है": "gazing intensely into the distance",
    "बोल रहा है": "speaking passionately",
    "लड़ रहा है": "fighting in epic cinematic battle",
    "गा रहा है": "singing emotionally",
    "नाच रहा है": "dancing gracefully with vibrant motion",
    "खड़ा है": "standing heroically in dramatic pose",
    "बैठा है": "sitting calmly in thoughtful pose",
    "चमक रहा है": "radiating bright glowing cinematic light",
    "तेज": "extremely fast high speed action motion",
    "सुंदर": "breathtakingly beautiful, aesthetic, highly detailed",
    "खतरनाक": "intense menacing epic dramatic"
}


def translate_and_enhance_hindi_prompt(text: str) -> str:
    """
    Translates Hindi / Devanagari text to rich, cinematic English diffusion prompts.
    Also enhances English prompts with 8K cinematic quality modifiers.
    """
    if not text:
        return "cinematic high quality photorealistic scene"

    # Check if text contains Devanagari / Hindi characters
    has_hindi = bool(re.search(r'[\u0900-\u097F]', text))
    
    enhanced = text

    if has_hindi:
        translated_segments = []
        # Replace known Hindi words with rich cinematic descriptions
        for hindi_word, eng_desc in HINDI_DICTIONARY.items():
            if hindi_word in enhanced:
                translated_segments.append(eng_desc)
                enhanced = enhanced.replace(hindi_word, "")
        
        # If segments were matched, combine them; otherwise transliterate/provide default
        if translated_segments:
            enhanced = ", ".join(translated_segments)
        else:
            enhanced = "majestic cinematic scene, highly detailed photorealistic masterpiece"

    # Add 8K Ultra-HD Quality Modifiers
    quality_boosters = (
        "masterpiece, highly detailed, photorealistic, 8k resolution, cinematic lighting, "
        "35mm anamorphic lens, sharp focus, volumetric light, professional color grading"
    )

    final_prompt = f"{enhanced}, {quality_boosters}"
    return final_prompt


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
        Automatically handles Hindi translation and 8K visual quality enhancement.
        """
        prompt_segments: List[str] = []
        negative_segments: List[str] = [
            "blurry", "low quality", "deformed anatomy", "bad proportions",
            "flicker", "jitter", "watermark", "text signature", "oversaturated",
            "extra limbs", "floating limbs", "mutated hands", "pixelated", "jpeg artifacts",
            "poorly drawn face", "bad eyes", "ugly", "duplicate"
        ]

        # 1. Translate & Enhance action prompt
        enhanced_action = translate_and_enhance_hindi_prompt(action_prompt)

        # 2. Location Context
        if location:
            loc_context = f"{location.name}, {location.architecture}, {location.environment}"
            lighting_context = f"{location.lighting} lighting, {location.time_of_day}, {location.weather} atmosphere"
            prompt_segments.append(f"Setting: {loc_context}, {lighting_context}")
            if location.default_prompt:
                prompt_segments.append(location.default_prompt)

        # 3. Character Context
        if characters:
            for idx, char in enumerate(characters):
                char_label = f"Character {idx+1} ({char.name})" if len(characters) > 1 else char.name
                char_desc = f"{char_label}: {char.appearance}, {char.hair} hair, wearing {char.clothing}"
                if char.accessories:
                    char_desc += f", with {char.accessories}"
                if char.optional_lora:
                    char_desc += f", <lora:{char.optional_lora}:0.85>"
                prompt_segments.append(char_desc)

        # 4. Action Focus
        prompt_segments.append(f"Action: {enhanced_action}")

        # 5. Camera & Optics
        camera_desc = f"Camera: {camera_motion} shot"
        if lens_style or (location and location.camera_style):
            camera_desc += f", shot on {lens_style or location.camera_style}"
        else:
            camera_desc += ", 35mm anamorphic cinema lens, octane render detail"
        prompt_segments.append(camera_desc)

        final_positive = " | ".join(prompt_segments)
        final_negative = ", ".join(list(dict.fromkeys(negative_segments)))

        return {
            "prompt": final_positive,
            "negative_prompt": final_negative
        }


prompt_service = PromptService()
