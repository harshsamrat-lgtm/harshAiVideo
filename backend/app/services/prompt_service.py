"""
Prompt Synthesizer & Dual-Track Audio/Visual Parser for Harsh AI Video Studio.
Splits compound prompts into clean visual scene directions and spoken Voice-over dialogue.
Optimized for Ancient Indian / Mythological epics, Modern Cinema, and Realistic Diffusion.
"""
import re
from typing import List, Optional, Dict, Any, Tuple
from app.models.schemas import CharacterResponse, LocationResponse


# Hindi to English lexicon for common terms
HINDI_DICTIONARY = {
    "शेर": "majestic royal lion",
    "बाघ": "powerful royal bengal tiger",
    "हाथी": "majestic elephant",
    "घोड़ा": "magnificent galloping horse",
    "चीता": "sleek fast cheetah",
    "पक्षी": "flock of birds flying across sky",
    "बर्फ": "deep white powdery snow",
    "जंगल": "dense ancient Vedic forest",
    "पहाड़": "majestic distant mountain peaks",
    "नदी": "winding sacred river with morning mist",
    "समुद्र": "dramatic ocean waves crashing",
    "सूर्य": "golden blazing sun",
    "सूर्यास्त": "dramatic golden hour sunset with orange sky",
    "सूर्योदय": "golden sunrise spreading light across landscape",
    "चांद": "luminous glowing full moon",
    "रात": "dark atmospheric night with starry sky",
    "दिन": "bright clear morning",
    "बारिश": "heavy cinematic rain pouring with water reflections",
    "तूफान": "dramatic thunderstorm with lightning",
    "हवा": "gentle morning breeze",
    "पेड़": "ancient tall banyan and sal trees",
    "गांव": "ancient small settlement with thatched cottages and smoke",
    "महल": "grand ancient Indian palace with stone carvings",
    "द्वापर युग": "ancient India during the legendary Dwapar Yuga era",
    "आर्यावर्त": "vast epic landscape of ancient Aryavarta with rivers and mountains",
    "महायुद्ध": "epic battlefield atmosphere of impending war with gathering storm clouds"
}


def parse_prompt_and_voiceover(raw_input: str) -> Tuple[str, Optional[str]]:
    """
    Separates user input into Visual Scene Prompt and Voice-over Dialogue.
    Example:
    Input: "Ancient India... Voice-over: “बहुत समय पहले…”"
    Returns: ("Ancient India...", "बहुत समय पहले…")
    """
    if not raw_input:
        return ("cinematic epic landscape, 8k resolution", None)

    # Patterns for voiceover / dialogue
    vo_patterns = [
        r'(?:Voice-over|Voiceover|Voice over|वॉइस ओवर|डायलॉग|Dialogue)\s*:\s*["“\']?(.*?)(?:["”\']?$)',
        r'["“]([\u0900-\u097F\s\.\,…!?-]+)["”]' # Devanagari inside quotes
    ]

    voiceover_text = None
    clean_visual = raw_input

    for pat in vo_patterns:
        match = re.search(pat, raw_input, re.IGNORECASE | re.DOTALL)
        if match:
            voiceover_text = match.group(1).strip().strip('"“\'”')
            # Remove voice-over portion from the visual prompt
            clean_visual = re.sub(pat, '', clean_visual, flags=re.IGNORECASE | re.DOTALL).strip()
            break

    # If raw input is entirely Devanagari (and no explicit voiceover tag), treat it as both
    if not voiceover_text and re.search(r'[\u0900-\u097F]', raw_input):
        voiceover_text = raw_input.strip()

    return (clean_visual.strip(), voiceover_text)


def translate_and_enhance_hindi_prompt(text: str) -> str:
    """
    Translates Hindi / Devanagari text to rich, cinematic English diffusion prompts.
    Also enhances mythological, ancient Indian, and epic cinematic themes.
    """
    if not text:
        return "majestic ancient Indian aerial landscape, golden sunrise, 8k resolution, cinematic lighting"

    enhanced = text
    p_lower = text.lower()

    # Check for Ancient Indian / Mythological keywords
    is_mythological = any(k in p_lower for k in [
        "ancient india", "dwapar", "dwapara", "aryavarta", "yuga", "vedic", "settlement", "war", "epic", "mytholog"
    ]) or any(k in text for k in ["द्वापर", "आर्यावर्त", "महायुद्ध", "प्राचीन"])

    # If text contains Devanagari, translate key terms
    if re.search(r'[\u0900-\u097F]', text):
        translated_segments = []
        for hindi_word, eng_desc in HINDI_DICTIONARY.items():
            if hindi_word in enhanced:
                translated_segments.append(eng_desc)
                enhanced = enhanced.replace(hindi_word, "")
        if translated_segments:
            enhanced = ", ".join(translated_segments)

    if is_mythological:
        quality_boosters = (
            "epic mythological Indian cinematic realism, Mahabharat and Baahubali visual scale, "
            "vast aerial drone view of ancient Aryavarta, dense virgin forests, winding holy rivers, "
            "distant misty mountains, small ancient Vedic settlements, golden sunrise beams, "
            "subtle storm clouds on horizon, flock of birds soaring, photorealistic, 4K resolution, "
            "35mm anamorphic cinema lens, atmospheric depth, octane render, masterpiece"
        )
    else:
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
    ) -> Dict[str, Any]:
        """
        Synthesizes visual and audio channels from prompt.
        """
        visual_raw, voiceover = parse_prompt_and_voiceover(action_prompt)
        enhanced_action = translate_and_enhance_hindi_prompt(visual_raw)

        negative_segments: List[str] = [
            "blurry", "low quality", "deformed anatomy", "bad proportions",
            "flicker", "jitter", "watermark", "text signature", "oversaturated",
            "extra limbs", "floating limbs", "mutated hands", "pixelated", "jpeg artifacts",
            "poorly drawn face", "bad eyes", "ugly", "duplicate", "modern buildings", "cars", "wires"
        ]

        prompt_segments = [enhanced_action]

        if location:
            prompt_segments.append(f"Setting: {location.name}, {location.architecture}, {location.environment}")
        if characters:
            for c in characters:
                prompt_segments.append(f"Character: {c.name} ({c.appearance})")

        return {
            "prompt": " | ".join(prompt_segments),
            "negative_prompt": ", ".join(list(dict.fromkeys(negative_segments))),
            "voiceover_text": voiceover
        }


prompt_service = PromptService()
