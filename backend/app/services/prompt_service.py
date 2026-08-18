"""
Prompt Synthesizer & Dual-Track Visual/Voice-over Parser for Harsh AI Video Studio.
Optimized for CogVideoX-5B text-to-video diffusion model.
Splits compound prompts into clean visual scene directions and spoken Voice-over dialogue.
Automatically enhances facial clarity, character features, and 8K visual precision.
"""
import re
from typing import List, Optional, Dict, Any, Tuple
from app.models.schemas import CharacterResponse, LocationResponse


# Hindi to English visual descriptor lexicon
HINDI_DICTIONARY = {
    "शेर": "a majestic lion with detailed fur walking powerfully through grasslands",
    "बाघ": "a powerful royal bengal tiger with sharp stripes stalking through jungle",
    "हाथी": "a majestic ancient elephant with golden ornaments",
    "घोड़ा": "a magnificent horse galloping at full speed",
    "चीता": "a sleek cheetah sprinting across open savanna",
    "पक्षी": "birds soaring gracefully across the sky",
    "बर्फ": "deep white snow covering a vast mountain landscape",
    "जंगल": "a dense ancient forest with massive banyan and sal trees",
    "पहाड़": "towering snow-capped Himalayan mountain peaks",
    "नदी": "a sacred winding river with golden reflections",
    "समुद्र": "dramatic ocean waves crashing against stone cliffs",
    "सूर्य": "the golden sun radiating warm light",
    "सूर्यास्त": "a dramatic golden hour sunset with crimson and amber sky",
    "सूर्योदय": "a glorious golden sunrise spreading dawn light across ancient terrain",
    "चांद": "a luminous glowing full moon in a dark sky",
    "रात": "a dark atmospheric night with a starry cosmic sky",
    "दिन": "a bright clear morning with pristine light",
    "बारिश": "heavy monsoon rain pouring down with water ripples on the ground",
    "तूफान": "a dramatic thunderstorm with dark swirling clouds and lightning",
    "हवा": "a gentle breeze swaying lush green foliage",
    "पेड़": "ancient sacred trees with twisting roots and dense canopy",
    "गांव": "a small ancient settlement with thatched cottages and curling woodsmoke",
    "महल": "a grand ancient stone palace with carved pillars and saffron flags",
    "गाड़ी": "a fast car driving along a scenic road",
    "शहर": "a vibrant modern city with tall buildings and busy streets",
    "अंतरिक्ष": "the vast dark expanse of outer space with distant stars",
    "योद्धा": "ancient Indian warriors with sharp facial features, royal golden crowns, bow and spear",
    "रथ": "an ornate golden chariot pulled by armored horses",
    "द्वापर युग": "ancient India during the legendary Dwapar Yuga era",
    "आर्यावर्त": "the vast epic landscape of ancient Aryavarta with holy rivers and mountains",
    "महायुद्ध": "an epic battlefield with warriors gathering under dramatic sunset sky",
    "कुरुक्षेत्र": "the vast plains of Kurukshetra at sunset with warriors and chariots marching",
}


def parse_prompt_and_voiceover(raw_input: str) -> Tuple[str, Optional[str]]:
    """
    Separates user input into Visual Scene Prompt and Voice-over Dialogue.
    """
    if not raw_input:
        return ("a cinematic landscape at golden hour, photorealistic, 4K", None)

    vo_patterns = [
        r'(?:Voice-over|Voiceover|Voice over|वॉइस ओवर|डायलॉग|Dialogue)\s*:\s*[""\']?(.*?)(?:[""\']?\s*$)',
        r'[""]([\u0900-\u097F\s\.\,…!?\-।]+)[""]'
    ]

    voiceover_text = None
    clean_visual = raw_input

    for pat in vo_patterns:
        match = re.search(pat, raw_input, re.IGNORECASE | re.DOTALL)
        if match:
            voiceover_text = match.group(1).strip().strip('""\'"')
            clean_visual = re.sub(pat, '', clean_visual, flags=re.IGNORECASE | re.DOTALL).strip()
            break

    if not voiceover_text and re.search(r'[\u0900-\u097F]', raw_input):
        has_english = bool(re.search(r'[a-zA-Z]{3,}', raw_input))
        if not has_english:
            voiceover_text = raw_input.strip()

    return (clean_visual.strip(), voiceover_text)


def translate_and_enhance_hindi_prompt(text: str) -> str:
    """
    Translates Hindi/Devanagari to rich cinematic English visual descriptions.
    Optimized for CogVideoX-5B model's prompt format.
    Enforces sharp facial clarity, realistic character details, and 8K visual precision.
    """
    if not text:
        return "A sweeping cinematic medium shot of ancient Indian warriors with sharp facial features and golden armor at sunset, photorealistic 4K quality."

    enhanced = text.strip()

    # Translate Hindi terms to English visual descriptions
    if re.search(r'[\u0900-\u097F]', enhanced):
        translated_parts = []
        remaining = enhanced
        for hindi_word, eng_desc in sorted(HINDI_DICTIONARY.items(), key=lambda x: -len(x[0])):
            if hindi_word in remaining:
                translated_parts.append(eng_desc)
                remaining = remaining.replace(hindi_word, "").strip()
        if translated_parts:
            enhanced = ". ".join(translated_parts)

    p_lower = enhanced.lower()
    has_character = any(c in p_lower for c in ["warrior", "man", "woman", "person", "people", "army", "character", "face", "commander", "king", "lord", "god", "krishna", "arjun", "karna", "hero"])

    # Enhance facial sharpness for character prompts
    if has_character:
        if "face" not in p_lower and "facial" not in p_lower:
            enhanced = f"{enhanced}. Cinematic medium shot, sharp detailed symmetrical facial features, clear expressive eyes, realistic skin texture, 85mm prime lens photography."
        else:
            enhanced = f"{enhanced}. Highly detailed symmetrical facial features, realistic eyes, cinematic dramatic lighting."

    has_quality = any(q in p_lower for q in ["4k", "8k", "photorealistic", "cinematic", "hd", "realistic"])
    if not has_quality:
        enhanced = f"{enhanced}. Cinematic masterpiece, photorealistic 8K resolution, sharp focus, masterwork."

    return enhanced


class PromptService:
    @staticmethod
    def synthesize_shot_prompt(
        action_prompt: str,
        characters: Optional[List[CharacterResponse]] = None,
        location: Optional[LocationResponse] = None,
        camera_motion: str = "cinematic tracking",
        lens_style: Optional[str] = None
    ) -> Dict[str, Any]:
        visual_raw, voiceover = parse_prompt_and_voiceover(action_prompt)
        enhanced_action = translate_and_enhance_hindi_prompt(visual_raw)

        negative_segments: List[str] = [
            "blurry face", "deformed face", "blurry", "low quality", "deformed anatomy", "bad proportions",
            "featureless face", "disfigured eyes", "watermark", "text", "extra limbs", "ugly", "duplicate", "jpeg artifacts"
        ]

        prompt_parts = [enhanced_action]

        if location:
            prompt_parts.append(f"Setting: {location.name}, {location.architecture}, {location.environment}")
        if characters:
            for c in characters:
                prompt_parts.append(f"Character: {c.name} ({c.appearance})")

        return {
            "prompt": ". ".join(prompt_parts),
            "negative_prompt": ", ".join(list(dict.fromkeys(negative_segments))),
            "voiceover_text": voiceover
        }


prompt_service = PromptService()
