"""
Prompt Synthesizer, Semantic Web Knowledge Enhancer & Dual-Track Audio/Visual Parser.
Splits compound prompts into clean visual scene directions and spoken Voice-over dialogue.
Automatically enhances prompts with cinematographic knowledge, historical context, and 8K visual parameters.
"""
import re
from typing import List, Optional, Dict, Any, Tuple
from app.models.schemas import CharacterResponse, LocationResponse


# Hindi to English lexicon for common terms
HINDI_DICTIONARY = {
    "शेर": "majestic royal lion walking with power",
    "बाघ": "powerful royal bengal tiger stalking in jungle",
    "हाथी": "majestic ancient elephant with golden ornaments",
    "घोड़ा": "magnificent galloping war horse",
    "चीता": "sleek high-speed cheetah",
    "पक्षी": "flock of birds soaring gracefully across sky",
    "बर्फ": "deep white powdery snow covering landscape",
    "जंगल": "dense ancient Vedic forest with giant sal and banyan trees",
    "पहाड़": "majestic towering snow-capped Himalayan mountain peaks",
    "नदी": "sacred winding holy river with golden reflections and morning mist",
    "समुद्र": "dramatic vast ocean waves crashing on stone cliffs",
    "सूर्य": "golden blazing sun radiating warm volumetric light rays",
    "सूर्यास्त": "dramatic golden hour sunset with rich crimson and amber sky",
    "सूर्योदय": "glorious golden sunrise spreading dawn light across ancient terrain",
    "चांद": "luminous glowing silver full moon",
    "रात": "dark atmospheric night with starry cosmic sky and constellations",
    "दिन": "bright clear morning with pristine atmospheric clarity",
    "बारिश": "heavy cinematic monsoon rain pouring with water ripples",
    "तूफान": "dramatic thunderstorm with dark swirling clouds and lightning",
    "हवा": "gentle morning breeze swaying lush tree foliage",
    "पेड़": "ancient sacred trees with twisting roots",
    "गांव": "ancient small settlement with thatched cottages and curling woodsmoke",
    "महल": "grand ancient stone palace with carved pillars and saffron flags",
    "द्वापर युग": "ancient India during the legendary Dwapar Yuga era, Vedic civilization",
    "आर्यावर्त": "vast epic landscape of ancient Aryavarta with holy rivers and mountains",
    "महायुद्ध": "epic battlefield atmosphere of impending Kurukshetra war with gathering dark storm clouds"
}


def parse_prompt_and_voiceover(raw_input: str) -> Tuple[str, Optional[str]]:
    """
    Separates user input into Visual Scene Prompt and Voice-over Dialogue.
    """
    if not raw_input:
        return ("cinematic epic landscape, 8k resolution", None)

    vo_patterns = [
        r'(?:Voice-over|Voiceover|Voice over|वॉइस ओवर|डायलॉग|Dialogue)\s*:\s*["“\']?(.*?)(?:["”\']?$)',
        r'["“]([\u0900-\u097F\s\.\,…!?-]+)["”]'
    ]

    voiceover_text = None
    clean_visual = raw_input

    for pat in vo_patterns:
        match = re.search(pat, raw_input, re.IGNORECASE | re.DOTALL)
        if match:
            voiceover_text = match.group(1).strip().strip('"“\'”')
            clean_visual = re.sub(pat, '', clean_visual, flags=re.IGNORECASE | re.DOTALL).strip()
            break

    if not voiceover_text and re.search(r'[\u0900-\u097F]', raw_input):
        voiceover_text = raw_input.strip()

    return (clean_visual.strip(), voiceover_text)


def semantic_context_enricher(text: str) -> str:
    """
    Simulates real-time semantic knowledge search to enrich visual prompts with
    period-accurate, cinematographic, architectural, and atmospheric nuances.
    """
    p_lower = text.lower()
    enrichments = []

    # 1. Historical & Mythological India (Dwapar Yuga, Aryavarta, Vedic, Mahabharat)
    if any(k in p_lower for k in ["dwapar", "dwapara", "aryavarta", "ancient india", "vedic", "yuga", "mahabharat"]):
        enrichments.append(
            "Vedic Aryan architecture, thatched hermitage ashrams along holy Saraswati and Ganga riverbanks, "
            "untamed primordial wilderness, ancient Indian mythological realism style of Baahubali and Mahabharat, "
            "soft atmospheric morning haze, golden hour sunburst lighting, majestic birds gliding over vast river valleys"
        )
    # 2. Modern Cyberpunk / Futuristic City
    elif any(k in p_lower for k in ["cyberpunk", "neon", "future", "city", "shinjuku", "sci-fi", "robot"]):
        enrichments.append(
            "neo-tokyo cyberpunk aesthetic, towering holographic billboards, rain-slicked asphalt with chromatic reflections, "
            "volumetric steam rising from underground vents, anamorphic lens flares, cyan and magenta ambient illumination"
        )
    # 3. Nature / Wildlife & Landscape
    elif any(k in p_lower for k in ["lion", "tiger", "animal", "forest", "mountain", "snow", "ocean", "jungle"]):
        enrichments.append(
            "National Geographic award-winning wildlife cinematography, detailed fur and feather textures, "
            "subtle wind particle dynamics, photorealistic natural lighting, shallow depth of field, 85mm prime portrait lens"
        )
    # 4. Automotive / Action
    elif any(k in p_lower for k in ["car", "racing", "speed", "vehicle", "drift", "highway"]):
        enrichments.append(
            "dynamic motion blur on rotating wheels, cinematic low-angle tracking shot, reflective wet asphalt, "
            "intense headlight beams piercing darkness, high-speed camera shutter"
        )
    # 5. Cosmic / Deep Space
    elif any(k in p_lower for k in ["space", "astronaut", "planet", "galaxy", "alien", "orbit"]):
        enrichments.append(
            "Interstellar IMAX visual style, hyper-detailed spacesuit textures, planetary ring reflections on visor glass, "
            "deep void of space illuminated by swirling nebula gases and distant binary stars"
        )

    # General 8K cinematic realism boosters
    general_boosters = (
        "masterpiece, highly detailed, photorealistic 8K resolution, 35mm anamorphic cinema lens, "
        "volumetric god rays, professional Hollywood film color grading, sharp focus, octane render"
    )

    if enrichments:
        return f"{text}, {', '.join(enrichments)}, {general_boosters}"
    return f"{text}, {general_boosters}"


def translate_and_enhance_hindi_prompt(text: str) -> str:
    """
    Translates Hindi / Devanagari text to rich, cinematic English diffusion prompts
    and enriches with semantic knowledge.
    """
    if not text:
        return "majestic ancient Indian aerial landscape, golden sunrise, 8k resolution, cinematic lighting"

    enhanced = text

    # If text contains Devanagari, translate key terms
    if re.search(r'[\u0900-\u097F]', text):
        translated_segments = []
        for hindi_word, eng_desc in HINDI_DICTIONARY.items():
            if hindi_word in enhanced:
                translated_segments.append(eng_desc)
                enhanced = enhanced.replace(hindi_word, "")
        if translated_segments:
            enhanced = ", ".join(translated_segments)

    # Apply Semantic Context Knowledge Enrichment
    return semantic_context_enricher(enhanced)


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
