"""
Prompt Synthesizer & Dual-Track Visual/Voice-over Parser for Harsh AI Video Studio.
Optimized for CogVideoX-5B and SANA-Video 2B text-to-video diffusion models.
Separates compound prompts into clean visual scene directions and spoken Voice-over dialogue.
Enforces photorealistic facial clarity, perfect 5-finger hand anatomy, and cinematic 8K coherence.
"""
import re
from typing import List, Optional, Dict, Any, Tuple
from app.models.schemas import CharacterResponse, LocationResponse


# Comprehensive Hindi to English visual descriptor lexicon
HINDI_DICTIONARY = {
    # People, Children, Family & Characters
    "बच्चे": "two cheerful cute Indian children with perfectly proportioned friendly faces, bright clear eyes, and anatomically perfect hands",
    "बच्चा": "a cute cheerful Indian child with a clear detailed expressive face and bright eyes",
    "बच्ची": "a cute cheerful Indian girl with a beautiful detailed smiling face and bright eyes",
    "बालक": "a young Indian boy with clear detailed facial features and bright expressive eyes",
    "लड़का": "a handsome young Indian boy with sharp detailed facial features",
    "लड़के": "cheerful young Indian boys with sharp detailed facial features and friendly smiles",
    "लड़की": "a beautiful young Indian girl with detailed facial features and graceful expression",
    "लड़कियां": "cheerful young Indian girls with detailed facial features and friendly expressions",
    "दोस्त": "close friends standing together with happy natural facial expressions",
    "मित्र": "dear friends with warm smiling expressions",
    "परिवार": "a loving Indian family together with detailed warm facial expressions",
    "माता": "a kind Indian mother with graceful gentle facial features",
    "पिता": "a caring Indian father with clear detailed facial features",
    "छात्र": "Indian students in neat school uniforms with bright curious expressions",
    "विद्यार्थी": "eager school students with friendly expressions",

    # Actions, Conversations & Expressions
    "बातें कर रहे": "conversing naturally with gentle friendly expressions and subtle natural mouth movement",
    "बातें": "talking pleasantly with cheerful subtle natural facial expressions",
    "बातचीत": "having a cheerful natural conversation with subtle friendly gestures",
    "बोल रहे": "speaking naturally with realistic facial expressions and steady gaze",
    "हंस रहे": "laughing joyfully with bright genuine smiles and clear teeth",
    "मुस्कुरा रहे": "smiling gently with warm expressive eyes",
    "खेल रहे": "playing joyfully with natural energetic posture and perfect body proportions",
    "दौड़ रहे": "running happily across the open ground with natural athletic motion",
    "चल रहे": "walking gracefully along the path with steady natural gait",
    "खड़े हैं": "standing gracefully with steady natural posture and sharp facial focus",
    "बैठे हैं": "sitting comfortably with relaxed natural posture",

    # Mythological & Epic Warriors
    "योद्धा": "ancient Indian warriors with sharp chiseled facial features, royal golden crowns, and gleaming armor",
    "रथ": "an ornate golden chariot pulled by armored royal white horses",
    "द्वापर युग": "ancient India during the legendary Dwapar Yuga era with grand architecture",
    "आर्यावर्त": "the vast epic landscape of ancient Aryavarta with sacred rivers and mountains",
    "महायुद्ध": "an epic battlefield under a dramatic crimson sunset sky",
    "कुरुक्षेत्र": "the sacred plains of Kurukshetra at sunset with banners waving in the wind",
    "राजा": "a royal Indian king with an ornate golden crown and majestic countenance",
    "सेना": "a disciplined ancient army in gleaming armor standing in formation",

    # Animals
    "शेर": "a majestic royal lion with detailed golden mane walking powerfully through grasslands",
    "बाघ": "a powerful royal bengal tiger with sharp stripes stalking through lush jungle",
    "हाथी": "a majestic decorated elephant with golden ceremonial ornaments",
    "घोड़ा": "a magnificent stallion galloping with flowing mane",
    "चीता": "a sleek cheetah sprinting across open savanna",
    "पक्षी": "colorful birds soaring gracefully across the clear sky",
    "मोर": "a vibrant Indian peacock displaying iridescent turquoise feathers",
    "गाय": "a sacred holy cow with gentle calm eyes in a green pasture",

    # Locations & Environment
    "स्कूल": "a picturesque sunny school courtyard with green trees and sunlight",
    "कक्षा": "a bright sunny classroom with wooden benches",
    "मैदान": "a lush green open playground with soft warm sunlight",
    "बगीचा": "a beautiful blooming garden with colorful flowers and green pathways",
    "पार्क": "a scenic public park with lush green grass and shady trees",
    "जंगल": "a dense scenic forest with towering trees and dappled sunbeams",
    "पहाड़": "towering snow-capped Himalayan mountain peaks under clear blue sky",
    "नदी": "a serene winding river with crystal clear flowing water and golden reflections",
    "समुद्र": "dramatic ocean waves crashing gently against the golden sandy shore",
    "गांव": "a peaceful traditional Indian village with rustic cottages and flowering trees",
    "महल": "a magnificent ancient stone palace with carved arches and golden domes",
    "शहर": "a vibrant modern city street with clean architecture and warm daylight",
    "घर": "a cozy traditional home with warm inviting atmosphere",

    # Nature & Sky
    "सूर्य": "the radiant golden sun casting warm cinematic light",
    "सूर्यास्त": "a breathtaking golden hour sunset with rich amber, crimson, and purple skies",
    "सूर्योदय": "a glorious golden dawn sunrise spreading warm morning light",
    "चांद": "a luminous full moon glowing serenely in a deep dark starry night sky",
    "रात": "a peaceful starry night with glowing moonlight and soft shadows",
    "दिन": "a bright crystal-clear sunny morning with pristine illumination",
    "बारिश": "gentle monsoon raindrops falling through golden sunbeams with ripples on water",
    "तूफान": "dramatic dark swirling storm clouds with cinematic lighting",
    "बर्फ": "pure white snow glistening under a soft winter sky",
    "हवा": "a gentle breeze swaying green leaves and flowing garments gracefully",
    "पेड़": "lush ancient banyan trees with hanging roots and dense green leaves",
}


def parse_prompt_and_voiceover(raw_input: str) -> Tuple[str, Optional[str]]:
    """
    Separates user input into Visual Scene Prompt and Voice-over Dialogue.
    Captures spoken dialogues cleanly for Neural Speech Synthesis.
    """
    if not raw_input:
        return ("a cinematic landscape at golden hour, photorealistic, 4K", None)

    vo_patterns = [
        r'(?:Voice-over|Voiceover|Voice over|वॉइस ओवर|डायलॉग|Dialogue|संवाद)\s*:\s*[""\']?(.*?)(?:[""\']?\s*$)',
        r'[""]([\u0900-\u097F\s\.\,…!?\-।]+)[""]',
        r'\'([\u0900-\u097F\s\.\,…!?\-।]+)\''
    ]

    voiceover_text = None
    clean_visual = raw_input

    for pat in vo_patterns:
        match = re.search(pat, raw_input, re.IGNORECASE | re.DOTALL)
        if match:
            voiceover_text = match.group(1).strip().strip('""\'"')
            clean_visual = re.sub(pat, '', clean_visual, flags=re.IGNORECASE | re.DOTALL).strip()
            break

    # If no explicit voiceover tag is used, but the input is in Hindi, use the full Hindi sentence as voiceover
    if not voiceover_text and re.search(r'[\u0900-\u097F]', raw_input):
        voiceover_text = raw_input.strip()

    return (clean_visual.strip(), voiceover_text)


def translate_and_enhance_hindi_prompt(text: str) -> str:
    """
    Translates Hindi/Devanagari to rich, anatomically stable English visual descriptions.
    Inforces rock-solid facial proportions, 5-finger hands, and cinematic coherence.
    """
    if not text:
        return "A cinematic medium shot of two joyful Indian children conversing in a sunny garden, sharp facial features, anatomically correct hands, photorealistic 8K quality."

    enhanced = text.strip()

    # Translate Hindi terms to descriptive English visual prompts
    if re.search(r'[\u0900-\u097F]', enhanced):
        translated_parts = []
        remaining = enhanced

        # Multi-word matching first
        for hindi_word, eng_desc in sorted(HINDI_DICTIONARY.items(), key=lambda x: -len(x[0])):
            if hindi_word in remaining:
                translated_parts.append(eng_desc)
                remaining = remaining.replace(hindi_word, " ").strip()

        if translated_parts:
            enhanced = ". ".join(translated_parts)
        else:
            # Fallback if Hindi words are not in dictionary
            enhanced = f"A beautiful cinematic scene of {enhanced}, photorealistic, sharp focus"

    p_lower = enhanced.lower()
    has_people = any(c in p_lower for c in ["child", "children", "boy", "girl", "people", "person", "man", "woman", "warrior", "student", "friend"])

    # Enforce anatomical stability for faces and hands
    if has_people:
        anatomical_stabilizer = (
            "Cinematic medium portrait shot, perfectly proportioned symmetrical facial features, "
            "clear expressive eyes, natural subtle smile, steady facial structure, "
            "anatomically correct hands with exactly five distinct fingers, gentle natural posture, "
            "85mm prime lens photography, soft natural daylight illumination, 8K masterpiece"
        )
        enhanced = f"{enhanced}. {anatomical_stabilizer}"
    else:
        enhanced = f"{enhanced}. Cinematic masterpiece, photorealistic 8K resolution, sharp focus, 35mm film grain, masterpiece lighting."

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
            "distorted face", "deformed mouth", "warped eyes", "asymmetrical face", "mutated facial features",
            "poorly drawn hands", "deformed hands", "extra fingers", "missing fingers", "fused fingers", "too many fingers",
            "deformed limbs", "disconnected limbs", "floating limbs", "bad anatomy", "bad proportions",
            "blurry face", "blurry eyes", "ghosting", "jitter", "flicker", "low quality", "morphing artifacts",
            "text", "watermark", "ugly", "duplicate", "jpeg artifacts"
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
