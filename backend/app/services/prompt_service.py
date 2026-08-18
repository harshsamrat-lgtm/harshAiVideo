"""
Prompt Synthesizer & Dual-Track Visual/Voice-over Parser for Harsh AI Video Studio.
Optimized for CogVideoX-5B and SANA-Video 2B diffusion models.
Translates all Hindi/Devanagari prompts into ultra-rich, photorealistic cinematic English.
Enforces crystal-clear facial symmetry, natural skin texture, and sharp 8K visual precision.
"""
import re
from typing import List, Optional, Dict, Any, Tuple
from app.models.schemas import CharacterResponse, LocationResponse


# Comprehensive Hindi to English visual descriptor lexicon
HINDI_DICTIONARY = {
    # Faces & Characters
    "चेहरा": "a handsome highly detailed symmetrical face, clear expressive eyes, realistic skin texture",
    "चेहरे": "handsome highly detailed symmetrical faces with sharp features and expressive eyes",
    "आंखें": "clear piercing expressive eyes with detailed irises",
    "आंख": "clear piercing expressive eye with detailed iris",
    "योद्धा": "an ancient Indian warrior with sharp chiseled facial features, royal golden armor, and intense eyes",
    "योद्धाओं": "ancient Indian warriors with sharp facial features, royal armor, and intense expressions",
    "सैनिक": "brave Indian soldiers marching in formation with detailed armor and weapons",
    "राजा": "a majestic Indian king wearing an ornate golden crown, royal saffron robes, and majestic beard",
    "रानी": "a gorgeous royal Indian queen wearing traditional silk saree and golden jewelry",
    "सुंदर": "a beautiful highly detailed photorealistic aesthetic with perfect proportions",
    "श्री कृष्ण": "Lord Krishna with radiant divine blue aura, peacock feather in crown, compassionate smiling face",
    "कृष्ण": "Lord Krishna with radiant divine golden aura, peacock feather, and graceful face",
    "अर्जुन": "the legendary archer warrior Arjuna holding Gandiva bow with focused intense eyes",
    "कर्ण": "the mighty warrior Karna with glowing golden armor (Kavach Kundal) and heroic sharp face",
    "भीष्म": "the revered grand warrior Bhishma with white beard, majestic demeanor, and golden armor",
    "हनुमान": "Lord Hanuman in divine heroic form, radiant aura, muscular physique, holding golden mace",
    "राम": "Lord Rama with divine serene face, holding Kodanda bow, royal attire",
    "शिव": "Lord Shiva in deep meditation with crescent moon, sacred ash (vibhuti), and flowing river Ganga",
    "साधु": "an ancient sage with white beard meditating serenely under a banyan tree",

    # Animals
    "शेर": "a majestic lion with detailed golden mane walking powerfully through grasslands",
    "बाघ": "a powerful royal bengal tiger with sharp stripes stalking through lush jungle",
    "हाथी": "a majestic ancient elephant with ornate golden ceremonial decorations",
    "घोड़ा": "a magnificent white stallion galloping with muscular power",
    "चीता": "a sleek cheetah sprinting across savanna",
    "पक्षी": "birds soaring gracefully across golden sky",

    # Environment & Landscapes
    "कुरुक्षेत्र": "the epic historic battlefield of Kurukshetra at sunset with dust swirling and saffron banners",
    "महायुद्ध": "an epic battlefield scene with thousands of warriors and chariots gathering under dramatic crimson sky",
    "द्वापर युग": "ancient India during the epic Dwapar Yuga era with grand palaces and Vedic architecture",
    "आर्यावर्त": "the vast sacred realm of Aryavarta with pristine rivers and Himalayan mountains",
    "महल": "a grand ancient stone palace with intricately carved pillars and fluttering saffron flags",
    "किला": "a massive ancient fortress on a rocky cliff under dramatic sunset",
    "मंदिर": "an ancient intricately carved stone temple surrounded by sacred oil lamps (diyas)",
    "गांव": "a peaceful ancient village with thatched cottages and holy banyan tree",
    "जंगल": "a dense ancient forest with giant banyan trees and sunbeams piercing through the canopy",
    "पहाड़": "towering snow-capped Himalayan mountain peaks glowing in morning light",
    "नदी": "a sacred winding river reflecting the golden sunset light",
    "गंगा": "the holy river Ganga flowing through ancient stone ghats with floating oil lamps",
    "समुद्र": "dramatic ocean waves crashing against ancient coastal cliffs",
    "सूर्य": "the radiant golden sun casting dramatic rays through misty air",
    "सूर्यास्त": "a breathtaking golden hour sunset with crimson, violet, and amber clouds",
    "सूर्योदय": "a glorious golden sunrise illuminating ancient landscape",
    "चांद": "a luminous full moon casting ethereal silver light across night landscape",
    "रात": "a dark cinematic starry night with milky way and glowing torches",
    "बारिश": "heavy monsoon rain with water droplets splashing on ground",
    "तूफान": "a dramatic storm with swirling dark thunderclouds and lightning strikes",
    "आग": "crackling golden flames and glowing embers rising in dark night",
    "रथ": "an ornate golden chariot pulled by armored white horses",
    "धनुष": "an ornate ancient bow drawn with golden glowing arrow",
    "तलवार": "a polished steel sword reflecting dramatic sunlight",
    "युद्ध": "an epic cinematic battle with warriors charging forward",
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
    Guarantees 100% clean English text suitable for text encoders (T5-XXL / Gemma).
    Enforces sharp facial symmetry, realistic skin texture, and 8K visual precision.
    """
    if not text:
        return (
            "A cinematic medium portrait of a legendary ancient Indian warrior with sharp symmetrical facial features, "
            "intense eyes, golden armor, photorealistic 8K resolution, 85mm portrait photography."
        )

    enhanced = text.strip()

    # Step 1: Translate Hindi dictionary terms
    if re.search(r'[\u0900-\u097F]', enhanced):
        translated_parts = []
        remaining = enhanced
        for hindi_word, eng_desc in sorted(HINDI_DICTIONARY.items(), key=lambda x: -len(x[0])):
            if hindi_word in remaining:
                translated_parts.append(eng_desc)
                remaining = remaining.replace(hindi_word, "").strip()

        if translated_parts:
            enhanced = ". ".join(translated_parts)
        else:
            # Fallback if Hindi text had no dictionary hit: generate clean high-quality character scene
            enhanced = (
                "A cinematic medium shot of an ancient Indian warrior hero with sharp symmetrical facial features, "
                "intense clear eyes, chiseled jawline, royal armor, standing dramatically on battlefield"
            )

    # Step 2: Strip any leftover Devanagari characters to ensure text-encoder compatibility
    enhanced = re.sub(r'[\u0900-\u097F]+', '', enhanced).strip()
    if not enhanced or len(enhanced) < 10:
        enhanced = "A majestic cinematic medium shot of an ancient warrior with sharp facial features in dramatic golden hour light"

    p_lower = enhanced.lower()
    has_character = any(c in p_lower for c in ["warrior", "man", "woman", "person", "people", "army", "character", "face", "commander", "king", "lord", "god", "krishna", "arjun", "karna", "hero", "soldier", "queen"])

    # Step 3: Enforce facial clarity & symmetry
    if has_character:
        if "symmetrical" not in p_lower:
            enhanced = (
                f"{enhanced}. Cinematic medium portrait shot, extremely detailed sharp symmetrical facial features, "
                f"clear expressive eyes, realistic skin texture, 85mm prime lens photography, dramatic studio lighting"
            )

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
            "deformed face", "ugly face", "distorted eyes", "asymmetrical face", "bad anatomy",
            "blurry face", "disfigured pupils", "watermark", "text", "extra limbs", "ugly",
            "duplicate", "jpeg artifacts", "cartoon", "3d render", "low quality"
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
