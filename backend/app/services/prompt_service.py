"""
Prompt Synthesizer & Dual-Track Visual/Voice-over Parser for Harsh AI Video Studio.
Optimized for CogVideoX-5B and SANA-Video 2B text-to-video diffusion models.
Separates compound prompts and multi-character dialogue scripts into:
  1. Clean visual scene directions with rock-solid facial and 5-finger hand anatomy.
  2. Multi-turn Neural Voice-over dialogue tracks with authentic Child and Adult vocal modulation.
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
    "आँगन": "a beautiful traditional Indian sunlit courtyard with terracotta floor and flowering plants",
    "आंगन": "a sunlit traditional Indian courtyard with terracotta tiles and green potted plants",
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


def parse_prompt_and_voiceover(raw_input: str) -> Tuple[str, Any]:
    """
    Separates user input into Visual Scene Prompt and Multi-character Voice-over Dialogues.
    Returns: (clean_visual_prompt, dialogue_data)
      dialogue_data can be:
        - List of dicts: [{'speaker': 'Kittu', 'text': '...', 'is_child': True, 'start': 0.0}, ...]
        - String: single dialogue / narration
        - None: no dialogue
    """
    if not raw_input:
        return ("a cinematic landscape at golden hour, photorealistic, 4K", None)

    # 1. Extract Hindi quotes from script / prompt
    # Matches “...”, "...", ‘...’, '...' containing Devanagari Hindi characters
    quote_matches = re.findall(r'[\"“\'‘]([\u0900-\u097F\s\.\,…!?\-।]+)[\"”\'’]', raw_input)

    is_child_scene = any(w in raw_input.lower() for w in ["बच्चे", "बच्चा", "बच्ची", "child", "children", "boy", "boys", "girl", "girls", "5-year-old", "kid", "kids", "किट्टू", "राघवेंद्र"])

    dialogues: List[Dict[str, Any]] = []

    if len(quote_matches) >= 2:
        # Multiple character dialogue exchange
        for i, q in enumerate(quote_matches[:2]):
            speaker_name = "Character 1" if i == 0 else "Character 2"
            if i == 0 and ("किट्टू" in raw_input or "kittu" in raw_input.lower()):
                speaker_name = "Kittu"
            elif i == 1 and ("राघवेंद्र" in raw_input or "raghavendra" in raw_input.lower()):
                speaker_name = "Raghavendra"

            dialogues.append({
                "speaker": speaker_name,
                "text": q.strip(),
                "is_child": is_child_scene,
                "turn": i
            })
    elif len(quote_matches) == 1:
        dialogues.append({
            "speaker": "Speaker",
            "text": quote_matches[0].strip(),
            "is_child": is_child_scene,
            "turn": 0
        })

    # If no quotes found, try explicit dialogue patterns
    if not dialogues:
        vo_patterns = [
            r'(?:Voice-over|Voiceover|Voice over|वॉइस ओवर|डायलॉग|Dialogue|संवाद)\s*:\s*[""\']?(.*?)(?:[""\']?\s*$)',
        ]
        for pat in vo_patterns:
            m = re.search(pat, raw_input, re.IGNORECASE | re.DOTALL)
            if m:
                dialogues.append({
                    "speaker": "Narrator",
                    "text": m.group(1).strip().strip('""\'"'),
                    "is_child": is_child_scene,
                    "turn": 0
                })
                break

    # If still no dialogue, and input is pure Hindi, extract only clean spoken content
    if not dialogues and re.search(r'[\u0900-\u097F]', raw_input):
        # Strip markdown, headers, script directions, timecodes — keep only natural speech
        clean_hindi = raw_input
        clean_hindi = re.sub(r'#.*?\n', ' ', clean_hindi)              # Remove markdown headers
        clean_hindi = re.sub(r'\*\*.*?\*\*', ' ', clean_hindi)         # Remove bold text
        clean_hindi = re.sub(r'\d+[–\-]\d+\s*सेकंड.*?:', ' ', clean_hindi)  # Remove timecodes
        clean_hindi = re.sub(r'दृश्य\s*:.*?\n', ' ', clean_hindi)      # Remove scene directions
        clean_hindi = re.sub(r'वीडियो\s*प्रॉम्प्ट\s*:.*', ' ', clean_hindi)  # Remove video prompt label
        clean_hindi = re.sub(r'[A-Za-z]{5,}', ' ', clean_hindi)       # Remove long English words (not spoken)
        clean_hindi = re.sub(r'[#\*\_\[\]\(\)]+', ' ', clean_hindi)    # Remove formatting chars
        clean_hindi = re.sub(r'\s+', ' ', clean_hindi).strip()
        
        # Only use as narration if there's meaningful Hindi content left
        if len(clean_hindi) > 10 and re.search(r'[\u0900-\u097F]{3,}', clean_hindi):
            dialogues.append({
                "speaker": "Narrator",
                "text": clean_hindi,
                "is_child": is_child_scene,
                "turn": 0
            })

    # 2. Clean visual prompt (remove script headers, timecodes, quotes)
    clean_visual = raw_input
    # Remove markdown headers like "# 8 सेकंड की वीडियो स्क्रिप्ट", "दृश्य:", "0-4 सेकंड"
    clean_visual = re.sub(r'#.*?\n', ' ', clean_visual)
    clean_visual = re.sub(r'\*\*.*?\*\*', ' ', clean_visual)
    clean_visual = re.sub(r'\d+–\d+\s*सेकंड.*?:', ' ', clean_visual)
    clean_visual = re.sub(r'दृश्य:.*?\n', ' ', clean_visual)
    clean_visual = re.sub(r'वीडियो प्रॉम्प्ट:.*?', ' ', clean_visual)

    # Remove quotes from visual prompt so diffusion gets pure scene description
    for q in quote_matches:
        clean_visual = clean_visual.replace(q, ' ')

    # If prompt contains English visual text, extract English description
    english_blocks = re.findall(r'([A-Za-z0-9\s,\.\-\'\"]{25,})', raw_input)
    if english_blocks:
        # Choose longest English visual description
        longest_english = max(english_blocks, key=len).strip()
        # Clean quotes inside English block
        longest_english = re.sub(r'[\"“\'‘][\u0900-\u097F\s\.\,…!?\-।]+[\"”\'’]', '', longest_english)
        clean_visual = longest_english

    return (clean_visual.strip(), dialogues if dialogues else None)


def translate_and_enhance_hindi_prompt(text: str) -> str:
    """
    Translates Hindi/Devanagari to rich, anatomically stable English visual descriptions.
    Enforces rock-solid facial proportions, 5-finger hands, and cinematic coherence.
    """
    if not text:
        return "A cinematic medium shot of two joyful 5-year-old Indian boys conversing in a sunny courtyard, sharp facial features, anatomically correct hands, photorealistic 8K quality."

    enhanced = text.strip()

    # Translate Hindi terms to descriptive English visual prompts
    if re.search(r'[\u0900-\u097F]', enhanced):
        translated_parts = []
        remaining = enhanced

        for hindi_word, eng_desc in sorted(HINDI_DICTIONARY.items(), key=lambda x: -len(x[0])):
            if hindi_word in remaining:
                translated_parts.append(eng_desc)
                remaining = remaining.replace(hindi_word, " ").strip()

        if translated_parts:
            enhanced = ". ".join(translated_parts)
        else:
            enhanced = f"A beautiful cinematic scene of {enhanced}, photorealistic, sharp focus"

    p_lower = enhanced.lower()
    has_people = any(c in p_lower for c in ["child", "children", "boy", "boys", "girl", "girls", "people", "person", "man", "woman", "warrior", "student", "friend", "kittu", "raghavendra"])

    # Enforce anatomical stability for faces, eyes, lips, and hands
    if has_people:
        anatomical_stabilizer = (
            "Cinematic medium portrait shot, perfectly proportioned symmetrical facial features, "
            "steady centered pupils with fixed eye gaze, natural well-defined lip contour with subtle natural mouth motion, "
            "anatomically correct hands with exactly five distinct steady fingers, resting natural posture without warping, "
            "85mm prime lens photography, soft warm natural daylight illumination, 8K resolution masterpiece"
        )
        enhanced = f"{enhanced}. {anatomical_stabilizer}"
    else:
        enhanced = f"{enhanced}. Cinematic masterpiece, photorealistic 8K resolution, sharp focus, 35mm film grain, masterpiece lighting."

    # Remove conflicting prompt tags like "no background music", "no subtitles" from visual prompt
    enhanced = re.sub(r'no\s+background\s+music', '', enhanced, flags=re.IGNORECASE)
    enhanced = re.sub(r'no\s+subtitles', '', enhanced, flags=re.IGNORECASE)
    enhanced = re.sub(r'no\s+text', '', enhanced, flags=re.IGNORECASE)

    return enhanced.strip()


class PromptService:
    @staticmethod
    def synthesize_shot_prompt(
        action_prompt: str,
        characters: Optional[List[CharacterResponse]] = None,
        location: Optional[LocationResponse] = None,
        camera_motion: str = "cinematic tracking",
        lens_style: Optional[str] = None
    ) -> Dict[str, Any]:
        visual_raw, dialogues = parse_prompt_and_voiceover(action_prompt)
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
            "voiceover_text": dialogues
        }


prompt_service = PromptService()
