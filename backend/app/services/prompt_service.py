"""
Prompt Synthesizer & Section-Aware Dual-Track Visual/Voice-over Parser for Harsh AI Video Studio.
Optimized for CogVideoX-5B, SANA-Video 2B, and high-end video diffusion models.
Handles:
  1. Multi-section Campaign Scripts (Master Prompt, Visual Breakdown, Voiceover, Music, Typography)
  2. Multi-turn Character Scripts (e.g. Kittu & Raghavendra child exchanges)
  3. Single-turn Narrations & Dialogue Prompts
  4. Hindi-to-English Cinematic Translation & Anatomical Quality Boosters
"""
import re
from typing import List, Optional, Dict, Any, Tuple
from app.models.schemas import CharacterResponse, LocationResponse


# Comprehensive Hindi to English visual descriptor lexicon
HINDI_DICTIONARY = {
    # Modern Infrastructure, Development & State Campaigns
    "उत्तर प्रदेश": "the progressive state of Uttar Pradesh India with world-class modern expressways, clean infrastructure, and lush green fields",
    "योगी": "progressive leadership, state development, and modern infrastructure governance in India",
    "सरकार": "modern Indian governance, welfare development, and public infrastructure",
    "योजनाएं": "progressive public welfare schemes, social prosperity, and modern development",
    "विकास": "futuristic modern infrastructure, wide clean expressways, state-of-the-art buildings, and progressive development",
    "एक्सप्रेसवे": "a magnificent multi-lane modern expressway with smooth traffic and sleek vehicles driving under golden morning sunlight",
    "गाड़ियाँ": "sleek modern cars and vehicles driving smoothly along the paved highway",
    "सड़कें": "wide clean modern paved roads with smart streetlights and green medians",
    "स्वच्छ शहर": "a clean modern Indian city with pristine streets, solar lights, and orderly architecture",
    "स्वच्छ": "clean pristine surroundings with orderly architecture and bright natural daylight",
    "अस्पताल": "a state-of-the-art clean modern hospital medical center with caring doctors",
    "रोजगार": "young Indian professionals working enthusiastically in modern clean skill training centers",
    "किसान": "a proud smiling Indian farmer standing in a flourishing green agricultural field under warm golden sunlight",
    "खेत": "lush flourishing green agricultural farmland and golden crops swaying under the blue sky",

    # People, Children, Family & Characters
    "बच्चे": "two cheerful cute Indian children with perfectly proportioned friendly faces, bright clear eyes, and anatomically perfect hands",
    "बच्चा": "a cute cheerful Indian child with a clear detailed expressive face and bright eyes",
    "बच्ची": "a cute cheerful Indian girl with a beautiful detailed smiling face and bright eyes",
    "बालक": "a young Indian boy with clear detailed facial features and bright expressive eyes",
    "लड़का": "a handsome young Indian boy with sharp detailed facial features",
    "लड़के": "cheerful young Indian boys with sharp detailed facial features and friendly smiles",
    "लड़की": "a beautiful young Indian girl with detailed facial features and graceful expression",
    "लड़कियां": "cheerful young Indian girls with detailed facial features and friendly expressions",
    "महिलाएं": "confident smiling Indian women in elegant traditional attire participating in community work",
    "महिला": "a graceful smiling Indian woman with warm confident expression",
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

    # Science, Space & Technology
    "वैज्ञानिक": "a brilliant visionary scientist in a sleek high-tech research laboratory with glowing futuristic holographic data displays and advanced scientific instruments",
    "विज्ञान": "futuristic scientific innovation with glowing data interfaces and advanced technology",
    "प्रयोगशाला": "a state-of-the-art futuristic research laboratory with glowing blue ambient lighting and precision instruments",
    "अंतरिक्ष": "the deep cosmic expanse with glowing spiral galaxies, colorful nebulas, and brilliant distant stars",
    "रॉकेट": "a powerful aerospace rocket blasting off towards space with glowing fiery thrust and dramatic smoke plumes",
    "कंप्यूटर": "a sleek modern computer workstation with high-resolution monitors and clean ambient lighting",
    "तकनीक": "cutting-edge technology with glowing digital circuits and futuristic interface",
    "रोबोट": "a sleek advanced humanoid AI robot with friendly expressive glowing optical sensors",

    # Professions, Daily Life & Emotions
    "डॉक्टर": "a compassionate skilled doctor in a clean white coat examining in a modern clinic with warm friendly smile",
    "इंजीनियर": "a smart dedicated engineer in safety helmet reviewing blueprints at a modern infrastructure project",
    "शिक्षक": "an inspiring kind teacher explaining concepts with enthusiasm to attentive students",
    "गुरु": "a wise revered spiritual teacher with serene gentle expression in a traditional serene setting",
    "सैनिक": "a brave disciplined soldier standing vigilant in uniform against a rugged mountain landscape",
    "खुश": "radiant genuine joy with bright smiling face, clear expressive eyes, and lively cheerful demeanor",
    "उत्साह": "dynamic inspiring enthusiasm with energetic posture and bright confident expression",
    "प्रेम": "warm gentle affectionate expression with tender soft cinematic lighting",
    "शांति": "profound peaceful serenity with calm relaxing posture and gentle ambient sunlight",

    # Indian Culture, Festivals & Heritage
    "दीपावली": "a magical Diwali festival evening with hundreds of glowing terracotta oil lamps (diyas) casting warm golden light",
    "दिवाली": "a radiant Diwali celebration with glowing diyas, marigold garlands, and festive golden illuminations",
    "होली": "a vibrant joyful Holi celebration with colorful natural organic gulal powder bursting in the bright sunny air",
    "मंदिर": "an ancient magnificent stone temple with intricately carved spires, glowing brass oil lamps, and divine serene atmosphere",
    "पूजा": "a sacred traditional prayer ceremony with fragrant incense smoke, fresh marigold flowers, and glowing Aarti lamps",
    "स्वतंत्रता": "proud Indian national celebration with the tricolor flag waving majestically against a crystal clear blue sky",

    # Nature & Sky
    "सुबह": "a glorious golden morning sunrise casting warm rays and pristine daylight",
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
    Separates user input into Visual Scene Prompt and Voice-over Dialogues.
    Understands:
      - Explicit `### Voiceover` sections in multi-section campaign scripts
      - Multi-character turn dialogues (Kittu & Raghavendra)
      - Master visual prompts & single scene directions
    """
    if not raw_input:
        return ("a cinematic landscape at golden hour, photorealistic, 4K", None)

    dialogues: List[Dict[str, Any]] = []

    # ── 1. CHECK FOR EXPLICIT VOICEOVER / DIALOGUE SECTION ──
    # Regex looks for `### Voiceover` or `Voice-over:` or `वॉइस ओवर:` block
    vo_section_match = re.search(
        r'(?:###?\s*(?:Voiceover|Voice-over|Voice over|वॉइस ओवर|डायलॉग|Dialogue|संवाद)|(?:\*\*|\b)(?:Voiceover|Voice-over|वॉइस ओवर)(?:\*\*|:))\s*([\s\S]*?)(?:###|\n\n\*\*Visual|\n\n\*\*Important|\Z)',
        raw_input,
        re.IGNORECASE
    )

    if vo_section_match:
        vo_block = vo_section_match.group(1).strip()
        # Find quoted text inside voiceover block
        vo_quotes = re.findall(r'[\"“\'‘]([\u0900-\u097F\s\.\,…!?\-।–]+)[\"”\'’]', vo_block)
        
        is_male = any(w in vo_block.lower() for w in ["पुरुष", "male", "man", "deep", "energetic male"])
        is_female = any(w in vo_block.lower() for w in ["महिला", "female", "woman"])
        is_child = any(w in vo_block.lower() for w in ["बच्चे", "बच्चा", "child", "kid"])

        if vo_quotes:
            vo_text = " ".join(vo_quotes).strip()
        else:
            # Clean markdown formatting inside voiceover block
            vo_text = re.sub(r'[\#\*\_\(\)]+', ' ', vo_block)
            vo_text = re.sub(r'(?:एक\s+)?(?:प्रभावशाली|स्पष्ट|ऊर्जावान|पुरुष|महिला|हिंदी|आवाज).*?:?', ' ', vo_text)
            vo_text = re.sub(r'[A-Za-z]+', ' ', vo_text)
            vo_text = re.sub(r'\s+', ' ', vo_text).strip()

        if vo_text and len(vo_text) > 3:
            dialogues.append({
                "speaker": "Male Voice" if is_male else ("Female Voice" if is_female else "Narrator"),
                "text": vo_text,
                "is_child": is_child,
                "turn": 0
            })

    # ── 2. IF NO EXPLICIT VOICEOVER SECTION, CHECK SCRIPT TURNS (E.G. KITTU & RAGHAVENDRA) ──
    if not dialogues:
        turn_matches = re.findall(r'(?:(\d+[–\-]\d+\s*सेकंड|किट्टू|राघवेंद्र|Speaker\s*\d+)[^\n\"]*?)[\"“\'‘]([\u0900-\u097F\s\.\,…!?\-।–]+)[\"”\'’]', raw_input)
        if turn_matches:
            for i, (spk, q) in enumerate(turn_matches[:2]):
                is_child_speaker = any(w in spk.lower() or w in raw_input.lower() for w in ["किट्टू", "राघवेंद्र", "बच्चे", "बच्चा", "child", "kittu"])
                dialogues.append({
                    "speaker": spk.strip(),
                    "text": q.strip(),
                    "is_child": is_child_speaker,
                    "turn": i
                })

    # ── 3. FALLBACK: GENERAL QUOTES IF MARKED AS DIALOGUE ──
    if not dialogues:
        all_quotes = re.findall(r'[\"“\'‘]([\u0900-\u097F\s\.\,…!?\-।–]+)[\"”\'’]', raw_input)
        # Filter out short slogans like "विकास की नई पहचान" if they look like screen titles
        meaningful_quotes = [q for q in all_quotes if len(q.strip()) > 10]
        if meaningful_quotes:
            is_child_scene = any(w in raw_input.lower() for w in ["बच्चे", "बच्चा", "child", "kid", "किट्टू", "राघवेंद्र"])
            for i, q in enumerate(meaningful_quotes[:2]):
                dialogues.append({
                    "speaker": f"Character {i+1}",
                    "text": q.strip(),
                    "is_child": is_child_scene,
                    "turn": i
                })

    # ── 4. EXTRACT & DISTILL THE MASTER VISUAL PROMPT FOR DIFFUSION ──
    # For complex multi-section prompts, extract the core visual scene
    visual_text = ""

    # Check for `MASTER VIDEO PROMPT:` or `वीडियो प्रॉम्प्ट:`
    master_prompt_match = re.search(
        r'(?:MASTER VIDEO PROMPT:|वीडियो प्रॉम्प्ट:|Visual Prompt:|Scene Description:)\s*([\s\S]*?)(?:###|\n\n\*\*|\n\n###|\Z)',
        raw_input,
        re.IGNORECASE
    )
    if master_prompt_match:
        visual_text = master_prompt_match.group(1).strip()

    # Check for opening scene `0-2 second` breakdown if available
    scene_match = re.search(
        r'(?:0[–\-]\d+\s*सेकंड[\s\S]*?:\s*)([^\n\r]+(?:\r?\n[^\n\r#\*\#]+)?)',
        raw_input
    )
    if scene_match:
        scene_desc = scene_match.group(1).strip()
        if visual_text:
            visual_text = f"{visual_text}. {scene_desc}"
        else:
            visual_text = scene_desc

    # If neither found, use raw input cleaned
    if not visual_text:
        visual_text = raw_input

    # Clean meta instructions, typography, slogans, headers from visual text
    visual_text = re.sub(r'#.*?\n', ' ', visual_text)
    visual_text = re.sub(r'###.*?\n', ' ', visual_text)
    visual_text = visual_text.replace('**', ' ').replace('*', ' ')
    visual_text = re.sub(r'MASTER VIDEO PROMPT:\s*', ' ', visual_text, flags=re.IGNORECASE)
    visual_text = re.sub(r'वीडियो प्रॉम्प्ट:\s*', ' ', visual_text)
    visual_text = re.sub(r'\d+[–\-]\d+\s*सेकंड.*?:', ' ', visual_text)
    visual_text = re.sub(r'दृश्य\s*:.*?\n', ' ', visual_text)
    visual_text = re.sub(r'Voiceover[\s\S]*?(?:###|\Z)', ' ', visual_text, flags=re.IGNORECASE)
    visual_text = re.sub(r'Background Music[\s\S]*?(?:###|\Z)', ' ', visual_text, flags=re.IGNORECASE)
    visual_text = re.sub(r'Important:.*', ' ', visual_text, flags=re.IGNORECASE)
    visual_text = re.sub(r'typography.*', ' ', visual_text, flags=re.IGNORECASE)
    visual_text = re.sub(r'स्लोगन.*', ' ', visual_text)
    visual_text = re.sub(r'[\"“\'‘][\u0900-\u097F\s\.\,…!?\-।–]+[\"”\'’]', ' ', visual_text)
    visual_text = re.sub(r'\s+', ' ', visual_text).strip()

    return (visual_text, dialogues if dialogues else None)


def translate_and_enhance_hindi_prompt(text: str) -> str:
    """
    Translates Hindi/Devanagari to rich, anatomically stable English visual descriptions.
    Enforces rock-solid facial proportions, 5-finger hands, and cinematic coherence.
    """
    if not text:
        return "A cinematic sweeping aerial shot of a modern multi-lane expressway in Uttar Pradesh India during golden sunrise, sleek vehicles driving smoothly, photorealistic 8K quality."

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

    # Enforce professional camera styling
    p_lower = enhanced.lower()
    has_people = any(c in p_lower for c in ["child", "children", "boy", "boys", "girl", "girls", "people", "person", "man", "woman", "warrior", "student", "friend", "farmer", "kittu", "raghavendra"])
    has_aerial = any(c in p_lower for c in ["aerial", "expressway", "highway", "city", "drone", "wide", "landscape", "infrastructure", "uttar pradesh"])

    if has_aerial:
        cinematic_style = (
            "Sweeping cinematic aerial drone shot, wide-angle 24mm master lens, "
            "radiant golden hour morning illumination, vivid vibrant natural color grading, "
            "ultra-sharp 8K resolution, dynamic smooth motion, photorealistic national campaign standard"
        )
        enhanced = f"{enhanced}. {cinematic_style}"
    elif has_people:
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
    enhanced = re.sub(r'create an \d+-second.*?(video|promotional)', '', enhanced, flags=re.IGNORECASE)

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
            "wobbling pupils", "shifting eyes", "moving eyeballs", "crossed eyes", "misaligned pupils",
            "teeth morphing", "split lips", "deformed fingers", "morphing fingers", "extra fingers", "missing fingers",
            "fused fingers", "six fingers", "poorly drawn hands", "deformed hands", "shifting hands",
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
