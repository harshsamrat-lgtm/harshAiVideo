"""
Tests for Character Bible and Location Bible persistence and querying.
"""
from fastapi.testclient import TestClient


def test_character_bible(client: TestClient):
    # 1. Create character
    char_payload = {
        "name": "Commander Vance",
        "description": "Veteran space fleet commander",
        "appearance": "Sharp silver-grey hair, stern jawline, cybernetic eye optic",
        "face_reference": "/projects/refs/vance_face.png",
        "body_description": "Broad-shouldered, 6ft 1in, tactical posture",
        "hair": "Short silver crop",
        "clothing": "Matte navy carbon-weave tactical uniform with gold insignia",
        "age_or_look": "Early 50s",
        "accessories": "Left eye cybernetic reticle",
        "default_prompt": "Commander Vance, commanding posture, cinematic illumination",
        "negative_prompt": "smiling, casual clothing, cartoon",
        "consistency_settings": {"face_weight": 0.85, "body_weight": 0.70}
    }
    create_res = client.post("/api/characters", json=char_payload)
    assert create_res.status_code == 201
    char_data = create_res.json()
    char_id = char_data["character_id"]
    assert char_data["name"] == "Commander Vance"

    # 2. Retrieve character
    get_res = client.get(f"/api/characters/{char_id}")
    assert get_res.status_code == 200
    assert get_res.json()["face_reference"] == "/projects/refs/vance_face.png"


def test_location_bible(client: TestClient):
    # 1. Create location
    loc_payload = {
        "name": "Orbital Command Bridge",
        "description": "Starship central control chamber",
        "architecture": "Brutalist titanium bulkheads, panoramic observation bay",
        "environment": "Holographic navigation consoles, stellar backdrop",
        "lighting": "Deep cyan ambient glow with sharp warm instrument highlights",
        "weather": "Space vacuum",
        "time_of_day": "Continuous stellar illumination",
        "camera_style": "Anamorphic 35mm wide lens"
    }
    create_res = client.post("/api/locations", json=loc_payload)
    assert create_res.status_code == 201
    loc_data = create_res.json()
    loc_id = loc_data["location_id"]
    assert loc_data["name"] == "Orbital Command Bridge"

    # 2. Retrieve location
    get_res = client.get(f"/api/locations/{loc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["architecture"] == loc_payload["architecture"]
