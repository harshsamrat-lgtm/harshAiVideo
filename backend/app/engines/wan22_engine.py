"""
Wan 2.2 I2V / T2V Video Engine — Powered by CogVideoX-5B Model Cascade.
Inherits full Neural AI Video Diffusion, HD Post-Processing, and Hindi Voice-over from LightX2VEngine.
"""
from typing import Dict, Any, Optional
from app.engines.lightx2v_engine import LightX2VEngine, _ACTIVE_MODEL_NAME
from app.core.config import settings


class Wan22Engine(LightX2VEngine):
    """
    Wan 2.2 Image-to-Video and Text-to-Video Engine.
    Uses same CogVideoX-5B → 2B → ModelScope cascade as LightX2V.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        self.name = "Wan2.2-CogVideoX-5B"
        self.checkpoint_dir = self.config.get("checkpoint_dir", settings.WAN22_CHECKPOINT_DIR)

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "active_model": _ACTIVE_MODEL_NAME or "not_loaded",
            "is_loaded": self.is_loaded,
            "checkpoint_dir": self.checkpoint_dir,
            "precision": "bfloat16",
            "accelerator": "wan_cogvideo_cuda",
            "target_hardware": "NVIDIA RTX 5090 (32GB VRAM)",
            "model_cascade": ["CogVideoX-5B", "CogVideoX-2B", "ModelScope-1.7B"],
            "supports_voiceover": True,
            "estimated_vram_peak_gb": 22.0
        }
