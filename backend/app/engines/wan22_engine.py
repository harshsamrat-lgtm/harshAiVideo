"""
Standard Wan 2.2 I2V A14B Video Generation Engine Implementation.
"""
from typing import Dict, Any, Optional
from app.engines.lightx2v_engine import LightX2VEngine
from app.core.config import settings


class Wan22Engine(LightX2VEngine):
    """
    Standard Wan 2.2 Image-to-Video Engine.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        self.name = "Wan2.2-I2V-14B"
        self.checkpoint_dir = self.config.get("checkpoint_dir", settings.WAN22_CHECKPOINT_DIR)

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "is_loaded": self.is_loaded,
            "checkpoint_dir": self.checkpoint_dir,
            "precision": "bfloat16",
            "accelerator": "standard_cuda",
            "estimated_vram_peak_gb": 22.0
        }
