"""
Video generation engines module export.
"""
from app.engines.base_engine import BaseVideoEngine
from app.engines.wan22_engine import Wan22Engine
from app.engines.lightx2v_engine import LightX2VEngine
from app.core.config import settings


def get_active_engine() -> BaseVideoEngine:
    """Factory to retrieve configured engine instance."""
    if settings.ENGINE.lower() == "lightx2v":
        return LightX2VEngine()
    return Wan22Engine()


__all__ = ["BaseVideoEngine", "Wan22Engine", "LightX2VEngine", "get_active_engine"]
