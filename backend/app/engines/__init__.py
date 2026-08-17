"""
Video generation engines module - singleton engine registry with dynamic engine resolution.
Supports LightX2V (NVFP4) and Wan 2.2 (14B / T2V / I2V).
"""
from typing import Optional
from app.engines.base_engine import BaseVideoEngine
from app.engines.wan22_engine import Wan22Engine
from app.engines.lightx2v_engine import LightX2VEngine
from app.core.config import settings

_engine_registry = {}


def get_active_engine(engine_name: Optional[str] = None) -> BaseVideoEngine:
    """
    Factory returning engine instance based on payload selection or default config.
    """
    selected = (engine_name or settings.ENGINE or "lightx2v").lower()

    if selected not in _engine_registry:
        if "wan" in selected:
            _engine_registry[selected] = Wan22Engine()
        else:
            _engine_registry[selected] = LightX2VEngine()

    return _engine_registry[selected]


__all__ = ["BaseVideoEngine", "Wan22Engine", "LightX2VEngine", "get_active_engine"]
