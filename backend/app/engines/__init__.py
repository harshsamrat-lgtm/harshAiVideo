"""
Video generation engines module - singleton factory with lazy loading.
"""
from app.engines.base_engine import BaseVideoEngine
from app.engines.wan22_engine import Wan22Engine
from app.engines.lightx2v_engine import LightX2VEngine
from app.core.config import settings

# Singleton engine instance - loaded once on first use
_engine_instance: BaseVideoEngine = None


def get_active_engine() -> BaseVideoEngine:
    """Factory returning the singleton engine instance."""
    global _engine_instance
    if _engine_instance is None:
        if settings.ENGINE.lower() == "lightx2v":
            _engine_instance = LightX2VEngine()
        else:
            _engine_instance = Wan22Engine()
    return _engine_instance


__all__ = ["BaseVideoEngine", "Wan22Engine", "LightX2VEngine", "get_active_engine"]
