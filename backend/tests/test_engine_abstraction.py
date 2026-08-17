"""
Tests for BaseVideoEngine interface contracts and LightX2V / Wan22 implementations.
"""
import pytest
from app.engines.base_engine import BaseVideoEngine
from app.engines.lightx2v_engine import LightX2VEngine
from app.engines.wan22_engine import Wan22Engine
from app.engines import get_active_engine


@pytest.mark.asyncio
async def test_lightx2v_engine_contract():
    engine = LightX2VEngine()
    assert isinstance(engine, BaseVideoEngine)
    assert engine.name == "LightX2V-Wan2.2-NVFP4"

    # Test load & unload
    loaded = await engine.load_model()
    assert loaded is True
    assert engine.is_loaded is True

    # Test generation abstraction
    result = await engine.generate_image_to_video(
        prompt="Test cinematic scene",
        duration_seconds=5.0,
        resolution="1280x720"
    )
    assert result["status"] == "COMPLETED"
    assert result["duration"] == 5.0
    assert result["resolution"] == "1280x720"
    assert result["precision"] == "nvfp4"

    # Test capabilities
    caps = engine.get_capabilities()
    assert caps["supports_nvfp4"] is True
    assert caps["supports_sparse_attention"] is True

    # Test unload
    unloaded = await engine.unload_model()
    assert unloaded is True
    assert engine.is_loaded is False


@pytest.mark.asyncio
async def test_wan22_engine_contract():
    engine = Wan22Engine()
    assert isinstance(engine, BaseVideoEngine)
    assert engine.name == "Wan2.2-I2V-14B"

    loaded = await engine.load_model()
    assert loaded is True

    result = await engine.generate_image_to_video(
        prompt="Test shot",
        duration_seconds=4.0
    )
    assert result["status"] == "COMPLETED"
    assert result["duration"] == 4.0
