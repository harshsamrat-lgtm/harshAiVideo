"""
Abstract Base Class for all AI Video Generation Engines in Harsh AI Video Studio.
Enforces contract for model loading, VRAM estimation, inference, cancellation, and hardware capability probing.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseVideoEngine(ABC):
    """
    Base abstraction interface for AI video generation backends.
    Isolates core pipeline logic from model-specific inference runtimes
    (e.g., LightX2V, Wan2.2, future video diffusion models).
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.is_loaded = False
        self.active_jobs: Dict[str, bool] = {}

    @abstractmethod
    async def load_model(self) -> bool:
        """Load model weights and initialize inference graph/pipeline."""
        pass

    @abstractmethod
    async def unload_model(self) -> bool:
        """Release GPU VRAM and unload model from memory."""
        pass

    @abstractmethod
    async def generate_image_to_video(
        self,
        prompt: str,
        reference_image_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        duration_seconds: float = 5.0,
        resolution: str = "1280x720",
        seed: int = -1,
        steps: int = 30,
        guidance_scale: float = 5.0,
        output_path: Optional[str] = None,
        callback: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute Image-to-Video generation inference.
        Returns metadata containing output_path, seed, duration, resolution, generation_time.
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return operational state, loaded weights, and memory metrics."""
        pass

    @abstractmethod
    async def cancel(self, job_id: str) -> bool:
        """Signal running inference to abort and cleanup resources."""
        pass

    @abstractmethod
    def validate_environment(self) -> Dict[str, Any]:
        """Check hardware compatibility (CUDA, driver, NVFP4, VRAM capacity)."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return supported resolutions, FPS, max duration, precision modes, acceleration flags."""
        pass

    @abstractmethod
    def estimate_vram_requirement(self, resolution: str = "1280x720", duration_seconds: float = 5.0) -> float:
        """Estimate peak VRAM memory requirement in Gigabytes (GB)."""
        pass
