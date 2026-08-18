import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    class _BaseSettings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=str(BASE_DIR / ".env"),
            env_file_encoding="utf-8",
            extra="ignore"
        )
except ImportError:
    class _BaseSettings(BaseModel):
        pass


class Settings(_BaseSettings):


    # Application
    APP_ENV: str = "development"
    APP_NAME: str = "Harsh AI Video Studio"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Execution Topology
    GPU_MODE: str = "remote"  # 'remote' | 'local'
    ENGINE: str = "sana-video-2b"   # Dedicated SANA-Video 2B Engine

    # Remote GPU Server Connectivity
    GPU_SERVER_URL: Optional[str] = None
    GPU_SERVER_TIMEOUT_SECONDS: int = 600

    # Security
    API_KEY: str = "harsh_studio_secure_dev_key_change_in_production"
    SECRET_KEY: str = "default_secret_key_change_in_production"
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    # Storage Roots
    MODEL_ROOT: str = str(BASE_DIR / "models")
    PROJECT_ROOT: str = str(BASE_DIR / "projects")
    OUTPUT_ROOT: str = str(BASE_DIR / "outputs")
    CACHE_ROOT: str = str(BASE_DIR / "cache")

    # Database & Redis
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/studio.sqlite3"
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_QUEUE_NAME: str = "video_generation_jobs"
    REDIS_QC_QUEUE_NAME: str = "video_qc_jobs"
    REDIS_RENDER_QUEUE_NAME: str = "video_render_jobs"

    # LightX2V Settings
    LIGHTX2V_PRECISION: str = "nvfp4"
    LIGHTX2V_SPARSE_ATTENTION: bool = True
    LIGHTX2V_ATTENTION_HEAD_RATIO: float = 0.75
    LIGHTX2V_USE_CUDA_GRAPH: bool = True
    LIGHTX2V_OFFLOAD_CPU: bool = False
    LIGHTX2V_MAX_BATCH_SIZE: int = 1

    # Wan 2.2 Model Settings
    WAN22_MODEL_NAME: str = "Wan2.2-I2V-14B-720P"
    WAN22_CHECKPOINT_DIR: str = str(BASE_DIR / "models" / "Wan2.2-I2V-14B-720P")
    WAN22_DEFAULT_STEPS: int = 30
    WAN22_DEFAULT_GUIDANCE_SCALE: float = 5.0
    WAN22_DEFAULT_FPS: int = 24
    WAN22_DEFAULT_RESOLUTION: str = "1280x720"

    # Quality Control (QC)
    MAX_REGENERATION_ATTEMPTS: int = 3
    QC_FACE_SIMILARITY_THRESHOLD: float = 0.75
    QC_BLACK_FRAME_TOLERANCE_PCT: float = 0.01
    QC_MIN_AUDIO_VIDEO_SYNC_MS: int = 50
    QC_ENABLE_AUTO_RETRY: bool = True

    # Video Processing & FFmpeg
    FFMPEG_BINARY_PATH: str = "ffmpeg"
    FFPROBE_BINARY_PATH: str = "ffprobe"
    DEFAULT_VIDEO_CODEC: str = "libx264"
    DEFAULT_AUDIO_CODEC: str = "aac"
    DEFAULT_RENDER_FPS: int = 24
    TARGET_FINAL_RESOLUTION: str = "1920x1080"


settings = Settings()
