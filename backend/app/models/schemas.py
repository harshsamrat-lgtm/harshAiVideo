"""
Pydantic Schemas for Harsh AI Video Studio.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum


class JobState(str, Enum):
    QUEUED = "QUEUED"
    LOADING = "LOADING"
    GENERATING = "GENERATING"
    QC = "QC"
    REGENERATING = "REGENERATING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class QCStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


# ==============================================================================
# CHARACTER BIBLE SCHEMAS
# ==============================================================================
class CharacterBase(BaseModel):
    name: str = Field(..., description="Character name")
    description: str = Field("", description="General personality and role overview")
    appearance: str = Field("", description="Detailed facial and visual traits")
    face_reference: Optional[str] = Field(None, description="Path or URL to primary master face reference")
    body_description: str = Field("", description="Height, build, posture, skin tone")
    hair: str = Field("", description="Hair style, length, color")
    clothing: str = Field("", description="Default wardrobe and color palette")
    age_or_look: str = Field("", description="Perceived age and facial structure")
    accessories: str = Field("", description="Glasses, jewelry, hats, scars, distinctive marks")
    default_prompt: str = Field("", description="Default prompt segment injected into shot prompts")
    negative_prompt: str = Field("", description="Character-specific negative prompt")
    voice_profile_id: Optional[str] = Field(None, description="Associated voice profile ID")
    consistency_settings: Dict[str, Any] = Field(default_factory=dict, description="Weights for face/body embedding matching")
    reference_images: List[str] = Field(default_factory=list, description="Array of auxiliary reference image paths")
    optional_lora: Optional[str] = Field(None, description="Optional LoRA adapter name or path")


class CharacterCreate(CharacterBase):
    pass


class CharacterResponse(CharacterBase):
    character_id: str
    project_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# LOCATION BIBLE SCHEMAS
# ==============================================================================
class LocationBase(BaseModel):
    name: str = Field(..., description="Location name (e.g. Market, House, School, Office)")
    description: str = Field("", description="Location background and narrative context")
    architecture: str = Field("", description="Structural style, walls, materials, era")
    environment: str = Field("", description="Natural or interior surroundings, terrain, props")
    lighting: str = Field("", description="Default lighting ambiance (e.g. golden hour, neon, overcast)")
    weather: str = Field("Clear", description="Atmospheric conditions")
    time_of_day: str = Field("Day", description="Morning, Afternoon, Sunset, Night")
    camera_style: str = Field("Cinematic 35mm", description="Preferred camera framing and color grade")
    reference_images: List[str] = Field(default_factory=list, description="Reference images for background consistency")
    default_prompt: str = Field("", description="Default location prompt segment")


class LocationCreate(LocationBase):
    pass


class LocationResponse(LocationBase):
    location_id: str
    project_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# SHOT SCHEMAS
# ==============================================================================
class ShotBase(BaseModel):
    scene_id: str
    prompt: str = Field(..., description="Full action and visual prompt for Wan 2.2 / LightX2V")
    negative_prompt: str = Field("", description="Negative prompt for artifacts exclusion")
    character_ids: List[str] = Field(default_factory=list, description="Characters appearing in this shot")
    location_id: Optional[str] = Field(None, description="Location ID for environment context")
    previous_shot_id: Optional[str] = Field(None, description="Previous shot ID for continuity chaining")
    previous_last_frame: Optional[str] = Field(None, description="Extracted last frame path of previous shot")
    seed: int = Field(default=-1, description="Random seed (-1 for randomized)")
    duration: float = Field(default=5.0, description="Shot duration in seconds (5-10s)")
    resolution: str = Field(default="1280x720", description="Internal generation resolution")
    camera_motion: str = Field("static", description="Camera motion: pan, zoom, tracking, dolly, static")
    action: str = Field("", description="Explicit action performed in shot")


class ShotCreate(ShotBase):
    pass


class ShotResponse(ShotBase):
    shot_id: str
    generation_status: JobState = JobState.QUEUED
    output_path: Optional[str] = None
    qc_status: Optional[QCStatus] = None
    attempts: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# SCENE SCHEMAS
# ==============================================================================
class SceneBase(BaseModel):
    location_id: str
    character_ids: List[str] = Field(default_factory=list, description="All characters present in the scene")
    dialogue: str = Field("", description="Full dialogue transcript for the scene")
    action: str = Field("", description="Scene narrative action overview")
    duration: float = Field(default=20.0, description="Total scene duration in seconds")
    camera: str = Field("Cinematic Medium Shot", description="Camera style & angle")
    lighting: str = Field("Natural daylight", description="Lighting conditions")
    weather: str = Field("Clear", description="Weather")
    continuity_requirements: str = Field("", description="Special continuity rules between shots")


class SceneCreate(SceneBase):
    project_id: str


class SceneResponse(SceneBase):
    scene_id: str
    project_id: str
    shots: List[ShotResponse] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# PROJECT SCHEMAS
# ==============================================================================
class ProjectBase(BaseModel):
    name: str = Field(..., description="Project title")
    description: str = Field("", description="Storyline & synopsis")
    target_duration: float = Field(default=300.0, description="Target total duration in seconds (e.g. 300s = 5min)")
    resolution: str = Field(default="1920x1080", description="Final target assembly resolution")
    fps: int = Field(default=24, description="Final frames per second")


class ProjectCreate(ProjectBase):
    pass


class ProjectResponse(ProjectBase):
    project_id: str
    characters: List[CharacterResponse] = Field(default_factory=list)
    locations: List[LocationResponse] = Field(default_factory=list)
    scenes: List[SceneResponse] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# JOB & GENERATION SCHEMAS
# ==============================================================================
class GenerationRequest(BaseModel):
    project_id: Optional[str] = None
    scene_id: Optional[str] = None
    shot_id: Optional[str] = None
    character_ids: List[str] = Field(default_factory=list)
    location_id: Optional[str] = None
    reference_image_path: Optional[str] = None
    prompt: str
    negative_prompt: Optional[str] = None
    duration: float = Field(default=5.0, ge=1.0, le=15.0)
    resolution: str = Field(default="1280x720")
    seed: int = Field(default=-1)
    engine: str = Field(default="lightx2v")
    auto_qc: bool = True
    auto_retry: bool = True


class JobResponse(BaseModel):
    job_id: str
    project_id: Optional[str] = None
    shot_id: Optional[str] = None
    status: JobState
    progress: float = 0.0
    engine: str = "lightx2v"
    attempts: int = 1
    max_attempts: int = 3
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# VOICE & LIPSYNC SCHEMAS
# ==============================================================================
class VoiceProfileCreate(BaseModel):
    name: str
    provider: str = "modular_tts"
    sample_audio_path: Optional[str] = None
    pitch: float = 1.0
    speed: float = 1.0
    emotion: str = "neutral"
    config: Dict[str, Any] = Field(default_factory=dict)


class VoiceProfileResponse(VoiceProfileCreate):
    voice_profile_id: str


class LipSyncRequest(BaseModel):
    video_path: str
    audio_path: str
    shot_id: Optional[str] = None
    character_id: Optional[str] = None
    provider: str = "modular_lipsync"


class LipSyncResponse(BaseModel):
    lipsync_id: str
    status: JobState
    output_video_path: Optional[str] = None


# ==============================================================================
# QUALITY CONTROL (QC) SCHEMAS
# ==============================================================================
class QCReport(BaseModel):
    shot_id: str
    video_path: str
    status: QCStatus
    duration_actual: float
    duration_expected: float
    resolution_actual: str
    black_frames_pct: float
    face_similarity_score: Optional[float] = None
    corruption_detected: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# FINAL RENDER & ASSEMBLY SCHEMAS
# ==============================================================================
class FinalRenderRequest(BaseModel):
    project_id: str
    target_resolution: str = "1920x1080"
    fps: int = 24
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    bgm_audio_path: Optional[str] = None
    bgm_volume: float = 0.2
    enable_subtitles: bool = False
    transition_style: str = "fade"


class FinalRenderResponse(BaseModel):
    render_job_id: str
    project_id: str
    status: JobState
    output_video_path: Optional[str] = None


# ==============================================================================
# SYSTEM & GPU TELEMETRY SCHEMAS
# ==============================================================================
class GPUTelemetry(BaseModel):
    gpu_id: int = 0
    gpu_name: str = "NVIDIA GeForce RTX 5090"
    gpu_utilization_pct: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 32768.0
    vram_used_pct: float = 0.0
    temperature_celsius: float = 0.0
    power_draw_watts: float = 0.0
    power_limit_watts: float = 600.0
    driver_version: str = "560.35.03"
    cuda_version: str = "12.4"


class SystemStatusResponse(BaseModel):
    status: str = "healthy"
    app_env: str
    gpu_mode: str
    active_engine: str
    gpus: List[GPUTelemetry] = Field(default_factory=list)
    current_active_job: Optional[str] = None
    queue_length: int = 0
    generation_progress_pct: float = 0.0
    eta_seconds: Optional[int] = None
    system_cpu_usage_pct: float = 0.0
    system_ram_used_gb: float = 0.0
    system_ram_total_gb: float = 0.0
