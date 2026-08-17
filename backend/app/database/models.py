"""
SQLAlchemy Async ORM Models for Harsh AI Video Studio.
Supports SQLite and PostgreSQL architectures.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Enum as SAEnum
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.database.base import Base
from app.models.schemas import JobState, QCStatus


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProjectModel(Base):
    __tablename__ = "projects"

    project_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    target_duration: Mapped[float] = mapped_column(Float, default=300.0)  # 300s = 5 minutes
    resolution: Mapped[str] = mapped_column(String(32), default="1920x1080")
    fps: Mapped[int] = mapped_column(Integer, default=24)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    characters = relationship("CharacterModel", back_populates="project", cascade="all, delete-orphan")
    locations = relationship("LocationModel", back_populates="project", cascade="all, delete-orphan")
    scenes = relationship("SceneModel", back_populates="project", cascade="all, delete-orphan")
    jobs = relationship("JobModel", back_populates="project", cascade="all, delete-orphan")


class CharacterModel(Base):
    __tablename__ = "characters"

    character_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    project_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    appearance: Mapped[str] = mapped_column(Text, default="")
    face_reference: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    body_description: Mapped[str] = mapped_column(Text, default="")
    hair: Mapped[str] = mapped_column(String(255), default="")
    clothing: Mapped[str] = mapped_column(Text, default="")
    age_or_look: Mapped[str] = mapped_column(String(128), default="")
    accessories: Mapped[str] = mapped_column(Text, default="")
    default_prompt: Mapped[str] = mapped_column(Text, default="")
    negative_prompt: Mapped[str] = mapped_column(Text, default="")
    voice_profile_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("voice_profiles.voice_profile_id", ondelete="SET NULL"), nullable=True)
    consistency_settings: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    reference_images: Mapped[List[str]] = mapped_column(JSON, default=list)
    optional_lora: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    project = relationship("ProjectModel", back_populates="characters")
    voice_profile = relationship("VoiceProfileModel", back_populates="characters")


class LocationModel(Base):
    __tablename__ = "locations"

    location_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    project_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    architecture: Mapped[str] = mapped_column(Text, default="")
    environment: Mapped[str] = mapped_column(Text, default="")
    lighting: Mapped[str] = mapped_column(String(255), default="")
    weather: Mapped[str] = mapped_column(String(128), default="Clear")
    time_of_day: Mapped[str] = mapped_column(String(128), default="Day")
    camera_style: Mapped[str] = mapped_column(String(255), default="Cinematic 35mm")
    reference_images: Mapped[List[str]] = mapped_column(JSON, default=list)
    default_prompt: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    project = relationship("ProjectModel", back_populates="locations")
    scenes = relationship("SceneModel", back_populates="location")


class SceneModel(Base):
    __tablename__ = "scenes"

    scene_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    project_id: Mapped[str] = mapped_column(String(64), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    location_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("locations.location_id", ondelete="SET NULL"), nullable=True)
    character_ids: Mapped[List[str]] = mapped_column(JSON, default=list)  # Multi-character IDs
    dialogue: Mapped[str] = mapped_column(Text, default="")
    action: Mapped[str] = mapped_column(Text, default="")
    duration: Mapped[float] = mapped_column(Float, default=20.0)
    camera: Mapped[str] = mapped_column(String(255), default="Cinematic Medium Shot")
    lighting: Mapped[str] = mapped_column(String(255), default="Natural daylight")
    weather: Mapped[str] = mapped_column(String(128), default="Clear")
    continuity_requirements: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    project = relationship("ProjectModel", back_populates="scenes")
    location = relationship("LocationModel", back_populates="scenes")
    shots = relationship("ShotModel", back_populates="scene", cascade="all, delete-orphan")


class ShotModel(Base):
    __tablename__ = "shots"

    shot_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    scene_id: Mapped[str] = mapped_column(String(64), ForeignKey("scenes.scene_id", ondelete="CASCADE"), nullable=False)
    sequence_order: Mapped[int] = mapped_column(Integer, default=1)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[str] = mapped_column(Text, default="")
    character_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    location_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    previous_shot_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    previous_last_frame: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)  # Frame chaining path
    seed: Mapped[int] = mapped_column(Integer, default=-1)
    duration: Mapped[float] = mapped_column(Float, default=5.0)  # 5 to 8 seconds
    resolution: Mapped[str] = mapped_column(String(32), default="1280x720")
    camera_motion: Mapped[str] = mapped_column(String(128), default="static")
    action: Mapped[str] = mapped_column(Text, default="")
    generation_status: Mapped[str] = mapped_column(String(32), default=JobState.QUEUED.value)
    output_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    qc_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    scene = relationship("SceneModel", back_populates="shots")
    qc_reports = relationship("QCReportModel", back_populates="shot", cascade="all, delete-orphan")


class VoiceProfileModel(Base):
    __tablename__ = "voice_profiles"

    voice_profile_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), default="modular_tts")
    sample_audio_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    pitch: Mapped[float] = mapped_column(Float, default=1.0)
    speed: Mapped[float] = mapped_column(Float, default=1.0)
    emotion: Mapped[str] = mapped_column(String(64), default="neutral")
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    characters = relationship("CharacterModel", back_populates="voice_profile")


class QCReportModel(Base):
    __tablename__ = "qc_reports"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    shot_id: Mapped[str] = mapped_column(String(64), ForeignKey("shots.shot_id", ondelete="CASCADE"), nullable=False)
    video_path: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default=QCStatus.PASS.value)
    duration_actual: Mapped[float] = mapped_column(Float, default=5.0)
    duration_expected: Mapped[float] = mapped_column(Float, default=5.0)
    resolution_actual: Mapped[str] = mapped_column(String(32), default="1280x720")
    black_frames_pct: Mapped[float] = mapped_column(Float, default=0.0)
    face_similarity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    corruption_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    shot = relationship("ShotModel", back_populates="qc_reports")


class JobModel(Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    project_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=True)
    shot_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default=JobState.QUEUED.value)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    engine: Mapped[str] = mapped_column(String(64), default="lightx2v")
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    result_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    logs: Mapped[List[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    project = relationship("ProjectModel", back_populates="jobs")


class RenderJobModel(Base):
    __tablename__ = "render_jobs"

    render_job_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    project_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default=JobState.QUEUED.value)
    target_resolution: Mapped[str] = mapped_column(String(32), default="1920x1080")
    fps: Mapped[int] = mapped_column(Integer, default=24)
    output_video_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    total_clips: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
