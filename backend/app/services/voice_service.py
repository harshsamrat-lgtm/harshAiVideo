"""
Modular Voice Profile and TTS Synthesis Service with Database Persistence.
Ensures Character A consistently uses Voice Profile A across all scenes and shots.
"""
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from app.database.session import AsyncSessionLocal
from app.database.models import VoiceProfileModel, CharacterModel
from app.models.schemas import VoiceProfileCreate, VoiceProfileResponse
from app.core.logging import logger


class VoiceService:
    @staticmethod
    async def create_voice_profile(data: VoiceProfileCreate) -> VoiceProfileResponse:
        voice_id = f"voice_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)
        
        async with AsyncSessionLocal() as session:
            voice = VoiceProfileModel(
                voice_profile_id=voice_id,
                name=data.name,
                provider=data.provider,
                sample_audio_path=data.sample_audio_path,
                pitch=data.pitch,
                speed=data.speed,
                emotion=data.emotion,
                config=data.config or {},
                created_at=now,
                updated_at=now
            )
            session.add(voice)
            await session.commit()
            await session.refresh(voice)
            
            logger.info(f"Registered voice profile: '{data.name}' (ID: {voice_id}, Pitch: {data.pitch}, Speed: {data.speed})")
            return VoiceProfileResponse(
                voice_profile_id=voice.voice_profile_id,
                name=voice.name,
                provider=voice.provider,
                sample_audio_path=voice.sample_audio_path,
                pitch=voice.pitch,
                speed=voice.speed,
                emotion=voice.emotion,
                config=voice.config or {}
            )

    @staticmethod
    async def get_voice_for_character(character_id: str) -> Optional[VoiceProfileResponse]:
        """
        Retrieves the exact fixed voice profile bound to a given character.
        """
        async with AsyncSessionLocal() as session:
            stmt = select(CharacterModel).where(CharacterModel.character_id == character_id)
            res = await session.execute(stmt)
            char = res.scalar_one_or_none()
            if not char or not char.voice_profile_id:
                return None
            
            voice_stmt = select(VoiceProfileModel).where(VoiceProfileModel.voice_profile_id == char.voice_profile_id)
            voice_res = await session.execute(voice_stmt)
            v = voice_res.scalar_one_or_none()
            if not v:
                return None
            return VoiceProfileResponse(
                voice_profile_id=v.voice_profile_id,
                name=v.name,
                provider=v.provider,
                sample_audio_path=v.sample_audio_path,
                pitch=v.pitch,
                speed=v.speed,
                emotion=v.emotion,
                config=v.config or {}
            )

    @staticmethod
    async def list_voice_profiles() -> List[VoiceProfileResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(VoiceProfileModel)
            res = await session.execute(stmt)
            voices = res.scalars().all()
            return [VoiceProfileResponse(
                voice_profile_id=v.voice_profile_id,
                name=v.name,
                provider=v.provider,
                sample_audio_path=v.sample_audio_path,
                pitch=v.pitch,
                speed=v.speed,
                emotion=v.emotion,
                config=v.config or {}
            ) for v in voices]


voice_service = VoiceService()
