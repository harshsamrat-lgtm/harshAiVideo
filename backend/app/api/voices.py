"""
Voice endpoints: Permanent voice profiles for character dialogue.
"""
from typing import List
from fastapi import APIRouter, status
from app.models.schemas import VoiceProfileCreate, VoiceProfileResponse
from app.services.voice_service import voice_service

router = APIRouter(prefix="/voices", tags=["Voices"])


@router.post("", response_model=VoiceProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_voice_profile(payload: VoiceProfileCreate):
    """Register a new character voice profile."""
    return await voice_service.create_voice_profile(payload)


@router.get("", response_model=List[VoiceProfileResponse])
async def list_voice_profiles():
    """List all character voice profiles."""
    return await voice_service.list_voice_profiles()
