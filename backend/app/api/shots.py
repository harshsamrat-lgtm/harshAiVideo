"""
Shot endpoints: Granular 5-10s shot clip planning and prompt management.
"""
from typing import List
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import ShotCreate, ShotResponse
from app.services.shot_service import shot_service

router = APIRouter(tags=["Shots"])


@router.post("/scenes/{scene_id}/shots", response_model=ShotResponse, status_code=status.HTTP_201_CREATED)
async def create_shot(scene_id: str, payload: ShotCreate):
    """Create a new shot clip within a scene."""
    return await shot_service.create_shot(scene_id, payload)


@router.get("/scenes/{scene_id}/shots", response_model=List[ShotResponse])
async def list_shots(scene_id: str):
    """List all shots belonging to a scene."""
    return await shot_service.list_shots(scene_id)


@router.get("/shots/{shot_id}", response_model=ShotResponse)
async def get_shot(shot_id: str):
    """Get single shot status, prompt, and output path."""
    shot = await shot_service.get_shot(shot_id)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    return shot
