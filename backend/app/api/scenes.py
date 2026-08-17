"""
Scene endpoints: Multi-character and location scene orchestration.
"""
from typing import List
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import SceneCreate, SceneResponse
from app.services.scene_service import scene_service

router = APIRouter(tags=["Scenes"])


@router.post("/projects/{project_id}/scenes", response_model=SceneResponse, status_code=status.HTTP_201_CREATED)
async def create_scene(project_id: str, payload: SceneCreate):
    """Create a new scene within a project."""
    return await scene_service.create_scene(project_id, payload)


@router.get("/projects/{project_id}/scenes", response_model=List[SceneResponse])
async def list_scenes(project_id: str):
    """List all scenes in a project."""
    return await scene_service.list_scenes(project_id)


@router.get("/scenes/{scene_id}", response_model=SceneResponse)
async def get_scene(scene_id: str):
    """Get scene details with assigned characters and location."""
    scene = await scene_service.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    return scene
