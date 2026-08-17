"""
Character endpoints: Permanent Character Bible registration and querying.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query
from app.models.schemas import CharacterCreate, CharacterResponse
from app.services.character_service import character_service

router = APIRouter(prefix="/characters", tags=["Characters"])


@router.post("", response_model=CharacterResponse, status_code=status.HTTP_201_CREATED)
async def create_character(
    payload: CharacterCreate,
    project_id: Optional[str] = Query(None, description="Optional project association")
):
    """Create a new character in the Character Bible."""
    return await character_service.create_character(project_id, payload)


@router.get("", response_model=List[CharacterResponse])
async def list_characters(project_id: Optional[str] = Query(None)):
    """List characters optionally filtered by project."""
    return await character_service.list_characters(project_id)


@router.get("/{character_id}", response_model=CharacterResponse)
async def get_character(character_id: str):
    """Retrieve character profile with consistency settings and reference assets."""
    char = await character_service.get_character(character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    return char
