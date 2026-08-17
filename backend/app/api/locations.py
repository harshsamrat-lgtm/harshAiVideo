"""
Location endpoints: Permanent Location Bible management.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query
from app.models.schemas import LocationCreate, LocationResponse
from app.services.location_service import location_service

router = APIRouter(prefix="/locations", tags=["Locations"])


@router.post("", response_model=LocationResponse, status_code=status.HTTP_201_CREATED)
async def create_location(
    payload: LocationCreate,
    project_id: Optional[str] = Query(None)
):
    """Create a new location in the Location Bible."""
    return await location_service.create_location(project_id, payload)


@router.get("", response_model=List[LocationResponse])
async def list_locations(project_id: Optional[str] = Query(None)):
    """List all registered locations."""
    return await location_service.list_locations(project_id)


@router.get("/{location_id}", response_model=LocationResponse)
async def get_location(location_id: str):
    """Retrieve detailed location profile and visual settings."""
    loc = await location_service.get_location(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")
    return loc
