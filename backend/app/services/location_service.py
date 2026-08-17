"""
Location Bible service for environmental and architectural continuity with database persistence.
"""
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from app.database.session import AsyncSessionLocal
from app.database.models import LocationModel
from app.models.schemas import LocationCreate, LocationResponse
from app.core.logging import logger


class LocationService:
    @staticmethod
    async def create_location(project_id: Optional[str], data: LocationCreate) -> LocationResponse:
        loc_id = f"loc_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)
        
        async with AsyncSessionLocal() as session:
            loc = LocationModel(
                location_id=loc_id,
                project_id=project_id,
                name=data.name,
                description=data.description,
                architecture=data.architecture,
                environment=data.environment,
                lighting=data.lighting,
                weather=data.weather,
                time_of_day=data.time_of_day,
                camera_style=data.camera_style,
                reference_images=data.reference_images or [],
                default_prompt=data.default_prompt,
                created_at=now,
                updated_at=now
            )
            session.add(loc)
            await session.commit()
            await session.refresh(loc)
            
            logger.info(f"Registered location in Bible: '{data.name}' (ID: {loc_id})")
            return LocationResponse(
                location_id=loc.location_id,
                project_id=loc.project_id,
                name=loc.name,
                description=loc.description,
                architecture=loc.architecture,
                environment=loc.environment,
                lighting=loc.lighting,
                weather=loc.weather,
                time_of_day=loc.time_of_day,
                camera_style=loc.camera_style,
                reference_images=loc.reference_images or [],
                default_prompt=loc.default_prompt,
                created_at=loc.created_at,
                updated_at=loc.updated_at
            )

    @staticmethod
    async def list_locations(project_id: Optional[str] = None) -> List[LocationResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(LocationModel)
            if project_id:
                stmt = stmt.where(
                    (LocationModel.project_id == project_id) | (LocationModel.project_id.is_(None))
                )
            result = await session.execute(stmt)
            locs = result.scalars().all()
            return [LocationResponse(
                location_id=l.location_id,
                project_id=l.project_id,
                name=l.name,
                description=l.description,
                architecture=l.architecture,
                environment=l.environment,
                lighting=l.lighting,
                weather=l.weather,
                time_of_day=l.time_of_day,
                camera_style=l.camera_style,
                reference_images=l.reference_images or [],
                default_prompt=l.default_prompt,
                created_at=l.created_at,
                updated_at=l.updated_at
            ) for l in locs]

    @staticmethod
    async def get_location(location_id: str) -> Optional[LocationResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(LocationModel).where(LocationModel.location_id == location_id)
            result = await session.execute(stmt)
            l = result.scalar_one_or_none()
            if not l:
                return None
            return LocationResponse(
                location_id=l.location_id,
                project_id=l.project_id,
                name=l.name,
                description=l.description,
                architecture=l.architecture,
                environment=l.environment,
                lighting=l.lighting,
                weather=l.weather,
                time_of_day=l.time_of_day,
                camera_style=l.camera_style,
                reference_images=l.reference_images or [],
                default_prompt=l.default_prompt,
                created_at=l.created_at,
                updated_at=l.updated_at
            )


location_service = LocationService()
