"""
Scene orchestration service managing multi-character assignment, locations, and DB persistence.
"""
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.database.session import AsyncSessionLocal
from app.database.models import SceneModel, ShotModel
from app.models.schemas import SceneCreate, SceneResponse, ShotResponse
from app.core.logging import logger


class SceneService:
    @staticmethod
    async def create_scene(project_id: str, data: SceneCreate) -> SceneResponse:
        scene_id = f"scene_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)
        
        async with AsyncSessionLocal() as session:
            scene = SceneModel(
                scene_id=scene_id,
                project_id=project_id,
                location_id=data.location_id,
                character_ids=data.character_ids or [],
                dialogue=data.dialogue,
                action=data.action,
                duration=data.duration,
                camera=data.camera,
                lighting=data.lighting,
                weather=data.weather,
                continuity_requirements=data.continuity_requirements,
                created_at=now,
                updated_at=now
            )
            session.add(scene)
            await session.commit()
            await session.refresh(scene)
            
            logger.info(f"Created scene {scene_id} in project {project_id} (Duration: {data.duration}s, Chars: {len(data.character_ids)})")
            return SceneResponse(
                scene_id=scene.scene_id,
                project_id=scene.project_id,
                location_id=scene.location_id or "",
                character_ids=scene.character_ids or [],
                dialogue=scene.dialogue,
                action=scene.action,
                duration=scene.duration,
                camera=scene.camera,
                lighting=scene.lighting,
                weather=scene.weather,
                continuity_requirements=scene.continuity_requirements,
                shots=[],
                created_at=scene.created_at,
                updated_at=scene.updated_at
            )

    @staticmethod
    async def list_scenes(project_id: str) -> List[SceneResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(SceneModel).where(SceneModel.project_id == project_id).options(selectinload(SceneModel.shots))
            result = await session.execute(stmt)
            scenes = result.scalars().all()
            return [SceneResponse(
                scene_id=s.scene_id,
                project_id=s.project_id,
                location_id=s.location_id or "",
                character_ids=s.character_ids or [],
                dialogue=s.dialogue,
                action=s.action,
                duration=s.duration,
                camera=s.camera,
                lighting=s.lighting,
                weather=s.weather,
                continuity_requirements=s.continuity_requirements,
                shots=[ShotResponse(
                    shot_id=sh.shot_id,
                    scene_id=sh.scene_id,
                    prompt=sh.prompt,
                    negative_prompt=sh.negative_prompt,
                    character_ids=sh.character_ids or [],
                    location_id=sh.location_id,
                    previous_shot_id=sh.previous_shot_id,
                    previous_last_frame=sh.previous_last_frame,
                    seed=sh.seed,
                    duration=sh.duration,
                    resolution=sh.resolution,
                    camera_motion=sh.camera_motion,
                    action=sh.action,
                    generation_status=sh.generation_status,
                    output_path=sh.output_path,
                    qc_status=sh.qc_status,
                    attempts=sh.attempts,
                    created_at=sh.created_at,
                    updated_at=sh.updated_at
                ) for sh in s.shots],
                created_at=s.created_at,
                updated_at=s.updated_at
            ) for s in scenes]

    @staticmethod
    async def get_scene(scene_id: str) -> Optional[SceneResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(SceneModel).where(SceneModel.scene_id == scene_id).options(selectinload(SceneModel.shots))
            result = await session.execute(stmt)
            s = result.scalar_one_or_none()
            if not s:
                return None
            return SceneResponse(
                scene_id=s.scene_id,
                project_id=s.project_id,
                location_id=s.location_id or "",
                character_ids=s.character_ids or [],
                dialogue=s.dialogue,
                action=s.action,
                duration=s.duration,
                camera=s.camera,
                lighting=s.lighting,
                weather=s.weather,
                continuity_requirements=s.continuity_requirements,
                shots=[ShotResponse(
                    shot_id=sh.shot_id,
                    scene_id=sh.scene_id,
                    prompt=sh.prompt,
                    negative_prompt=sh.negative_prompt,
                    character_ids=sh.character_ids or [],
                    location_id=sh.location_id,
                    previous_shot_id=sh.previous_shot_id,
                    previous_last_frame=sh.previous_last_frame,
                    seed=sh.seed,
                    duration=sh.duration,
                    resolution=sh.resolution,
                    camera_motion=sh.camera_motion,
                    action=sh.action,
                    generation_status=sh.generation_status,
                    output_path=sh.output_path,
                    qc_status=sh.qc_status,
                    attempts=sh.attempts,
                    created_at=sh.created_at,
                    updated_at=sh.updated_at
                ) for sh in s.shots],
                created_at=s.created_at,
                updated_at=s.updated_at
            )


scene_service = SceneService()
