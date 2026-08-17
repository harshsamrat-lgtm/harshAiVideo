"""
Project management business logic with database persistence.
"""
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.database.session import AsyncSessionLocal
from app.database.models import ProjectModel, CharacterModel, LocationModel, SceneModel, ShotModel
from app.models.schemas import ProjectCreate, ProjectResponse, CharacterResponse, LocationResponse, SceneResponse, ShotResponse
from app.core.logging import logger


class ProjectService:
    @staticmethod
    async def create_project(data: ProjectCreate) -> ProjectResponse:
        project_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        async with AsyncSessionLocal() as session:
            project = ProjectModel(
                project_id=project_id,
                name=data.name,
                description=data.description,
                target_duration=data.target_duration,
                resolution=data.resolution,
                fps=data.fps,
                created_at=now,
                updated_at=now
            )
            session.add(project)
            await session.commit()
            await session.refresh(project)
            
            logger.info(f"Created project: '{data.name}' (ID: {project_id}, Target: {data.target_duration}s)")
            return ProjectResponse(
                project_id=project.project_id,
                name=project.name,
                description=project.description,
                target_duration=project.target_duration,
                resolution=project.resolution,
                fps=project.fps,
                characters=[],
                locations=[],
                scenes=[],
                created_at=project.created_at,
                updated_at=project.updated_at
            )

    @staticmethod
    async def list_projects() -> List[ProjectResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(ProjectModel).options(
                selectinload(ProjectModel.characters),
                selectinload(ProjectModel.locations),
                selectinload(ProjectModel.scenes).selectinload(SceneModel.shots)
            )
            result = await session.execute(stmt)
            projects = result.scalars().all()
            
            response = []
            for p in projects:
                response.append(ProjectResponse(
                    project_id=p.project_id,
                    name=p.name,
                    description=p.description,
                    target_duration=p.target_duration,
                    resolution=p.resolution,
                    fps=p.fps,
                    characters=[CharacterResponse(
                        character_id=c.character_id,
                        project_id=c.project_id,
                        name=c.name,
                        description=c.description,
                        appearance=c.appearance,
                        face_reference=c.face_reference,
                        body_description=c.body_description,
                        hair=c.hair,
                        clothing=c.clothing,
                        age_or_look=c.age_or_look,
                        accessories=c.accessories,
                        default_prompt=c.default_prompt,
                        negative_prompt=c.negative_prompt,
                        voice_profile_id=c.voice_profile_id,
                        consistency_settings=c.consistency_settings or {},
                        reference_images=c.reference_images or [],
                        optional_lora=c.optional_lora,
                        created_at=c.created_at,
                        updated_at=c.updated_at
                    ) for c in p.characters],
                    locations=[LocationResponse(
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
                    ) for l in p.locations],
                    scenes=[SceneResponse(
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
                    ) for s in p.scenes],
                    created_at=p.created_at,
                    updated_at=p.updated_at
                ))
            return response

    @staticmethod
    async def get_project(project_id: str) -> Optional[ProjectResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(ProjectModel).where(ProjectModel.project_id == project_id).options(
                selectinload(ProjectModel.characters),
                selectinload(ProjectModel.locations),
                selectinload(ProjectModel.scenes).selectinload(SceneModel.shots)
            )
            result = await session.execute(stmt)
            p = result.scalar_one_or_none()
            if not p:
                return None
            return ProjectResponse(
                project_id=p.project_id,
                name=p.name,
                description=p.description,
                target_duration=p.target_duration,
                resolution=p.resolution,
                fps=p.fps,
                characters=[CharacterResponse(
                    character_id=c.character_id,
                    project_id=c.project_id,
                    name=c.name,
                    description=c.description,
                    appearance=c.appearance,
                    face_reference=c.face_reference,
                    body_description=c.body_description,
                    hair=c.hair,
                    clothing=c.clothing,
                    age_or_look=c.age_or_look,
                    accessories=c.accessories,
                    default_prompt=c.default_prompt,
                    negative_prompt=c.negative_prompt,
                    voice_profile_id=c.voice_profile_id,
                    consistency_settings=c.consistency_settings or {},
                    reference_images=c.reference_images or [],
                    optional_lora=c.optional_lora,
                    created_at=c.created_at,
                    updated_at=c.updated_at
                ) for c in p.characters],
                locations=[LocationResponse(
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
                ) for l in p.locations],
                scenes=[SceneResponse(
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
                ) for s in p.scenes],
                created_at=p.created_at,
                updated_at=p.updated_at
            )


project_service = ProjectService()
