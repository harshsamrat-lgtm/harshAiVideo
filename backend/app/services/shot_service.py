"""
Shot planning and frame continuity service for 5-8 second clips and long-form video chaining.
"""
from typing import List, Optional, Dict, Any
import uuid
import math
from datetime import datetime, timezone
from sqlalchemy import select
from app.database.session import AsyncSessionLocal
from app.database.models import ShotModel, SceneModel
from app.models.schemas import ShotCreate, ShotResponse, JobState
from app.core.logging import logger


class ShotService:
    @staticmethod
    async def create_shot(scene_id: str, data: ShotCreate) -> ShotResponse:
        shot_id = f"shot_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)
        
        # Enforce 5.0s to 8.0s clip duration for maximum AI video stability
        duration = max(5.0, min(8.0, data.duration))
        
        async with AsyncSessionLocal() as session:
            # Determine sequence order if not provided
            stmt = select(ShotModel).where(ShotModel.scene_id == scene_id)
            res = await session.execute(stmt)
            existing_shots = res.scalars().all()
            sequence_order = len(existing_shots) + 1
            
            # Automatically link to previous shot in scene if not explicitly set
            previous_shot_id = data.previous_shot_id
            if not previous_shot_id and existing_shots:
                previous_shot_id = existing_shots[-1].shot_id

            shot = ShotModel(
                shot_id=shot_id,
                scene_id=scene_id,
                sequence_order=sequence_order,
                prompt=data.prompt,
                negative_prompt=data.negative_prompt,
                character_ids=data.character_ids or [],
                location_id=data.location_id,
                previous_shot_id=previous_shot_id,
                previous_last_frame=data.previous_last_frame,
                seed=data.seed,
                duration=duration,
                resolution=data.resolution,
                camera_motion=data.camera_motion,
                action=data.action,
                generation_status=JobState.QUEUED.value,
                output_path=None,
                qc_status=None,
                attempts=0,
                created_at=now,
                updated_at=now
            )
            session.add(shot)
            await session.commit()
            await session.refresh(shot)
            
            logger.info(f"Created shot {shot_id} in scene {scene_id} [Seq #{sequence_order}, {duration}s, Prev: {previous_shot_id}]")
            return ShotResponse(
                shot_id=shot.shot_id,
                scene_id=shot.scene_id,
                prompt=shot.prompt,
                negative_prompt=shot.negative_prompt,
                character_ids=shot.character_ids or [],
                location_id=shot.location_id,
                previous_shot_id=shot.previous_shot_id,
                previous_last_frame=shot.previous_last_frame,
                seed=shot.seed,
                duration=shot.duration,
                resolution=shot.resolution,
                camera_motion=shot.camera_motion,
                action=shot.action,
                generation_status=shot.generation_status,
                output_path=shot.output_path,
                qc_status=shot.qc_status,
                attempts=shot.attempts,
                created_at=shot.created_at,
                updated_at=shot.updated_at
            )

    @staticmethod
    async def generate_scene_shots_breakdown(
        scene_id: str,
        total_scene_duration: float = 20.0,
        clip_duration: float = 6.0,
        base_prompt: str = "Cinematic shot",
        character_ids: Optional[List[str]] = None,
        location_id: Optional[str] = None
    ) -> List[ShotResponse]:
        """
        Automatically slices a long scene duration (e.g. 20s or 60s) into sequential 5-8s chained shots.
        """
        num_shots = max(1, math.ceil(total_scene_duration / clip_duration))
        created_shots: List[ShotResponse] = []
        prev_id = None
        
        for i in range(num_shots):
            dur = clip_duration if (i + 1) * clip_duration <= total_scene_duration else (total_scene_duration - (i * clip_duration))
            if dur < 3.0 and created_shots:
                # Append remainder duration to previous shot if remainder is too short
                break
            
            shot_payload = ShotCreate(
                scene_id=scene_id,
                prompt=f"{base_prompt} - Part {i+1}/{num_shots}",
                character_ids=character_ids or [],
                location_id=location_id,
                previous_shot_id=prev_id,
                duration=max(5.0, min(8.0, dur)),
                resolution="1280x720",
                camera_motion="dynamic tracking" if i > 0 else "static establishing",
                action=f"Narrative action beat {i+1}"
            )
            new_shot = await ShotService.create_shot(scene_id, shot_payload)
            created_shots.append(new_shot)
            prev_id = new_shot.shot_id
            
        logger.info(f"Auto-generated {len(created_shots)} chained shots for scene {scene_id} ({total_scene_duration}s total)")
        return created_shots

    @staticmethod
    async def build_5min_story_pipeline(
        project_id: str,
        scenes_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Builds an entire 5-minute production pipeline across multiple scenes and locations.
        Generates 50+ sequential 5-8s clips with frame chaining.
        """
        all_shots: List[ShotResponse] = []
        last_shot_id = None
        
        for scene_info in scenes_data:
            scene_id = scene_info["scene_id"]
            scene_dur = scene_info.get("duration", 30.0)
            base_prompt = scene_info.get("prompt", "Cinematic scene")
            char_ids = scene_info.get("character_ids", [])
            loc_id = scene_info.get("location_id")
            
            shots = await ShotService.generate_scene_shots_breakdown(
                scene_id=scene_id,
                total_scene_duration=scene_dur,
                clip_duration=6.0,
                base_prompt=base_prompt,
                character_ids=char_ids,
                location_id=loc_id
            )
            
            # Cross-scene frame chaining link
            if last_shot_id and shots and not shots[0].previous_shot_id:
                async with AsyncSessionLocal() as session:
                    stmt = select(ShotModel).where(ShotModel.shot_id == shots[0].shot_id)
                    res = await session.execute(stmt)
                    first_sh = res.scalar_one_or_none()
                    if first_sh:
                        first_sh.previous_shot_id = last_shot_id
                        await session.commit()
                        shots[0].previous_shot_id = last_shot_id
                        
            if shots:
                last_shot_id = shots[-1].shot_id
            all_shots.extend(shots)
            
        total_duration = sum(s.duration for s in all_shots)
        logger.info(f"5-Minute Pipeline built: {len(all_shots)} chained shots across {len(scenes_data)} scenes ({total_duration}s total)")
        return {
            "project_id": project_id,
            "total_shots": len(all_shots),
            "total_duration_seconds": total_duration,
            "shots": all_shots
        }

    @staticmethod
    async def list_shots(scene_id: str) -> List[ShotResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(ShotModel).where(ShotModel.scene_id == scene_id).order_by(ShotModel.sequence_order)
            result = await session.execute(stmt)
            shots = result.scalars().all()
            return [ShotResponse(
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
            ) for sh in shots]

    @staticmethod
    async def get_shot(shot_id: str) -> Optional[ShotResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(ShotModel).where(ShotModel.shot_id == shot_id)
            result = await session.execute(stmt)
            sh = result.scalar_one_or_none()
            if not sh:
                return None
            return ShotResponse(
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
            )


shot_service = ShotService()
