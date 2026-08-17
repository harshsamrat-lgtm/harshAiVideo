"""
Generation service orchestrating prompt synthesis, job dispatching, and queue dispatch with DB persistence.
"""
from typing import Dict, Any, Optional
import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from app.database.session import AsyncSessionLocal
from app.database.models import JobModel, ShotModel
from app.models.schemas import GenerationRequest, JobResponse, JobState
from app.services.queue_service import queue_service
from app.core.logging import logger
from app.core.config import settings


class GenerationService:
    @staticmethod
    async def submit_generation(data: GenerationRequest) -> JobResponse:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        
        async with AsyncSessionLocal() as session:
            job = JobModel(
                job_id=job_id,
                project_id=data.project_id,
                shot_id=data.shot_id,
                status=JobState.QUEUED.value,
                progress=0.0,
                engine=data.engine or settings.ENGINE,
                attempts=1,
                max_attempts=settings.MAX_REGENERATION_ATTEMPTS if data.auto_retry else 1,
                result_url=None,
                error_message=None,
                logs=[f"Job enqueued at {now.isoformat()}"],
                created_at=now,
                updated_at=now
            )
            session.add(job)
            
            # If linked to a specific shot, update shot status
            if data.shot_id:
                shot_stmt = select(ShotModel).where(ShotModel.shot_id == data.shot_id)
                res = await session.execute(shot_stmt)
                shot = res.scalar_one_or_none()
                if shot:
                    shot.generation_status = JobState.QUEUED.value
                    shot.attempts += 1
            
            await session.commit()
            await session.refresh(job)
            
            # Dispatch to Redis / Worker Queue
            job_payload = {
                "job_id": job.job_id,
                "project_id": data.project_id,
                "shot_id": data.shot_id,
                "prompt": data.prompt,
                "negative_prompt": data.negative_prompt,
                "duration": data.duration,
                "resolution": data.resolution,
                "seed": data.seed,
                "engine": job.engine,
                "auto_qc": data.auto_qc,
                "auto_retry": data.auto_retry,
                "created_at": now.isoformat()
            }
            await queue_service.enqueue_job(settings.REDIS_QUEUE_NAME, job_payload)
            
            logger.info(f"Enqueued generation job {job_id} to queue '{settings.REDIS_QUEUE_NAME}' using engine '{data.engine}'")
            return JobResponse(
                job_id=job.job_id,
                project_id=job.project_id,
                shot_id=job.shot_id,
                status=JobState(job.status),
                progress=job.progress,
                engine=job.engine,
                attempts=job.attempts,
                max_attempts=job.max_attempts,
                result_url=job.result_url,
                error_message=job.error_message,
                logs=job.logs or [],
                created_at=job.created_at,
                updated_at=job.updated_at
            )

    @staticmethod
    async def get_job(job_id: str) -> Optional[JobResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(JobModel).where(JobModel.job_id == job_id)
            res = await session.execute(stmt)
            job = res.scalar_one_or_none()
            if not job:
                return None
            return JobResponse(
                job_id=job.job_id,
                project_id=job.project_id,
                shot_id=job.shot_id,
                status=JobState(job.status),
                progress=job.progress,
                engine=job.engine,
                attempts=job.attempts,
                max_attempts=job.max_attempts,
                result_url=job.result_url,
                error_message=job.error_message,
                logs=job.logs or [],
                created_at=job.created_at,
                updated_at=job.updated_at
            )

    @staticmethod
    async def cancel_job(job_id: str) -> bool:
        async with AsyncSessionLocal() as session:
            stmt = select(JobModel).where(JobModel.job_id == job_id)
            res = await session.execute(stmt)
            job = res.scalar_one_or_none()
            if not job:
                return False
            job.status = JobState.CANCELLED.value
            job.updated_at = datetime.now(timezone.utc)
            await session.commit()
            logger.info(f"Cancelled job {job_id}")
            return True


generation_service = GenerationService()
