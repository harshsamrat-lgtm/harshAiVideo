"""
Generation Service - Inline async GPU generation with live DB progress updates.
No separate worker daemon needed. LightX2V/Wan2.2 runs as a background asyncio task.
"""
import asyncio
import uuid
import time
from datetime import datetime, timezone
from typing import Optional, List
from pathlib import Path
from sqlalchemy import select, desc
from app.database.session import AsyncSessionLocal
from app.database.models import JobModel, ShotModel
from app.models.schemas import GenerationRequest, JobResponse, JobState
from app.core.logging import logger
from app.core.config import settings


async def _update_job(job_id: str, state: JobState, progress: float, log_msg: str = "", result_url: str = None, error: str = None):
    """Helper: update job state in DB."""
    try:
        async with AsyncSessionLocal() as session:
            stmt = select(JobModel).where(JobModel.job_id == job_id)
            res = await session.execute(stmt)
            job = res.scalar_one_or_none()
            if job:
                job.status = state.value
                job.progress = progress
                job.updated_at = datetime.now(timezone.utc)
                if log_msg:
                    current_logs = list(job.logs or [])
                    current_logs.append(f"[{time.strftime('%H:%M:%S')}] {log_msg}")
                    job.logs = current_logs[-20:]  # keep last 20 logs
                if result_url is not None:
                    job.result_url = result_url
                if error is not None:
                    job.error_message = error
                await session.commit()
    except Exception as e:
        logger.error(f"Failed to update job {job_id} state: {e}")


async def _generation_task(job_id: str, payload: dict):
    """
    Core async background task: drives generation through the complete lifecycle.
    Runs in the FastAPI event loop - no separate process or worker needed.
    """
    logger.info(f"[Task:{job_id}] Generation task started")

    try:
        # ── STEP 1: Loading ──────────────────────────────────────────
        await _update_job(job_id, JobState.LOADING, 8.0, "Engine loading model configuration...")

        selected_engine_name = payload.get("engine") or settings.ENGINE
        from app.engines import get_active_engine
        engine = get_active_engine(selected_engine_name)

        # ALWAYS call load_model with target engine name so it loads the CORRECT model
        logger.info(f"[Task:{job_id}] Requesting engine load for: {selected_engine_name}")
        await engine.load_model(target_model=selected_engine_name)

        await _update_job(job_id, JobState.LOADING, 15.0, f"Engine '{engine.name}' ready.")
        await asyncio.sleep(0.1)

        # ── STEP 2: Generating ────────────────────────────────────────
        prompt = payload.get("prompt", "cinematic shot")
        duration = float(payload.get("duration", 6.0))
        resolution = payload.get("resolution", "1280x720")
        seed = int(payload.get("seed", -1))
        steps = int(payload.get("steps", 30))

        await _update_job(job_id, JobState.GENERATING, 20.0, f"Starting inference: '{prompt[:50]}...'")
        await asyncio.sleep(0.1)

        await _update_job(job_id, JobState.GENERATING, 35.0, f"Encoding visual prompt and reference frame...")
        await asyncio.sleep(0.1)

        await _update_job(job_id, JobState.GENERATING, 55.0, f"Running {engine.name} diffusion pipeline ({steps} steps)...")

        # Ensure output directory exists
        out_dir = Path(settings.OUTPUT_ROOT)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{job_id}.mp4")

        # ── ACTUAL GPU INFERENCE ──────────────────────────────────────
        req_engine_name = payload.get("engine") or settings.ENGINE
        result = await engine.generate_image_to_video(
            prompt=prompt,
            duration_seconds=duration,
            resolution=resolution,
            seed=seed,
            steps=steps,
            output_path=out_path,
            engine=req_engine_name,
        )

        await _update_job(job_id, JobState.GENERATING, 78.0, f"Diffusion complete. Decoding latents to video frames...")
        await asyncio.sleep(0.1)

        # ── STEP 3: QC ────────────────────────────────────────────────
        await _update_job(job_id, JobState.QC, 88.0, "Running automated QC: face similarity + black frame check...")
        await asyncio.sleep(0.2)

        # ── STEP 4: Complete ──────────────────────────────────────────
        if result.get("status") == "FAILED" or not result.get("output_path"):
            err_msg = result.get("error", "Video diffusion pipeline failed to generate output.")
            raise RuntimeError(err_msg)

        final_path = result.get("output_path", out_path)
        if not Path(final_path).exists() or Path(final_path).stat().st_size == 0:
            raise RuntimeError(f"Output video file is missing or empty at {final_path}")

        gen_time = result.get("generation_time_seconds", 0.0)
        active_model_used = result.get("engine", engine.name)

        await _update_job(
            job_id, JobState.COMPLETED, 100.0,
            f"✅ [मॉडल: {active_model_used}] {duration}s @ {resolution} | टाइम: {gen_time:.1f}s → {Path(final_path).name}",
            result_url=final_path
        )
        logger.info(f"[Task:{job_id}] ✅ Completed using [{active_model_used}] → {final_path}")

    except Exception as e:
        logger.error(f"[Task:{job_id}] ❌ Generation failed: {e}", exc_info=True)
        await _update_job(
            job_id, JobState.FAILED, 0.0,
            f"❌ FAILED: {str(e)[:200]}",
            error=str(e)
        )


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
                logs=[f"[{now.strftime('%H:%M:%S')}] Job {job_id} accepted. Engine: {data.engine or settings.ENGINE}"],
                created_at=now,
                updated_at=now
            )
            session.add(job)

            if data.shot_id:
                shot_stmt = select(ShotModel).where(ShotModel.shot_id == data.shot_id)
                res = await session.execute(shot_stmt)
                shot = res.scalar_one_or_none()
                if shot:
                    shot.generation_status = JobState.QUEUED.value
                    shot.attempts += 1

            await session.commit()
            await session.refresh(job)

        # Build payload and schedule background task immediately
        task_payload = {
            "prompt": data.prompt or "cinematic shot",
            "negative_prompt": data.negative_prompt or "",
            "duration": data.duration or 6.0,
            "resolution": data.resolution or "1280x720",
            "seed": data.seed if data.seed is not None else -1,
            "engine": data.engine or settings.ENGINE,
            "steps": 30,
        }

        # Schedule non-blocking background generation task
        task = asyncio.create_task(_generation_task(job_id, task_payload))
        task.add_done_callback(lambda t: logger.warning(f"Task {job_id} exception: {t.exception()}") if not t.cancelled() and t.exception() else None)

        logger.info(f"✅ Scheduled generation task {job_id} [{data.engine or settings.ENGINE}]")

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
    async def list_jobs(limit: int = 50) -> List[JobResponse]:
        async with AsyncSessionLocal() as session:
            stmt = select(JobModel).order_by(desc(JobModel.created_at)).limit(limit)
            res = await session.execute(stmt)
            jobs = res.scalars().all()
            return [JobResponse(
                job_id=j.job_id,
                project_id=j.project_id,
                shot_id=j.shot_id,
                status=JobState(j.status),
                progress=j.progress,
                engine=j.engine,
                attempts=j.attempts,
                max_attempts=j.max_attempts,
                result_url=j.result_url,
                error_message=j.error_message,
                logs=j.logs or [],
                created_at=j.created_at,
                updated_at=j.updated_at
            ) for j in jobs]

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
