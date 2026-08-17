"""
Harsh AI Video Studio - Asynchronous GPU Worker.
Listens on Redis / Async Queue, processes Video Diffusion generation via LightX2V / Wan 2.2,
executes automated Quality Control (QC), and manages auto-regeneration loops.
"""
import sys
import os
import time
import json
import signal
import asyncio
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.config import settings
from app.core.logging import logger
from app.engines import get_active_engine
from app.services.qc_service import qc_service
from app.services.queue_service import queue_service
from app.database.session import AsyncSessionLocal
from app.database.models import JobModel, ShotModel
from app.models.schemas import JobState, QCStatus
from sqlalchemy import select

RUNNING = True


def handle_shutdown(signum, frame):
    global RUNNING
    logger.info(f"Received termination signal ({signum}). Initiating graceful worker shutdown...")
    RUNNING = False


async def update_job_status(job_id: str, status: JobState, progress: float = 0.0, error_message: str = None, result_url: str = None):
    async with AsyncSessionLocal() as session:
        stmt = select(JobModel).where(JobModel.job_id == job_id)
        res = await session.execute(stmt)
        job = res.scalar_one_or_none()
        if job:
            job.status = status.value
            job.progress = progress
            if error_message:
                job.error_message = error_message
            if result_url:
                job.result_url = result_url
            job.logs = (job.logs or []) + [f"[{time.strftime('%X')}] State -> {status.value} (progress={progress}%)"]
            await session.commit()


async def run_worker():
    """Main worker loop for Redis queue polling and GPU inference execution."""
    global RUNNING
    
    # Register signal handlers
    try:
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info(f"Starting GPU Worker for {settings.APP_NAME}")
    logger.info(f"Queue: {settings.REDIS_QUEUE_NAME} | Engine: {settings.ENGINE}")
    logger.info("=" * 60)

    engine = get_active_engine()
    logger.info(f"Loading AI Video Engine: {engine.name}...")
    await engine.load_model()

    logger.info(f"GPU Worker ready and polling for tasks on Redis queue '{settings.REDIS_QUEUE_NAME}'...")

    while RUNNING:
        try:
            # 1. Poll next job from Queue (2-second timeout)
            job_payload = await queue_service.pop_job(settings.REDIS_QUEUE_NAME, timeout_seconds=2)
            if not job_payload:
                await asyncio.sleep(0.5)
                continue

            job_id = job_payload.get("job_id")
            shot_id = job_payload.get("shot_id")
            prompt = job_payload.get("prompt", "")
            duration = float(job_payload.get("duration", 5.0))
            resolution = job_payload.get("resolution", "1280x720")
            seed = int(job_payload.get("seed", -1))
            max_attempts = int(job_payload.get("max_attempts", 3))

            logger.info(f"Processing Job {job_id} for Shot {shot_id}: '{prompt[:45]}...' ({duration}s)")

            # 2. Update state to LOADING -> GENERATING
            await update_job_status(job_id, JobState.LOADING, progress=10.0)
            await asyncio.sleep(0.2)
            await update_job_status(job_id, JobState.GENERATING, progress=30.0)

            # 3. Execute Image-to-Video generation through BaseVideoEngine
            output_file = f"{settings.OUTPUT_ROOT}/{shot_id or job_id}_gen.mp4"
            gen_result = await engine.generate_image_to_video(
                prompt=prompt,
                duration_seconds=duration,
                resolution=resolution,
                seed=seed,
                output_path=output_file
            )

            await update_job_status(job_id, JobState.QC, progress=80.0)

            # 4. Automated Quality Control (QC)
            qc_report = await qc_service.evaluate_shot_video(
                shot_id=shot_id or job_id,
                video_path=output_file,
                expected_duration=duration,
                simulated_face_score=0.88,
                simulated_black_frames=0.001
            )

            if qc_report.status == QCStatus.PASS:
                await update_job_status(job_id, JobState.COMPLETED, progress=100.0, result_url=output_file)
                logger.info(f"Job {job_id} successfully COMPLETED! Output: {output_file}")
            else:
                logger.warning(f"Job {job_id} failed QC. Marking REGENERATING...")
                await update_job_status(job_id, JobState.REGENERATING, progress=50.0)
            
        except Exception as e:
            logger.error(f"Error in GPU worker execution loop: {e}", exc_info=True)
            await asyncio.sleep(1.0)

    logger.info("Unloading engine and releasing GPU memory...")
    await engine.unload_model()
    logger.info("GPU Worker terminated cleanly.")


if __name__ == "__main__":
    asyncio.run(run_worker())
