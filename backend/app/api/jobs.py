"""
Jobs API - Full CRUD + Live SSE streaming for real-time progress tracking.
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from app.models.schemas import JobResponse
from app.services.generation_service import generation_service
import asyncio, json, time

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("", response_model=list)
async def list_jobs():
    """List all jobs with current status and progress."""
    return await generation_service.list_jobs(limit=50)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Poll job status, progress percentage, and logs."""
    job = await generation_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return job


@router.get("/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """
    Server-Sent Events (SSE) stream for live job progress.
    Frontend polls this endpoint and receives real-time updates without page refresh.
    """
    async def event_generator():
        terminal_states = {"COMPLETED", "FAILED", "CANCELLED"}
        while True:
            job = await generation_service.get_job(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            payload = {
                "job_id": job.job_id,
                "status": job.status.value if hasattr(job.status, 'value') else job.status,
                "progress": job.progress,
                "result_url": job.result_url,
                "error_message": job.error_message,
                "logs": job.logs[-5:] if job.logs else [],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if payload["status"] in terminal_states:
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_job(job_id: str):
    """Cancel a queued or running generation job."""
    success = await generation_service.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or already completed")
    return {"message": f"Job {job_id} cancelled successfully", "job_id": job_id}
