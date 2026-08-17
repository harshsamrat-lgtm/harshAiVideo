"""
Job endpoints: Asynchronous task tracking, queue polling, and job cancellation.
"""
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import JobResponse
from app.services.generation_service import generation_service

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Poll job status, progress percentage, ETA, and logs."""
    job = await generation_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return job


@router.post("/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_job(job_id: str):
    """Cancel a queued or running generation job."""
    success = await generation_service.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or already completed")
    return {"message": f"Job {job_id} cancelled successfully", "job_id": job_id}
