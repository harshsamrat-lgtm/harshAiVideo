"""
Generation endpoints: Submits AI video generation jobs to Redis queue / GPU engine.
"""
from fastapi import APIRouter, status
from app.models.schemas import GenerationRequest, JobResponse
from app.services.generation_service import generation_service

router = APIRouter(prefix="/generate", tags=["Generation"])


@router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_generation(payload: GenerationRequest):
    """
    Enqueue an asynchronous Image-to-Video generation job.
    Dispatches to LightX2V / Wan 2.2 GPU worker.
    """
    return await generation_service.submit_generation(payload)
