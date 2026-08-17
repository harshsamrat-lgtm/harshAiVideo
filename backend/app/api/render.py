"""
Render endpoints: Triggers master FFmpeg stitching, 1080p upscaling, and final video delivery.
"""
from fastapi import APIRouter, status
from app.models.schemas import FinalRenderRequest, FinalRenderResponse
from app.services.render_service import render_service

router = APIRouter(prefix="/render", tags=["Render & Assembly"])


@router.post("/final", response_model=FinalRenderResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_final_render(payload: FinalRenderRequest):
    """
    Trigger final FFmpeg assembly for the complete 5-minute project.
    Applies clip concatenation, audio sync, transitions, and 1080p master export.
    """
    return await render_service.trigger_final_render(payload)
