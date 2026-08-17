"""
Lip-sync endpoints: Executes character mouth synchronization against generated audio.
"""
from fastapi import APIRouter, status
from app.models.schemas import LipSyncRequest, LipSyncResponse
from app.services.lipsync_service import lipsync_service

router = APIRouter(prefix="/lipsync", tags=["Lip-Sync"])


@router.post("", response_model=LipSyncResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_lipsync(payload: LipSyncRequest):
    """Enqueue lip-sync task for video clip and character audio."""
    return await lipsync_service.process_lipsync(payload)
