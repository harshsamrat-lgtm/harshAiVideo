"""
Generation endpoints: Submits AI video generation jobs & handles reference image uploads.
"""
from fastapi import APIRouter, status, UploadFile, File, HTTPException
import shutil
import uuid
from pathlib import Path
from app.core.config import settings
from app.models.schemas import GenerationRequest, JobResponse
from app.services.generation_service import generation_service

router = APIRouter(prefix="/generate", tags=["Generation"])


@router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_generation(payload: GenerationRequest):
    """
    Enqueue an asynchronous Image-to-Video generation job.
    Dispatches to LightX2V / CogVideoX / Wan 2.2 GPU worker.
    """
    return await generation_service.submit_generation(payload)


@router.post("/upload-reference", tags=["Generation"])
async def upload_reference_image(file: UploadFile = File(...)):
    """
    Upload a reference image (Character / Actor / Location / Starting Frame).
    Returns the server file path and preview URL.
    """
    upload_dir = Path(settings.OUTPUT_ROOT) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "image.png").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".png"
    file_name = f"ref_{uuid.uuid4().hex[:10]}{ext}"
    dest_path = upload_dir / file_name
    
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "status": "success",
        "file_path": str(dest_path),
        "file_name": file_name,
        "file_url": f"/api/outputs/uploads/{file_name}"
    }
