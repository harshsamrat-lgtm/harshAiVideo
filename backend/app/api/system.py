"""
System & GPU telemetry endpoints: Real-time NVIDIA hardware metrics, VRAM, and health checks.
"""
from typing import List
from fastapi import APIRouter
from app.models.schemas import GPUTelemetry, SystemStatusResponse
from app.services.gpu_service import gpu_service

router = APIRouter(prefix="/system", tags=["System & Telemetry"])


@router.get("/gpu", response_model=List[GPUTelemetry])
async def get_gpu_metrics():
    """Retrieve live NVIDIA GPU telemetry (VRAM, temperature, power, utilization)."""
    return gpu_service.get_gpu_telemetry()


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Retrieve full system health, CPU, RAM, active jobs, and queue status."""
    return await gpu_service.get_system_status()
