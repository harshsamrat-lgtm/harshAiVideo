"""
API package router registration.
"""
from fastapi import APIRouter
from app.api.projects import router as projects_router
from app.api.characters import router as characters_router
from app.api.locations import router as locations_router
from app.api.scenes import router as scenes_router
from app.api.shots import router as shots_router
from app.api.generation import router as generation_router
from app.api.jobs import router as jobs_router
from app.api.system import router as system_router
from app.api.voices import router as voices_router
from app.api.lipsync import router as lipsync_router
from app.api.render import router as render_router

api_router = APIRouter(prefix="/api")
api_router.include_router(projects_router)
api_router.include_router(characters_router)
api_router.include_router(locations_router)
api_router.include_router(scenes_router)
api_router.include_router(shots_router)
api_router.include_router(generation_router)
api_router.include_router(jobs_router)
api_router.include_router(system_router)
api_router.include_router(voices_router)
api_router.include_router(lipsync_router)
api_router.include_router(render_router)

__all__ = ["api_router"]
