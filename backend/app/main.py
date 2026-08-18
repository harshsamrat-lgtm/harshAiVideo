"""
Harsh AI Video Studio - FastAPI Application Entry Point.
"""
from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path

# Ensure backend root is always in Python module search path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from app.core.config import settings
from app.core.logging import logger
from app.api import api_router
from app.database import init_db

STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown event management."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.APP_ENV} | GPU Mode: {settings.GPU_MODE}")
    logger.info(f"Active AI Video Engine: {settings.ENGINE}")
    logger.info("=" * 60)

    # Ensure required storage directories exist
    for directory in [settings.MODEL_ROOT, settings.PROJECT_ROOT, settings.OUTPUT_ROOT, settings.CACHE_ROOT]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Initialize SQLite / PostgreSQL Database tables
    logger.info("Initializing async database schemas...")
    await init_db()
    logger.info("Database schemas initialized successfully.")
    
    yield
    
    logger.info(f"Shutting down {settings.APP_NAME}...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise Private AI Video Studio for Wan 2.2 I2V A14B + LightX2V NVFP4.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if settings.cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System & Telemetry"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.APP_ENV,
        "gpu_mode": settings.GPU_MODE,
        "engine": settings.ENGINE
    }


@app.get("/", tags=["Studio UI"], response_class=HTMLResponse)
async def studio_ui():
    """Harsh AI Video Studio Web Dashboard."""
    if STATIC_INDEX.exists():
        return HTMLResponse(content=STATIC_INDEX.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Harsh AI Video Studio Backend Online</h1><p><a href='/docs'>Swagger API Docs</a></p>")


# Include all modular API routers under /api
app.include_router(api_router)


# Video Output file server - serves generated MP4 clips directly to browser/player
from fastapi.responses import FileResponse
from fastapi import HTTPException as FastHTTPException

@app.get("/api/outputs/{filename:path}", tags=["Outputs"])
async def serve_output_file(filename: str):
    """Serve a generated video or uploaded reference image."""
    output_path = Path(settings.OUTPUT_ROOT) / filename
    if not output_path.exists():
        raise FastHTTPException(status_code=404, detail=f"File '{filename}' not found.")
    media_type = "video/mp4"
    if output_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
        media_type = f"image/{output_path.suffix.lower().replace('.', '')}"
    return FileResponse(str(output_path), media_type=media_type)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred in video studio backend."}
    )
