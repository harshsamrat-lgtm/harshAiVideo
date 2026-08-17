"""
Project endpoints: CRUD operations for video production projects.
"""
from typing import List
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import ProjectCreate, ProjectResponse
from app.services.project_service import project_service

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(payload: ProjectCreate):
    """Create a new long-form video studio project."""
    return await project_service.create_project(payload)


@router.get("", response_model=List[ProjectResponse])
async def list_projects():
    """List all projects in the studio."""
    return await project_service.list_projects()


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    """Retrieve details for a specific project."""
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project
