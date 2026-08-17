"""
Root level repository integrity test.
Verifies all required folders, configs, schemas, and entry points exist.
"""
import os
from pathlib import Path


def test_repository_structure():
    root = Path(__file__).resolve().parent.parent
    
    required_paths = [
        root / "backend" / "app" / "main.py",
        root / "backend" / "app" / "core" / "config.py",
        root / "backend" / "app" / "engines" / "base_engine.py",
        root / "backend" / "app" / "engines" / "lightx2v_engine.py",
        root / "backend" / "app" / "engines" / "wan22_engine.py",
        root / "workers" / "gpu_worker.py",
        root / "frontend" / "package.json",
        root / "docker" / "Dockerfile.backend",
        root / "docker" / "Dockerfile.worker",
        root / "docker" / "Dockerfile.frontend",
        root / "docker-compose.yml",
        root / ".env.example",
        root / ".gitignore",
        root / "README.md",
        root / "docs" / "ARCHITECTURE.md",
        root / "docs" / "DEPLOYMENT_GUIDE.md",
        root / "docs" / "REMOTE_GPU_TEST_PLAN.md",
        root / "docs" / "DEVELOPMENT_ROADMAP.md",
        root / "docs" / "API_SPECIFICATION.md",
    ]

    for p in required_paths:
        assert p.exists(), f"Missing required project file: {p}"
