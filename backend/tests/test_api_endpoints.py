"""
Tests for core FastAPI endpoints and API contract compliance.
"""
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Harsh AI Video Studio" in data["app"]


def test_system_status(client: TestClient):
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert len(data["gpus"]) > 0


def test_gpu_metrics(client: TestClient):
    response = client.get("/api/system/gpu")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "NVIDIA" in data[0]["gpu_name"]


def test_project_crud(client: TestClient):
    # 1. Create project
    payload = {
        "name": "Cyberpunk Neon Chronicle",
        "description": "5-minute cyberpunk short story",
        "target_duration": 300.0,
        "resolution": "1920x1080",
        "fps": 24
    }
    create_res = client.post("/api/projects", json=payload)
    assert create_res.status_code == 201
    created_data = create_res.json()
    project_id = created_data["project_id"]
    assert created_data["name"] == payload["name"]

    # 2. List projects
    list_res = client.get("/api/projects")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    # 3. Get single project
    get_res = client.get(f"/api/projects/{project_id}")
    assert get_res.status_code == 200
    assert get_res.json()["project_id"] == project_id
