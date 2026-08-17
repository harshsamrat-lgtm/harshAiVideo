"""
Tests for generation submission, QC evaluation, and job lifecycle.
"""
import pytest
from fastapi.testclient import TestClient
from app.services.qc_service import qc_service
from app.models.schemas import QCStatus


def test_generation_job_submission(client: TestClient):
    payload = {
        "prompt": "Cinematic shot of hero standing in rain, neon city lights",
        "duration": 5.0,
        "resolution": "1280x720",
        "seed": 42,
        "engine": "lightx2v",
        "auto_qc": True,
        "auto_retry": True
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 202
    data = response.json()
    job_id = data["job_id"]
    assert data["status"] == "QUEUED"
    assert data["engine"] == "lightx2v"

    # Poll status
    job_res = client.get(f"/api/jobs/{job_id}")
    assert job_res.status_code == 200
    assert job_res.json()["job_id"] == job_id


@pytest.mark.asyncio
async def test_automated_qc_service():
    report = await qc_service.evaluate_shot_video(
        shot_id="shot_test_101",
        video_path="/workspace/outputs/shot_101.mp4",
        expected_duration=5.0
    )
    assert report.status == QCStatus.PASS
    assert report.duration_actual == 5.0
    assert report.black_frames_pct <= 0.01
    assert report.face_similarity_score >= 0.75
