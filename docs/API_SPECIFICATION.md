# Harsh AI Video Studio - REST API Specification

Base Path: `/api`  
Interactive Swagger UI: `http://localhost:8000/docs`

---

## 1. Projects
- `POST /api/projects`: Create a new video studio project.
- `GET /api/projects`: List all studio projects.
- `GET /api/projects/{project_id}`: Retrieve full project details, including linked characters, locations, and scenes.

---

## 2. Character Bible
- `POST /api/characters`: Register a permanent character (appearance, face reference, hair, clothing, voice profile, LoRA).
- `GET /api/characters`: List registered characters.
- `GET /api/characters/{character_id}`: Get specific character profile.

---

## 3. Location Bible
- `POST /api/locations`: Register a permanent location (architecture, lighting, weather, time of day).
- `GET /api/locations`: List all locations.
- `GET /api/locations/{location_id}`: Get specific location details.

---

## 4. Scenes & Shots
- `POST /api/projects/{project_id}/scenes`: Create a new scene with assigned characters and location.
- `GET /api/projects/{project_id}/scenes`: List scenes in a project.
- `GET /api/scenes/{scene_id}`: Get scene details.
- `POST /api/scenes/{scene_id}/shots`: Create a granular 5-10 second shot clip.
- `GET /api/scenes/{scene_id}/shots`: List shots for a scene.
- `GET /api/shots/{shot_id}`: Get single shot details.

---

## 5. Generation & Job Queue
- `POST /api/generate`: Enqueue an asynchronous Image-to-Video generation job.
- `GET /api/jobs/{job_id}`: Poll job progress percentage, state (`QUEUED`, `GENERATING`, `QC`, `COMPLETED`, `FAILED`), and logs.
- `POST /api/jobs/{job_id}/cancel`: Abort a queued or executing job.

---

## 6. Voice, Lip-Sync, & Render
- `POST /api/voices`: Create a character voice profile.
- `GET /api/voices`: List available voice profiles.
- `POST /api/lipsync`: Dispatch video and audio lip-sync task.
- `POST /api/render/final`: Trigger FFmpeg master assembly for complete 5-minute video export (1080p).

---

## 7. System & GPU Telemetry
- `GET /api/system/gpu`: Real-time NVIDIA GPU telemetry (VRAM used/total, GPU utilization %, temperature, power draw).
- `GET /api/system/status`: Full host metrics (CPU, RAM, active jobs, queue length).
- `GET /health`: Fast service health check.
