# Harsh AI Video Studio (Private AI Video Studio)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg?style=flat&logo=python)](https://www.python.org)
[![Wan 2.2](https://img.shields.io/badge/Model-Wan%202.2%20I2V%20A14B-FF6B6B.svg)](https://github.com/Wan-Video/Wan2.1)
[![LightX2V](https://img.shields.io/badge/Acceleration-LightX2V%20NVFP4-76B900.svg?logo=nvidia)](https://github.com/ModelTC/LightX2V)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED.svg?logo=docker)](https://www.docker.com)

**Harsh AI Video Studio** is an enterprise-grade, private AI video studio infrastructure engineered for long-form, multi-character, multi-location consistent video production (5-minute 1080p final videos). 

The platform leverages **Wan 2.2 I2V A14B** accelerated by **LightX2V (NVFP4, Sparse Attention, RTX Blackwell / 50-series optimizations)** with an automated continuity pipeline, character bible, location bible, automated quality control (QC), voice synthesis, lip-synchronization, and FFmpeg assembly.

---

## 🏛 Architecture & Deployment Topology

```
+-----------------------------------------------------------------------------------+
|                        ANTIGRAVITY / LAPTOP (CONTROL PLANE)                       |
|   - Web UI (Next.js / React)                                                      |
|   - FastAPI Backend (Project/Scene/Shot/Character/Location API)                   |
|   - Redis Job Broker & Database (SQLite/PostgreSQL)                               |
|   - NO AI Model weights or heavy GPU loads on laptop                              |
+-----------------------------------------------------------------------------------+
                                         |
                                         | Git Push / Authenticated API
                                         v
+-----------------------------------------------------------------------------------+
|                     RENTED GPU SERVER (REAL AI TESTING PHASE)                     |
|   - Ubuntu + NVIDIA Drivers + NVIDIA Container Toolkit                            |
|   - LightX2V Engine + Wan 2.2 I2V A14B (NVFP4 + Sparse Attention)                 |
|   - 17-Step Systematic GPU Testing Plan                                           |
|   - Automated QC & Regeneration Workers                                           |
+-----------------------------------------------------------------------------------+
                                         |
                                         | Verified & Hardened
                                         v
+-----------------------------------------------------------------------------------+
|                   DEDICATED NVIDIA RTX 5090 SERVER (PRODUCTION)                   |
|   - Identical codebase & Docker containers                                        |
|   - Blackwell NVFP4 acceleration for ultra-fast 5-minute video workflows          |
+-----------------------------------------------------------------------------------+
```

---

## 🎬 5-Minute Video Assembly Pipeline

Rather than risking single-pass hallucination, long videos are structured hierarchically:

```
SCRIPT / STORY
      ↓
SCENES (1 to N with linked Characters & Locations)
      ↓
SHOTS (5 to 10 second manageable clips)
      ↓
I2V GENERATION (Wan 2.2 + LightX2V NVFP4 with Character/Location/Frame Reference)
      ↓
AUTOMATIC QUALITY CONTROL (QC: Black frame, face similarity, duration, corruption)
      ↓ (Auto-Regenerate on Fail, max 3 attempts)
VOICE SYNTHESIS (Consistent Character Voice Profiles)
      ↓
LIP-SYNC GENERATION (Audio-Driven Lip Sync)
      ↓
UPSCALING & FFMPEG ASSEMBLY (Transitions, BGM, Audio Sync -> 1080p Final Video)
```

---

## 🚀 Quick Start (Development Laptop)

### 1. Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose (optional for local Redis)

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/private-ai-video-studio.git
cd private-ai-video-studio

# Copy environment template
cp .env.example .env

# Create python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 3. Launch Backend
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation available at: `http://localhost:8000/docs`

---

## 📁 Repository Structure

```
private-ai-video-studio/
├── backend/
│   ├── app/
│   │   ├── api/             # REST Endpoints (projects, characters, scenes, shots, etc.)
│   │   ├── core/            # Configuration, security, logging
│   │   ├── database/        # Async DB session, engine, model base
│   │   ├── engines/         # BaseVideoEngine, Wan22Engine, LightX2VEngine
│   │   ├── models/          # Pydantic schemas & SQLAlchemy ORM models
│   │   ├── services/        # Business logic (character, location, QC, render, etc.)
│   │   └── main.py          # FastAPI application entry point
│   ├── requirements.txt
│   └── tests/               # Backend unit and integration tests
├── workers/
│   └── gpu_worker.py        # Asynchronous Redis GPU worker
├── frontend/                # Next.js / React Web UI
├── docker/                  # Dockerfiles & deployment compose configurations
├── docs/                    # Architecture, Deployment Guide, Test Plan, Roadmap
├── projects/                # Persistent project data (ignored by git)
├── models/                  # AI model checkpoints (ignored by git)
├── outputs/                 # Final rendered videos & clips (ignored by git)
├── cache/                   # Intermediate generation cache (ignored by git)
├── scripts/                 # Utility scripts & healthchecks
├── .env.example
├── .gitignore
├── docker-compose.yml
└── README.md
```

---

## 🗺 24-Phase Development Roadmap

| Phase | Milestone | Description | Status |
|---|---|---|---|
| **Phase 1** | **Repository Structure & Skeleton** | Foundations, Docker, Docs, Architecture | 🟢 Completed |
| Phase 2 | FastAPI Backend Core | Full routing, middleware, error handling | ⏳ Next |
| Phase 3 | Database Models & Migrations | Character, Location, Scene, Shot, Job schema | ⏳ Pending |
| Phase 4 | Character Bible System | Identity references, clothing, voice profiles | ⏳ Pending |
| Phase 5 | Location Bible System | Environments, architecture, lighting styles | ⏳ Pending |
| Phase 6 | Scene & Shot Engine | Multi-character assignment, frame continuity | ⏳ Pending |
| Phase 7 | Job Queue & Telemetry | Redis queue, async dispatch, state tracking | ⏳ Pending |
| Phase 8 | Engine Abstraction | BaseVideoEngine, LightX2V & Wan2.2 interfaces | ⏳ Pending |
| Phase 9 | Docker Containers | Dev, Remote-GPU, Local-GPU Compose profiles | ⏳ Pending |
| Phase 10 | Web UI Dashboard | Complete responsive studio interface | ⏳ Pending |
| Phase 11 | Software Test Suite | End-to-end API, Queue, and Service tests | ⏳ Pending |
| Phase 12 | Remote Deployment Package | Tarball/scripts ready for Rented GPU server | ⏳ Pending |
| Phase 13 | Rented GPU Server Connect | SSH handshake, NVML verification, Docker setup | ⏳ Pending |
| Phase 14 | Model Ingestion & LightX2V | Wan 2.2 + LightX2V NVFP4 setup on GPU | ⏳ Pending |
| Phase 15 | First 5s Real I2V Test | Initial 5-second video generation on GPU | ⏳ Pending |
| Phase 16 | Character Consistency Test | Face similarity and visual continuity checks | ⏳ Pending |
| Phase 17 | Multi-Character Test | Simultaneous multi-character scenes | ⏳ Pending |
| Phase 18 | Multi-Location Test | Location switching across scenes | ⏳ Pending |
| Phase 19 | Voice Consistency Test | Multi-character TTS synthesis | ⏳ Pending |
| Phase 20 | Lip-Sync Test | Audio-driven mouth movement synchronization | ⏳ Pending |
| Phase 21 | 30-Second Video Test | Multi-shot sequential assembly | ⏳ Pending |
| Phase 22 | 1-Minute Video Test | Full scene transitions & audio mixing | ⏳ Pending |
| Phase 23 | 5-Minute Video Test | Complete 60-shot production pipeline run | ⏳ Pending |
| Phase 24 | RTX 5090 Hardening | Final migration & optimization for RTX 5090 | ⏳ Pending |

---

## 🛡 License & Safety
Private and confidential. Built for proprietary AI video production workflows. Ensure all voice profiles and character likenesses possess proper authorized usage rights.
