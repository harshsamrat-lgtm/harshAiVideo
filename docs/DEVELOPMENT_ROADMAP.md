# Harsh AI Video Studio - Master Development Roadmap (Phases 1 - 24)

This roadmap governs the entire development and deployment lifecycle. Work is executed strictly one phase at a time with rigorous testing and verification before advancing.

---

### Part I: Software Architecture & Local Development Plane
- **PHASE 1: Repository Structure & Foundation** *(Current)*
  - Directory skeleton, Docker configurations, README, .env.example, gitignore, docs.
- **PHASE 2: FastAPI Backend Core**
  - Robust exception handling, CORS, async middleware, healthchecks.
- **PHASE 3: Database Models & Migrations**
  - SQLAlchemy async models for Character Bible, Location Bible, Scenes, Shots, Projects, Jobs.
- **PHASE 4: Character Bible System**
  - Permanent character registry, likeness embeddings, wardrobe, voice linkage, LoRAs.
- **PHASE 5: Location Bible System**
  - Environment templates, lighting setups, architectural style definitions.
- **PHASE 6: Scene & Shot Engine**
  - Multi-character assignment, shot continuity framing, duration rules.
- **PHASE 7: Job Queue & Telemetry**
  - Redis broker integration, async job submission, cancellation, progress streaming.
- **PHASE 8: AI Engine Abstraction**
  - BaseVideoEngine interface contracts, LightX2V and Wan 2.2 modular wrappers.
- **PHASE 9: Docker Multi-Profile Configuration**
  - Docker Compose configs for dev, remote-gpu, and local-gpu.
- **PHASE 10: Web UI Dashboard**
  - Complete Next.js / React studio interface with live GPU monitor and shot planner.
- **PHASE 11: Comprehensive Software Test Suite**
  - Automated integration tests, mocking remote GPU responses, verifying end-to-end API flows.

---

### Part II: Remote GPU Cluster Testing & Model Optimization
- **PHASE 12: Remote GPU Deployment Package**
  - Bundled scripts, environment configs, automated dependency installers for cloud nodes.
- **PHASE 13: Rented GPU Connection & Handshake**
  - Host configuration, NVIDIA Container Toolkit verification, SSH handshake.
- **PHASE 14: Wan 2.2 + LightX2V Installation**
  - Model weights ingestion, NVFP4 quantization verification, sparse attention ops.
- **PHASE 15: First Real 5s Video Generation**
  - Single-shot test inference, latency profiling, baseline quality evaluation.
- **PHASE 16: Character Consistency Engine Test**
  - Automated face similarity checks, consecutive shot likeness retention.
- **PHASE 17: Multi-Character Scene Test**
  - 2 to 3 characters in simultaneous interaction within a single shot.
- **PHASE 18: Multi-Location Scene Test**
  - Sequential scene transitions across diverse environment bibles.
- **PHASE 19: Character Voice Consistency Test**
  - Multi-character TTS dialogue generation with fixed acoustic profiles.
- **PHASE 20: Lip-Sync Pipeline Test**
  - Audio-driven phoneme mouth alignment with video generation clips.

---

### Part III: Long-Form Video Assembly & Production Hardening
- **PHASE 21: 30-Second Video Pipeline**
  - Multi-shot concatenation, transitions, and audio sync.
- **PHASE 22: 1-Minute Video Pipeline**
  - Full multi-scene story assembly with BGM and sound effects.
- **PHASE 23: 5-Minute Master Production**
  - End-to-end 60-shot video generation with automated QC, auto-regeneration, and 1080p final master export.
- **PHASE 24: RTX 5090 Server Production Migration**
  - Deploy final verified container stack to dedicated NVIDIA RTX 5090 server.
