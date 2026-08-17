# Harsh AI Video Studio - Deployment Guide

This guide details how to deploy the **exact same codebase** across three environments:
1. **Local Development (Laptop)**
2. **Rented GPU Server (Testing Phase)**
3. **Dedicated NVIDIA RTX 5090 Server (Production)**

---

## 💻 1. Local Development Setup (Laptop)

The laptop acts exclusively as the control and development plane. No AI model weights or NVIDIA GPUs are required.

```bash
# 1. Clone repository
git clone https://github.com/your-org/private-ai-video-studio.git
cd private-ai-video-studio

# 2. Configure environment
cp .env.example .env

# Set GPU_MODE=remote in .env
# Set APP_ENV=development in .env

# 3. Launch with Docker Compose
docker compose --profile dev up -d
```

Access:
- **Web UI**: `http://localhost:3000`
- **FastAPI Docs**: `http://localhost:8000/docs`

---

## ☁️ 2. Rented GPU Server Setup (Testing Node)

### Step 1: Base Host Configuration (Ubuntu 22.04 / 24.04)
```bash
# Install NVIDIA Drivers and Container Toolkit
sudo apt-get update && sudo apt-get install -y nvidia-driver-560 nvidia-container-toolkit
sudo systemctl restart docker
```

### Step 2: Clone & Configure
```bash
git clone https://github.com/your-org/private-ai-video-studio.git /opt/harsh-studio
cd /opt/harsh-studio
cp .env.example .env

# Configure .env:
# GPU_MODE=local
# ENGINE=lightx2v
# MODEL_ROOT=/opt/harsh-studio/models
```

### Step 3: Launch GPU Cluster
```bash
docker compose --profile remote-gpu up -d --build
```

---

## ⚡ 3. RTX 5090 Server Setup (Production Node)

The dedicated RTX 5090 (32GB VRAM Blackwell) runs the identical Docker stack:

```bash
# Launch full production stack with local GPU acceleration
docker compose --profile local-gpu up -d --build
```

### Verification:
```bash
# Verify NVFP4 and GPU telemetry
curl http://localhost:8000/api/system/gpu
```
