# Harsh AI Video Studio - Architecture Specification

## 1. Executive Summary

**Harsh AI Video Studio** is an enterprise-grade, distributed AI video creation platform engineered for generating long-form (up to 5-minute), narrative videos with consistent characters, environments, voices, and lip-synchronized dialogue.

The platform is designed around the **Wan 2.2 I2V A14B** diffusion model accelerated by **LightX2V (NVFP4 quantization, sparse attention, and Blackwell kernel fusion)**.

---

## 2. Distributed Execution Plane

The system uses a decoupled control-plane / execution-plane topology:

```
+-------------------------------------------------------------------------+
|                  CONTROL PLANE (Antigravity Dev Laptop)                 |
|   - Next.js Web UI & FastAPI Gateway                                    |
|   - Relational Database (SQLite / PostgreSQL)                           |
|   - Redis Broker & Job Dispatcher                                       |
|   - ZERO Model Weights / ZERO GPU Inference                             |
+-------------------------------------------------------------------------+
                                    |
                                    | Secure API / Redis Queue
                                    v
+-------------------------------------------------------------------------+
|                   EXECUTION PLANE (Rented GPU / RTX 5090)               |
|   - Asynchronous Python GPU Worker (`workers/gpu_worker.py`)            |
|   - LightX2V Engine (NVFP4 + Sparse Attention)                          |
|   - Automated Quality Control (QC) & Auto-Regeneration (max 3 tries)    |
|   - FFmpeg Video Assembly, Transitions, & 1080p Upscaling               |
+-------------------------------------------------------------------------+
```

---

## 3. Core Engine Abstraction (`BaseVideoEngine`)

To insulate business logic from underlying video diffusion implementations, all model pipelines derive from `BaseVideoEngine`:

```python
class BaseVideoEngine(ABC):
    @abstractmethod
    async def load_model(self) -> bool: ...
    
    @abstractmethod
    async def unload_model(self) -> bool: ...
    
    @abstractmethod
    async def generate_image_to_video(
        self,
        prompt: str,
        reference_image_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        duration_seconds: float = 5.0,
        resolution: str = "1280x720",
        seed: int = -1,
        **kwargs
    ) -> Dict[str, Any]: ...
```

### Implementations:
1. `LightX2VEngine`: Target production engine on RTX 5090 (32GB) utilizing NVFP4 quantization and sparse attention.
2. `Wan22Engine`: Reference standard implementation for baseline comparison.

---

## 4. Multi-Character & Multi-Location Bible

### Character Bible
- **Identity & Likeness**: Master face reference, body description, hair, wardrobe, accessories.
- **Voice Profile**: Dedicated voice identifier, TTS pitch/speed, authorized sample reference.
- **LoRA Adapter**: Optional character-specific LoRA weights.

### Location Bible
- **Environment**: Architectural features, terrain, lighting ambiance, weather, time of day.
- **Camera Style**: Default focal length, color grading, perspective framing.

### Chaining Continuity
Each shot in a scene leverages:
$$\text{Input} = \text{Master Character Reference} + \text{Previous Shot Last Frame} + \text{Location Context} + \text{Shot Prompt} + \text{Seed}$$

---

## 5. Automated Quality Control (QC) & Regeneration Loop

Every generated shot clip passes through automated evaluation:
1. **File Integrity**: Codec validation, frame count, corrupt block detection.
2. **Black Frame Detection**: Fails if black frame ratio exceeds 1%.
3. **Duration Accuracy**: Tolerance $\pm 0.1$s against target duration.
4. **Face Identity Consistency**: Evaluates cosine similarity of facial embeddings against the Master Character Reference.

If any check **FAILS**, the worker automatically re-samples with an adjusted seed up to `MAX_REGENERATION_ATTEMPTS` (default: 3).
