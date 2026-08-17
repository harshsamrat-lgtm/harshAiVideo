"""
Harsh AI Video Studio - Master Test Suite (Phases 1 through 12).
Validates Database ORM, LightX2V NVFP4 & Wan 2.2 Deep Engine Abstractions,
Prompt Synthesizer, 5-Minute 50-Shot Slicing, Redis Job Queue, GPU Worker Loop,
QC Auto-Regeneration, and Remote GPU Deployment Package.
"""
import sys
import unittest
import asyncio
from pathlib import Path

# Setup paths
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db
from app.services.project_service import project_service
from app.services.character_service import character_service
from app.services.location_service import location_service
from app.services.scene_service import scene_service
from app.services.shot_service import shot_service
from app.services.voice_service import voice_service
from app.services.prompt_service import prompt_service
from app.services.queue_service import queue_service
from app.services.generation_service import generation_service
from app.services.qc_service import qc_service
from app.services.render_service import render_service
from app.engines.lightx2v_engine import LightX2VEngine
from app.engines.wan22_engine import Wan22Engine
from app.core.config import settings
from app.models.schemas import (
    ProjectCreate, CharacterCreate, LocationCreate, SceneCreate,
    ShotCreate, VoiceProfileCreate, GenerationRequest, FinalRenderRequest, QCStatus, JobState
)


class TestHarshVideoStudioPhases1To12(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asyncio.run(init_db())
        cls.client = TestClient(app)

    def test_01_system_integrity(self):
        """Verify presence of all repository structure components."""
        required = [
            root_dir / "backend" / "app" / "main.py",
            root_dir / "backend" / "app" / "core" / "config.py",
            root_dir / "backend" / "app" / "database" / "models.py",
            root_dir / "backend" / "app" / "services" / "prompt_service.py",
            root_dir / "backend" / "app" / "services" / "queue_service.py",
            root_dir / "backend" / "app" / "engines" / "base_engine.py",
            root_dir / "backend" / "app" / "engines" / "lightx2v_engine.py",
            root_dir / "backend" / "app" / "engines" / "wan22_engine.py",
            root_dir / "workers" / "gpu_worker.py",
            root_dir / "frontend" / "package.json",
            root_dir / "docker" / "Dockerfile.backend",
            root_dir / "docker" / "Dockerfile.worker",
            root_dir / "docker" / "Dockerfile.frontend",
            root_dir / "docker-compose.yml",
            root_dir / ".env.example",
            root_dir / ".gitignore",
            root_dir / "README.md",
            root_dir / "docs" / "ARCHITECTURE.md",
            root_dir / "docs" / "DEPLOYMENT_GUIDE.md",
            root_dir / "docs" / "REMOTE_GPU_TEST_PLAN.md",
            root_dir / "docs" / "DEVELOPMENT_ROADMAP.md",
            root_dir / "docs" / "API_SPECIFICATION.md",
            root_dir / "scripts" / "deploy_remote_gpu.sh",
            root_dir / "scripts" / "install_lightx2v_env.sh",
            root_dir / "scripts" / "download_models.sh",
        ]
        for p in required:
            self.assertTrue(p.exists(), f"Missing required file: {p}")

    def test_02_engine_deep_abstraction_and_vram(self):
        """Test LightX2V and Wan22 engine contracts, capabilities, and VRAM estimation."""
        lightx2v = LightX2VEngine()
        self.assertEqual(lightx2v.precision, "nvfp4")
        self.assertTrue(lightx2v.sparse_attention)
        
        # Test VRAM Estimation (RTX 5090 32GB threshold)
        vram_720p = lightx2v.estimate_vram_requirement("1280x720", 6.0)
        self.assertLessEqual(vram_720p, 28.0, f"LightX2V VRAM exceeds RTX 5090 budget: {vram_720p} GB")
        
        caps = lightx2v.get_capabilities()
        self.assertTrue(caps["supports_nvfp4"])
        self.assertTrue(caps["supports_sparse_attention"])
        self.assertTrue(caps["supports_cuda_graph"])

        wan = Wan22Engine()
        vram_wan = wan.estimate_vram_requirement("1280x720", 6.0)
        self.assertEqual(vram_wan, 46.0)

    def test_03_multi_character_prompt_synthesis(self):
        """Test Prompt Synthesizer with Multi-Character presence and LoRAs."""
        char_a = CharacterCreate(
            name="Commander Vance",
            appearance="Silver crop hair, stern jaw, cybernetic eye",
            clothing="Matte navy carbon-weave tactical uniform",
            hair="Silver",
            optional_lora="vance_v2"
        )
        char_b = CharacterCreate(
            name="Dr. Elena Thorne",
            appearance="Auburn hair, brass glasses, lab coat",
            clothing="White holographic lab coat",
            hair="Auburn",
            optional_lora="elena_v1"
        )
        loc = LocationCreate(
            name="Neon Cyber Market",
            architecture="Dense neon billboards, rain-slicked asphalt",
            environment="Holographic vendor stalls, rising steam",
            lighting="Cyan and magenta neon glow",
            weather="Rain",
            time_of_day="Midnight",
            camera_style="35mm Anamorphic"
        )

        async def run_test():
            saved_a = await character_service.create_character(None, char_a)
            saved_b = await character_service.create_character(None, char_b)
            saved_loc = await location_service.create_location(None, loc)

            synthesized = prompt_service.synthesize_shot_prompt(
                action_prompt="Vance confronts informant while Elena scans the security node",
                characters=[saved_a, saved_b],
                location=saved_loc,
                camera_motion="dolly zoom"
            )

            self.assertIn("Commander Vance", synthesized["prompt"])
            self.assertIn("<lora:vance_v2:0.85>", synthesized["prompt"])
            self.assertIn("Dr. Elena Thorne", synthesized["prompt"])
            self.assertIn("<lora:elena_v1:0.85>", synthesized["prompt"])
            self.assertIn("Neon Cyber Market", synthesized["prompt"])
            self.assertIn("blurry", synthesized["negative_prompt"])
            self.assertIn("flicker", synthesized["negative_prompt"])

        asyncio.run(run_test())

    def test_04_multi_location_scene_switching(self):
        """Test Multi-Location switching across 5 distinct narrative scenes."""
        async def run_test():
            locations_data = [
                ("Market", "Neon street market with steam"),
                ("House", "Cozy high-tech suburban living room"),
                ("School", "Futuristic academy lecture hall"),
                ("Road", "High-speed magnetic highway under rain"),
                ("Office", "Corporate executive boardroom with skyline view")
            ]
            
            created_loc_ids = []
            for name, desc in locations_data:
                loc = await location_service.create_location(None, LocationCreate(
                    name=name,
                    description=desc,
                    lighting="Dynamic cinematic lighting"
                ))
                created_loc_ids.append(loc.location_id)

            self.assertEqual(len(created_loc_ids), 5)
            all_locs = await location_service.list_locations()
            self.assertGreaterEqual(len(all_locs), 5)

        asyncio.run(run_test())

    def test_05_full_5minute_story_breakdown(self):
        """
        CRITICAL TEST: Build complete 5-minute story (300 seconds)
        across 5 scenes, automatically sliced into 50 chained 6-second shots.
        """
        async def run_test():
            proj = await project_service.create_project(ProjectCreate(
                name="Operation Eclipse: 5-Minute Feature",
                description="Epic sci-fi short film",
                target_duration=300.0
            ))

            scenes_spec = [
                {"scene_id": f"scene_mkt_{proj.project_id[:6]}", "duration": 60.0, "prompt": "Market chase"},
                {"scene_id": f"scene_hse_{proj.project_id[:6]}", "duration": 60.0, "prompt": "Safehouse planning"},
                {"scene_id": f"scene_rd_{proj.project_id[:6]}", "duration": 60.0, "prompt": "Highway pursuit"},
                {"scene_id": f"scene_off_{proj.project_id[:6]}", "duration": 60.0, "prompt": "Corporate infiltration"},
                {"scene_id": f"scene_hlp_{proj.project_id[:6]}", "duration": 60.0, "prompt": "Helipad extraction"}
            ]

            for s in scenes_spec:
                await scene_service.create_scene(proj.project_id, SceneCreate(
                    project_id=proj.project_id,
                    location_id="loc_test",
                    duration=s["duration"],
                    dialogue="Narrative dialogue transcript",
                    action=s["prompt"]
                ))

            pipeline_result = await shot_service.build_5min_story_pipeline(proj.project_id, scenes_spec)
            
            self.assertEqual(pipeline_result["total_shots"], 50)
            self.assertEqual(pipeline_result["total_duration_seconds"], 300.0)

            shots = pipeline_result["shots"]
            self.assertIsNone(shots[0].previous_shot_id)
            for i in range(1, len(shots)):
                self.assertEqual(shots[i].previous_shot_id, shots[i-1].shot_id, f"Break in frame chain at shot {i}")

        asyncio.run(run_test())

    def test_06_redis_job_queue_and_worker_dispatch(self):
        """Test asynchronous job submission to Redis queue and status lifecycle."""
        async def run_test():
            req = GenerationRequest(
                prompt="Commander Vance surveying the city from rooftop",
                duration=6.0,
                resolution="1280x720",
                seed=42,
                engine="lightx2v",
                auto_qc=True,
                auto_retry=True
            )
            job = await generation_service.submit_generation(req)
            self.assertEqual(job.status, JobState.QUEUED)
            self.assertEqual(job.engine, "lightx2v")

            q_len = await queue_service.get_queue_length(settings.REDIS_QUEUE_NAME)
            self.assertGreaterEqual(q_len, 1)

            popped = await queue_service.pop_job(settings.REDIS_QUEUE_NAME, timeout_seconds=1)
            self.assertIsNotNone(popped)
            self.assertEqual(popped["job_id"], job.job_id)

        asyncio.run(run_test())

    def test_07_voice_profile_isolation(self):
        """Test Character Voice Profile mapping."""
        async def run_test():
            v_a = await voice_service.create_voice_profile(VoiceProfileCreate(
                name="Vance_Voice_Profile",
                pitch=0.9
            ))
            v_b = await voice_service.create_voice_profile(VoiceProfileCreate(
                name="Elena_Voice_Profile",
                pitch=1.1
            ))

            c_a = await character_service.create_character(None, CharacterCreate(
                name="Vance_Actor",
                voice_profile_id=v_a.voice_profile_id
            ))
            c_b = await character_service.create_character(None, CharacterCreate(
                name="Elena_Actor",
                voice_profile_id=v_b.voice_profile_id
            ))

            res_a = await voice_service.get_voice_for_character(c_a.character_id)
            self.assertEqual(res_a.name, "Vance_Voice_Profile")

            res_b = await voice_service.get_voice_for_character(c_b.character_id)
            self.assertEqual(res_b.name, "Elena_Voice_Profile")

        asyncio.run(run_test())

    def test_08_qc_scoring_auto_regeneration(self):
        """Test QC pass and auto-regeneration failure handling."""
        async def run_test():
            pass_rep = await qc_service.evaluate_shot_video(
                shot_id="shot_pass_01",
                video_path="/outputs/pass.mp4",
                expected_duration=6.0,
                simulated_face_score=0.89,
                simulated_black_frames=0.001
            )
            self.assertEqual(pass_rep.status, QCStatus.PASS)

            fail_rep = await qc_service.evaluate_shot_video(
                shot_id="shot_fail_01",
                video_path="/outputs/fail.mp4",
                expected_duration=6.0,
                simulated_face_score=0.52,
                simulated_black_frames=0.001
            )
            self.assertEqual(fail_rep.status, QCStatus.FAIL)

        asyncio.run(run_test())

    def test_09_master_1080p_render_assembly(self):
        """Test master FFmpeg stitching job submission for multi-clip assembly."""
        async def run_test():
            p = await project_service.create_project(ProjectCreate(name="Master Feature", target_duration=300.0))
            render_req = FinalRenderRequest(
                project_id=p.project_id,
                target_resolution="1920x1080",
                fps=24,
                video_codec="libx264",
                audio_codec="aac"
            )
            render_resp = await render_service.trigger_final_render(render_req)
            self.assertEqual(render_resp.status, JobState.PROCESSING)
            self.assertIn("final_1080p.mp4", render_resp.output_video_path)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main(verbosity=2)
