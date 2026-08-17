"""
Redis Asynchronous Job Queue and Telemetry Service for Harsh AI Video Studio.
Manages job submission, state transitions, queue length metrics, and worker task polling.
"""
from typing import Dict, Any, Optional, List
import json
import asyncio
from datetime import datetime, timezone
from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import JobState

# Local queue fallback when Redis service is not connected on laptop
_IN_MEMORY_QUEUE: asyncio.Queue = asyncio.Queue()


class QueueService:
    def __init__(self):
        self.redis_client = None
        self._init_redis()

    def _init_redis(self):
        try:
            import redis.asyncio as aioredis
            self.redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            logger.info(f"Connected to Redis broker at {settings.REDIS_URL}")
        except Exception as e:
            logger.warning(f"Redis not available ({e}). Initializing In-Memory Async Queue fallback for dev plane.")
            self.redis_client = None

    async def enqueue_job(self, queue_name: str, payload: Dict[str, Any]) -> bool:
        """Pushes job payload into Redis queue or in-memory queue."""
        payload_str = json.dumps(payload)
        if self.redis_client:
            try:
                await self.redis_client.rpush(queue_name, payload_str)
                logger.info(f"Enqueued job {payload.get('job_id')} to Redis queue '{queue_name}'")
                return True
            except Exception as e:
                logger.error(f"Redis rpush failed: {e}. Falling back to in-memory queue.")
        
        await _IN_MEMORY_QUEUE.put(payload)
        logger.info(f"Enqueued job {payload.get('job_id')} to In-Memory Queue (size={_IN_MEMORY_QUEUE.qsize()})")
        return True

    async def pop_job(self, queue_name: str, timeout_seconds: int = 2) -> Optional[Dict[str, Any]]:
        """Pops next job from Redis queue or in-memory queue."""
        if self.redis_client:
            try:
                res = await self.redis_client.blpop(queue_name, timeout=timeout_seconds)
                if res:
                    _, data_str = res
                    return json.loads(data_str)
            except Exception as e:
                logger.warning(f"Redis blpop error: {e}. Checking in-memory queue...")

        try:
            return await asyncio.wait_for(_IN_MEMORY_QUEUE.get(), timeout=float(timeout_seconds))
        except asyncio.TimeoutError:
            return None

    async def get_queue_length(self, queue_name: str) -> int:
        """Returns total pending jobs in the queue."""
        if self.redis_client:
            try:
                return await self.redis_client.llen(queue_name)
            except Exception:
                pass
        return _IN_MEMORY_QUEUE.qsize()


queue_service = QueueService()
