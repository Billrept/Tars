"""Optimization job dispatcher — submits jobs to ARQ and streams progress.

Bridges the API layer with the background optimization workers.
Uses Redis pub/sub to stream intermediate results back to WebSocket clients.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncGenerator

import redis.asyncio as aioredis
from arq import create_pool
from arq.connections import RedisSettings, ArqRedis

from config import settings
from optimizer.gmpa import (
    OptimizationProgress,
    OptimizationRequest,
    MultiLegOptimizationProgress,
    MultiLegOptimizationRequest,
)

logger = logging.getLogger("tars.dispatcher")

# Redis key prefixes
JOB_PREFIX = "tars:job:"
CHANNEL_PREFIX = "tars:progress:"


def _redis_settings() -> RedisSettings:
    """Parse REDIS_URL into ARQ RedisSettings."""
    url = settings.redis_url
    # redis://host:port or redis://host:port/db
    parts = url.replace("redis://", "").split(":")
    host = parts[0] if parts[0] else "localhost"
    port = int(parts[1].split("/")[0]) if len(parts) > 1 else 6379
    return RedisSettings(host=host, port=port)


async def get_arq_pool() -> ArqRedis:
    """Create and return an ARQ Redis connection pool."""
    return await create_pool(_redis_settings())


async def get_redis() -> aioredis.Redis:
    """Create a raw async Redis client."""
    return aioredis.from_url(settings.redis_url, decode_responses=True)


async def submit_optimization(
    request: OptimizationRequest,
) -> str:
    """Submit an optimization job to the ARQ worker.

    Returns a job_id that clients can use to subscribe to progress.
    """
    job_id = str(uuid.uuid4())

    pool = await get_arq_pool()

    # Serialize request
    request_data = {
        "origin_id": request.origin_id,
        "target_id": request.target_id,
        "dep_start_jd": request.dep_start_jd,
        "dep_end_jd": request.dep_end_jd,
        "tof_min_days": request.tof_min_days,
        "tof_max_days": request.tof_max_days,
        "population_size": request.population_size,
        "max_iterations": request.max_iterations,
        "prograde": request.prograde,
        "mode": request.mode,
    }

    # Store job metadata
    r = await get_redis()
    await r.hset(f"{JOB_PREFIX}{job_id}", mapping={
        "status": "queued",
        "request": json.dumps(request_data),
        "result": "",
    })

    # Enqueue the ARQ task
    await pool.enqueue_job(
        "run_optimization",
        job_id=job_id,
        request_data=request_data,
        _job_id=job_id,
    )

    await pool.close()
    await r.close()

    logger.info("Submitted optimization job %s", job_id)
    return job_id


async def get_job_status(job_id: str) -> dict:
    """Get the current status of an optimization job."""
    r = await get_redis()
    data = await r.hgetall(f"{JOB_PREFIX}{job_id}")
    await r.close()

    if not data:
        return {"status": "not_found", "job_id": job_id}

    result = {
        "job_id": job_id,
        "status": data.get("status", "unknown"),
    }

    result_str = data.get("result", "")
    if result_str:
        try:
            result["result"] = json.loads(result_str)
        except json.JSONDecodeError:
            pass

    return result


async def stream_progress(job_id: str) -> AsyncGenerator[dict, None]:
    """Subscribe to optimization progress via Redis pub/sub.

    Yields progress dicts as they arrive. Terminates when the job
    publishes a "complete" or "failed" status.
    """
    r = await get_redis()
    pubsub = r.pubsub()
    channel = f"{CHANNEL_PREFIX}{job_id}"

    await pubsub.subscribe(channel)
    logger.info("Subscribed to progress channel: %s", channel)

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            try:
                data = json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                continue

            yield data

            # Check for terminal states
            status = data.get("status", "")
            if status in ("complete", "failed"):
                break
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()
        await r.close()


async def publish_progress(
    job_id: str,
    progress: OptimizationProgress,
    status: str = "running",
) -> None:
    """Publish optimization progress to the Redis channel.

    Called by the worker during optimization.
    """
    r = await get_redis()
    channel = f"{CHANNEL_PREFIX}{job_id}"

    payload = {
        "status": status,
        "job_id": job_id,
        **_progress_to_dict(progress),
    }

    await r.publish(channel, json.dumps(payload))

    # Also update the stored job state
    await r.hset(f"{JOB_PREFIX}{job_id}", mapping={
        "status": status,
        "result": json.dumps(_progress_to_dict(progress)),
    })

    await r.close()


def _progress_to_dict(p: OptimizationProgress) -> dict:
    """Convert OptimizationProgress to a JSON-safe dict."""
    return {
        "iteration": p.iteration,
        "max_iterations": p.max_iterations,
        "best_dv_total": p.best_dv_total if p.best_dv_total != float("inf") else None,
        "best_departure_jd": p.best_departure_jd,
        "best_tof_days": p.best_tof_days,
        "best_dv_departure": p.best_dv_departure if p.best_dv_departure != float("inf") else None,
        "best_dv_arrival": p.best_dv_arrival if p.best_dv_arrival != float("inf") else None,
        "converged": p.converged,
        "population_best_dvs": [
            v if v != float("inf") else None for v in p.population_best_dvs
        ],
        "population_positions": p.population_positions,
        "pareto_front": p.pareto_front,
        "mode": p.mode,
    }


# --------------------------------------------------------------------------- #
#  Multi-leg optimization
# --------------------------------------------------------------------------- #


async def submit_multileg_optimization(
    request: MultiLegOptimizationRequest,
) -> str:
    """Submit a multi-leg optimization job to the ARQ worker.

    Returns a job_id that clients can use to subscribe to progress.
    """
    job_id = str(uuid.uuid4())
    pool = await get_arq_pool()

    # Serialize request
    request_data = {
        "body_names": request.body_names,
        "dep_start_jd": request.dep_start_jd,
        "dep_end_jd": request.dep_end_jd,
        "leg_tof_bounds": [[b[0], b[1]] for b in request.leg_tof_bounds],
        "population_size": request.population_size,
        "max_iterations": request.max_iterations,
        "max_c3": request.max_c3,
        "mode": request.mode,
    }

    # Store job metadata
    r = await get_redis()
    await r.hset(f"{JOB_PREFIX}{job_id}", mapping={
        "status": "queued",
        "type": "multileg",
        "request": json.dumps(request_data),
        "result": "",
    })

    # Enqueue the ARQ task
    await pool.enqueue_job(
        "run_multileg_optimization",
        job_id=job_id,
        request_data=request_data,
        _job_id=job_id,
    )

    await pool.close()
    await r.close()

    logger.info("Submitted multi-leg optimization job %s", job_id)
    return job_id


async def publish_multileg_progress(
    job_id: str,
    progress: MultiLegOptimizationProgress,
    status: str = "running",
) -> None:
    """Publish multi-leg optimization progress to the Redis channel."""
    r = await get_redis()
    channel = f"{CHANNEL_PREFIX}{job_id}"

    payload = {
        "status": status,
        "job_id": job_id,
        "type": "multileg",
        **_multileg_progress_to_dict(progress),
    }

    await r.publish(channel, json.dumps(payload))

    await r.hset(f"{JOB_PREFIX}{job_id}", mapping={
        "status": status,
        "result": json.dumps(_multileg_progress_to_dict(progress)),
    })

    await r.close()


def _multileg_progress_to_dict(p: MultiLegOptimizationProgress) -> dict:
    """Convert MultiLegOptimizationProgress to a JSON-safe dict."""
    def _safe(v: float) -> float | None:
        return v if v != float("inf") and v < 1e11 else None

    return {
        "iteration": p.iteration,
        "max_iterations": p.max_iterations,
        "best_dv_total": _safe(p.best_dv_total),
        "best_departure_jd": p.best_departure_jd,
        "best_leg_tof_days": p.best_leg_tof_days,
        "best_total_tof_days": p.best_total_tof_days,
        "best_dv_departure": _safe(p.best_dv_departure),
        "best_dv_arrival": _safe(p.best_dv_arrival),
        "best_dv_flyby": _safe(p.best_dv_flyby),
        "converged": p.converged,
        "body_sequence": p.body_sequence,
        "population_best_dvs": [
            v if v != float("inf") and v < 1e11 else None
            for v in p.population_best_dvs
        ],
        "population_positions": p.population_positions,
        "pareto_front": p.pareto_front,
        "mode": p.mode,
    }
