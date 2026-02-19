"""ARQ worker — runs optimization jobs in the background.

This worker process is started separately (via `arq workers.worker.WorkerSettings`)
and picks up jobs from the Redis queue.
"""

from __future__ import annotations

import asyncio
import json
import logging

from arq.connections import RedisSettings  # noqa: F401 (used by WorkerSettings indirectly)

from config import settings
from ephemeris.spline_cache import EphemerisCache
from optimizer.dispatcher import publish_progress, publish_multileg_progress, JOB_PREFIX, get_redis, _redis_settings
from optimizer.gmpa import (
    GreyWolfOptimizer,
    OptimizationRequest,
    MultiLegGreyWolfOptimizer,
    MultiLegOptimizationRequest,
)

logger = logging.getLogger("tars.worker")

# Worker-local ephemeris cache (loaded once at worker startup)
_cache: EphemerisCache | None = None


async def startup(ctx: dict) -> None:
    """Called once when the worker starts. Warms ephemeris cache."""
    global _cache
    logger.info("Worker starting — warming ephemeris cache ...")
    _cache = EphemerisCache()
    await _cache.warm(
        start=settings.ephemeris_start,
        end=settings.ephemeris_end,
        step_days=settings.ephemeris_step_days,
    )
    logger.info("Worker ephemeris cache ready — %d bodies", len(_cache))
    ctx["cache"] = _cache


async def shutdown(ctx: dict) -> None:
    """Called when the worker shuts down."""
    logger.info("Worker shutting down")


async def run_optimization(ctx: dict, job_id: str, request_data: dict) -> dict:
    """Execute an optimization job and stream progress via Redis pub/sub.

    This is the ARQ task function registered with the worker.
    """
    cache: EphemerisCache = ctx["cache"]

    request = OptimizationRequest(
        origin_id=request_data["origin_id"],
        target_id=request_data["target_id"],
        dep_start_jd=request_data["dep_start_jd"],
        dep_end_jd=request_data["dep_end_jd"],
        tof_min_days=request_data["tof_min_days"],
        tof_max_days=request_data["tof_max_days"],
        population_size=request_data.get("population_size", 30),
        max_iterations=request_data.get("max_iterations", 200),
        prograde=request_data.get("prograde", True),
        mode=request_data.get("mode", "pareto"),
    )

    logger.info("Starting optimization job %s: %s -> %s",
                job_id, request.origin_id, request.target_id)

    # Update job status
    r = await get_redis()
    try:
        await r.hset(f"{JOB_PREFIX}{job_id}", "status", "running")
    finally:
        await r.close()

    optimizer = GreyWolfOptimizer(request, cache)
    last_progress = None

    try:
        for progress in optimizer.run():
            last_progress = progress

            # Determine status
            is_final = (progress.iteration >= progress.max_iterations)
            status = "complete" if is_final else "running"

            # Publish to Redis pub/sub for WebSocket streaming
            await publish_progress(job_id, progress, status=status)

            # Yield control to the event loop so other tasks can run
            await asyncio.sleep(0)

        # Mark complete
        if last_progress:
            await publish_progress(job_id, last_progress, status="complete")

        logger.info("Optimization job %s complete — best dv: %.3f km/s",
                     job_id, last_progress.best_dv_total if last_progress else float("inf"))

        return {"status": "complete", "job_id": job_id}

    except Exception as e:
        logger.error("Optimization job %s failed: %s", job_id, e)

        r = await get_redis()
        try:
            await r.hset(f"{JOB_PREFIX}{job_id}", "status", "failed")
            error_payload = json.dumps({"status": "failed", "job_id": job_id, "error": str(e)})
            await r.publish(f"tars:progress:{job_id}", error_payload)
        finally:
            await r.close()

        return {"status": "failed", "job_id": job_id, "error": str(e)}


async def run_multileg_optimization(ctx: dict, job_id: str, request_data: dict) -> dict:
    """Execute a multi-leg optimization job and stream progress via Redis pub/sub."""
    cache: EphemerisCache = ctx["cache"]

    request = MultiLegOptimizationRequest(
        body_names=request_data["body_names"],
        dep_start_jd=request_data["dep_start_jd"],
        dep_end_jd=request_data["dep_end_jd"],
        leg_tof_bounds=[(b[0], b[1]) for b in request_data["leg_tof_bounds"]],
        population_size=request_data.get("population_size", 40),
        max_iterations=request_data.get("max_iterations", 300),
        max_c3=request_data.get("max_c3"),
        mode=request_data.get("mode", "pareto"),
    )

    logger.info("Starting multi-leg optimization job %s: %s",
                job_id, " -> ".join(request.body_names))

    # Update job status
    r = await get_redis()
    try:
        await r.hset(f"{JOB_PREFIX}{job_id}", "status", "running")
    finally:
        await r.close()

    optimizer = MultiLegGreyWolfOptimizer(request, cache)
    last_progress = None

    try:
        for progress in optimizer.run():
            last_progress = progress

            is_final = (progress.iteration >= progress.max_iterations)
            status = "complete" if is_final else "running"

            await publish_multileg_progress(job_id, progress, status=status)
            await asyncio.sleep(0)

        if last_progress:
            await publish_multileg_progress(job_id, last_progress, status="complete")

        logger.info("Multi-leg optimization job %s complete — best dv: %.3f km/s",
                     job_id, last_progress.best_dv_total if last_progress else float("inf"))

        return {"status": "complete", "job_id": job_id}

    except Exception as e:
        logger.error("Multi-leg optimization job %s failed: %s", job_id, e)

        r = await get_redis()
        try:
            await r.hset(f"{JOB_PREFIX}{job_id}", "status", "failed")
            error_payload = json.dumps({"status": "failed", "job_id": job_id, "type": "multileg", "error": str(e)})
            await r.publish(f"tars:progress:{job_id}", error_payload)
        finally:
            await r.close()

        return {"status": "failed", "job_id": job_id, "error": str(e)}


class WorkerSettings:
    """ARQ worker settings class."""
    functions = [run_optimization, run_multileg_optimization]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _redis_settings()
    max_jobs = 4
    job_timeout = 600  # 10 minutes max per optimization
