from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.routes_http import router as http_router
from api.routes_ws import router as ws_router
from ephemeris.spline_cache import EphemerisCache

logger = logging.getLogger("tars")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# Global ephemeris cache instance
ephemeris_cache = EphemerisCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm ephemeris cache. Shutdown: cleanup."""
    logger.info("Warming ephemeris cache ...")
    await ephemeris_cache.warm(
        start=settings.ephemeris_start,
        end=settings.ephemeris_end,
        step_days=settings.ephemeris_step_days,
    )
    logger.info("Ephemeris cache ready — %d bodies loaded", len(ephemeris_cache))
    app.state.ephemeris = ephemeris_cache
    yield
    logger.info("Shutting down Tars")


app = FastAPI(
    title="Tars — Interplanetary Trajectory Designer",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(http_router)
app.include_router(ws_router)
