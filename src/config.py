from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Redis
    redis_url: str = "redis://localhost:6379"

    # Ephemeris
    ephemeris_start: str = "2025-01-01"
    ephemeris_end: str = "2060-01-01"
    ephemeris_step_days: int = 1
    ephemeris_cache_dir: Path = Path("data/cache")

    # Scene scaling: 1 AU = this many scene units
    scene_scale_au: float = 1000.0

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

# Ensure cache directory exists
settings.ephemeris_cache_dir.mkdir(parents=True, exist_ok=True)
