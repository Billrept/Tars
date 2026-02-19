#!/usr/bin/env python3
"""Pre-warm the ephemeris cache by fetching data from JPL Horizons.

Run this before starting the server for the first time to avoid
a slow startup while it fetches all planetary data.

Usage:
    python scripts/fetch_ephemeris.py
    python scripts/fetch_ephemeris.py --start 2025-01-01 --end 2035-01-01 --step 1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

# Add src to path
sys.path.insert(0, "src")

from config import settings
from ephemeris.spline_cache import EphemerisCache
from ephemeris.bodies import ALL_BODIES, PLANETS


async def main(start: str, end: str, step_days: int, planets_only: bool) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("fetch_ephemeris")

    bodies = PLANETS if planets_only else ALL_BODIES

    logger.info("Fetching ephemeris for %d bodies [%s → %s] step=%dd",
                len(bodies), start, end, step_days)
    logger.info("Cache directory: %s", settings.ephemeris_cache_dir)

    t0 = time.time()
    cache = EphemerisCache()
    await cache.warm(start=start, end=end, step_days=step_days, bodies=bodies)
    elapsed = time.time() - t0

    logger.info("Done. %d bodies cached in %.1f seconds.", len(cache), elapsed)
    logger.info("Cache files stored in: %s", settings.ephemeris_cache_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-warm ephemeris cache")
    parser.add_argument("--start", default="2025-01-01", help="Start date (ISO)")
    parser.add_argument("--end", default="2060-01-01", help="End date (ISO)")
    parser.add_argument("--step", type=int, default=1, help="Step size in days")
    parser.add_argument("--planets-only", action="store_true",
                        help="Only fetch planets (skip moons/dwarf planets)")
    args = parser.parse_args()

    asyncio.run(main(args.start, args.end, args.step, args.planets_only))
