"""Async client for JPL Horizons REST API.

Fetches heliocentric ecliptic state vectors (position + velocity) for
celestial bodies over a date range, returning structured numpy arrays.

API docs: https://ssd-api.jpl.nasa.gov/doc/horizons.html
"""

from __future__ import annotations

import logging

import httpx
import numpy as np

logger = logging.getLogger("tars.horizons")

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Horizons vector table settings
# COORD_TYPE  = ecliptic heliocentric
# REF_PLANE   = ecliptic (ECLIPJ2000)
# VEC_TABLE   = 2  (state vectors: x,y,z,vx,vy,vz)
# OUT_UNITS   = KM-S  (km and km/s)
# CSV_FORMAT  = YES


async def fetch_state_vectors(
    horizons_id: str,
    start: str,
    stop: str,
    step_days: int = 1,
    center: str = "500@10",
) -> dict:
    """Fetch state vectors from JPL Horizons for a single body.

    Parameters
    ----------
    horizons_id : str
        Horizons command string (e.g. "399" for Earth).
    start : str
        Start date, ISO format "YYYY-MM-DD".
    stop : str
        End date, ISO format "YYYY-MM-DD".
    step_days : int
        Step size in days.
    center : str
        Horizons CENTER parameter. Default "500@10" (heliocentric).
        Use "500@{parent_naif_id}" for parent-centric (e.g. "500@399" for Earth-centric).

    Returns
    -------
    dict with keys:
        "epochs" : np.ndarray[float64]   — Julian dates
        "positions" : np.ndarray[float64] shape (N, 3) — x, y, z in km
        "velocities" : np.ndarray[float64] shape (N, 3) — vx, vy, vz in km/s
    """
    params = {
        "format": "text",
        "COMMAND": f"'{horizons_id}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": f"'{center}'",  # heliocentric by default, parent-centric for moons
        "REF_PLANE": "ECLIPTIC",
        "REF_SYSTEM": "J2000",
        "VEC_TABLE": "2",
        "VEC_LABELS": "NO",
        "OUT_UNITS": "'KM-S'",
        "CSV_FORMAT": "YES",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step_days}d'",
    }

    logger.info("Fetching ephemeris for %s  [%s → %s]", horizons_id, start, stop)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(HORIZONS_URL, params=params)
        resp.raise_for_status()

    return _parse_horizons_response(resp.text)


def _parse_horizons_response(text: str) -> dict:
    """Parse the Horizons text response into numpy arrays.

    The data block sits between $$SOE and $$EOE markers.
    With VEC_TABLE=2 and CSV_FORMAT=YES, each record is ONE line:
        JDTDB, CalendarDate, X, Y, Z, VX, VY, VZ,
    """
    lines = text.split("\n")

    # Find data block boundaries
    soe_idx = None
    eoe_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "$$SOE":
            soe_idx = i + 1
        elif line.strip() == "$$EOE":
            eoe_idx = i
            break

    if soe_idx is None or eoe_idx is None:
        raise ValueError("Could not locate $$SOE / $$EOE markers in Horizons response")

    data_lines = lines[soe_idx:eoe_idx]

    epochs = []
    positions = []
    velocities = []

    for line in data_lines:
        line = line.strip()
        if not line:
            continue

        # Split by comma
        parts = [p.strip() for p in line.split(",")]

        # Filter out empty strings from trailing commas
        parts = [p for p in parts if p]

        # Format: JDTDB, CalendarDate, X, Y, Z, VX, VY, VZ
        # CalendarDate contains spaces like "A.D. 2025-Jan-01 00:00:00.0000"
        # so it won't parse as float. We identify numeric fields only.
        numeric_vals = []
        for p in parts:
            try:
                numeric_vals.append(float(p))
            except ValueError:
                continue  # skip the calendar date string

        # We expect 7 numeric values: JDTDB, X, Y, Z, VX, VY, VZ
        if len(numeric_vals) < 7:
            logger.warning("Skipping line with %d numeric fields: %s",
                           len(numeric_vals), line[:80])
            continue

        jdtdb = numeric_vals[0]
        x, y, z = numeric_vals[1], numeric_vals[2], numeric_vals[3]
        vx, vy, vz = numeric_vals[4], numeric_vals[5], numeric_vals[6]

        epochs.append(jdtdb)
        positions.append([x, y, z])
        velocities.append([vx, vy, vz])

    if len(epochs) == 0:
        raise ValueError("No data records parsed from Horizons response")

    logger.info("Parsed %d ephemeris records", len(epochs))

    return {
        "epochs": np.array(epochs, dtype=np.float64),
        "positions": np.array(positions, dtype=np.float64),
        "velocities": np.array(velocities, dtype=np.float64),
    }
