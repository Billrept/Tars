"""Coordinate transforms and unit conversions for the 3D scene.

Handles:
- Real units (km) <-> Scene units (1 AU = configurable scale)
- Julian Date <-> ISO datetime conversion
- Ecliptic <-> Equatorial frame rotation
- Adaptive trajectory sampling for keyframe generation
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
from numba import njit

from config import settings
from ephemeris.bodies import AU_KM

# --------------------------------------------------------------------------- #
#  Unit conversions
# --------------------------------------------------------------------------- #
SCENE_SCALE = settings.scene_scale_au  # 1 AU = this many scene units


def km_to_scene(pos_km: np.ndarray) -> np.ndarray:
    """Convert position from km to scene units (scaled by AU)."""
    return pos_km * (SCENE_SCALE / AU_KM)


def scene_to_km(pos_scene: np.ndarray) -> np.ndarray:
    """Convert position from scene units back to km."""
    return pos_scene * (AU_KM / SCENE_SCALE)


def km_to_au(pos_km: np.ndarray) -> np.ndarray:
    """Convert position from km to AU."""
    return pos_km / AU_KM


def au_to_km(pos_au: np.ndarray) -> np.ndarray:
    """Convert position from AU to km."""
    return pos_au * AU_KM


# --------------------------------------------------------------------------- #
#  Epoch conversions  (Julian Date <-> Python datetime)
# --------------------------------------------------------------------------- #
_J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)  # JD 2451545.0
_JD_J2000 = 2451545.0


def jd_to_datetime(jd: float) -> datetime:
    """Convert Julian Date to Python datetime (UTC)."""
    delta_days = jd - _JD_J2000
    return _J2000_EPOCH + timedelta(days=delta_days)


def datetime_to_jd(dt: datetime) -> float:
    """Convert Python datetime (UTC) to Julian Date."""
    delta = (dt - _J2000_EPOCH).total_seconds() / 86400.0
    return _JD_J2000 + delta


def iso_to_jd(iso_str: str) -> float:
    """Convert ISO date string to Julian Date."""
    dt = datetime.fromisoformat(iso_str)
    return datetime_to_jd(dt)


def jd_to_iso(jd: float) -> str:
    """Convert Julian Date to ISO date string."""
    return jd_to_datetime(jd).isoformat()


# --------------------------------------------------------------------------- #
#  Frame rotations
# --------------------------------------------------------------------------- #
# Obliquity of the ecliptic at J2000 (23.4393 degrees)
_OBLIQUITY_RAD = math.radians(23.4392911)
_COS_OBL = math.cos(_OBLIQUITY_RAD)
_SIN_OBL = math.sin(_OBLIQUITY_RAD)

# Rotation matrix: ecliptic -> equatorial
_R_ECL_TO_EQ = np.array([
    [1.0, 0.0, 0.0],
    [0.0, _COS_OBL, -_SIN_OBL],
    [0.0, _SIN_OBL, _COS_OBL],
], dtype=np.float64)

# Inverse: equatorial -> ecliptic
_R_EQ_TO_ECL = _R_ECL_TO_EQ.T


def ecliptic_to_equatorial(vec: np.ndarray) -> np.ndarray:
    """Rotate a vector from ecliptic J2000 to equatorial J2000."""
    return _R_ECL_TO_EQ @ vec


def equatorial_to_ecliptic(vec: np.ndarray) -> np.ndarray:
    """Rotate a vector from equatorial J2000 to ecliptic J2000."""
    return _R_EQ_TO_ECL @ vec


# --------------------------------------------------------------------------- #
#  Adaptive trajectory sampling
# --------------------------------------------------------------------------- #
def sample_trajectory_adaptive(
    positions: np.ndarray,
    epochs: np.ndarray,
    angle_threshold_deg: float = 2.0,
    min_points: int = 20,
    max_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptively sample a trajectory, keeping points where curvature is high.

    This reduces the number of points sent to the frontend while preserving
    visual fidelity at maneuvers and curved sections.

    Parameters
    ----------
    positions : (N, 3) array of positions
    epochs : (N,) array of Julian Dates
    angle_threshold_deg : keep a point if the direction change exceeds this
    min_points : always keep at least this many uniformly-spaced points
    max_points : never return more than this many points

    Returns
    -------
    (sampled_positions, sampled_epochs) — reduced arrays
    """
    n = len(positions)
    if n <= min_points:
        return positions, epochs

    # Always keep first and last
    keep = set()
    keep.add(0)
    keep.add(n - 1)

    # Add uniformly-spaced baseline
    step = max(1, n // min_points)
    for i in range(0, n, step):
        keep.add(i)

    # Add points where curvature is high
    threshold_rad = math.radians(angle_threshold_deg)
    for i in range(1, n - 1):
        d1 = positions[i] - positions[i - 1]
        d2 = positions[i + 1] - positions[i]
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        if n1 > 1e-10 and n2 > 1e-10:
            cos_angle = np.dot(d1, d2) / (n1 * n2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle)
            if angle > threshold_rad:
                keep.add(i)

    # Sort and limit
    indices = sorted(keep)
    if len(indices) > max_points:
        step = len(indices) // max_points
        indices = indices[::step]
        if indices[-1] != n - 1:
            indices.append(n - 1)

    idx = np.array(indices)
    return positions[idx], epochs[idx]


def positions_to_scene(positions_km: np.ndarray) -> np.ndarray:
    """Batch convert (N, 3) positions from km to scene units."""
    return positions_km * (SCENE_SCALE / AU_KM)
