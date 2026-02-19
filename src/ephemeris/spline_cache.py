"""Ephemeris spline cache — fast in-RAM planetary position lookups.

On warm-up, fetches state vectors from JPL Horizons for all bodies,
fits cubic B-splines to (x, y, z) and (vx, vy, vz) as functions of
Julian Date, and stores the spline coefficients in memory.

Subsequent position queries evaluate the spline in ~microseconds.
Spline coefficients are also persisted to disk for fast restarts.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from config import settings
from ephemeris.bodies import ALL_BODIES, CelestialBody
from ephemeris.horizons_client import fetch_state_vectors

logger = logging.getLogger("tars.ephemeris")


class EphemerisRangeError(ValueError):
    """Raised when a query epoch is outside the valid ephemeris range."""

    def __init__(self, naif_id: int, epoch_jd: float, epoch_start: float, epoch_end: float):
        self.naif_id = naif_id
        self.epoch_jd = epoch_jd
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        super().__init__(
            f"Epoch JD {epoch_jd:.2f} is outside the valid ephemeris range "
            f"[{epoch_start:.2f}, {epoch_end:.2f}] for body {naif_id}"
        )


@dataclass
class BodySpline:
    """Spline interpolators for a single body's ephemeris."""

    body: CelestialBody
    epoch_start: float  # JD
    epoch_end: float  # JD
    pos_splines: tuple  # (CubicSpline_x, CubicSpline_y, CubicSpline_z)
    vel_splines: tuple  # (CubicSpline_vx, CubicSpline_vy, CubicSpline_vz)


class EphemerisCache:
    """In-memory cache of spline-interpolated ephemeris data."""

    def __init__(self) -> None:
        self._splines: dict[int, BodySpline] = {}
        self._cache_dir: Path = settings.ephemeris_cache_dir

    def __len__(self) -> int:
        return len(self._splines)

    def __contains__(self, naif_id: int) -> bool:
        return naif_id in self._splines

    async def warm(
        self,
        start: str,
        end: str,
        step_days: int = 1,
        bodies: list[CelestialBody] | None = None,
    ) -> None:
        """Load or fetch ephemeris for all bodies and fit splines.

        Tries disk cache first. On miss or if the cached range doesn't cover
        the requested range, fetches from Horizons and saves.
        """
        if bodies is None:
            bodies = ALL_BODIES

        from mechanics.transforms import iso_to_jd
        requested_start_jd = iso_to_jd(start)
        requested_end_jd = iso_to_jd(end)

        for body in bodies:
            cache_path = self._cache_dir / f"spline_{body.naif_id}.pkl"

            if cache_path.exists():
                try:
                    self._load_from_disk(body, cache_path)
                    # Verify cached data covers the requested range
                    spline = self._splines.get(body.naif_id)
                    if spline and spline.epoch_start <= requested_start_jd and spline.epoch_end >= requested_end_jd:
                        logger.info("Loaded cached spline for %s (covers requested range)", body.name)
                        continue
                    else:
                        if spline:
                            logger.info(
                                "Cached data for %s covers [%.1f, %.1f] but need [%.1f, %.1f] — re-fetching",
                                body.name, spline.epoch_start, spline.epoch_end,
                                requested_start_jd, requested_end_jd,
                            )
                        # Remove stale spline
                        self._splines.pop(body.naif_id, None)
                except Exception as e:
                    logger.warning("Cache load failed for %s: %s", body.name, e)

            # Fetch from Horizons
            try:
                # Determine CENTER for Horizons query
                # Moons use parent-centric coordinates, then we combine
                if body.parent_id is not None:
                    center = f"500@{body.parent_id}"
                else:
                    center = "500@10"  # heliocentric

                data = await fetch_state_vectors(
                    horizons_id=body.horizons_id,
                    start=start,
                    stop=end,
                    step_days=step_days,
                    center=center,
                )

                # For moons, convert parent-relative to heliocentric
                if body.parent_id is not None and body.parent_id in self._splines:
                    parent_spline = self._splines[body.parent_id]
                    epochs = data["epochs"]
                    parent_pos = np.column_stack(
                        [s(epochs) for s in parent_spline.pos_splines]
                    )
                    parent_vel = np.column_stack(
                        [s(epochs) for s in parent_spline.vel_splines]
                    )
                    data["positions"] = data["positions"] + parent_pos
                    data["velocities"] = data["velocities"] + parent_vel
                    logger.info(
                        "Converted %s from parent-relative to heliocentric",
                        body.name,
                    )

                self._fit_and_store(body, data)
                self._save_to_disk(body, cache_path, data)
                logger.info("Fetched & cached ephemeris for %s (%d points)",
                            body.name, len(data["epochs"]))
            except Exception as e:
                logger.error("Failed to fetch ephemeris for %s: %s", body.name, e)

    def _fit_and_store(self, body: CelestialBody, data: dict) -> None:
        """Fit cubic splines to the raw state vector data."""
        epochs = data["epochs"]
        pos = data["positions"]  # (N, 3)
        vel = data["velocities"]  # (N, 3)

        pos_splines = tuple(
            CubicSpline(epochs, pos[:, i], extrapolate=False) for i in range(3)
        )
        vel_splines = tuple(
            CubicSpline(epochs, vel[:, i], extrapolate=False) for i in range(3)
        )

        self._splines[body.naif_id] = BodySpline(
            body=body,
            epoch_start=epochs[0],
            epoch_end=epochs[-1],
            pos_splines=pos_splines,
            vel_splines=vel_splines,
        )

    def _save_to_disk(self, body: CelestialBody, path: Path, data: dict) -> None:
        """Persist raw data + spline to disk for fast restart."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "naif_id": body.naif_id,
                "epochs": data["epochs"],
                "positions": data["positions"],
                "velocities": data["velocities"],
            }, f)

    def _load_from_disk(self, body: CelestialBody, path: Path) -> None:
        """Load cached data from disk and refit splines."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._fit_and_store(body, data)

    # ----- Public query API ----- #

    def validate_epoch(self, naif_id: int, epoch_jd: float) -> None:
        """Raise EphemerisRangeError if epoch is outside the valid range."""
        spline = self._splines[naif_id]
        if epoch_jd < spline.epoch_start or epoch_jd > spline.epoch_end:
            raise EphemerisRangeError(
                naif_id, epoch_jd, spline.epoch_start, spline.epoch_end,
            )

    def validate_epoch_range(self, naif_id: int, start_jd: float, end_jd: float) -> None:
        """Raise EphemerisRangeError if any part of the range is outside valid bounds."""
        spline = self._splines[naif_id]
        if start_jd < spline.epoch_start or end_jd > spline.epoch_end:
            bad_jd = start_jd if start_jd < spline.epoch_start else end_jd
            raise EphemerisRangeError(
                naif_id, bad_jd, spline.epoch_start, spline.epoch_end,
            )

    def get_position(self, naif_id: int, epoch_jd: float) -> np.ndarray:
        """Get position (x, y, z) in km at a Julian Date. ~microseconds."""
        spline = self._splines[naif_id]
        self.validate_epoch(naif_id, epoch_jd)
        return np.array([s(epoch_jd) for s in spline.pos_splines], dtype=np.float64)

    def get_velocity(self, naif_id: int, epoch_jd: float) -> np.ndarray:
        """Get velocity (vx, vy, vz) in km/s at a Julian Date."""
        spline = self._splines[naif_id]
        self.validate_epoch(naif_id, epoch_jd)
        return np.array([s(epoch_jd) for s in spline.vel_splines], dtype=np.float64)

    def get_state(self, naif_id: int, epoch_jd: float) -> tuple[np.ndarray, np.ndarray]:
        """Get full state vector (position, velocity) at a Julian Date."""
        self.validate_epoch(naif_id, epoch_jd)
        spline = self._splines[naif_id]
        pos = np.array([s(epoch_jd) for s in spline.pos_splines], dtype=np.float64)
        vel = np.array([s(epoch_jd) for s in spline.vel_splines], dtype=np.float64)
        return pos, vel

    def get_positions_batch(
        self, naif_id: int, epochs_jd: np.ndarray
    ) -> np.ndarray:
        """Get positions for multiple epochs at once. Returns (N, 3) array."""
        spline = self._splines[naif_id]
        if len(epochs_jd) > 0:
            if epochs_jd[0] < spline.epoch_start or epochs_jd[-1] > spline.epoch_end:
                bad_jd = epochs_jd[0] if epochs_jd[0] < spline.epoch_start else epochs_jd[-1]
                raise EphemerisRangeError(
                    naif_id, float(bad_jd), spline.epoch_start, spline.epoch_end,
                )
        return np.column_stack([s(epochs_jd) for s in spline.pos_splines])

    def get_epoch_range(self, naif_id: int) -> tuple[float, float]:
        """Return the valid epoch range (JD) for a body."""
        spline = self._splines[naif_id]
        return spline.epoch_start, spline.epoch_end

    def available_bodies(self) -> list[int]:
        """Return list of NAIF IDs with loaded ephemeris."""
        return list(self._splines.keys())
