"""Multi-leg trajectory solver — chained Lambert arcs with flyby analysis.

Computes patched-conic trajectories through a sequence of bodies.
At each intermediate body, calculates the gravity-assist geometry:
  - v_infinity_in / v_infinity_out
  - flyby turning angle
  - powered delta-v if pure gravity assist is insufficient
  - flyby periapsis altitude

This is the core feature for planning Cassini-like trajectories
(e.g., Earth -> Venus -> Venus -> Earth -> Jupiter -> Saturn).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

import numpy as np

from ephemeris.bodies import GM_SUN, BODY_BY_ID, BODY_BY_NAME, CelestialBody
from ephemeris.spline_cache import EphemerisCache, EphemerisRangeError
from mechanics.lambert import compute_transfer_dv
from mechanics.kepler import propagate_kepler
from mechanics.transforms import km_to_scene, jd_to_iso

logger = logging.getLogger("tars.multileg")


@dataclass
class FlybyInfo:
    """Gravity-assist flyby geometry at an intermediate body."""
    body_name: str
    body_naif_id: int
    epoch_jd: float
    epoch_iso: str

    # Incoming & outgoing v-infinity (heliocentric transfer vel - planet vel)
    v_inf_in_km_s: float      # |v_inf_in|
    v_inf_out_km_s: float     # |v_inf_out|

    # Turning angle (angle between v_inf_in and v_inf_out)
    turning_angle_deg: float

    # Minimum flyby radius for an unpowered flyby (km)
    # Computed from the turning angle and v_inf
    flyby_periapsis_km: float
    flyby_altitude_km: float   # periapsis - body radius

    # Powered delta-v needed if pure gravity assist can't achieve the turn
    # (0 if unpowered flyby is feasible)
    powered_dv_km_s: float

    # Is this flyby geometrically feasible without powered assist?
    feasible_unpowered: bool


@dataclass
class LegResult:
    """Result for a single leg of a multi-leg trajectory."""
    leg_index: int
    origin_name: str
    target_name: str

    departure_jd: float
    arrival_jd: float
    departure_iso: str
    arrival_iso: str
    tof_days: float

    dv_departure_km_s: float
    dv_arrival_km_s: float

    # Transfer orbit velocities
    v1_transfer: np.ndarray
    v2_transfer: np.ndarray

    # Planet velocities at endpoints
    v1_planet: np.ndarray
    v2_planet: np.ndarray

    # Trajectory points for visualization (in scene units)
    trajectory_points: list[dict] = field(default_factory=list)

    converged: bool = True


@dataclass
class MultiLegResult:
    """Complete multi-leg trajectory solution."""
    body_sequence: list[str]        # e.g., ["Earth", "Venus", "Earth", "Jupiter"]
    legs: list[LegResult]
    flybys: list[FlybyInfo]         # one per intermediate body

    # Mission totals
    total_dv_km_s: float            # departure + all flybys + arrival
    departure_dv_km_s: float        # initial burn
    arrival_dv_km_s: float          # final capture/insertion
    flyby_dv_km_s: float            # sum of powered flyby delta-v

    total_tof_days: float
    departure_jd: float
    arrival_jd: float


def compute_multileg_trajectory(
    body_names: list[str],
    departure_jd: float,
    leg_tof_days: list[float],
    cache: EphemerisCache,
    mu: float = GM_SUN,
    n_traj_points: int = 100,
) -> MultiLegResult:
    """Compute a multi-leg patched-conic trajectory.

    Parameters
    ----------
    body_names : sequence of body names, e.g. ["earth", "venus", "earth", "jupiter"]
    departure_jd : Julian Date of initial departure
    leg_tof_days : TOF in days for each leg (len = len(body_names) - 1)
    cache : warm EphemerisCache
    mu : central body gravitational parameter
    n_traj_points : number of trajectory points per leg for visualization

    Returns
    -------
    MultiLegResult with all legs, flybys, and totals

    Raises
    ------
    ValueError : if inputs are invalid
    EphemerisRangeError : if any epoch is outside the ephemeris range
    """
    n_legs = len(body_names) - 1
    if len(leg_tof_days) != n_legs:
        raise ValueError(
            f"Expected {n_legs} TOF values for {len(body_names)} bodies, "
            f"got {len(leg_tof_days)}"
        )
    if n_legs < 1:
        raise ValueError("Need at least 2 bodies for a trajectory")

    # Resolve all bodies
    resolved_bodies: list[CelestialBody] = []
    for name in body_names:
        name_lower = name.lower().strip()
        if name_lower in BODY_BY_NAME:
            resolved_bodies.append(BODY_BY_NAME[name_lower])
        else:
            try:
                naif_id = int(name)
                if naif_id in BODY_BY_ID:
                    resolved_bodies.append(BODY_BY_ID[naif_id])
                else:
                    raise ValueError(f"Unknown body: {name}")
            except ValueError:
                raise ValueError(f"Unknown body: {name}")

    # Compute each leg
    legs: list[LegResult] = []
    current_jd = departure_jd

    for i in range(n_legs):
        origin = resolved_bodies[i]
        target = resolved_bodies[i + 1]
        tof_days = leg_tof_days[i]
        tof_seconds = tof_days * 86400.0
        arr_jd = current_jd + tof_days

        # Get states
        r1, v1_planet = cache.get_state(origin.naif_id, current_jd)
        r2, v2_planet = cache.get_state(target.naif_id, arr_jd)

        # Solve Lambert
        result = compute_transfer_dv(r1, v1_planet, r2, v2_planet, tof_seconds, mu)

        if not result["converged"]:
            logger.warning(
                "Lambert solver did not converge for leg %d: %s -> %s (TOF=%.1f days)",
                i, origin.name, target.name, tof_days,
            )

        # Generate trajectory points for this leg (skip if not converged to avoid NaN)
        if result["converged"]:
            traj_points = _generate_leg_trajectory(
                r1, result["v1_transfer"], tof_seconds, current_jd, mu, n_traj_points,
            )
        else:
            traj_points = []

        # Sanitize inf values for JSON serialization
        dv_dep = result["dv_departure"] if np.isfinite(result["dv_departure"]) else 9999.0
        dv_arr = result["dv_arrival"] if np.isfinite(result["dv_arrival"]) else 9999.0

        leg = LegResult(
            leg_index=i,
            origin_name=origin.name,
            target_name=target.name,
            departure_jd=current_jd,
            arrival_jd=arr_jd,
            departure_iso=jd_to_iso(current_jd),
            arrival_iso=jd_to_iso(arr_jd),
            tof_days=tof_days,
            dv_departure_km_s=dv_dep,
            dv_arrival_km_s=dv_arr,
            v1_transfer=result["v1_transfer"],
            v2_transfer=result["v2_transfer"],
            v1_planet=v1_planet,
            v2_planet=v2_planet,
            trajectory_points=traj_points,
            converged=result["converged"],
        )
        legs.append(leg)
        current_jd = arr_jd

    # Compute flyby geometry at intermediate bodies
    flybys: list[FlybyInfo] = []
    for i in range(1, n_legs):
        # At body i, incoming from leg (i-1), outgoing to leg i
        prev_leg = legs[i - 1]
        next_leg = legs[i]
        body = resolved_bodies[i]

        flyby = _compute_flyby(
            body=body,
            epoch_jd=prev_leg.arrival_jd,
            v_in_transfer=prev_leg.v2_transfer,
            v_out_transfer=next_leg.v1_transfer,
            v_planet=prev_leg.v2_planet,
        )
        flybys.append(flyby)

    # Compute mission totals
    departure_dv = legs[0].dv_departure_km_s
    arrival_dv = legs[-1].dv_arrival_km_s
    flyby_dv = sum(fb.powered_dv_km_s for fb in flybys)
    total_dv = departure_dv + arrival_dv + flyby_dv
    total_tof = sum(leg.tof_days for leg in legs)

    return MultiLegResult(
        body_sequence=[b.name for b in resolved_bodies],
        legs=legs,
        flybys=flybys,
        total_dv_km_s=total_dv,
        departure_dv_km_s=departure_dv,
        arrival_dv_km_s=arrival_dv,
        flyby_dv_km_s=flyby_dv,
        total_tof_days=total_tof,
        departure_jd=departure_jd,
        arrival_jd=departure_jd + total_tof,
    )


def _compute_flyby(
    body: CelestialBody,
    epoch_jd: float,
    v_in_transfer: np.ndarray,
    v_out_transfer: np.ndarray,
    v_planet: np.ndarray,
) -> FlybyInfo:
    """Compute gravity-assist flyby geometry at an intermediate body.

    v_inf_in = v_in_transfer - v_planet  (incoming excess velocity)
    v_inf_out = v_out_transfer - v_planet  (outgoing excess velocity)

    The turning angle delta is the angle between v_inf_in and v_inf_out.

    For an unpowered flyby, the turning angle is related to the periapsis
    distance r_p by:  sin(delta/2) = 1 / (1 + r_p * v_inf^2 / mu_body)

    If the required turning angle exceeds what's achievable at the body's
    surface (r_p = radius), a powered flyby is needed.
    """
    v_inf_in = v_in_transfer - v_planet
    v_inf_out = v_out_transfer - v_planet

    v_inf_in_mag = float(np.linalg.norm(v_inf_in))
    v_inf_out_mag = float(np.linalg.norm(v_inf_out))

    # Turning angle
    if v_inf_in_mag > 1e-10 and v_inf_out_mag > 1e-10:
        cos_delta = np.dot(v_inf_in, v_inf_out) / (v_inf_in_mag * v_inf_out_mag)
        cos_delta = max(-1.0, min(1.0, float(cos_delta)))
        turning_angle = math.acos(cos_delta)
    else:
        turning_angle = 0.0

    turning_angle_deg = math.degrees(turning_angle)

    # Compute minimum flyby periapsis for unpowered flyby
    # sin(delta/2) = 1 / (1 + r_p * v_inf^2 / mu_body)
    # => r_p = mu_body / v_inf^2 * (1/sin(delta/2) - 1)
    v_inf_avg = (v_inf_in_mag + v_inf_out_mag) / 2.0
    mu_body = body.gm

    half_delta = turning_angle / 2.0
    sin_half = math.sin(half_delta) if half_delta > 1e-10 else 1e-10

    if mu_body > 0 and v_inf_avg > 1e-10 and sin_half > 1e-10:
        rp = mu_body / (v_inf_avg ** 2) * (1.0 / sin_half - 1.0)
    else:
        rp = body.radius  # fallback

    flyby_altitude = rp - body.radius
    feasible = flyby_altitude >= 0  # periapsis above surface

    # Powered delta-v needed if not feasible
    # This is the difference in v_inf magnitudes as a first-order estimate.
    # A more accurate model would compute the powered flyby at the surface.
    if not feasible:
        # Compute max turning angle achievable at body surface
        rp_min = body.radius * 1.1  # 10% altitude margin
        if mu_body > 0 and v_inf_avg > 1e-10:
            sin_max = 1.0 / (1.0 + rp_min * v_inf_avg ** 2 / mu_body)
            max_turn = 2.0 * math.asin(min(1.0, sin_max))
            # The remaining turn must come from a powered maneuver
            remaining = turning_angle - max_turn
            # Simple estimate: dv ~ v_inf * 2 * sin(remaining/2)
            powered_dv = v_inf_avg * 2.0 * math.sin(remaining / 2.0) if remaining > 0 else 0.0
        else:
            powered_dv = abs(v_inf_out_mag - v_inf_in_mag)
    else:
        powered_dv = 0.0

    return FlybyInfo(
        body_name=body.name,
        body_naif_id=body.naif_id,
        epoch_jd=epoch_jd,
        epoch_iso=jd_to_iso(epoch_jd),
        v_inf_in_km_s=v_inf_in_mag,
        v_inf_out_km_s=v_inf_out_mag,
        turning_angle_deg=turning_angle_deg,
        flyby_periapsis_km=rp,
        flyby_altitude_km=flyby_altitude,
        powered_dv_km_s=powered_dv,
        feasible_unpowered=feasible,
    )


def _generate_leg_trajectory(
    r0: np.ndarray,
    v0: np.ndarray,
    tof_seconds: float,
    departure_jd: float,
    mu: float,
    n_points: int,
) -> list[dict]:
    """Generate trajectory points for a single leg (scene units)."""
    if n_points <= 0:
        return []
    points = []
    for i in range(n_points + 1):
        t = i * tof_seconds / n_points
        if i == 0:
            pos = r0
        else:
            pos, vel = propagate_kepler(r0, v0, t, mu)
        scene_pos = km_to_scene(pos)
        points.append({
            "x": float(scene_pos[0]),
            "y": float(scene_pos[1]),
            "z": float(scene_pos[2]),
            "epoch_jd": float(departure_jd + t / 86400.0),
        })
    return points


def multileg_result_to_dict(result: MultiLegResult) -> dict:
    """Convert MultiLegResult to a JSON-serializable dict."""
    legs_out = []
    for leg in result.legs:
        legs_out.append({
            "leg_index": leg.leg_index,
            "origin": leg.origin_name,
            "target": leg.target_name,
            "departure_jd": leg.departure_jd,
            "arrival_jd": leg.arrival_jd,
            "departure_iso": leg.departure_iso,
            "arrival_iso": leg.arrival_iso,
            "tof_days": leg.tof_days,
            "dv_departure_km_s": leg.dv_departure_km_s,
            "dv_arrival_km_s": leg.dv_arrival_km_s,
            "trajectory_points": leg.trajectory_points,
            "converged": leg.converged,
        })

    flybys_out = []
    for fb in result.flybys:
        flybys_out.append({
            "body": fb.body_name,
            "naif_id": fb.body_naif_id,
            "epoch_jd": fb.epoch_jd,
            "epoch_iso": fb.epoch_iso,
            "v_inf_in_km_s": round(fb.v_inf_in_km_s, 4),
            "v_inf_out_km_s": round(fb.v_inf_out_km_s, 4),
            "turning_angle_deg": round(fb.turning_angle_deg, 2),
            "flyby_periapsis_km": round(fb.flyby_periapsis_km, 1),
            "flyby_altitude_km": round(fb.flyby_altitude_km, 1),
            "powered_dv_km_s": round(fb.powered_dv_km_s, 4),
            "feasible_unpowered": fb.feasible_unpowered,
        })

    return {
        "body_sequence": result.body_sequence,
        "n_legs": len(result.legs),
        "legs": legs_out,
        "flybys": flybys_out,
        "total_dv_km_s": round(result.total_dv_km_s, 4),
        "departure_dv_km_s": round(result.departure_dv_km_s, 4),
        "arrival_dv_km_s": round(result.arrival_dv_km_s, 4),
        "flyby_dv_km_s": round(result.flyby_dv_km_s, 4),
        "total_tof_days": round(result.total_tof_days, 2),
        "departure_jd": result.departure_jd,
        "arrival_jd": result.arrival_jd,
        "departure_iso": jd_to_iso(result.departure_jd),
        "arrival_iso": jd_to_iso(result.arrival_jd),
    }
