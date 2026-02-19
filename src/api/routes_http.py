"""HTTP REST endpoints for the Tars API.

- /health                — Health check
- /epoch-range           — Get valid ephemeris date range
- /bodies                — List all supported celestial bodies
- /bodies/{id}/ephemeris — Get position data for a body
- /optimize              — Submit optimization job
- /optimize/{id}/status  — Poll job status
- /porkchop             — Generate pork-chop plot data
- /lambert              — Single Lambert transfer calculation
- /multileg             — Multi-leg trajectory (gravity assists)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, field_validator

from ephemeris.bodies import ALL_BODIES, BODY_BY_ID, BODY_BY_NAME, CelestialBody
from ephemeris.spline_cache import EphemerisCache, EphemerisRangeError
from mechanics.lambert import compute_transfer_dv, solve_lambert
from mechanics.multileg import compute_multileg_trajectory, multileg_result_to_dict
from mechanics.transforms import (
    iso_to_jd,
    jd_to_iso,
    km_to_scene,
    positions_to_scene,
    sample_trajectory_adaptive,
)
from optimizer.dispatcher import get_job_status, submit_optimization, submit_multileg_optimization
from optimizer.gmpa import OptimizationRequest, MultiLegOptimizationRequest
from optimizer.objective import delta_v_objective, porkchop_grid
from serialization.encoder import encode_porkchop

logger = logging.getLogger("tars.api")
router = APIRouter()


from planner.engine import plan_routes

# --------------------------------------------------------------------------- #
#  Shared validators
# --------------------------------------------------------------------------- #

def _validate_iso_date(v: str) -> str:
    """Validate that a string is a parseable ISO date."""
    try:
        datetime.fromisoformat(v)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid ISO date: '{v}'. Expected format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
    return v


def _safe_iso_to_jd(iso_date: str) -> float:
    """Convert ISO date to JD, raising HTTPException on bad input."""
    try:
        return iso_to_jd(iso_date)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid date format: '{iso_date}'. Expected ISO format: YYYY-MM-DD",
        )


# --------------------------------------------------------------------------- #
#  Pydantic models for request/response
# --------------------------------------------------------------------------- #

class BodyOut(BaseModel):
    naif_id: int
    name: str
    gm: float
    radius: float
    color: str
    parent_id: int | None = None


class EphemerisQuery(BaseModel):
    start: str = Field(description="Start date ISO, e.g. 2026-01-01")
    end: str = Field(description="End date ISO, e.g. 2027-01-01")
    step_days: int = Field(default=10, ge=1, le=365)
    scene_units: bool = Field(default=True, description="Convert to scene units")

    @field_validator("start", "end")
    @classmethod
    def check_iso_date(cls, v: str) -> str:
        return _validate_iso_date(v)


class OptimizeRequest(BaseModel):
    origin: str = Field(description="Origin body name or NAIF ID, e.g. 'earth' or '399'")
    target: str = Field(description="Target body name or NAIF ID, e.g. 'mars' or '499'")
    dep_start: str = Field(description="Departure window start, ISO date")
    dep_end: str = Field(description="Departure window end, ISO date")
    tof_min_days: float = Field(default=60, ge=1)
    tof_max_days: float = Field(default=400, ge=1)
    population_size: int = Field(default=30, ge=5, le=200)
    max_iterations: int = Field(default=200, ge=10, le=2000)
    mode: str = Field(default="pareto", description="Optimization mode: 'min_dv', 'min_tof', or 'pareto'")

    @field_validator("dep_start", "dep_end")
    @classmethod
    def check_iso_date(cls, v: str) -> str:
        return _validate_iso_date(v)

    @field_validator("mode")
    @classmethod
    def check_mode(cls, v: str) -> str:
        if v not in ("min_dv", "min_tof", "pareto"):
            raise ValueError(f"mode must be 'min_dv', 'min_tof', or 'pareto', got '{v}'")
        return v


class LambertRequest(BaseModel):
    origin: str
    target: str
    departure_date: str
    tof_days: float = Field(ge=1)

    @field_validator("departure_date")
    @classmethod
    def check_iso_date(cls, v: str) -> str:
        return _validate_iso_date(v)


class PorkchopQuery(BaseModel):
    origin: str
    target: str
    dep_start: str
    dep_end: str
    tof_min_days: float = Field(default=60, ge=1)
    tof_max_days: float = Field(default=400, ge=1)
    dep_steps: int = Field(default=80, ge=10, le=300)
    tof_steps: int = Field(default=80, ge=10, le=300)
    binary: bool = Field(default=False, description="Return binary protobuf instead of JSON")

    @field_validator("dep_start", "dep_end")
    @classmethod
    def check_iso_date(cls, v: str) -> str:
        return _validate_iso_date(v)


class MultiLegRequest(BaseModel):
    body_sequence: list[str] = Field(
        description="Ordered list of body names, e.g. ['earth', 'venus', 'earth', 'jupiter']",
        min_length=2,
    )
    departure_date: str = Field(description="Departure date ISO, e.g. 2028-06-01")
    leg_tof_days: list[float] = Field(
        description="TOF in days for each leg (length = len(body_sequence) - 1)",
        min_length=1,
    )

    @field_validator("departure_date")
    @classmethod
    def check_iso_date(cls, v: str) -> str:
        return _validate_iso_date(v)


class MultiLegOptimizeRequest(BaseModel):
    body_sequence: list[str] = Field(
        description="Ordered list of body names, e.g. ['earth', 'venus', 'earth', 'jupiter']",
        min_length=2,
    )
    dep_start: str = Field(description="Departure window start, ISO date")
    dep_end: str = Field(description="Departure window end, ISO date")
    leg_tof_bounds: list[list[float]] = Field(
        description="TOF bounds [[min, max]] for each leg, in days",
        min_length=1,
    )
    population_size: int = Field(default=40, ge=5, le=200)
    max_iterations: int = Field(default=300, ge=10, le=2000)
    max_c3: float | None = Field(default=None, description="Max departure C3 (km^2/s^2)")
    mode: str = Field(default="pareto", description="Optimization mode: 'min_dv', 'min_tof', or 'pareto'")

    @field_validator("dep_start", "dep_end")
    @classmethod
    def check_iso_date(cls, v: str) -> str:
        return _validate_iso_date(v)

    @field_validator("leg_tof_bounds")
    @classmethod
    def check_tof_bounds(cls, v: list[list[float]]) -> list[list[float]]:
        for i, bounds in enumerate(v):
            if len(bounds) != 2:
                raise ValueError(f"leg_tof_bounds[{i}] must have exactly 2 elements [min, max], got {len(bounds)}")
            if bounds[0] >= bounds[1]:
                raise ValueError(f"leg_tof_bounds[{i}]: min ({bounds[0]}) must be < max ({bounds[1]})")
        return v

    @field_validator("mode")
    @classmethod
    def check_mode(cls, v: str) -> str:
        if v not in ("min_dv", "min_tof", "pareto"):
            raise ValueError(f"mode must be 'min_dv', 'min_tof', or 'pareto', got '{v}'")
        return v


class PlanRequest(BaseModel):
    origin: str
    target: str
    mode: str = Field(default="min_dv", description="'min_dv' or 'min_tof'")
    dep_start: str | None = None
    dep_end: str | None = None

    @field_validator("dep_start", "dep_end")
    @classmethod
    def check_iso_date(cls, v: str | None) -> str | None:
        if v:
            return _validate_iso_date(v)
        return v


# --------------------------------------------------------------------------- #
#  Helper to resolve body name/id
# --------------------------------------------------------------------------- #

def _resolve_body(identifier: str) -> CelestialBody:
    """Resolve a body by name or NAIF ID string."""
    # Try as NAIF ID
    try:
        naif_id = int(identifier)
        if naif_id in BODY_BY_ID:
            return BODY_BY_ID[naif_id]
    except ValueError:
        pass

    # Try as name (case-insensitive)
    name_lower = identifier.lower().strip()
    if name_lower in BODY_BY_NAME:
        return BODY_BY_NAME[name_lower]

    raise HTTPException(status_code=404, detail=f"Unknown body: {identifier}")


def _get_cache(request: Request) -> EphemerisCache:
    """Get the ephemeris cache from the app state."""
    cache: EphemerisCache = request.app.state.ephemeris
    return cache


# --------------------------------------------------------------------------- #
#  Endpoints
# --------------------------------------------------------------------------- #

@router.post("/plan")
async def plan_mission(req: PlanRequest, request: Request):
    """Find top route options for a mission."""
    cache = _get_cache(request)
    
    # Ensure bodies exist
    origin = _resolve_body(req.origin)
    target = _resolve_body(req.target)
    
    if origin.naif_id not in cache or target.naif_id not in cache:
        raise HTTPException(status_code=503, detail="Ephemeris not loaded")

    # Run planner engine
    results = plan_routes(
        req.origin, req.target, req.mode,
        req.dep_start, req.dep_end,
        cache
    )
    
    return {
        "origin": origin.name,
        "target": target.name,
        "mode": req.mode,
        "count": len(results),
        "routes": results,
    }


@router.get("/health")
async def health():
    return {"status": "ok", "service": "tars"}


@router.get("/epoch-range")
async def epoch_range(request: Request):
    """Return the valid ephemeris epoch range (useful for frontend date pickers)."""
    cache = _get_cache(request)
    body_ids = cache.available_bodies()
    if not body_ids:
        raise HTTPException(status_code=503, detail="Ephemeris not loaded")
    # Use the first body's range (all bodies share the same fetch window)
    start_jd, end_jd = cache.get_epoch_range(body_ids[0])
    return {
        "start_jd": start_jd,
        "end_jd": end_jd,
        "start_iso": jd_to_iso(start_jd)[:10],
        "end_iso": jd_to_iso(end_jd)[:10],
    }


@router.get("/bodies", response_model=list[BodyOut])
async def list_bodies():
    """List all supported celestial bodies."""
    return [
        BodyOut(
            naif_id=b.naif_id,
            name=b.name,
            gm=b.gm,
            radius=b.radius,
            color=b.color,
            parent_id=b.parent_id,
        )
        for b in ALL_BODIES
    ]


@router.get("/bodies/{identifier}/ephemeris")
async def get_ephemeris(
    identifier: str,
    request: Request,
    start: str = Query(default="2026-01-01"),
    end: str = Query(default="2027-01-01"),
    step_days: int = Query(default=10, ge=1, le=365),
    scene_units: bool = Query(default=True),
):
    """Get position data for a celestial body over a date range.

    Returns an array of [epoch_jd, x, y, z] points suitable for
    3D rendering. Uses adaptive sampling if step is small.
    """
    body = _resolve_body(identifier)
    cache = _get_cache(request)

    if body.naif_id not in cache:
        raise HTTPException(status_code=503, detail=f"Ephemeris not loaded for {body.name}")

    start_jd = _safe_iso_to_jd(start)
    end_jd = _safe_iso_to_jd(end)

    # Validate epoch range
    try:
        cache.validate_epoch_range(body.naif_id, start_jd, end_jd)
    except EphemerisRangeError as e:
        epoch_start, epoch_end = cache.get_epoch_range(body.naif_id)
        raise HTTPException(
            status_code=422,
            detail=f"Requested dates are outside ephemeris range. "
                   f"Valid range: JD {epoch_start:.2f} to {epoch_end:.2f} "
                   f"({jd_to_iso(epoch_start)[:10]} to {jd_to_iso(epoch_end)[:10]}). "
                   f"Error: {e}",
        )

    # Generate epochs
    n_points = max(2, int((end_jd - start_jd) / step_days))
    epochs = np.linspace(start_jd, end_jd, n_points)

    # Batch query
    positions = cache.get_positions_batch(body.naif_id, epochs)

    if scene_units:
        positions = positions_to_scene(positions)

    # Adaptive sampling to reduce payload
    positions, epochs = sample_trajectory_adaptive(positions, epochs)

    # Build response
    points = []
    for i in range(len(epochs)):
        points.append({
            "epoch_jd": float(epochs[i]),
            "epoch_iso": jd_to_iso(float(epochs[i])),
            "x": float(positions[i, 0]),
            "y": float(positions[i, 1]),
            "z": float(positions[i, 2]),
        })

    return {
        "body": body.name,
        "naif_id": body.naif_id,
        "n_points": len(points),
        "scene_units": scene_units,
        "points": points,
    }


@router.post("/lambert")
async def lambert_transfer(req: LambertRequest, request: Request):
    """Compute a single Lambert transfer between two bodies."""
    origin = _resolve_body(req.origin)
    target = _resolve_body(req.target)
    cache = _get_cache(request)

    dep_jd = _safe_iso_to_jd(req.departure_date)
    tof_seconds = req.tof_days * 86400.0
    arr_jd = dep_jd + req.tof_days

    # Validate origin != target
    if origin.naif_id == target.naif_id:
        raise HTTPException(status_code=422, detail="Origin and target must be different bodies")

    # Validate epochs are within ephemeris range
    try:
        cache.validate_epoch(origin.naif_id, dep_jd)
        cache.validate_epoch(target.naif_id, arr_jd)
    except EphemerisRangeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Transfer dates outside ephemeris range: {e}",
        )

    r1, v1_planet = cache.get_state(origin.naif_id, dep_jd)
    r2, v2_planet = cache.get_state(target.naif_id, arr_jd)

    result = compute_transfer_dv(r1, v1_planet, r2, v2_planet, tof_seconds)

    if not result["converged"]:
        raise HTTPException(status_code=422, detail="Lambert solver did not converge")

    # Generate trajectory points for visualization
    from mechanics.kepler import propagate_kepler
    from ephemeris.bodies import GM_SUN

    n_steps = min(200, max(20, int(req.tof_days / 2)))
    dt_step = tof_seconds / n_steps
    traj_points = []
    traj_epochs = []

    for i in range(n_steps + 1):
        t = i * dt_step
        if i == 0:
            pos = r1
        else:
            pos, vel = propagate_kepler(r1, result["v1_transfer"], t, GM_SUN)
        scene_pos = km_to_scene(pos)
        traj_points.append({
            "x": float(scene_pos[0]),
            "y": float(scene_pos[1]),
            "z": float(scene_pos[2]),
        })
        traj_epochs.append(float(dep_jd + t / 86400.0))

    return {
        "origin": origin.name,
        "target": target.name,
        "departure_jd": dep_jd,
        "departure_iso": jd_to_iso(dep_jd),
        "arrival_jd": arr_jd,
        "arrival_iso": jd_to_iso(arr_jd),
        "tof_days": req.tof_days,
        "dv_departure_km_s": result["dv_departure"],
        "dv_arrival_km_s": result["dv_arrival"],
        "dv_total_km_s": result["dv_total"],
        "trajectory_points": traj_points,
        "trajectory_epochs": traj_epochs,
    }


class AssessRequest(BaseModel):
    origin: str
    target: str
    departure_date: str
    tof_days: float = Field(ge=1)

    @field_validator("departure_date")
    @classmethod
    def check_iso_date(cls, v: str) -> str:
        return _validate_iso_date(v)


@router.post("/assess")
async def assess_launch_window(req: AssessRequest, request: Request):
    """Assess whether a specific departure date is good for launch.

    Returns trajectory details plus a 'rating' field indicating quality:
    - 'excellent': dv_total < 4 km/s
    - 'good': dv_total < 6 km/s
    - 'moderate': dv_total < 8 km/s
    - 'poor': dv_total < 12 km/s
    - 'bad': dv_total >= 12 km/s

    Also returns 'suggestion' with a nearby better departure date if available.
    """
    origin = _resolve_body(req.origin)
    target = _resolve_body(req.target)
    cache = _get_cache(request)

    for body in [origin, target]:
        if body.naif_id not in cache:
            raise HTTPException(status_code=503, detail=f"Ephemeris not loaded for {body.name}")

    dep_jd = _safe_iso_to_jd(req.departure_date)
    arr_jd = dep_jd + req.tof_days

    try:
        cache.validate_epoch(origin.naif_id, dep_jd)
        cache.validate_epoch(target.naif_id, arr_jd)
    except EphemerisRangeError as e:
        raise HTTPException(status_code=422, detail=f"Date outside ephemeris range: {e}")

    result = delta_v_objective(
        dep_jd, req.tof_days,
        origin.naif_id, target.naif_id, cache,
    )

    dv_total = result["dv_total"]

    if dv_total < 4:
        rating = "excellent"
    elif dv_total < 6:
        rating = "good"
    elif dv_total < 8:
        rating = "moderate"
    elif dv_total < 12:
        rating = "poor"
    else:
        rating = "bad"

    suggestion = None
    if rating in ("poor", "bad", "moderate"):
        best_dv = dv_total
        best_jd = dep_jd
        for delta in [-30, -15, -7, 7, 15, 30]:
            test_jd = dep_jd + delta
            try:
                cache.validate_epoch(origin.naif_id, test_jd)
                cache.validate_epoch(target.naif_id, test_jd + req.tof_days)
            except EphemerisRangeError:
                continue
            test_result = delta_v_objective(test_jd, req.tof_days, origin.naif_id, target.naif_id, cache)
            if test_result["dv_total"] < best_dv - 0.5:
                best_dv = test_result["dv_total"]
                best_jd = test_jd
        if best_jd != dep_jd:
            suggestion = {
                "departure_date": jd_to_iso(best_jd)[:10],
                "improvement_km_s": round(dv_total - best_dv, 2),
            }

    # Sanitize inf/nan for JSON serialization (Lambert may not converge)
    def _safe_float(v: float) -> float | None:
        return v if np.isfinite(v) else None

    return {
        "origin": origin.name,
        "target": target.name,
        "departure_date": req.departure_date,
        "departure_jd": dep_jd,
        "arrival_date": jd_to_iso(arr_jd)[:10],
        "arrival_jd": arr_jd,
        "tof_days": req.tof_days,
        "dv_total_km_s": _safe_float(result["dv_total"]),
        "dv_departure_km_s": _safe_float(result["dv_departure"]),
        "dv_arrival_km_s": _safe_float(result["dv_arrival"]),
        "converged": result["converged"],
        "rating": rating,
        "suggestion": suggestion,
    }


@router.post("/optimize")
async def start_optimization(req: OptimizeRequest, request: Request):
    """Submit an optimization job. Returns a job_id for tracking."""
    origin = _resolve_body(req.origin)
    target = _resolve_body(req.target)
    cache = _get_cache(request)

    # Validate bodies are loaded
    for body in [origin, target]:
        if body.naif_id not in cache:
            raise HTTPException(status_code=503, detail=f"Ephemeris not loaded for {body.name}")

    # Validate epoch ranges
    dep_start_jd = _safe_iso_to_jd(req.dep_start)
    dep_end_jd = _safe_iso_to_jd(req.dep_end)

    # Validate ordering
    if dep_start_jd >= dep_end_jd:
        raise HTTPException(status_code=422, detail="dep_start must be before dep_end")
    if req.tof_min_days >= req.tof_max_days:
        raise HTTPException(status_code=422, detail="tof_min_days must be less than tof_max_days")

    max_arr_jd = dep_end_jd + req.tof_max_days
    try:
        cache.validate_epoch_range(origin.naif_id, dep_start_jd, dep_end_jd)
        cache.validate_epoch_range(target.naif_id, dep_start_jd + req.tof_min_days, max_arr_jd)
    except EphemerisRangeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Optimization window extends outside ephemeris range: {e}",
        )

    opt_req = OptimizationRequest(
        origin_id=origin.naif_id,
        target_id=target.naif_id,
        dep_start_jd=dep_start_jd,
        dep_end_jd=dep_end_jd,
        tof_min_days=req.tof_min_days,
        tof_max_days=req.tof_max_days,
        population_size=req.population_size,
        max_iterations=req.max_iterations,
        mode=req.mode,
    )

    job_id = await submit_optimization(opt_req)

    return {
        "job_id": job_id,
        "status": "queued",
        "origin": origin.name,
        "target": target.name,
        "message": f"Optimization job submitted. Connect to WS /ws/trajectory/{job_id} for live updates.",
    }


@router.get("/optimize/{job_id}/status")
async def optimization_status(job_id: str):
    """Poll the current status of an optimization job."""
    result = await get_job_status(job_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return result


@router.post("/porkchop")
async def compute_porkchop(req: PorkchopQuery, request: Request):
    """Compute a pork-chop plot (departure date vs TOF vs delta-v).

    Can return JSON or binary protobuf (set binary=true).
    """
    origin = _resolve_body(req.origin)
    target = _resolve_body(req.target)
    cache = _get_cache(request)

    # Validate epoch ranges
    dep_start_jd = _safe_iso_to_jd(req.dep_start)
    dep_end_jd = _safe_iso_to_jd(req.dep_end)

    # Validate ordering
    if dep_start_jd >= dep_end_jd:
        raise HTTPException(status_code=422, detail="dep_start must be before dep_end")

    max_arr_jd = dep_end_jd + req.tof_max_days
    try:
        cache.validate_epoch_range(origin.naif_id, dep_start_jd, dep_end_jd)
        cache.validate_epoch_range(target.naif_id, dep_start_jd + req.tof_min_days, max_arr_jd)
    except EphemerisRangeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Pork-chop window extends outside ephemeris range: {e}",
        )

    # Run in thread pool to avoid blocking the event loop (CPU-bound)
    result = await asyncio.to_thread(
        porkchop_grid,
        origin_id=origin.naif_id,
        target_id=target.naif_id,
        dep_start_jd=dep_start_jd,
        dep_end_jd=dep_end_jd,
        tof_min_days=req.tof_min_days,
        tof_max_days=req.tof_max_days,
        cache=cache,
        dep_steps=req.dep_steps,
        tof_steps=req.tof_steps,
    )

    if req.binary:
        data = encode_porkchop(
            result["departure_jds"], result["tof_days"], result["dv_grid"],
        )
        return Response(content=data, media_type="application/octet-stream")

    # JSON response — replace inf with null
    dv_grid = result["dv_grid"].tolist()
    for i in range(len(dv_grid)):
        for j in range(len(dv_grid[i])):
            if dv_grid[i][j] == float("inf"):
                dv_grid[i][j] = None

    return {
        "origin": origin.name,
        "target": target.name,
        "departure_jds": result["departure_jds"].tolist(),
        "departure_isos": [jd_to_iso(jd) for jd in result["departure_jds"]],
        "tof_days": result["tof_days"].tolist(),
        "dv_grid": dv_grid,
        "dep_steps": req.dep_steps,
        "tof_steps": req.tof_steps,
    }


@router.post("/multileg")
async def multileg_transfer(req: MultiLegRequest, request: Request):
    """Compute a multi-leg trajectory with gravity assists.

    Chains Lambert solves through a sequence of bodies and computes
    flyby geometry (turning angle, periapsis, powered dv) at each
    intermediate body.

    Example: Earth -> Venus -> Earth -> Jupiter (VEGA trajectory)
    """
    cache = _get_cache(request)

    if len(req.leg_tof_days) != len(req.body_sequence) - 1:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(req.body_sequence) - 1} TOF values for "
                   f"{len(req.body_sequence)} bodies, got {len(req.leg_tof_days)}",
        )

    dep_jd = _safe_iso_to_jd(req.departure_date)

    try:
        result = compute_multileg_trajectory(
            body_names=req.body_sequence,
            departure_jd=dep_jd,
            leg_tof_days=req.leg_tof_days,
            cache=cache,
        )
    except EphemerisRangeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Multi-leg trajectory extends outside ephemeris range: {e}",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return multileg_result_to_dict(result)


@router.post("/optimize/multileg")
async def start_multileg_optimization(req: MultiLegOptimizeRequest, request: Request):
    """Submit a multi-leg trajectory optimization job.

    Searches over departure date and leg TOFs to minimize total delta-v
    for a gravity-assist trajectory. Returns a job_id for tracking progress
    via WebSocket at /ws/trajectory/{job_id}.
    """
    cache = _get_cache(request)
    n_legs = len(req.body_sequence) - 1

    if len(req.leg_tof_bounds) != n_legs:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {n_legs} TOF bound pairs for {len(req.body_sequence)} bodies, "
                   f"got {len(req.leg_tof_bounds)}",
        )

    # Validate all bodies exist
    for name in req.body_sequence:
        _resolve_body(name)

    # Validate epoch range covers the worst-case window
    dep_start_jd = _safe_iso_to_jd(req.dep_start)
    dep_end_jd = _safe_iso_to_jd(req.dep_end)

    # Validate ordering
    if dep_start_jd >= dep_end_jd:
        raise HTTPException(status_code=422, detail="dep_start must be before dep_end")

    max_total_tof = sum(bounds[1] for bounds in req.leg_tof_bounds)
    max_arr_jd = dep_end_jd + max_total_tof

    try:
        first_body = _resolve_body(req.body_sequence[0])
        last_body = _resolve_body(req.body_sequence[-1])
        cache.validate_epoch_range(first_body.naif_id, dep_start_jd, dep_end_jd)
        cache.validate_epoch(last_body.naif_id, max_arr_jd)
    except EphemerisRangeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Optimization window extends outside ephemeris range: {e}",
        )

    opt_req = MultiLegOptimizationRequest(
        body_names=req.body_sequence,
        dep_start_jd=dep_start_jd,
        dep_end_jd=dep_end_jd,
        leg_tof_bounds=[(b[0], b[1]) for b in req.leg_tof_bounds],
        population_size=req.population_size,
        max_iterations=req.max_iterations,
        max_c3=req.max_c3,
        mode=req.mode,
    )

    job_id = await submit_multileg_optimization(opt_req)

    return {
        "job_id": job_id,
        "status": "queued",
        "type": "multileg",
        "body_sequence": req.body_sequence,
        "message": f"Multi-leg optimization job submitted. Connect to WS /ws/trajectory/{job_id} for live updates.",
    }
