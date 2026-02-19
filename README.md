# Interplanetary Trajectory Optimizer

(Will be)Interactive 3D multi-leg gravity-assist trajectory optimizer for a space science course.
Backend in Python (FastAPI + Redis + ARQ), frontend in Three.js.

## Quick Start

```bash
# Clone and run
cp .env.example .env
docker compose up --build

# Wait ~1-2 min for first boot (fetches ephemeris from JPL Horizons for 22 bodies)
# After that, cached to disk -- subsequent starts take ~5s

# Open
# Frontend:  http://localhost:3000
# API:       http://localhost:8000
# API docs:  http://localhost:8000/docs  (auto-generated Swagger UI)
```

Requirements: Docker + Docker Compose. Nothing else.

---

## Architecture

```
                      +-----------+
                      |  Frontend |  Three.js (nginx :3000)
                      +-----+-----+
                            |
                   HTTP + WebSocket
                            |
                      +-----v-----+
                      | API Server|  FastAPI (uvicorn :8000)
                      |           |
                      | - Routes  |  11 HTTP + 2 WebSocket endpoints
                      | - Ephem   |  Spline-interpolated planetary data in RAM
                      +-----+-----+
                            |
                       Redis pub/sub
                            |
                      +-----v-----+
                      |  Worker   |  ARQ background optimizer
                      +-----------+
```

**Three-layer design:**

| Layer | Role | Key files |
|-------|------|-----------|
| API Gateway | Validates input, serves queries, streams WS | `src/api/routes_http.py`, `routes_ws.py` |
| Hot Cache | In-RAM cubic B-spline ephemeris for 22 bodies | `src/ephemeris/spline_cache.py` |
| Cold Worker | Background optimization via ARQ + Redis | `src/workers/worker.py`, `src/optimizer/` |

---

## Project Structure

```
src/
  main.py                    # FastAPI app, CORS, lifespan (cache warm-up)
  config.py                  # Settings (env vars, ephemeris range, Redis URL)
  api/
    routes_http.py           # All HTTP endpoints + Pydantic models
    routes_ws.py             # WebSocket endpoints (optimization stream, ephemeris stream)
  ephemeris/
    bodies.py                # 22 celestial body definitions (planets, Pluto, Ceres, moons)
    horizons_client.py       # JPL Horizons REST API client (async)
    spline_cache.py          # In-RAM spline cache with parent-relative moon handling
  mechanics/
    lambert.py               # Lambert solver (universal variable, Numba JIT)
    kepler.py                # Kepler propagation (Numba JIT)
    multileg.py              # Multi-leg trajectory solver + flyby geometry
    transforms.py            # Coordinate transforms, JD<->ISO, km<->scene units
  optimizer/
    objective.py             # Single-leg delta-v objective, porkchop grid
    multileg_objective.py    # Multi-leg objective function (with optional C3 constraint)
    gmpa.py                  # Grey Wolf Optimizer (PLACEHOLDER -- see below)
    dispatcher.py            # Job submission, Redis pub/sub progress streaming
  serialization/
    encoder.py               # Protobuf encoder with manual binary fallback
  workers/
    worker.py                # ARQ worker: run_optimization, run_multileg_optimization
frontend/
  index.html                 # UI layout
  css/style.css              # Styling
  js/app.js                  # Three.js scene, API calls, orbit rendering
```

---

## For the Tata and Pino (Swapping in The real from research paper)

The Grey Wolf Optimizer in `src/optimizer/gmpa.py` is a **placeholder**. It works but is
intentionally simple. You need to replace it with your research-paper algorithm.

### What you need to implement

There are **two optimizer classes**, one for single-leg and one for multi-leg:

#### 1. Single-leg optimizer

```python
# File: src/optimizer/gmpa.py (or your own file)

class YourOptimizer:
    """Replace GreyWolfOptimizer with this."""

    def __init__(self, request: OptimizationRequest, cache: EphemerisCache):
        self.req = request    # Has: origin_id, target_id, dep_start_jd, dep_end_jd,
                              #      tof_min_days, tof_max_days, population_size,
                              #      max_iterations
        self.cache = cache    # EphemerisCache -- call cache.get_state(naif_id, jd) to
                              # get (position, velocity) for any body at any epoch

    def run(self) -> Generator[OptimizationProgress, None, None]:
        """Yield progress at each iteration. MUST yield at least once at the end."""
        # Search space: 2D -- [departure_jd, tof_days]
        # Objective: minimize total delta-v

        for iteration in range(self.req.max_iterations):
            # ... your algorithm ...

            yield OptimizationProgress(
                iteration=iteration,
                max_iterations=self.req.max_iterations,
                best_dv_total=...,          # float, km/s
                best_departure_jd=...,      # float, Julian Date
                best_tof_days=...,          # float
                best_dv_departure=...,      # float, km/s
                best_dv_arrival=...,        # float, km/s
                converged=True,             # bool
                population_best_dvs=[...],  # top-3 dv values (for frontend display)
            )
```

#### 2. Multi-leg optimizer (gravity assists)

```python
class YourMultiLegOptimizer:
    """Replace MultiLegGreyWolfOptimizer with this."""

    def __init__(self, request: MultiLegOptimizationRequest, cache: EphemerisCache):
        self.req = request    # Has: body_names (list[str]), dep_start_jd, dep_end_jd,
                              #      leg_tof_bounds (list[tuple[min,max]]),
                              #      population_size, max_iterations, max_c3 (optional)
        self.cache = cache

    def run(self) -> Generator[MultiLegOptimizationProgress, None, None]:
        """Yield progress at each iteration."""
        # Search space: (N+1)-D -- [departure_jd, tof_leg0, tof_leg1, ...]
        # N = len(body_names) - 1

        for iteration in range(self.req.max_iterations):
            yield MultiLegOptimizationProgress(
                iteration=iteration,
                max_iterations=self.req.max_iterations,
                best_dv_total=...,          # float
                best_departure_jd=...,      # float
                best_leg_tof_days=[...],    # list[float], one per leg
                best_total_tof_days=...,    # float
                best_dv_departure=...,      # float
                best_dv_arrival=...,        # float
                best_dv_flyby=...,          # float (sum of powered flyby dv)
                converged=True,
                body_sequence=self.req.body_names,
                population_best_dvs=[...],
            )
```

### Objective functions (already implemented, you can reuse)

```python
from optimizer.objective import delta_v_objective
from optimizer.multileg_objective import multileg_objective, multileg_objective_full

# Single-leg: returns float (total dv, or inf if infeasible)
dv = delta_v_objective(dep_jd, tof_days, origin_id, target_id, cache)

# Multi-leg: returns float (total dv with optional C3 penalty)
# pos = np.array([departure_jd, tof_leg0, tof_leg1, ...])
dv = multileg_objective(pos, body_names, cache, max_c3=None)

# Multi-leg with full result (returns MultiLegResult or None)
result = multileg_objective_full(pos, body_names, cache, n_traj_points=0)
```

### Where to wire it in

1. Edit your optimizer class in `src/optimizer/gmpa.py` (or create a new file)
2. Update imports in `src/workers/worker.py` (lines 18-22)
3. That's it. The dispatcher, API routes, and WebSocket streaming all work unchanged.

### Key constraint

Your `run()` generator must:
- Yield `OptimizationProgress` / `MultiLegOptimizationProgress` dataclasses
- Yield at least every ~10 iterations (so the WebSocket doesn't time out)
- Yield a final progress with `iteration >= max_iterations` to signal completion

---

## For the Nano Role

### Scene coordinate system

Backend returns ecliptic coordinates (km) converted to scene units (1 AU = 1000 units).
Three.js mapping:
```
three.x = ecliptic.x
three.y = ecliptic.z   (ecliptic Z becomes "up")
three.z = -ecliptic.y  (flip for right-handed coords)
```

### Key API interactions

All API calls go to `http://localhost:8000`. See the API Reference section below.

The frontend connects two WebSockets:
1. `/ws/ephemeris/stream` -- real-time planetary positions for animation
2. `/ws/trajectory/{job_id}` -- optimization progress streaming

---

## API Reference

Base URL: `http://localhost:8000`

### GET /health

Health check.

**Response:** `{"status": "ok", "service": "tars"}`

---

### GET /epoch-range

Returns the valid date range for all queries (the ephemeris cache window).

**Response:**
```json
{
  "start_jd": 2460676.5,
  "end_jd": 2473459.5,
  "start_iso": "2025-01-01",
  "end_iso": "2060-01-01"
}
```

---

### GET /bodies

List all 22 supported celestial bodies.

**Response:** Array of:
```json
{
  "naif_id": 399,
  "name": "Earth",
  "gm": 398600.4418,
  "radius": 6371.0,
  "color": "#6B93D6",
  "parent_id": null
}
```

`parent_id` is set for moons (e.g., `301` Moon has `parent_id: 399` Earth).

---

### GET /bodies/{identifier}/ephemeris

Get position data for a body over a date range. `identifier` can be a name (`"earth"`) or NAIF ID (`"399"`).

**Query params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| start | str | "2026-01-01" | Start date (ISO) |
| end | str | "2027-01-01" | End date (ISO) |
| step_days | int | 10 | Sample interval |
| scene_units | bool | true | Convert to scene units |

**Response:**
```json
{
  "body": "Earth",
  "naif_id": 399,
  "n_points": 37,
  "scene_units": true,
  "points": [
    {"epoch_jd": 2461041.5, "epoch_iso": "2026-01-01T00:00:00", "x": -175.2, "y": 0.03, "z": 985.1}
  ]
}
```

---

### POST /assess

Assess whether a specific departure date is a good launch window. Returns a quality rating and an optional suggestion for a better nearby date.

**Request body:**
```json
{
  "origin": "earth",
  "target": "mars",
  "departure_date": "2026-09-01",
  "tof_days": 200
}
```

**Response:**
```json
{
  "origin": "Earth",
  "target": "Mars",
  "departure_date": "2026-09-01",
  "departure_jd": 2461327.5,
  "arrival_date": "2027-03-20",
  "arrival_jd": 2461527.5,
  "tof_days": 200.0,
  "dv_total_km_s": 5.83,
  "dv_departure_km_s": 3.12,
  "dv_arrival_km_s": 2.71,
  "converged": true,
  "rating": "good",
  "suggestion": null
}
```

Rating thresholds: `excellent` (<4), `good` (<6), `moderate` (<8), `poor` (<12), `bad` (>=12 km/s).
When rating is `moderate`, `poor`, or `bad`, nearby dates (+/-7,15,30 days) are tested and a `suggestion` is returned if a better date is found.

---

### POST /lambert

Compute a single Lambert transfer between two bodies.

**Request body:**
```json
{
  "origin": "earth",
  "target": "mars",
  "departure_date": "2028-11-15",
  "tof_days": 259
}
```

**Response:**
```json
{
  "origin": "Earth",
  "target": "Mars",
  "departure_jd": 2462090.5,
  "departure_iso": "2028-11-15T00:00:00",
  "arrival_jd": 2462349.5,
  "arrival_iso": "2029-08-01T00:00:00",
  "tof_days": 259.0,
  "dv_departure_km_s": 3.385,
  "dv_arrival_km_s": 3.970,
  "dv_total_km_s": 7.356,
  "trajectory_points": [{"x": 598.7, "y": 787.3, "z": -0.05}, ...],
  "trajectory_epochs": [2462090.5, ...]
}
```

`dv_departure_km_s` = |v_transfer - v_planet| at departure = hyperbolic excess velocity.
So `C3 = dv_departure_km_s^2`.

---

### POST /multileg

Compute a multi-leg trajectory with gravity assists.

**Request body:**
```json
{
  "body_sequence": ["earth", "venus", "earth", "jupiter"],
  "departure_date": "2030-01-15",
  "leg_tof_days": [140, 340, 580]
}
```

**Response:**
```json
{
  "body_sequence": ["Earth", "Venus", "Earth", "Jupiter"],
  "n_legs": 3,
  "total_dv_km_s": 64.37,
  "departure_dv_km_s": 5.99,
  "arrival_dv_km_s": 9.38,
  "flyby_dv_km_s": 0.47,
  "total_tof_days": 1060.0,
  "departure_jd": 2462516.5,
  "arrival_jd": 2463576.5,
  "departure_iso": "2030-01-15T00:00:00",
  "arrival_iso": "2032-12-10T00:00:00",
  "legs": [
    {
      "leg_index": 0,
      "origin": "Earth",
      "target": "Venus",
      "departure_jd": 2462516.5,
      "arrival_jd": 2462656.5,
      "departure_iso": "2030-01-15T00:00:00",
      "arrival_iso": "2030-06-04T00:00:00",
      "tof_days": 140.0,
      "dv_departure_km_s": 5.99,
      "dv_arrival_km_s": 8.66,
      "trajectory_points": [{"x": ..., "y": ..., "z": ..., "epoch_jd": ...}, ...],
      "converged": true
    }
  ],
  "flybys": [
    {
      "body": "Venus",
      "naif_id": 299,
      "epoch_jd": 2462656.5,
      "epoch_iso": "2030-06-04T00:00:00",
      "v_inf_in_km_s": 8.66,
      "v_inf_out_km_s": 6.38,
      "turning_angle_deg": 42.1,
      "flyby_periapsis_km": 12500.0,
      "flyby_altitude_km": 6448.2,
      "powered_dv_km_s": 0.0,
      "feasible_unpowered": true
    }
  ]
}
```

---

### POST /porkchop

Generate a porkchop plot grid (departure date vs TOF vs delta-v).

**Request body:**
```json
{
  "origin": "earth",
  "target": "mars",
  "dep_start": "2028-01-01",
  "dep_end": "2029-01-01",
  "tof_min_days": 150,
  "tof_max_days": 350,
  "dep_steps": 80,
  "tof_steps": 80,
  "binary": false
}
```

**Response:**
```json
{
  "origin": "Earth",
  "target": "Mars",
  "departure_jds": [2461941.5, ...],
  "departure_isos": ["2028-01-01T00:00:00", ...],
  "tof_days": [150.0, 152.5, ...],
  "dv_grid": [[7.2, 8.1, null, ...], ...],
  "dep_steps": 80,
  "tof_steps": 80
}
```

`dv_grid[i][j]` = total delta-v for departure `i`, TOF `j`. `null` = did not converge.
Set `binary: true` to get protobuf encoding instead.

---

### POST /optimize

Submit a single-leg optimization job. Returns immediately with a `job_id`.

**Mode options:** `"min_dv"` (minimize delta-v), `"min_tof"` (minimize time of flight), `"pareto"` (build Pareto front of dv vs tof). Default: `"pareto"`.

**Request body:**
```json
{
  "origin": "earth",
  "target": "mars",
  "dep_start": "2028-01-01",
  "dep_end": "2029-01-01",
  "tof_min_days": 150,
  "tof_max_days": 350,
  "population_size": 30,
  "max_iterations": 200,
  "mode": "pareto"
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "origin": "Earth",
  "target": "Mars",
  "message": "Optimization job submitted. Connect to WS /ws/trajectory/{job_id} for live updates."
}
```

---

### POST /optimize/multileg

Submit a multi-leg optimization job.

**Mode options:** `"min_dv"` (minimize delta-v), `"min_tof"` (minimize time of flight), `"pareto"` (build Pareto front). Default: `"pareto"`.

**Request body:**
```json
{
  "body_sequence": ["earth", "venus", "earth", "jupiter"],
  "dep_start": "2029-06-01",
  "dep_end": "2030-06-01",
  "leg_tof_bounds": [[100, 200], [250, 450], [400, 700]],
  "population_size": 40,
  "max_iterations": 300,
  "max_c3": null,
  "mode": "pareto"
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "type": "multileg",
  "body_sequence": ["earth", "venus", "earth", "jupiter"],
  "message": "Multi-leg optimization job submitted. Connect to WS /ws/trajectory/{job_id} for live updates."
}
```

---

### GET /optimize/{job_id}/status

Poll the current status of an optimization job (alternative to WebSocket).

**Response (running):**
```json
{
  "job_id": "...",
  "status": "running",
  "result": {
    "iteration": 50,
    "max_iterations": 300,
    "best_dv_total": 12.47,
    ...
  }
}
```

**Response (complete):**
```json
{
  "job_id": "...",
  "status": "complete",
  "result": { ... }
}
```

---

### WebSocket: /ws/trajectory/{job_id}

Stream real-time optimization progress. Connect after submitting a `POST /optimize` or `/optimize/multileg`.

**Messages received (JSON):**

Single-leg progress:
```json
{
  "status": "running",
  "iteration": 42,
  "max_iterations": 200,
  "best_dv_total": 5.63,
  "best_departure_jd": 2462050.5,
  "best_tof_days": 220.0,
  "best_dv_departure": 3.2,
  "best_dv_arrival": 2.43,
  "converged": true,
  "population_best_dvs": [5.63, 5.89, 6.12]
}
```

Multi-leg progress:
```json
{
  "status": "running",
  "type": "multileg",
  "iteration": 42,
  "max_iterations": 300,
  "best_dv_total": 12.47,
  "best_departure_jd": 2462288.5,
  "best_leg_tof_days": [140, 340, 580],
  "best_total_tof_days": 1060,
  "best_dv_departure": 3.9,
  "best_dv_arrival": 8.1,
  "best_dv_flyby": 0.47,
  "converged": true,
  "body_sequence": ["earth", "venus", "earth", "jupiter"],
  "population_best_dvs": [12.47, 13.2, 14.1]
}
```

Final message has `"status": "complete"`. Connection closes after.

---

### WebSocket: /ws/ephemeris/stream

Stream real-time planetary positions for 3D animation.

**Send config (JSON) on connect:**
```json
{
  "body_ids": [399, 499, 599],
  "start_jd": 2461041.5,
  "speed": 1.0,
  "fps": 30,
  "scene_units": true,
  "binary": false
}
```

All fields optional. Defaults: all planets, 2026-01-01, 1 day/sec, 30 FPS, scene units, JSON.

**Messages received:**
```json
{
  "epoch_jd": 2461041.5,
  "epoch_iso": "2026-01-01T00:00:00",
  "bodies": [
    {"body_id": 399, "name": "Earth", "position": [-175.2, 0.03, 985.1]}
  ]
}
```

**Control messages (send as text):** `"pause"`, `"resume"`, `"stop"`, or a new config JSON.

---

## Bodies Catalog (22 total)

| NAIF ID | Name | Type |
|---------|------|------|
| 10 | Sun | Star |
| 199 | Mercury | Planet |
| 299 | Venus | Planet |
| 399 | Earth | Planet |
| 499 | Mars | Planet |
| 599 | Jupiter | Planet |
| 699 | Saturn | Planet |
| 799 | Uranus | Planet |
| 899 | Neptune | Planet |
| 999 | Pluto | Dwarf planet |
| 2000001 | Ceres | Dwarf planet |
| 301 | Moon | Moon (Earth) |
| 401 | Phobos | Moon (Mars) |
| 402 | Deimos | Moon (Mars) |
| 501 | Io | Moon (Jupiter) |
| 502 | Europa | Moon (Jupiter) |
| 503 | Ganymede | Moon (Jupiter) |
| 504 | Callisto | Moon (Jupiter) |
| 606 | Titan | Moon (Saturn) |
| 602 | Enceladus | Moon (Saturn) |
| 801 | Triton | Moon (Neptune) |
| 901 | Charon | Moon (Pluto) |

---

## Configuration (.env)

```bash
REDIS_URL=redis://redis:6379       # Redis connection
EPHEMERIS_START=2025-01-01          # Cache start date
EPHEMERIS_END=2060-01-01            # Cache end date
EPHEMERIS_STEP_DAYS=1               # Sample interval (days)
```

---

## Development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Need Redis running locally
redis-server &

# Start API
cd src && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Start worker (separate terminal)
cd src && arq workers.worker.WorkerSettings

# Serve frontend (separate terminal)
cd frontend && python -m http.server 3000
```

---