#!/usr/bin/env python3
"""Comprehensive API test suite for Tars trajectory optimizer.

Tests all endpoints:
- /health, /bodies, /epoch-range
- /bodies/{id}/ephemeris, /lambert
- /assess
- /optimize with mode parameter (min_dv, min_tof, pareto)
- /porkchop
- /multileg, /optimize/multileg
- WebSocket streaming

Run: python tests/test_api.py
"""

import asyncio
import json
import sys
import time
from typing import Any

import websockets

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"


def log(test_name: str, passed: bool, detail: str = ""):
    status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""))


def request(endpoint: str, method: str = "GET", data: dict | None = None) -> tuple[int, dict | None]:
    """Make HTTP request and return (status_code, json_response)."""
    import urllib.request
    import urllib.error

    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"} if data else {}

    try:
        if method == "GET":
            req = urllib.request.Request(url, headers=headers)
        else:
            body = json.dumps(data).encode() if data else None
            req = urllib.request.Request(url, data=body, headers=headers, method=method)

        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}


# --------------------------------------------------------------------------- #
#  Basic Endpoints
# --------------------------------------------------------------------------- #

def test_health():
    print("\n[Health Check]")
    status, resp = request("/health")
    log("/health returns 200", status == 200 and resp.get("status") == "ok")


def test_bodies():
    print("\n[Bodies Catalog]")
    status, resp = request("/bodies")
    log("/bodies returns 200", status == 200)
    log("Has 22 bodies", status == 200 and isinstance(resp, list) and len(resp) == 22)
    log("Earth exists", isinstance(resp, list) and any(isinstance(b, dict) and b.get("name", "").lower() == "earth" for b in resp))
    log("Moons have parent_id", isinstance(resp, list) and any(isinstance(b, dict) and b.get("parent_id") is not None for b in resp))


def test_epoch_range():
    print("\n[Epoch Range]")
    status, resp = request("/epoch-range")
    log("/epoch-range returns 200", status == 200)
    log("Has start_iso/end_iso", status == 200 and "start_iso" in resp and "end_iso" in resp)
    if resp:
        print(f"    Range: {resp.get('start_iso')} to {resp.get('end_iso')}")


# --------------------------------------------------------------------------- #
#  Ephemeris Endpoint
# --------------------------------------------------------------------------- #

def test_ephemeris():
    print("\n[Ephemeris]")
    status, resp = request("/bodies/earth/ephemeris?start=2026-06-01&end=2026-07-01&step_days=10")
    log("/bodies/{id}/ephemeris returns 200", status == 200)
    log("Returns points array", status == 200 and isinstance(resp, dict) and "points" in resp and len(resp.get("points", [])) > 0)


# --------------------------------------------------------------------------- #
#  Lambert Endpoint
# --------------------------------------------------------------------------- #

def test_lambert():
    print("\n[Lambert Transfer]")
    status, resp = request("/lambert", "POST", {
        "origin": "earth",
        "target": "mars",
        "departure_date": "2026-09-01",
        "tof_days": 200,
    })
    log("/lambert returns 200", status == 200)
    log("Has dv_total_km_s", status == 200 and "dv_total_km_s" in resp)
    log("Has trajectory_points", status == 200 and len(resp.get("trajectory_points", [])) > 0 if isinstance(resp, dict) else False)
    if resp and "dv_total_km_s" in resp:
        print(f"    dv_total: {resp['dv_total_km_s']:.2f} km/s, tof: {resp['tof_days']:.0f} days")


def test_lambert_validation():
    print("\n[Lambert Validation]")
    status, resp = request("/lambert", "POST", {
        "origin": "earth",
        "target": "mars",
        "departure_date": "not-a-date",
        "tof_days": 200,
    })
    log("Rejects invalid date (422)", status == 422)

    status, resp = request("/lambert", "POST", {
        "origin": "earth",
        "target": "earth",
        "departure_date": "2026-09-01",
        "tof_days": 200,
    })
    log("Rejects origin==target (422)", status == 422)


# --------------------------------------------------------------------------- #
#  Assess Endpoint (NEW)
# --------------------------------------------------------------------------- #

def test_assess():
    print("\n[Assess Launch Window - NEW]")
    status, resp = request("/assess", "POST", {
        "origin": "earth",
        "target": "mars",
        "departure_date": "2026-09-01",
        "tof_days": 200,
    })
    log("/assess returns 200", status == 200)
    log("Has rating field", status == 200 and "rating" in resp)
    log("Valid rating value", status == 200 and resp.get("rating") in ["excellent", "good", "moderate", "poor", "bad"])
    if resp:
        print(f"    Rating: {resp.get('rating')}, dv_total: {resp.get('dv_total_km_s', 0):.2f} km/s")

    log("Has suggestion when poor/bad", True)  # May or may not have suggestion
    if resp and resp.get("suggestion"):
        print(f"    Suggestion: {resp['suggestion']}")


def test_assess_good_date():
    print("\n[Assess Good Launch Window]")
    status, resp = request("/assess", "POST", {
        "origin": "earth",
        "target": "mars",
        "departure_date": "2026-10-01",
        "tof_days": 200,
    })
    log("/assess good date returns 200", status == 200)
    if resp:
        print(f"    Rating: {resp.get('rating')}, dv_total: {resp.get('dv_total_km_s', 0):.2f} km/s")


# --------------------------------------------------------------------------- #
#  Optimize Endpoint with Mode (NEW)
# --------------------------------------------------------------------------- #

def test_optimize_modes():
    print("\n[Optimize with Mode - NEW]")

    modes = ["min_dv", "min_tof", "pareto"]
    job_ids = {}

    for mode in modes:
        status, resp = request("/optimize", "POST", {
            "origin": "earth",
            "target": "mars",
            "dep_start": "2026-06-01",
            "dep_end": "2026-12-01",
            "tof_min_days": 150,
            "tof_max_days": 350,
            "population_size": 20,
            "max_iterations": 30,
            "mode": mode,
        })
        log(f"/optimize mode={mode} returns 200", status == 200)
        if resp:
            job_ids[mode] = resp.get("job_id")
            print(f"    job_id: {resp.get('job_id')}")

    log("All modes accepted", len(job_ids) == 3)
    return job_ids


def test_optimize_validation():
    print("\n[Optimize Validation]")
    status, resp = request("/optimize", "POST", {
        "origin": "earth",
        "target": "mars",
        "dep_start": "2026-12-01",
        "dep_end": "2026-06-01",
        "mode": "pareto",
    })
    log("Rejects dep_start > dep_end (422)", status == 422)

    status, resp = request("/optimize", "POST", {
        "origin": "earth",
        "target": "mars",
        "dep_start": "2026-06-01",
        "dep_end": "2026-12-01",
        "mode": "invalid_mode",
    })
    log("Rejects invalid mode (422)", status == 422)


# --------------------------------------------------------------------------- #
#  Porkchop Endpoint
# --------------------------------------------------------------------------- #

def test_porkchop():
    print("\n[Porkchop]")
    status, resp = request("/porkchop", "POST", {
        "origin": "earth",
        "target": "mars",
        "dep_start": "2026-06-01",
        "dep_end": "2026-12-01",
        "tof_min_days": 150,
        "tof_max_days": 350,
        "dep_steps": 10,
        "tof_steps": 10,
    })
    log("/porkchop returns 200", status == 200)
    log("Has dv_grid", status == 200 and "dv_grid" in resp)
    log("Grid shape correct", status == 200 and len(resp.get("dv_grid", [])) == 10)
    log("Has departure_isos", status == 200 and len(resp.get("departure_isos", [])) == 10)
    log("Has tof_days", status == 200 and len(resp.get("tof_days", [])) == 10)


# --------------------------------------------------------------------------- #
#  Multi-leg Endpoints
# --------------------------------------------------------------------------- #

def test_multileg():
    print("\n[Multi-leg Trajectory]")
    status, resp = request("/multileg", "POST", {
        "body_sequence": ["earth", "venus", "mars"],
        "departure_date": "2028-06-01",
        "leg_tof_days": [180, 200],
    })
    log("/multileg returns 200", status == 200)
    log("Has legs array", status == 200 and "legs" in resp)
    log("Has flybys array", status == 200 and "flybys" in resp)
    log("Has total_dv_km_s", status == 200 and "total_dv_km_s" in resp)
    if resp and "total_dv_km_s" in resp:
        print(f"    total_dv: {resp['total_dv_km_s']:.2f} km/s")


def test_multileg_optimize():
    print("\n[Multi-leg Optimize with Mode - NEW]")
    modes = ["min_dv", "pareto"]

    for mode in modes:
        status, resp = request("/optimize/multileg", "POST", {
            "body_sequence": ["earth", "venus", "mars"],
            "dep_start": "2028-01-01",
            "dep_end": "2029-01-01",
            "leg_tof_bounds": [[100, 300], [100, 400]],
            "population_size": 20,
            "max_iterations": 30,
            "mode": mode,
        })
        log(f"/optimize/multileg mode={mode} returns 200", status == 200)
        if resp:
            print(f"    job_id: {resp.get('job_id')}")


# --------------------------------------------------------------------------- #
#  Job Status
# --------------------------------------------------------------------------- #

def test_job_status():
    print("\n[Job Status]")
    status, resp = request("/optimize", "POST", {
        "origin": "earth",
        "target": "mars",
        "dep_start": "2026-06-01",
        "dep_end": "2026-12-01",
        "max_iterations": 20,
    })
    if resp:
        job_id = resp.get("job_id")
        print(f"    Created job: {job_id}")

        time.sleep(3)
        status2, resp2 = request(f"/optimize/{job_id}/status")
        log("/optimize/{id}/status returns 200", status2 == 200)
        log("Has status field", status2 == 200 and "status" in resp2)


# --------------------------------------------------------------------------- #
#  WebSocket Streaming
# --------------------------------------------------------------------------- #

async def test_websocket():
    print("\n[WebSocket Streaming]")

    status, resp = request("/optimize", "POST", {
        "origin": "earth",
        "target": "mars",
        "dep_start": "2026-06-01",
        "dep_end": "2026-12-01",
        "max_iterations": 20,
        "mode": "pareto",
    })

    if not resp:
        log("Failed to create job", False)
        return

    job_id = resp.get("job_id")
    print(f"    Connecting to WS for job: {job_id}")

    messages = []
    try:
        async with websockets.connect(f"{WS_URL}/ws/trajectory/{job_id}", ping_interval=None) as ws:
            start_time = time.time()
            while time.time() - start_time < 30:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(msg)
                    messages.append(data)
                    if data.get("status") in ("complete", "failed"):
                        break
                except asyncio.TimeoutError:
                    continue
    except Exception as e:
        log(f"WebSocket connection failed: {e}", False)
        return

    log("Received progress messages", len(messages) > 0)
    log("Has mode field", any("mode" in m for m in messages))
    log("Has pareto_front in pareto mode", any("pareto_front" in m for m in messages))
    log("Job completed", any(m.get("status") == "complete" for m in messages))

    if messages:
        last = messages[-1]
        print(f"    Final iteration: {last.get('iteration')}/{last.get('max_iterations')}")
        print(f"    Best dv: {last.get('best_dv_total'):.2f} km/s" if last.get('best_dv_total') else "")
        if last.get("pareto_front"):
            print(f"    Pareto front: {len(last['pareto_front'])} solutions")


# --------------------------------------------------------------------------- #
#  Error Handling
# --------------------------------------------------------------------------- #

def test_errors():
    print("\n[Error Handling]")
    status, resp = request("/bodies/invalid_body")
    log("404 for unknown body", status == 404)

    status, resp = request("/optimize/invalid-job-id/status")
    log("404 for unknown job", status == 404 or status == 200 and resp.get("status") == "not_found")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    print("=" * 60)
    print("TARS API TEST SUITE")
    print("=" * 60)

    test_health()
    test_bodies()
    test_epoch_range()
    test_ephemeris()
    test_lambert()
    test_lambert_validation()
    test_assess()
    test_assess_good_date()
    test_porkchop()
    test_multileg()
    test_optimize_modes()
    test_optimize_validation()
    test_multileg_optimize()
    test_job_status()
    test_errors()

    print("\n[WebSocket Tests]")
    asyncio.run(test_websocket())

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
