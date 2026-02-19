"""Mission planning engine — coarse search for optimal launch windows."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from ephemeris.bodies import BODY_BY_NAME, GM_SUN
from ephemeris.spline_cache import EphemerisCache
from mechanics.lambert import compute_transfer_dv
from mechanics.multileg import compute_multileg_trajectory, multileg_result_to_dict
from mechanics.transforms import jd_to_iso, iso_to_jd
from planner.catalog import GA_CATALOG, ORBITAL_PERIODS_DAYS


def plan_routes(
    origin: str,
    target: str,
    mode: str,  # "min_dv" or "min_tof"
    dep_start_iso: str | None,
    dep_end_iso: str | None,
    cache: EphemerisCache,
) -> list[dict]:
    """Find top route options for the given mission parameters."""
    origin = origin.lower()
    target = target.lower()
    
    # 1. Determine search window
    if dep_start_iso and dep_end_iso:
        start_jd = iso_to_jd(dep_start_iso)
        end_jd = iso_to_jd(dep_end_iso)
    else:
        # Default: Search next 15 years
        # Use a fixed date if 'now' is not available, but cache should be loaded.
        # We can use the cache start date or just 2025-01-01
        start_jd = 2460676.5  # ~2025-01-01
        end_jd = start_jd + 15 * 365.25

    # 2. Identify candidate routes (Direct + Catalog)
    routes_to_eval = []
    
    # Always add Direct
    routes_to_eval.append({
        "type": "direct",
        "name": f"Direct {origin.title()}-{target.title()}",
        "sequence": [origin, target],
        "legs_ratio": [1.0],
    })

    # Add catalog options if available
    if target in GA_CATALOG and origin == "earth":
        for route in GA_CATALOG[target]:
            r = route.copy()
            r["type"] = "multileg"
            routes_to_eval.append(r)

    # 3. Find launch windows for each route type
    results = []
    
    for route in routes_to_eval:
        # Find optimal departure dates using a coarse 1D scan
        # For multi-leg, we scan based on the FIRST leg's target (usually Venus or Mars)
        first_target = route["sequence"][1]
        
        # Estimate Hohmann TOF for the first leg
        tof_guess = _estimate_tof(origin, first_target)
        
        # Scan step: 15 days is fine for coarse search
        scan_step = 15
        windows = _scan_for_windows(
            origin, first_target, start_jd, end_jd, tof_guess, scan_step, cache
        )
        
        # For each window, evaluate the full route
        for win_jd in windows:
            if route["type"] == "direct":
                # Evaluate direct transfer
                res = _eval_direct(origin, target, win_jd, tof_guess, cache)
                if res:
                    res["name"] = route["name"]
                    results.append(res)
            else:
                # Evaluate multi-leg
                # We need to distribute total TOF across legs
                total_tof = route.get("typical_tof_days", 1000)
                leg_tofs = [total_tof * r for r in route["legs_ratio"]]
                
                res = _eval_multileg(route["sequence"], win_jd, leg_tofs, route["name"], cache)
                if res:
                    results.append(res)

    # 4. Sort and filter results
    if mode == "min_tof":
        results.sort(key=lambda x: x["total_tof_days"])
    else:
        results.sort(key=lambda x: x["total_dv_km_s"])

    # Dedup: remove results with very similar dates (keep best)
    unique_results = []
    for r in results:
        is_dup = False
        for u in unique_results:
            # Same route name and departure within 60 days
            if u["name"] == r["name"] and abs(u["departure_jd"] - r["departure_jd"]) < 60:
                is_dup = True
                break
        if not is_dup:
            unique_results.append(r)
            
    return unique_results[:5]  # Top 5


def _scan_for_windows(
    origin: str, target: str, start_jd: float, end_jd: float, tof: float, step: int, cache: EphemerisCache
) -> list[float]:
    """Scan date range for local dV minima (Hohmann-like windows)."""
    jds = np.arange(start_jd, end_jd, step)
    dvs = []
    
    try:
        origin_id = BODY_BY_NAME[origin].naif_id
        target_id = BODY_BY_NAME[target].naif_id
    except KeyError:
        return []
    
    for jd in jds:
        try:
            r1, v1p = cache.get_state(origin_id, jd)
            r2, v2p = cache.get_state(target_id, jd + tof)
            res = compute_transfer_dv(r1, v1p, r2, v2p, tof * 86400.0, GM_SUN)
            if res["converged"]:
                dvs.append(res["dv_total"])
            else:
                dvs.append(1e9)
        except Exception:
            dvs.append(1e9)
            
    dvs = np.array(dvs)
    
    # Find local minima
    windows = []
    for i in range(1, len(dvs) - 1):
        if dvs[i] < dvs[i-1] and dvs[i] < dvs[i+1]:
            # It's a valley. Check if it's not insanely high
            if dvs[i] < 20.0:  # Relaxed threshold
                windows.append(jds[i])
                
    return windows


def _estimate_tof(origin: str, target: str) -> float:
    """Return rough Hohmann transfer time in days."""
    pair = tuple(sorted((origin, target)))
    if pair == ("earth", "mars"): return 260.0
    if pair == ("earth", "venus"): return 160.0
    if pair == ("earth", "jupiter"): return 800.0
    if pair == ("earth", "saturn"): return 1500.0
    if pair == ("earth", "mercury"): return 110.0
    
    T1 = ORBITAL_PERIODS_DAYS.get(origin, 365.25)
    T2 = ORBITAL_PERIODS_DAYS.get(target, 687.0)
    return 0.5 * abs(T1 + T2) / 2


def _eval_direct(origin: str, target: str, dep_jd: float, tof: float, cache: EphemerisCache) -> dict | None:
    try:
        origin_id = BODY_BY_NAME[origin].naif_id
        target_id = BODY_BY_NAME[target].naif_id
    except KeyError:
        return None
    
    # Search +/- 20 days around the guess to refine TOF
    best_dv = 1e9
    best_res = None
    
    # Also vary departure slightly
    for dt_dep in [-5, 0, 5]:
        d = dep_jd + dt_dep
        for dt_tof in [-20, -10, 0, 10, 20]:
            t = tof + dt_tof
            if t <= 10: continue
            try:
                r1, v1p = cache.get_state(origin_id, d)
                r2, v2p = cache.get_state(target_id, d + t)
                res = compute_transfer_dv(r1, v1p, r2, v2p, t * 86400.0, GM_SUN)
                
                if res["converged"] and res["dv_total"] < best_dv:
                    best_dv = res["dv_total"]
                    
                    # Generate points
                    from mechanics.kepler import propagate_kepler
                    from mechanics.transforms import km_to_scene
                    points = []
                    for k in range(30):
                        tm = k * (t * 86400.0) / 29
                        p, _ = propagate_kepler(r1, res["v1_transfer"], tm, GM_SUN)
                        sp = km_to_scene(p)
                        points.append({"x": float(sp[0]), "y": float(sp[1]), "z": float(sp[2])})
                    
                    best_res = {
                        "type": "direct",
                        "origin": origin,
                        "target": target,
                        "departure_jd": d,
                        "arrival_jd": d + t,
                        "departure_iso": jd_to_iso(d),
                        "arrival_iso": jd_to_iso(d + t),
                        "total_tof_days": t,
                        "total_dv_km_s": res["dv_total"],
                        "dv_departure_km_s": res["dv_departure"],
                        "dv_arrival_km_s": res["dv_arrival"],
                        "flyby_dv_km_s": 0.0,
                        "legs": [], 
                        "trajectory_points": points,
                        "flybys": [],
                        "rating": _rate_dv(res["dv_total"]),
                        "mode": "planner"
                    }
            except Exception:
                pass
            
    return best_res


def _eval_multileg(sequence: list[str], dep_jd: float, leg_tofs: list[float], name: str, cache: EphemerisCache) -> dict | None:
    try:
        # Just run one-shot with the catalog guesses
        res = compute_multileg_trajectory(
            sequence, dep_jd, leg_tofs, cache, n_traj_points=30
        )
        if not np.isfinite(res.total_dv_km_s):
            return None
            
        d = multileg_result_to_dict(res)
        d["name"] = name
        d["type"] = "multileg"
        d["rating"] = _rate_dv(res.total_dv_km_s)
        d["mode"] = "planner"
        return d
    except Exception:
        return None


def _rate_dv(dv: float) -> str:
    if dv < 6.0: return "excellent"
    if dv < 9.0: return "good"
    if dv < 12.0: return "moderate"
    if dv < 16.0: return "poor"
    return "bad"
