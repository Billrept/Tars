"""Objective / cost functions for trajectory optimization.

Evaluates the total delta-v for a given (departure_epoch, tof) pair
using the Lambert solver and ephemeris cache.
"""

from __future__ import annotations

import numpy as np

from ephemeris.bodies import GM_SUN
from ephemeris.spline_cache import EphemerisCache, EphemerisRangeError
from mechanics.lambert import compute_transfer_dv


def delta_v_objective(
    departure_jd: float,
    tof_days: float,
    origin_id: int,
    target_id: int,
    cache: EphemerisCache,
    mu: float = GM_SUN,
    prograde: bool = True,
) -> dict:
    """Compute total delta-v for a single-leg transfer.

    Parameters
    ----------
    departure_jd : Julian Date of departure
    tof_days : time of flight in days
    origin_id : NAIF ID of departure body
    target_id : NAIF ID of arrival body
    cache : warm EphemerisCache instance
    mu : central body GM (default Sun)
    prograde : prograde transfer

    Returns
    -------
    dict with "dv_total", "dv_departure", "dv_arrival", "converged",
    "v1_transfer", "v2_transfer", "departure_jd", "arrival_jd", "tof_days"
    """
    tof_seconds = tof_days * 86400.0
    arrival_jd = departure_jd + tof_days

    # Clamp to valid epoch range
    if tof_seconds <= 0:
        return _inf_result(departure_jd, arrival_jd, tof_days)

    try:
        r1, v1_planet = cache.get_state(origin_id, departure_jd)
        r2, v2_planet = cache.get_state(target_id, arrival_jd)
    except (KeyError, EphemerisRangeError):
        return _inf_result(departure_jd, arrival_jd, tof_days)

    result = compute_transfer_dv(r1, v1_planet, r2, v2_planet, tof_seconds, mu, prograde)

    result["departure_jd"] = departure_jd
    result["arrival_jd"] = arrival_jd
    result["tof_days"] = tof_days
    return result


def _inf_result(dep: float, arr: float, tof: float) -> dict:
    return {
        "dv_departure": np.inf,
        "dv_arrival": np.inf,
        "dv_total": np.inf,
        "v1_transfer": np.zeros(3),
        "v2_transfer": np.zeros(3),
        "converged": False,
        "departure_jd": dep,
        "arrival_jd": arr,
        "tof_days": tof,
    }


def porkchop_grid(
    origin_id: int,
    target_id: int,
    dep_start_jd: float,
    dep_end_jd: float,
    tof_min_days: float,
    tof_max_days: float,
    cache: EphemerisCache,
    dep_steps: int = 100,
    tof_steps: int = 100,
    mu: float = GM_SUN,
) -> dict:
    """Compute a pork-chop plot grid of delta-v values.

    Returns
    -------
    dict with:
        "departure_jds" : (M,) array
        "tof_days" : (N,) array
        "dv_grid" : (M, N) array of total delta-v (inf where Lambert fails)
        "dv_dep_grid" : (M, N) departure delta-v
        "dv_arr_grid" : (M, N) arrival delta-v
    """
    dep_jds = np.linspace(dep_start_jd, dep_end_jd, dep_steps)
    tofs = np.linspace(tof_min_days, tof_max_days, tof_steps)

    dv_grid = np.full((dep_steps, tof_steps), np.inf)

    for i, dep_jd in enumerate(dep_jds):
        for j, tof in enumerate(tofs):
            res = delta_v_objective(dep_jd, tof, origin_id, target_id, cache, mu)
            if res["converged"]:
                dv_grid[i, j] = res["dv_total"]

    return {
        "departure_jds": dep_jds,
        "tof_days": tofs,
        "dv_grid": dv_grid,
    }
