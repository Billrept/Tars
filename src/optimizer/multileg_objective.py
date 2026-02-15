"""Multi-leg trajectory objective function for optimization.

Maps an N+1 dimensional search vector [departure_jd, tof_0, tof_1, ..., tof_{N-1}]
to a scalar cost (total delta-v) via the multi-leg Lambert solver.

The optimizer treats this as a black-box function: it only sees the search vector
and the returned cost. All orbital mechanics happen inside.
"""

from __future__ import annotations

import logging

import numpy as np

from ephemeris.spline_cache import EphemerisCache, EphemerisRangeError
from mechanics.multileg import compute_multileg_trajectory, MultiLegResult

logger = logging.getLogger("tars.optimizer.multileg")

# Penalty value for infeasible solutions
INF_COST = 1e12


def multileg_objective(
    x: np.ndarray,
    body_names: list[str],
    cache: EphemerisCache,
    max_c3: float | None = None,
) -> float:
    """Evaluate multi-leg trajectory cost for a given search vector.

    Parameters
    ----------
    x : array of shape (N+1,) where N = len(body_names) - 1
        x[0] = departure Julian Date
        x[1:] = TOF in days for each leg
    body_names : ordered list of body names for the trajectory
    cache : warm EphemerisCache
    max_c3 : optional maximum departure C3 constraint (km²/s²).
        If the departure C3 exceeds this, a penalty is added.

    Returns
    -------
    Total delta-v in km/s (departure + powered flybys + arrival).
    Returns INF_COST if the trajectory is infeasible.
    """
    departure_jd = float(x[0])
    leg_tofs = [float(x[i + 1]) for i in range(len(body_names) - 1)]

    # Sanity: all TOFs must be positive
    if any(t <= 0 for t in leg_tofs):
        return INF_COST

    try:
        result = compute_multileg_trajectory(
            body_names=body_names,
            departure_jd=departure_jd,
            leg_tof_days=leg_tofs,
            cache=cache,
            n_traj_points=0,  # skip trajectory generation during optimization
        )
    except (ValueError, EphemerisRangeError):
        return INF_COST
    except Exception as e:
        logger.debug("Multi-leg objective exception: %s", e)
        return INF_COST

    # Check convergence on all legs
    if not all(leg.converged for leg in result.legs):
        return INF_COST

    cost = result.total_dv_km_s

    # Optional C3 constraint (penalty method)
    if max_c3 is not None:
        # C3 = v_inf_departure^2
        leg0 = result.legs[0]
        v_inf_dep = leg0.v1_transfer - leg0.v1_planet
        c3_dep = float(np.dot(v_inf_dep, v_inf_dep))
        if c3_dep > max_c3:
            # Quadratic penalty
            cost += 10.0 * (c3_dep - max_c3)

    return cost


def multileg_objective_full(
    x: np.ndarray,
    body_names: list[str],
    cache: EphemerisCache,
    n_traj_points: int = 100,
) -> MultiLegResult | None:
    """Evaluate and return the full MultiLegResult for visualization.

    Used to get the detailed result for the best solution found by the optimizer.

    Returns None if the trajectory is infeasible.
    """
    departure_jd = float(x[0])
    leg_tofs = [float(x[i + 1]) for i in range(len(body_names) - 1)]

    if any(t <= 0 for t in leg_tofs):
        return None

    try:
        result = compute_multileg_trajectory(
            body_names=body_names,
            departure_jd=departure_jd,
            leg_tof_days=leg_tofs,
            cache=cache,
            n_traj_points=n_traj_points,
        )
    except (ValueError, EphemerisRangeError):
        return None
    except Exception:
        return None

    if not all(leg.converged for leg in result.legs):
        return None

    return result
