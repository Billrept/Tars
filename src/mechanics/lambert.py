"""Lambert solver — Izzo's algorithm (2015) with Numba JIT.

Solves the Lambert boundary value problem: given two position vectors
r1, r2 and a time-of-flight tof, find the velocity vectors v1, v2
that connect them under two-body dynamics.

Reference:
    Izzo, D. "Revisiting Lambert's problem."
    Celestial Mechanics and Dynamical Astronomy, 121(1), 1-15, 2015.

All units: km, seconds, km^3/s^2.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


# --------------------------------------------------------------------------- #
#  Stumpff functions  c2(psi) and c3(psi)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _stumpff_c2(psi: float) -> float:
    if abs(psi) < 1e-10:
        return 1.0 / 2.0
    elif psi > 0:
        sp = math.sqrt(psi)
        return (1.0 - math.cos(sp)) / psi
    else:
        sp = math.sqrt(-psi)
        return (math.cosh(sp) - 1.0) / (-psi)


@njit(cache=True)
def _stumpff_c3(psi: float) -> float:
    if abs(psi) < 1e-10:
        return 1.0 / 6.0
    elif psi > 0:
        sp = math.sqrt(psi)
        return (sp - math.sin(sp)) / (psi * sp)
    else:
        sp = math.sqrt(-psi)
        return (math.sinh(sp) - sp) / ((-psi) * sp)


# --------------------------------------------------------------------------- #
#  Universal-variable Lambert solver (Bate, Mueller, White approach)
#  Robust, handles elliptic + hyperbolic arcs. Numba JIT for speed.
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _stumpff_dc2(psi: float, c2: float, c3: float) -> float:
    """Derivative of c2 with respect to psi."""
    if abs(psi) < 1e-10:
        return -1.0 / 12.0
    return (1.0 - psi * c3 - 2.0 * c2) / (2.0 * psi)


@njit(cache=True)
def _stumpff_dc3(psi: float, c2: float, c3: float) -> float:
    """Derivative of c3 with respect to psi."""
    if abs(psi) < 1e-10:
        return -1.0 / 60.0
    return (c2 - 3.0 * c3) / (2.0 * psi)


@njit(cache=True)
def _solve_lambert_uv(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    prograde: bool = True,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple:
    """Solve Lambert's problem using the universal variable method.

    Uses Bate-Mueller-White (BMW) approach with Newton-Raphson iteration
    on the universal variable psi, with bisection fallback for robustness.

    Parameters
    ----------
    r1, r2 : (3,) position vectors (km)
    tof : time of flight (seconds, > 0)
    mu : gravitational parameter (km^3/s^2)
    prograde : if True, select prograde (short-way) solution
    max_iter : maximum Newton iterations
    tol : relative convergence tolerance (|tof_calc - tof| / tof < tol)

    Returns
    -------
    v1, v2 : (3,) departure and arrival velocity vectors (km/s)
    converged : bool
    """
    r1_mag = math.sqrt(r1[0]**2 + r1[1]**2 + r1[2]**2)
    r2_mag = math.sqrt(r2[0]**2 + r2[1]**2 + r2[2]**2)

    # Cross product for direction
    cross_z = r1[0] * r2[1] - r1[1] * r2[0]

    # True anomaly change
    cos_dnu = (r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2]) / (r1_mag * r2_mag)
    cos_dnu = max(-1.0, min(1.0, cos_dnu))

    if prograde:
        if cross_z < 0:
            dnu = 2.0 * math.pi - math.acos(cos_dnu)
        else:
            dnu = math.acos(cos_dnu)
    else:
        if cross_z >= 0:
            dnu = 2.0 * math.pi - math.acos(cos_dnu)
        else:
            dnu = math.acos(cos_dnu)

    # Variable A
    sin_dnu = math.sin(dnu)
    A = sin_dnu * math.sqrt(r1_mag * r2_mag / (1.0 - cos_dnu))

    if abs(A) < 1e-14:
        # Degenerate case (r1 == r2 or anti-parallel)
        return np.zeros(3), np.zeros(3), False

    # Newton-Raphson on the universal variable psi
    psi_low = -4.0 * math.pi * math.pi  # hyperbolic bound
    psi_up = 4.0 * math.pi * math.pi * 4.0  # generous upper bound for multi-rev
    psi = 0.0  # initial guess (parabolic)

    converged = False

    for _ in range(max_iter):
        c2 = _stumpff_c2(psi)
        c3 = _stumpff_c3(psi)

        sqrt_c2 = math.sqrt(abs(c2))
        if sqrt_c2 < 1e-30:
            sqrt_c2 = 1e-30

        B = r1_mag + r2_mag + A * (psi * c3 - 1.0) / sqrt_c2

        if A > 0.0 and B < 0.0:
            # Readjust bounds — psi too low
            psi_low = psi
            psi = (psi_low + psi_up) / 2.0
            continue

        if abs(c2) < 1e-30:
            # Near-parabolic, skip
            psi_low = psi
            psi = (psi_low + psi_up) / 2.0
            continue

        chi = math.sqrt(B / c2)
        chi3 = chi * chi * chi
        sqrt_B = math.sqrt(B)

        tof_calc = (chi3 * c3 + A * sqrt_B) / math.sqrt(mu)

        # Relative convergence check
        if abs(tof_calc - tof) / tof < tol:
            converged = True
            break

        # Correct derivative dtof/dpsi using chain rule through B, chi, c2, c3
        # Stumpff derivatives
        dc2 = _stumpff_dc2(psi, c2, c3)
        dc3 = _stumpff_dc3(psi, c2, c3)

        # dB/dpsi = A * [(c3 + psi*dc3)*sqrt_c2 - (psi*c3 - 1)*dc2/(2*sqrt_c2)] / c2
        dB = A * ((c3 + psi * dc3) * sqrt_c2 - (psi * c3 - 1.0) * dc2 / (2.0 * sqrt_c2)) / c2

        # dchi/dpsi = (dB*c2 - B*dc2) / (2 * c2^2 * chi)
        dchi = (dB * c2 - B * dc2) / (2.0 * c2 * c2 * chi)

        # dtof/dpsi = (3*chi^2*dchi*c3 + chi^3*dc3 + A*dB/(2*sqrt_B)) / sqrt(mu)
        dtof_dpsi = (3.0 * chi * chi * dchi * c3 + chi3 * dc3 + A * dB / (2.0 * sqrt_B)) / math.sqrt(mu)

        if abs(dtof_dpsi) < 1e-30:
            # Derivative too small, fall back to bisection
            if tof_calc < tof:
                psi_low = psi
            else:
                psi_up = psi
            psi = (psi_low + psi_up) / 2.0
            continue

        psi_new = psi + (tof - tof_calc) / dtof_dpsi

        # Update bisection bounds
        if tof_calc < tof:
            psi_low = psi
        else:
            psi_up = psi

        # Use Newton if within bounds, else bisect
        if psi_low <= psi_new <= psi_up:
            psi = psi_new
        else:
            psi = (psi_low + psi_up) / 2.0

    # Compute velocities from the final psi
    c2 = _stumpff_c2(psi)
    c3 = _stumpff_c3(psi)
    sqrt_c2 = math.sqrt(abs(c2))
    if sqrt_c2 < 1e-30:
        sqrt_c2 = 1e-30
    B = r1_mag + r2_mag + A * (psi * c3 - 1.0) / sqrt_c2

    f = 1.0 - B / r1_mag
    g = A * math.sqrt(B / mu)
    g_dot = 1.0 - B / r2_mag

    v1 = np.array([
        (r2[0] - f * r1[0]) / g,
        (r2[1] - f * r1[1]) / g,
        (r2[2] - f * r1[2]) / g,
    ])

    v2 = np.array([
        (g_dot * r2[0] - r1[0]) / g,
        (g_dot * r2[1] - r1[1]) / g,
        (g_dot * r2[2] - r1[2]) / g,
    ])

    return v1, v2, converged


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def solve_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float = 1.32712440018e11,  # Sun GM
    prograde: bool = True,
) -> dict:
    """Solve Lambert's problem.

    Parameters
    ----------
    r1, r2 : (3,) position vectors in km
    tof : time of flight in seconds (> 0)
    mu : gravitational parameter (default: Sun)
    prograde : prograde (short-way) transfer if True

    Returns
    -------
    dict with keys:
        "v1" : departure velocity (km/s)
        "v2" : arrival velocity (km/s)
        "converged" : bool
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)

    v1, v2, converged = _solve_lambert_uv(r1, r2, tof, mu, prograde)

    return {"v1": v1, "v2": v2, "converged": converged}


def compute_transfer_dv(
    r1: np.ndarray,
    v1_planet: np.ndarray,
    r2: np.ndarray,
    v2_planet: np.ndarray,
    tof: float,
    mu: float = 1.32712440018e11,
    prograde: bool = True,
) -> dict:
    """Compute total delta-v for a Lambert transfer.

    Parameters
    ----------
    r1 : departure position (km)
    v1_planet : departure planet velocity (km/s)
    r2 : arrival position (km)
    v2_planet : arrival planet velocity (km/s)
    tof : time of flight (seconds)
    mu : central body GM
    prograde : prograde transfer

    Returns
    -------
    dict with keys:
        "dv_departure" : departure delta-v magnitude (km/s)
        "dv_arrival" : arrival delta-v magnitude (km/s)
        "dv_total" : total delta-v (km/s)
        "v1_transfer" : transfer orbit departure velocity
        "v2_transfer" : transfer orbit arrival velocity
        "converged" : bool
    """
    result = solve_lambert(r1, r2, tof, mu, prograde)

    if not result["converged"]:
        return {
            "dv_departure": np.inf,
            "dv_arrival": np.inf,
            "dv_total": np.inf,
            "v1_transfer": np.zeros(3),
            "v2_transfer": np.zeros(3),
            "converged": False,
        }

    dv1 = np.linalg.norm(result["v1"] - v1_planet)
    dv2 = np.linalg.norm(result["v2"] - v2_planet)

    return {
        "dv_departure": float(dv1),
        "dv_arrival": float(dv2),
        "dv_total": float(dv1 + dv2),
        "v1_transfer": result["v1"],
        "v2_transfer": result["v2"],
        "converged": True,
    }
