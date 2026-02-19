"""Keplerian orbital mechanics — two-body propagation and orbital elements.

All functions operate in km / km/s / seconds unless noted.
Performance-critical inner loops are JIT-compiled with Numba.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ephemeris.bodies import GM_SUN


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
#  State vector <-> Classical orbital elements
# --------------------------------------------------------------------------- #
@njit(cache=True)
def state_to_elements(r: np.ndarray, v: np.ndarray, mu: float) -> tuple:
    """Convert state vector (r, v) to classical Keplerian elements.

    Parameters
    ----------
    r : (3,) position in km
    v : (3,) velocity in km/s
    mu : gravitational parameter km^3/s^2

    Returns
    -------
    (a, e, i, raan, argp, nu)
        a    — semi-major axis (km)
        e    — eccentricity
        i    — inclination (rad)
        raan — right ascension of ascending node (rad)
        argp — argument of periapsis (rad)
        nu   — true anomaly (rad)
    """
    r_mag = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    v_mag = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

    # Specific angular momentum
    h = np.array([
        r[1] * v[2] - r[2] * v[1],
        r[2] * v[0] - r[0] * v[2],
        r[0] * v[1] - r[1] * v[0],
    ])
    h_mag = np.sqrt(h[0]**2 + h[1]**2 + h[2]**2)

    # Node vector
    n = np.array([-h[1], h[0], 0.0])
    n_mag = np.sqrt(n[0]**2 + n[1]**2)

    # Eccentricity vector
    rdotv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2]
    e_vec = np.array([
        (v_mag**2 - mu / r_mag) * r[0] / mu - rdotv * v[0] / mu,
        (v_mag**2 - mu / r_mag) * r[1] / mu - rdotv * v[1] / mu,
        (v_mag**2 - mu / r_mag) * r[2] / mu - rdotv * v[2] / mu,
    ])
    ecc = np.sqrt(e_vec[0]**2 + e_vec[1]**2 + e_vec[2]**2)

    # Semi-major axis
    energy = v_mag**2 / 2.0 - mu / r_mag
    if abs(ecc - 1.0) > 1e-10:
        a = -mu / (2.0 * energy)
    else:
        a = np.inf  # parabolic

    # Inclination
    inc = math.acos(max(-1.0, min(1.0, h[2] / h_mag)))

    # RAAN
    if n_mag > 1e-10:
        raan = math.acos(max(-1.0, min(1.0, n[0] / n_mag)))
        if n[1] < 0:
            raan = TWO_PI - raan
    else:
        raan = 0.0

    # Argument of periapsis
    if n_mag > 1e-10 and ecc > 1e-10:
        cos_argp = (n[0] * e_vec[0] + n[1] * e_vec[1] + n[2] * e_vec[2]) / (n_mag * ecc)
        argp = math.acos(max(-1.0, min(1.0, cos_argp)))
        if e_vec[2] < 0:
            argp = TWO_PI - argp
    else:
        argp = 0.0

    # True anomaly
    if ecc > 1e-10:
        cos_nu = (e_vec[0] * r[0] + e_vec[1] * r[1] + e_vec[2] * r[2]) / (ecc * r_mag)
        nu = math.acos(max(-1.0, min(1.0, cos_nu)))
        if rdotv < 0:
            nu = TWO_PI - nu
    else:
        nu = 0.0

    return a, ecc, inc, raan, argp, nu


@njit(cache=True)
def elements_to_state(a: float, ecc: float, inc: float, raan: float,
                      argp: float, nu: float, mu: float) -> tuple:
    """Convert classical orbital elements to state vector (r, v).

    Returns (r, v) each as (3,) arrays in km and km/s.
    """
    # Semi-latus rectum
    if ecc < 1.0:
        p = a * (1.0 - ecc**2)
    else:
        p = a * (ecc**2 - 1.0)

    r_mag = p / (1.0 + ecc * math.cos(nu))

    # Position and velocity in perifocal frame
    r_pf = np.array([r_mag * math.cos(nu), r_mag * math.sin(nu), 0.0])
    v_pf = np.array([
        -math.sqrt(mu / p) * math.sin(nu),
        math.sqrt(mu / p) * (ecc + math.cos(nu)),
        0.0,
    ])

    # Rotation matrix: perifocal -> inertial (ecliptic J2000)
    cos_raan = math.cos(raan)
    sin_raan = math.sin(raan)
    cos_argp = math.cos(argp)
    sin_argp = math.sin(argp)
    cos_inc = math.cos(inc)
    sin_inc = math.sin(inc)

    r = np.array([
        (cos_raan * cos_argp - sin_raan * sin_argp * cos_inc) * r_pf[0]
        + (-cos_raan * sin_argp - sin_raan * cos_argp * cos_inc) * r_pf[1],

        (sin_raan * cos_argp + cos_raan * sin_argp * cos_inc) * r_pf[0]
        + (-sin_raan * sin_argp + cos_raan * cos_argp * cos_inc) * r_pf[1],

        (sin_argp * sin_inc) * r_pf[0] + (cos_argp * sin_inc) * r_pf[1],
    ])

    v = np.array([
        (cos_raan * cos_argp - sin_raan * sin_argp * cos_inc) * v_pf[0]
        + (-cos_raan * sin_argp - sin_raan * cos_argp * cos_inc) * v_pf[1],

        (sin_raan * cos_argp + cos_raan * sin_argp * cos_inc) * v_pf[0]
        + (-sin_raan * sin_argp + cos_raan * cos_argp * cos_inc) * v_pf[1],

        (sin_argp * sin_inc) * v_pf[0] + (cos_argp * sin_inc) * v_pf[1],
    ])

    return r, v


# --------------------------------------------------------------------------- #
#  Kepler's equation solvers
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _solve_kepler_elliptic(M: float, ecc: float, tol: float = 1e-12) -> float:
    """Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson."""
    E = M + ecc * math.sin(M)  # initial guess
    for _ in range(50):
        dE = (E - ecc * math.sin(E) - M) / (1.0 - ecc * math.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


@njit(cache=True)
def _solve_kepler_hyperbolic(M: float, ecc: float, tol: float = 1e-12) -> float:
    """Solve hyperbolic Kepler's equation M = e*sinh(H) - H via Newton-Raphson."""
    H = M  # initial guess
    for _ in range(50):
        dH = (ecc * math.sinh(H) - H - M) / (ecc * math.cosh(H) - 1.0)
        H -= dH
        if abs(dH) < tol:
            break
    return H


# --------------------------------------------------------------------------- #
#  Two-body propagation
# --------------------------------------------------------------------------- #
@njit(cache=True)
def propagate_kepler(r0: np.ndarray, v0: np.ndarray, dt: float, mu: float) -> tuple:
    """Propagate state vector (r0, v0) forward by dt seconds under two-body dynamics.

    Uses universal variable formulation for robustness across orbit types.

    Returns (r, v) at time t0 + dt.
    """
    r0_mag = np.sqrt(r0[0]**2 + r0[1]**2 + r0[2]**2)
    v0_mag = np.sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)

    # Specific energy -> orbit type
    energy = v0_mag**2 / 2.0 - mu / r0_mag

    # Convert to elements, propagate mean anomaly, convert back
    a_e_i_raan_argp_nu = state_to_elements(r0, v0, mu)
    a = a_e_i_raan_argp_nu[0]
    ecc = a_e_i_raan_argp_nu[1]
    inc = a_e_i_raan_argp_nu[2]
    raan = a_e_i_raan_argp_nu[3]
    argp = a_e_i_raan_argp_nu[4]
    nu0 = a_e_i_raan_argp_nu[5]

    if ecc < 1.0 - 1e-10:
        # Elliptic
        n = math.sqrt(mu / abs(a)**3)  # mean motion

        # True anomaly -> Eccentric anomaly
        E0 = 2.0 * math.atan2(
            math.sqrt(1.0 - ecc) * math.sin(nu0 / 2.0),
            math.sqrt(1.0 + ecc) * math.cos(nu0 / 2.0),
        )
        M0 = E0 - ecc * math.sin(E0)
        M = M0 + n * dt

        # Solve Kepler's equation
        E = _solve_kepler_elliptic(M, ecc)

        # Eccentric anomaly -> True anomaly
        nu = 2.0 * math.atan2(
            math.sqrt(1.0 + ecc) * math.sin(E / 2.0),
            math.sqrt(1.0 - ecc) * math.cos(E / 2.0),
        )
    elif ecc > 1.0 + 1e-10:
        # Hyperbolic
        n = math.sqrt(mu / abs(a)**3)

        # True anomaly -> Hyperbolic anomaly
        H0 = 2.0 * math.atanh(
            math.sqrt((ecc - 1.0) / (ecc + 1.0)) * math.tan(nu0 / 2.0)
        )
        M0 = ecc * math.sinh(H0) - H0
        M = M0 + n * dt

        H = _solve_kepler_hyperbolic(M, ecc)

        nu = 2.0 * math.atan2(
            math.sqrt(ecc + 1.0) * math.sinh(H / 2.0),
            math.sqrt(ecc - 1.0) * math.cosh(H / 2.0),
        )
    else:
        # Near-parabolic: use a small-eccentricity offset to avoid singularity
        # Treat as slightly elliptic or slightly hyperbolic
        ecc_eff = 1.0 - 1e-8 if ecc <= 1.0 else 1.0 + 1e-8
        a_eff = -mu / (2.0 * energy) if abs(energy) > 1e-15 else 1e12
        n = math.sqrt(mu / abs(a_eff)**3)

        if ecc_eff < 1.0:
            E0 = 2.0 * math.atan2(
                math.sqrt(1.0 - ecc_eff) * math.sin(nu0 / 2.0),
                math.sqrt(1.0 + ecc_eff) * math.cos(nu0 / 2.0),
            )
            M0 = E0 - ecc_eff * math.sin(E0)
            M = M0 + n * dt
            E = _solve_kepler_elliptic(M, ecc_eff)
            nu = 2.0 * math.atan2(
                math.sqrt(1.0 + ecc_eff) * math.sin(E / 2.0),
                math.sqrt(1.0 - ecc_eff) * math.cos(E / 2.0),
            )
        else:
            H0 = 2.0 * math.atanh(
                math.sqrt((ecc_eff - 1.0) / (ecc_eff + 1.0)) * math.tan(nu0 / 2.0)
            )
            M0 = ecc_eff * math.sinh(H0) - H0
            M = M0 + n * dt
            H = _solve_kepler_hyperbolic(M, ecc_eff)
            nu = 2.0 * math.atan2(
                math.sqrt(ecc_eff + 1.0) * math.sinh(H / 2.0),
                math.sqrt(ecc_eff - 1.0) * math.cosh(H / 2.0),
            )

    return elements_to_state(a, ecc, inc, raan, argp, nu, mu)
