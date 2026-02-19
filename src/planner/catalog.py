"""Catalog of known routes and orbital parameters for mission planning."""

from __future__ import annotations

# Approximate orbital periods (days) for synodic period calculation
# T = 2*pi * sqrt(a^3/mu)
ORBITAL_PERIODS_DAYS = {
    "mercury": 87.97,
    "venus": 224.70,
    "earth": 365.25,
    "mars": 686.98,
    "jupiter": 4332.59,
    "saturn": 10759.22,
    "uranus": 30685.4,
    "neptune": 60189.0,
    "pluto": 90560.0,
}

# Catalog of known routes
# Direct is always implied for all pairs.
# These are *additional* multi-leg options.
GA_CATALOG = {
    "mercury": [
        {
            "name": "MESSENGER-like (E-V-V-M)",
            "sequence": ["earth", "venus", "venus", "mercury"],
            "legs_ratio": [0.4, 0.4, 0.2],  # Rough TOF split
            "typical_tof_days": 1200,
        },
    ],
    "jupiter": [
        {
            "name": "VEGA (E-V-E-J)",
            "sequence": ["earth", "venus", "earth", "jupiter"],
            "legs_ratio": [0.15, 0.35, 0.5],
            "typical_tof_days": 1000,
        },
        {
            "name": "EGA (E-E-J)",
            "sequence": ["earth", "earth", "jupiter"],
            "legs_ratio": [0.4, 0.6],
            "typical_tof_days": 900,
        },
    ],
    "saturn": [
        {
            "name": "VVEJGA (Cassini)",
            "sequence": ["earth", "venus", "venus", "earth", "jupiter", "saturn"],
            "legs_ratio": [0.05, 0.15, 0.15, 0.25, 0.4],
            "typical_tof_days": 2400,
        },
        {
            "name": "VEGA (E-V-E-S)",
            "sequence": ["earth", "venus", "earth", "saturn"],
            "legs_ratio": [0.1, 0.3, 0.6],
            "typical_tof_days": 1800,
        },
    ],
    "uranus": [
        {
            "name": "Jupiter Assist",
            "sequence": ["earth", "jupiter", "uranus"],
            "legs_ratio": [0.3, 0.7],
            "typical_tof_days": 4000,
        },
    ],
    "neptune": [
        {
            "name": "Jupiter Assist",
            "sequence": ["earth", "jupiter", "neptune"],
            "legs_ratio": [0.3, 0.7],
            "typical_tof_days": 5000,
        },
    ],
    "pluto": [
        {
            "name": "Jupiter Assist (New Horizons)",
            "sequence": ["earth", "jupiter", "pluto"],
            "legs_ratio": [0.2, 0.8],
            "typical_tof_days": 4500,
        },
    ],
}
