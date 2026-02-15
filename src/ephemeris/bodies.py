"""Celestial body catalog with NAIF IDs and physical parameters.

GM values (gravitational parameter, km^3/s^2) from JPL DE440/441.
Radii in km.  Horizons command strings for querying.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CelestialBody:
    naif_id: int
    name: str
    gm: float  # km^3 / s^2
    radius: float  # km (mean volumetric)
    color: str  # hex hint for frontend
    horizons_id: str  # JPL Horizons command / ID string
    parent_id: int | None = None  # NAIF ID of parent body (None = Sun-orbiting)


# --------------------------------------------------------------------------- #
#  Sun
# --------------------------------------------------------------------------- #
SUN = CelestialBody(
    naif_id=10, name="Sun", gm=1.32712440018e11,
    radius=695_700.0, color="#FDB813", horizons_id="10",
)

# --------------------------------------------------------------------------- #
#  Planets  (barycenters for Horizons queries, planet centers for GM/radius)
# --------------------------------------------------------------------------- #
MERCURY = CelestialBody(
    naif_id=199, name="Mercury", gm=2.2032e4,
    radius=2_439.7, color="#B5B5B5", horizons_id="199",
)
VENUS = CelestialBody(
    naif_id=299, name="Venus", gm=3.24859e5,
    radius=6_051.8, color="#E8CDA0", horizons_id="299",
)
EARTH = CelestialBody(
    naif_id=399, name="Earth", gm=3.986004418e5,
    radius=6_371.0, color="#6B93D6", horizons_id="399",
)
MARS = CelestialBody(
    naif_id=499, name="Mars", gm=4.282837e4,
    radius=3_389.5, color="#C1440E", horizons_id="499",
)
JUPITER = CelestialBody(
    naif_id=599, name="Jupiter", gm=1.26686534e8,
    radius=69_911.0, color="#C88B3A", horizons_id="599",
)
SATURN = CelestialBody(
    naif_id=699, name="Saturn", gm=3.7931187e7,
    radius=58_232.0, color="#E8D191", horizons_id="699",
)
URANUS = CelestialBody(
    naif_id=799, name="Uranus", gm=5.793939e6,
    radius=25_362.0, color="#D1E7E7", horizons_id="799",
)
NEPTUNE = CelestialBody(
    naif_id=899, name="Neptune", gm=6.836529e6,
    radius=24_622.0, color="#5B5DDF", horizons_id="899",
)

# --------------------------------------------------------------------------- #
#  Dwarf planets
# --------------------------------------------------------------------------- #
PLUTO = CelestialBody(
    naif_id=999, name="Pluto", gm=8.71e2,
    radius=1_188.3, color="#C2B280", horizons_id="999",
)
CERES = CelestialBody(
    naif_id=2000001, name="Ceres", gm=6.263e1,
    radius=473.0, color="#8C8C8C", horizons_id="Ceres",
)

# --------------------------------------------------------------------------- #
#  Major moons
# --------------------------------------------------------------------------- #
MOON = CelestialBody(
    naif_id=301, name="Moon", gm=4.9028e3,
    radius=1_737.4, color="#CCCCCC", horizons_id="301",
    parent_id=399,
)
PHOBOS = CelestialBody(
    naif_id=401, name="Phobos", gm=7.087546066894452e-04,
    radius=11.1, color="#917E6E", horizons_id="401",
    parent_id=499,
)
DEIMOS = CelestialBody(
    naif_id=402, name="Deimos", gm=9.615569648120313e-05,
    radius=6.2, color="#B5A78C", horizons_id="402",
    parent_id=499,
)
IO = CelestialBody(
    naif_id=501, name="Io", gm=5.959916e3,
    radius=1_821.6, color="#FFFF00", horizons_id="501",
    parent_id=599,
)
EUROPA = CelestialBody(
    naif_id=502, name="Europa", gm=3.202739e3,
    radius=1_560.8, color="#B0A890", horizons_id="502",
    parent_id=599,
)
GANYMEDE = CelestialBody(
    naif_id=503, name="Ganymede", gm=9.887834e3,
    radius=2_631.2, color="#8C7E6C", horizons_id="503",
    parent_id=599,
)
CALLISTO = CelestialBody(
    naif_id=504, name="Callisto", gm=7.179289e3,
    radius=2_410.3, color="#707060", horizons_id="504",
    parent_id=599,
)
TITAN = CelestialBody(
    naif_id=606, name="Titan", gm=8.978138e3,
    radius=2_574.7, color="#D4A017", horizons_id="606",
    parent_id=699,
)
ENCELADUS = CelestialBody(
    naif_id=602, name="Enceladus", gm=7.211454e0,
    radius=252.1, color="#FFFFFF", horizons_id="602",
    parent_id=699,
)
TRITON = CelestialBody(
    naif_id=801, name="Triton", gm=1.427598e3,
    radius=1_353.4, color="#B0C4DE", horizons_id="801",
    parent_id=899,
)
CHARON = CelestialBody(
    naif_id=901, name="Charon", gm=1.058799e2,
    radius=606.0, color="#9B9B9B", horizons_id="901",
    parent_id=999,
)

# --------------------------------------------------------------------------- #
#  Lookup tables
# --------------------------------------------------------------------------- #
ALL_BODIES: list[CelestialBody] = [
    SUN,
    MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE,
    PLUTO, CERES,
    MOON, PHOBOS, DEIMOS,
    IO, EUROPA, GANYMEDE, CALLISTO,
    TITAN, ENCELADUS,
    TRITON, CHARON,
]

BODY_BY_ID: dict[int, CelestialBody] = {b.naif_id: b for b in ALL_BODIES}
BODY_BY_NAME: dict[str, CelestialBody] = {b.name.lower(): b for b in ALL_BODIES}

# Planets only (for default trajectory endpoints)
PLANETS: list[CelestialBody] = [
    MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO,
]

# Sun GM — used as central body for heliocentric transfers
GM_SUN: float = SUN.gm  # km^3/s^2

# 1 AU in km
AU_KM: float = 1.495978707e8
