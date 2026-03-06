// js/planetData.js

// ── Display Configuration ──────────────────────────────────────────────────
export const DISPLAY_RADIUS = {
  10: 6.0,       // Sun
  199: 1.2,      // Mercury
  299: 1.6,      // Venus
  399: 1.8,      // Earth
  499: 1.5,      // Mars
  599: 4.0,      // Jupiter
  699: 3.5,      // Saturn
  799: 2.5,      // Uranus
  899: 2.4,      // Neptune
  999: 1.0,      // Pluto
  2000001: 0.7   // Ceres
};

export const TEXTURE_MAP = {
  10:  '2k_sun.jpg',
  199: '2k_mercury.jpg',
  299: '2k_venus_surface.jpg',
  399: '2k_earth_daymap.jpg',
  '399_clouds': '2k_earth_clouds.jpg',     // Cloud Map
  '399_specular': '2k_earth_specular_map.tif', // Water reflection map (optional, uses .tif or .jpg)
  '399_night': '2k_earth_nightmap.jpg',    // (Optional)
  499: '2k_mars.jpg',
  599: '2k_jupiter.jpg',
  699: '2k_saturn.jpg',
  '699_ring': '2k_saturn_ring_alpha.png',  // Your ring texture
  799: '2k_uranus.jpg',
  899: '2k_neptune.jpg',
  999: '2k_pluto.jpg',
  2000001: '2k_ceres_fictional.jpg' 
};

export const ROTATION_PERIODS = {
  10:  25.0,
  199: 58.6,
  299: -243.0,
  399: 1.0,
  499: 1.03,
  599: 0.41,
  699: 0.45,
  799: 0.72,
  899: 0.67,
  999: 6.39,
  2000001: 0.375
};

// ── Rich Info Dictionary ──────────────────────────────────────────────────
export const PLANET_INFO = {
  10: { 
    type: 'Star', 
    gravity: '27.9 g',
    moons: 'N/A',
    atm: 'Hydrogen, Helium',
    radius: '696,340 km', day: '25 days', year: '230 M yr', temp: '5,500°C', 
    desc: 'The star at the center of our Solar System. Its gravity holds the solar system together.' 
  },
  199: { 
    type: 'Terrestrial Planet', 
    gravity: '0.38 g',
    moons: '0',
    atm: 'Minimal (Exosphere)',
    radius: '2,439 km', day: '59 days', year: '88 days', temp: '167°C', 
    desc: 'The smallest planet. It has a huge iron core and is shrinking as it cools.' 
  },
  299: { 
    type: 'Terrestrial Planet', 
    gravity: '0.90 g',
    moons: '0',
    atm: 'CO₂ (Thick)',
    radius: '6,051 km', day: '243 days', year: '225 days', temp: '464°C', 
    desc: 'Runaway greenhouse effect makes it the hottest planet. It spins backward compared to Earth.' 
  },
  399: { 
    type: 'Terrestrial Planet', 
    gravity: '1.0 g',
    moons: '1',
    atm: 'N₂, O₂',
    radius: '6,371 km', day: '24h', year: '365.25 d', temp: '15°C', 
    desc: 'The only known planet to harbor life. 70% of the surface is covered in water.' 
  },
  499: { 
    type: 'Terrestrial Planet', 
    gravity: '0.38 g',
    moons: '2',
    atm: 'CO₂ (Thin)',
    radius: '3,389 km', day: '24h 37m', year: '687 days', temp: '-65°C', 
    desc: 'Home to Olympus Mons, the largest volcano in the solar system. The goal of future colonization.' 
  },
  599: { 
    type: 'Gas Giant', 
    gravity: '2.52 g',
    moons: '95',
    atm: 'H₂, He',
    radius: '69,911 km', day: '9h 56m', year: '12 years', temp: '-110°C', 
    desc: 'The largest planet. Its Great Red Spot is a storm that has raged for centuries.' 
  },
  699: { 
    type: 'Gas Giant', 
    gravity: '1.06 g',
    moons: '146',
    atm: 'H₂, He',
    radius: '58,232 km', day: '10h 42m', year: '29 years', temp: '-140°C', 
    desc: 'Famous for its complex ring system composed of ice and rock particles.' 
  },
  799: { 
    type: 'Ice Giant', 
    gravity: '0.88 g',
    moons: '28',
    atm: 'H₂, He, Methane',
    radius: '25,362 km', day: '17h 14m', year: '84 years', temp: '-195°C', 
    desc: 'The "sideways planet" rotates at a 98-degree tilt, likely due to a massive collision.' 
  },
  899: { 
    type: 'Ice Giant', 
    gravity: '1.14 g',
    moons: '16',
    atm: 'H₂, He, Methane',
    radius: '24,622 km', day: '16h 6m', year: '165 years', temp: '-200°C', 
    desc: 'Dark, cold, and whipped by supersonic winds. It was the first planet found by math, not observation.' 
  },
  999: { 
    type: 'Dwarf Planet', 
    gravity: '0.06 g',
    moons: '5',
    atm: 'N₂, Methane',
    radius: '1,188 km', day: '153 hours', year: '248 years', temp: '-225°C', 
    desc: 'Located in the Kuiper belt. It has a heart-shaped glacier made of nitrogen ice.' 
  },
  2000001: { 
    type: 'Dwarf Planet', 
    gravity: '0.03 g',
    moons: '0',
    atm: 'Water Vapor (Thin)',
    radius: '473 km', day: '9 hours', year: '4.6 years', temp: '-105°C', 
    desc: 'The largest object in the asteroid belt. It may hold a subsurface ocean.' 
  }
};
