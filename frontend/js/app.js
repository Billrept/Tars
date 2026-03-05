// ═══════════════════════════════════════════════════════════════════════════
//  TARS Frontend — Interplanetary Trajectory Designer
//  Three.js 3D scene + API client + UI
// ═══════════════════════════════════════════════════════════════════════════

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { jdToIso, pointToUnixMs, findClosestIndexByTime } from './utils/dateUtils.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const API = 'http://localhost:8000';
const WS = 'ws://localhost:8000';

// ── Viridis colormap (reversed: low=yellow/bright, high=purple/dark) ──────
const VIRIDIS = [
  [0.993, 0.906, 0.144], [0.741, 0.873, 0.150], [0.478, 0.821, 0.318],
  [0.267, 0.749, 0.441], [0.135, 0.659, 0.518], [0.128, 0.567, 0.551],
  [0.164, 0.471, 0.558], [0.207, 0.372, 0.553], [0.253, 0.265, 0.530],
  [0.282, 0.141, 0.458], [0.267, 0.004, 0.329],
];

// ── Mass modeling defaults ────────────────────────────────────────────────
const MASS_M0 = 2000;           // wet mass, kg
const MASS_ISP = 320;            // bipropellant Isp, seconds
const G0_KMS = 9.80665e-3;     // gravitational accel, km/s²

function massEstimate(dvKms, m0 = MASS_M0, isp = MASS_ISP) {
  const ve = isp * G0_KMS;
  const mf = m0 * Math.exp(-dvKms / ve);
  return { final: mf, propellant: m0 - mf, ratio: m0 / mf };
}

// ── State ──────────────────────────────────────────────────────────────────
let bodies = [];                    // from /bodies
let bodyMeshes = {};                // naif_id -> THREE.Mesh
let transferLine = null;            // current Lambert arc
let multiLegLines = [];             // multi-leg arc lines
let scene, camera, renderer, labelRenderer, controls;
let ephemerisWs = null;             // WebSocket for streaming
let simPaused = false;
let simSpeed = 10;
let epochRange = { start: '2025-01-01', end: '2059-12-31' }; // updated from backend
let stepDays = 3;
let currentDate = new Date(epochRange.start);
let orbitPoints = {};
let planetTrails = {};      // naif_id -> THREE.Points
let planetTrailHist = {};   // naif_id -> Array<THREE.Vector3>
const TRAIL_MAX_POINTS = 100; // how long the trail is (increase for longer)
let lastTrailSimTime = {};
const TRAIL_SIM_INTERVAL_MS = 6 * 60 * 60 * 1000; 
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();
let focusedBodyId = null; 
let previousBodyPosition = new THREE.Vector3(); // To track delta movement
let hoveredBodyId = null;

// Planet display radii (scene units) — exaggerated for visibility
const DISPLAY_RADIUS = {
  10: 6,       // Sun
  199: 1.2,    // Mercury
  299: 1.6,    // Venus
  399: 1.8,    // Earth
  499: 1.5,    // Mars
  599: 4,      // Jupiter
  699: 3.5,    // Saturn
  799: 2.5,    // Uranus
  899: 2.4,    // Neptune
  999: 1.0,    // Pluto
};

// ── Planet Data Dictionary ────────────────────────────────────────────────
const PLANET_INFO = {
  10:  { type: 'Star', radius: '696,340 km', day: '25 days', year: '230 M yr', temp: '5,500°C', desc: 'The star at the center of our Solar System.' },
  199: { type: 'Planet', radius: '2,439 km', day: '59 days', year: '88 days', temp: '167°C', desc: 'The smallest planet in the Solar System and the closest to the Sun.' },
  299: { type: 'Planet', radius: '6,051 km', day: '243 days', year: '225 days', temp: '464°C', desc: 'Second planet from the Sun. It has a thick atmosphere trapping heat.' },
  399: { type: 'Planet', radius: '6,371 km', day: '24 hours', year: '365 days', temp: '15°C', desc: 'Our home. The only known planet to harbor life.' },
  499: { type: 'Planet', radius: '3,389 km', day: '24h 37m', year: '687 days', temp: '-65°C', desc: 'The Red Planet. Dusty, cold, desert world with a very thin atmosphere.' },
  599: { type: 'Gas Giant', radius: '69,911 km', day: '9h 56m', year: '12 years', temp: '-110°C', desc: 'The largest planet in the Solar System.' },
  699: { type: 'Gas Giant', radius: '58,232 km', day: '10h 42m', year: '29 years', temp: '-140°C', desc: 'Adorned with a dazzling, complex system of icy rings.' },
  799: { type: 'Ice Giant', radius: '25,362 km', day: '17h 14m', year: '84 years', temp: '-195°C', desc: 'Rotates at a nearly 90-degree angle from the plane of its orbit.' },
  899: { type: 'Ice Giant', radius: '24,622 km', day: '16h 6m', year: '165 years', temp: '-200°C', desc: 'The most distant major planet, dark, cold, and whipped by supersonic winds.' },
  999: { type: 'Dwarf Planet', radius: '1,188 km', day: '153 hours', year: '248 years', temp: '-225°C', desc: 'A dwarf planet in the Kuiper belt, a ring of bodies beyond Neptune.' }
};

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  initScene();
  await fetchBodies();
  await fetchEpochRange();
  populateSelects();
  fetchOrbits();
  connectEphemeris();
  bindEvents();
  animate();
}

// ── Three.js Scene Setup ───────────────────────────────────────────────────
function initScene() {
  const container = document.getElementById('viewport');
  scene = new THREE.Scene();

  // Camera — looking down the ecliptic
  camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100000);
  camera.position.set(0, 1800, 2200);
  camera.lookAt(0, 0, 0);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  container.appendChild(renderer.domElement);

  // CSS2D labels renderer
  labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.domElement.style.position = 'absolute';
  labelRenderer.domElement.style.top = '0';
  labelRenderer.domElement.style.left = '0';
  labelRenderer.domElement.style.pointerEvents = 'none';
  container.appendChild(labelRenderer.domElement);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 50;
  controls.maxDistance = 30000;

  // Ambient light
  scene.add(new THREE.AmbientLight(0x334466, 0.5));

  // Sun point light
  const sunLight = new THREE.PointLight(0xfff8e7, 2, 0, 0.5);
  sunLight.position.set(0, 0, 0);
  scene.add(sunLight);

  // Starfield background
  createStarfield();

  // Grid helper (ecliptic plane reference, very faint)
  const grid = new THREE.GridHelper(8000, 40, 0x1a1a3e, 0x111122);
  grid.material.transparent = true;
  grid.material.opacity = 0.3;
  scene.add(grid);

  // Resize
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    labelRenderer.setSize(window.innerWidth, window.innerHeight);
  });
}

function createStarfield() {
  const geo = new THREE.BufferGeometry();
  const count = 6000;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const r = 15000 + Math.random() * 30000;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    positions[i * 3 + 2] = r * Math.cos(phi);
    const b = 0.5 + Math.random() * 0.5;
    colors[i * 3] = b;
    colors[i * 3 + 1] = b;
    colors[i * 3 + 2] = b + Math.random() * 0.2;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.PointsMaterial({ size: 1.5, vertexColors: true, sizeAttenuation: false });
  scene.add(new THREE.Points(geo, mat));
}

// ── API ────────────────────────────────────────────────────────────────────
async function fetchEpochRange() {
  try {
    const res = await fetch(`${API}/epoch-range`);
    if (res.ok) {
      const data = await res.json();
      epochRange.start = data.start_iso;
      epochRange.end = data.end_iso;
    }
  } catch (_) {}
}

async function fetchBodies() {
  try {
    const res = await fetch(`${API}/bodies`);
    bodies = await res.json();
    setStatus('ok', `${bodies.length} bodies loaded`);
  } catch (e) {
    setStatus('err', 'API unreachable');
    console.error(e);
  }
}

function getPlanets() {
  // Only top-level bodies (no moons) for selects
  return bodies.filter(b => b.parent_id === null && b.naif_id !== 10);
}

function populateSelects() {
  const planets = getPlanets();
  const selectors = [
    'pc-origin', 'pc-target',
    'plan-origin', 'plan-target',
  ];
  for (const id of selectors) {
    const sel = document.getElementById(id);
    if (!sel) continue;
    sel.innerHTML = '';
    for (const p of planets) {
      const opt = document.createElement('option');
      opt.value = p.name.toLowerCase();
      opt.textContent = p.name;
      sel.appendChild(opt);
    }
  }
  // Defaults
  document.getElementById('plan-origin').value = 'earth';
  document.getElementById('plan-target').value = 'mars';
}

async function fetchOrbits() {
  const planets = bodies.filter(b => b.parent_id === null);
  for (const body of planets) {
    try {
      const res = await fetch(
        `${API}/bodies/${body.name.toLowerCase()}/ephemeris?start=${epochRange.start}&end=${epochRange.end}&step_days=${stepDays}&scene_units=true`
      );
      const data = await res.json();

      orbitPoints[data.body] = data.points;
    } catch (e) {
      console.warn(`Failed to fetch orbit for ${body.name}:`, e);
    }
  }
}

function removePlanetTrail(bodyNaifId) {
  const prev = planetTrails[bodyNaifId];
  if (!prev) return;
  prev.parent?.remove(prev);
  prev.geometry?.dispose?.();
  prev.material?.dispose?.();
  planetTrails[bodyNaifId] = null;
}

function updatePlanetTrail(body, worldPosVec3) {
  const id = body.naif_id;
  if (!planetTrailHist[id]) planetTrailHist[id] = [];
  const hist = planetTrailHist[id];

  // push newest position
  hist.push(worldPosVec3.clone());
  while (hist.length > TRAIL_MAX_POINTS) hist.shift();

  // rebuild the trail geometry (simple + works)
  removePlanetTrail(id);

  const positions = [];
  const colors = [];
  const base = new THREE.Color(body.color);
  const n = hist.length;

  for (let i = 0; i < n; i++) {
    const v = hist[i];
    positions.push(v.x, v.y, v.z);

    // fade: old dim -> new bright
    const t = n > 1 ? i / (n - 1) : 1;
    const brightness = 0.05 + 0.95 * t;
    colors.push(base.r * brightness, base.g * brightness, base.b * brightness);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

  const mat = new THREE.ShaderMaterial({
    uniforms: {},
    vertexShader: `
      varying vec3 vColor;
      void main() {
        vColor = color;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = 3.0;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      void main() {
        float d = length(gl_PointCoord - vec2(0.5));
        if (d > 0.5) discard;
        gl_FragColor = vec4(vColor, 1.0);
      }
    `,
    vertexColors: true,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });

  const pts = new THREE.Points(geo, mat);
  pts.userData.type = "planetTrail";
  pts.userData.naifId = id;

  scene.add(pts);
  planetTrails[id] = pts;
}

function createCss2DLabel(text) {
  const div = document.createElement('div');
  div.className = 'planet-label';
  div.textContent = text;

  const label = new CSS2DObject(div);
  label.userData.isPlanetLabel = true;
  return label;
}

// ── Create Body Meshes ─────────────────────────────────────────────────────
function ensureBodyMesh(body) {
  if (bodyMeshes[body.naif_id]) return bodyMeshes[body.naif_id];

  const radius = DISPLAY_RADIUS[body.naif_id] || 0.8;
  const color = new THREE.Color(body.color);

  let mesh;
  if (body.naif_id === 10) {
    // Sun — emissive sphere + glow
    const geo = new THREE.SphereGeometry(radius, 32, 32);
    const mat = new THREE.MeshBasicMaterial({ color });
    mesh = new THREE.Mesh(geo, mat);

    // Glow sprite
    const spriteMat = new THREE.SpriteMaterial({
      map: createGlowTexture(),
      color: 0xfff0c0,
      transparent: true,
      opacity: 0.4,
      blending: THREE.AdditiveBlending,
    });
    const glow = new THREE.Sprite(spriteMat);
    glow.scale.set(radius * 8, radius * 8, 1);
    mesh.add(glow);
  } else {
    const geo = new THREE.SphereGeometry(radius, 24, 24);
    const mat = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.8,
      metalness: 0.1,
      emissive: color,
      emissiveIntensity: 0.05,
    });
    mesh = new THREE.Mesh(geo, mat);
  }

  // Create an invisible hitbox (2x to 4x larger than the visual planet)
  // For very small planets, we ensure a minimum clickable size
  const hitboxRadius = Math.max(radius * 10, 3.0); 
  const hitboxGeo = new THREE.SphereGeometry(hitboxRadius, 16, 16);
  const hitboxMat = new THREE.MeshBasicMaterial({ 
    visible: false, // Invisible!
    color: 0xff0000,
    wireframe: true // Helpful for debugging if you set visible: true
  });
  
  const hitbox = new THREE.Mesh(hitboxGeo, hitboxMat);
  hitbox.userData = { isHitbox: true, parentBody: body }; // Tag it
  mesh.add(hitbox); // Attach to planet so it moves with it

  // Name label (CSS2D)
  const label = createCss2DLabel(body.name);
  label.position.set(0, radius - 25, 0);
  mesh.add(label);

  mesh.userData = { body };
  scene.add(mesh);
  bodyMeshes[body.naif_id] = mesh;
  return mesh;
}

function createGlowTexture() {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,248,200,1)');
  grad.addColorStop(0.3, 'rgba(255,220,100,0.4)');
  grad.addColorStop(1, 'rgba(255,200,50,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  return tex;
}

function createTextSprite(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold 42px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.8;
  ctx.fillText(text, 128, 64);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  return new THREE.Sprite(mat);
}

// ── WebSocket Ephemeris Streaming ──────────────────────────────────────────
function connectEphemeris() {
  if (ephemerisWs) {
    try { ephemerisWs.close(); } catch (_) {}
  }

  ephemerisWs = new WebSocket(`${WS}/ws/ephemeris/stream`);

  ephemerisWs.onopen = () => {
    setStatus('ok', 'Streaming');
    // Send config — only planets (no moons for now to keep it clean)
    const planetIds = bodies.filter(b => b.parent_id === null).map(b => b.naif_id);
    ephemerisWs.send(JSON.stringify({
      body_ids: planetIds,
      start_jd: 2460676.5,   // ~2025-01-01
      speed: simSpeed,
      fps: 30,
      scene_units: true,
    }));
  };

  ephemerisWs.onmessage = (event) => {
    const snap = JSON.parse(event.data);
    currentDate = new Date(snap.epoch_iso);
    const currentSimMs = currentDate.getTime();

    updatePlanetPositions(snap);
  };

  ephemerisWs.onclose = () => {
    setStatus('err', 'Disconnected');
    // Reconnect after 3s
    setTimeout(connectEphemeris, 3000);
  };

  ephemerisWs.onerror = () => {
    setStatus('err', 'WS error');
  };
}

function updatePlanetPositions(snapshot) {
  // Update epoch display
  const dateStr = snapshot.epoch_iso ? snapshot.epoch_iso.split('T')[0] : '';
  document.getElementById('status-text').textContent = `Streaming  ${dateStr}`;

  for (const bd of snapshot.bodies) {
    const bodyInfo = bodies.find(b => b.naif_id === bd.body_id);
    if (!bodyInfo) continue;

    const mesh = ensureBodyMesh(bodyInfo);
    // Backend: x=ecliptic-x, y=ecliptic-y, z=ecliptic-z
    // Three.js: x=right, y=up, z=toward-camera
    // Map: three.x = ecliptic.x, three.y = ecliptic.z, three.z = -ecliptic.y
    mesh.position.set(bd.position[0], bd.position[2], -bd.position[1]);
    const simMs = new Date(snapshot.epoch_iso).getTime();
    const id = bodyInfo.naif_id;

    if (!lastTrailSimTime[id]) {
      lastTrailSimTime[id] = simMs;
      updatePlanetTrail(bodyInfo, mesh.position);
    } else {
      const delta = simMs - lastTrailSimTime[id];
      if (delta >= TRAIL_SIM_INTERVAL_MS) {
        lastTrailSimTime[id] = simMs;
        updatePlanetTrail(bodyInfo, mesh.position);
      }
    }
  }
}

function onMouseClick(event) {
  // 1. Calculate mouse position
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);

  // 2. Intersect with planets
  const meshes = Object.values(bodyMeshes);
  const intersects = raycaster.intersectObjects(meshes, true);

  if (intersects.length > 0) {
    // Get the first object hit
    const hitObj = intersects[0].object;

    // Check if it's our invisible hitbox
    if (hitObj.userData.isHitbox) {
      // Use the parent body data we stored
      focusOnBody(hitObj.userData.parentBody.naif_id);
      return;
    }

    // Standard check (if they clicked the visual mesh directly)
    let object = hitObj;
    while(object.parent && !object.userData.body) {
      object = object.parent;
    }

    if (object.userData.body) {
      focusOnBody(object.userData.body.naif_id);
    }
  } else {
    // Clicked empty space
    if (focusedBodyId !== null) {
      unlockCamera();
    }
  }
}

function unlockCamera() {
  focusedBodyId = null;
  // Hide (Add .hidden class to trigger CSS transition)
  document.getElementById('planet-info-panel').classList.add('hidden');
}



function focusOnBody(naifId) {
  focusedBodyId = naifId;
  const mesh = bodyMeshes[naifId];
  
  if (!mesh) return;

  // Store the current position so we can calculate the delta in the next frame
  previousBodyPosition.copy(mesh.position);

  // OPTIONAL: Snap camera closer immediately upon click
  // This moves the camera to a fixed offset (e.g., 100 units away)
  // If you prefer to keep the camera where it is and just start following, remove these 4 lines:
  const offset = new THREE.Vector3(50, 50, 50); 
  camera.position.copy(mesh.position).add(offset);
  controls.target.copy(mesh.position);
  controls.update();
  updatePlanetInfoPanel(naifId, mesh.userData.body);

  const bodyData = mesh.userData.body;
  const info = PLANET_INFO[naifId] || { 
    type: 'Unknown', radius: '?', day: '?', year: '?', temp: '?', desc: 'No data available.' 
  };

  // Populate
  document.getElementById('pi-name').textContent = bodyData.name;
  const swatch = document.getElementById('pi-color-swatch');
  swatch.style.backgroundColor = bodyData.color || '#fff';
  swatch.style.boxShadow = `0 0 15px ${bodyData.color || '#fff'}`; // Add glow to swatch
  
  document.getElementById('pi-type').textContent = info.type;
  document.getElementById('pi-radius').textContent = info.radius;
  document.getElementById('pi-day').textContent = info.day;
  document.getElementById('pi-year').textContent = info.year;
  document.getElementById('pi-temp').textContent = info.temp;
  document.getElementById('pi-desc').textContent = info.desc;

  // Show (Remove .hidden class to trigger CSS transition)
  document.getElementById('planet-info-panel').classList.remove('hidden');
}

function updatePlanetInfoPanel(id, bodyData) {
  const panel = document.getElementById('planet-info-panel');
  const info = PLANET_INFO[id] || { 
    type: 'Unknown', radius: '?', day: '?', year: '?', temp: '?', desc: 'No data available.' 
  };

  // Populate fields
  document.getElementById('pi-name').textContent = bodyData.name;
  document.getElementById('pi-color-swatch').style.backgroundColor = bodyData.color || '#fff';
  document.getElementById('pi-type').textContent = info.type;
  document.getElementById('pi-radius').textContent = info.radius;
  document.getElementById('pi-day').textContent = info.day;
  document.getElementById('pi-year').textContent = info.year;
  document.getElementById('pi-temp').textContent = info.temp;
  document.getElementById('pi-desc').textContent = info.desc;

  // Show panel
  panel.classList.remove('hidden');
}

function onMouseMove(event) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const meshes = Object.values(bodyMeshes);
  const intersects = raycaster.intersectObjects(meshes, true);

  if (intersects.length > 0) {
    const hitObj = intersects[0].object;
    let bodyId = null;
    let bodyName = "";

    // Check Hitbox
    if (hitObj.userData.isHitbox) {
      bodyId = hitObj.userData.parentBody.naif_id;
      bodyName = hitObj.userData.parentBody.name;
    } 
    // Check Visual Mesh
    else {
      let object = hitObj;
      while (object.parent && !object.userData.body) {
        object = object.parent;
      }
      if (object.userData.body) {
        bodyId = object.userData.body.naif_id;
        bodyName = object.userData.body.name;
      }
    }

    if (bodyId) {
      if (hoveredBodyId !== bodyId) {
        resetHover();
        hoveredBodyId = bodyId;
        document.body.style.cursor = 'pointer';
        highlightBody(hoveredBodyId, true);
        showTooltip(event, bodyName);
      }
      updateTooltipPosition(event);
    }
  } else {
    if (hoveredBodyId !== null) {
      resetHover();
    }
  }
}


function resetHover() {
  if (hoveredBodyId !== null) {
    highlightBody(hoveredBodyId, false);
    hoveredBodyId = null;
    document.body.style.cursor = 'auto';
    hideTooltip();
  }
}

function highlightBody(naifId, isHovered) {
  const mesh = bodyMeshes[naifId];
  if (!mesh) return;

  // The Sun (ID 10) is a BasicMaterial and already glows, so we skip it or handle differently
  if (naifId === 10) return; 

  // For planets (StandardMaterial)
  if (mesh.material && mesh.material.emissive) {
    // Save original intensity if not saved yet
    if (mesh.userData.originalEmissive === undefined) {
      mesh.userData.originalEmissive = mesh.material.emissiveIntensity;
      mesh.userData.originalColor = mesh.material.color.getHex();
    }

    if (isHovered) {
      // Make it glow brighter
      mesh.material.emissiveIntensity = 0.8; 
      // Optional: lighten the color slightly
      mesh.material.color.setHex(0xffffff); 
    } else {
      // Reset
      mesh.material.emissiveIntensity = mesh.userData.originalEmissive;
      mesh.material.color.setHex(mesh.userData.originalColor);
    }
  }
}

function showTooltip(event, text) {
  const tooltip = document.getElementById('body-tooltip');
  tooltip.textContent = text;
  tooltip.classList.remove('hidden');
  updateTooltipPosition(event);
}

function updateTooltipPosition(event) {
  const tooltip = document.getElementById('body-tooltip');
  if (!tooltip.classList.contains('hidden')) {
    tooltip.style.left = (event.clientX + 15) + 'px';
    tooltip.style.top = (event.clientY + 15) + 'px';
  }
}

function hideTooltip() {
  const tooltip = document.getElementById('body-tooltip');
  tooltip.classList.add('hidden');
}


// ── Lambert Transfer ───────────────────────────────────────────────────────
// (Manual UI removed, helper used by optimizer)
async function drawOptimalTransfer(origin, target, depJd, tofDays) {
  // Convert JD to ISO date
  const jdToDate = (jd) => {
    const ms = (jd - 2440587.5) * 86400000;
    return new Date(ms).toISOString().split('T')[0];
  };
  try {
    const res = await fetch(`${API}/lambert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        origin,
        target,
        departure_date: jdToDate(depJd),
        tof_days: tofDays,
      }),
    });
    if (res.ok) {
      const data = await res.json();
      clearMarkers();
      drawTransferArc(data.trajectory_points);
    }
  } catch (_) { }
}

function drawTransferArc(points) {
  // Remove existing
  if (transferLine) {
    scene.remove(transferLine);
    transferLine.geometry.dispose();
    transferLine.material.dispose();
    transferLine = null;
  }
  clearMarkers();

  const positions = [];
  for (const p of points) {
    positions.push(p.x, p.z, -p.y);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

  // Gradient color line
  const mat = new THREE.LineBasicMaterial({
    color: 0x4ea8de,
    linewidth: 2,
  });

  transferLine = new THREE.Line(geo, mat);
  scene.add(transferLine);

  // Also add departure and arrival markers
  addMarker(points[0], 0x34d399, 2);               // green — departure
  addMarker(points[points.length - 1], 0xef4444, 2); // red — arrival
}

const markers = [];
function addMarker(point, color, size) {
  const geo = new THREE.SphereGeometry(size, 12, 12);
  const mat = new THREE.MeshBasicMaterial({ color });
  const m = new THREE.Mesh(geo, mat);
  m.position.set(point.x, point.z, -point.y);
  scene.add(m);
  markers.push(m);
}

function clearMarkers() {
  for (const m of markers) {
    scene.remove(m);
    m.geometry.dispose();
    m.material.dispose();
  }
  markers.length = 0;
}



// ── Multi-Leg Visualization ────────────────────────────────────────────────

function clearMultiLegArcs() {
  for (const line of multiLegLines) {
    scene.remove(line);
    line.geometry.dispose();
    line.material.dispose();
  }
  multiLegLines = [];
}

function drawMultiLegArcs(legs) {
  clearMultiLegArcs();
  clearMarkers();

  if (!legs || legs.length === 0) return;

  // Draw each leg
  legs.forEach((leg, index) => {
    const points = leg.trajectory_points;
    if (!points || points.length === 0) return;

    const positions = [];
    for (const p of points) {
      positions.push(p.x, p.z, -p.y);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

    // Color cycle for legs to distinguish them
    const colors = [0x4ea8de, 0x6930c3, 0x48bfe3, 0x64dfdf];
    const color = colors[index % colors.length];

    const mat = new THREE.LineBasicMaterial({
      color: color,
      linewidth: 2,
    });

    const line = new THREE.Line(geo, mat);
    scene.add(line);
    multiLegLines.push(line);

    // Markers
    if (index === 0) {
      addMarker(points[0], 0x34d399, 2); // Start (Green)
    }
    // End of current leg (which is start of next or final destination)
    // If it's the last leg, make it red. If intermediate (flyby), make it yellow/orange.
    const isLast = index === legs.length - 1;
    const markerColor = isLast ? 0xef4444 : 0xfbbf24;
    addMarker(points[points.length - 1], markerColor, 2);
  });
}

// ── Pork-Chop Plot ─────────────────────────────────────────────────────────
async function generatePorkchop(config) {
  const payload = {
    origin: config.origin,
    target: config.target,
    dep_start: config.dep_start,
    dep_end: config.dep_end,
    tof_min_days: config.tof_min_days || 100,
    tof_max_days: config.tof_max_days || 400,
    dep_steps: 60,
    tof_steps: 60,
  };

  try {
    const res = await fetch(`${API}/porkchop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    document.getElementById('porkchop-overlay').classList.remove('hidden');
    renderPorkchop(data);
  } catch (e) {
    alert('Pork-chop error: ' + e.message);
  }
}

function renderPorkchop(data) {
  const canvas = document.getElementById('porkchop-canvas');
  const container = document.getElementById('porkchop-container');

  // Set canvas resolution — container must be visible (overlay shown first)
  const size = Math.min(container.clientWidth, 640) || 640;
  canvas.width = size;
  canvas.height = size;
  canvas.style.width = size + 'px';
  canvas.style.height = size + 'px';

  const ctx = canvas.getContext('2d');
  const grid = data.dv_grid;
  const depCount = grid.length;        // departure steps (row index i)
  const tofCount = grid[0].length;     // TOF steps (col index j)

  // Find min/max dv (exclude null)
  let minDv = Infinity, maxDv = 0;
  for (const row of grid) {
    for (const v of row) {
      if (v != null && v < 50) {
        minDv = Math.min(minDv, v);
        maxDv = Math.max(maxDv, v);
      }
    }
  }
  // Cap max for better contrast
  maxDv = Math.min(maxDv, minDv * 4);

  // Departure on x-axis, TOF on y-axis (conventional porkchop orientation)
  const cellW = size / depCount;
  const cellH = size / tofCount;

  for (let i = 0; i < depCount; i++) {
    for (let j = 0; j < tofCount; j++) {
      const v = grid[i][j];
      if (v == null || v > maxDv * 1.5) {
        ctx.fillStyle = '#0a0a12';
      } else {
        const t = Math.max(0, Math.min(1, (v - minDv) / (maxDv - minDv)));
        ctx.fillStyle = dvColor(t);
      }
      // x = departure index, y = TOF index (inverted so low TOF at bottom)
      ctx.fillRect(i * cellW, (tofCount - 1 - j) * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  // Axis labels
  const depIsos = data.departure_isos;
  const tofs = data.tof_days;

  ctx.fillStyle = '#707090';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';

  // Departure date labels (x-axis at bottom) — iterate over depCount
  const depStep = Math.max(1, Math.floor(depCount / 5));
  for (let i = 0; i < depCount; i += depStep) {
    const label = depIsos[i] ? depIsos[i].split('T')[0].slice(5) : '';
    ctx.fillText(label, i * cellW + cellW / 2, size - 4);
  }

  // TOF labels (y-axis on left) — iterate over tofCount
  ctx.textAlign = 'left';
  const tofStep = Math.max(1, Math.floor(tofCount / 5));
  for (let j = 0; j < tofCount; j += tofStep) {
    ctx.fillText(`${tofs[j].toFixed(0)}d`, 4, (tofCount - 1 - j) * cellH + cellH / 2 + 3);
  }

  // Legend
  document.getElementById('porkchop-legend').textContent =
    `${data.origin} → ${data.target}  |  Min ΔV: ${minDv.toFixed(2)} km/s  |  Range: ${minDv.toFixed(1)} – ${maxDv.toFixed(1)} km/s`;

  // Tooltip on hover
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (size / rect.width);
    const y = (e.clientY - rect.top) * (size / rect.height);
    const depIdx = Math.floor(x / cellW);
    const tofIdx = tofCount - 1 - Math.floor(y / cellH);
    if (depIdx >= 0 && depIdx < depCount && tofIdx >= 0 && tofIdx < tofCount) {
      const v = grid[depIdx][tofIdx];
      const dep = depIsos[depIdx] ? depIsos[depIdx].split('T')[0] : '?';
      const tof = tofs[tofIdx] ? tofs[tofIdx].toFixed(0) : '?';
      const tip = document.getElementById('porkchop-tooltip');
      tip.textContent = `Dep: ${dep}  TOF: ${tof}d  ΔV: ${v != null ? v.toFixed(2) + ' km/s' : 'N/A'}`;
      tip.style.left = (e.clientX + 12) + 'px';
      tip.style.top = (e.clientY - 24) + 'px';
      tip.classList.remove('hidden');
    }
  };
  canvas.onmouseleave = () => {
    document.getElementById('porkchop-tooltip').classList.add('hidden');
  };
}

function dvColor(t) {
  // Perceptually-uniform Viridis interpolation (reversed: low = yellow, high = purple)
  const n = VIRIDIS.length - 1;
  const idx = Math.min(t, 1 - 1e-9) * n;
  const i = Math.floor(idx);
  const f = idx - i;
  const a = VIRIDIS[i], b = VIRIDIS[Math.min(i + 1, n)];
  const r = Math.floor((a[0] + (b[0] - a[0]) * f) * 255);
  const g = Math.floor((a[1] + (b[1] - a[1]) * f) * 255);
  const bl = Math.floor((a[2] + (b[2] - a[2]) * f) * 255);
  return `rgb(${r},${g},${bl})`;
}

// ── Optimizer ──────────────────────────────────────────────────────────────
async function runOptimizer(config, onProgress, onComplete) {
  const payload = {
    origin: config.origin,
    target: config.target,
    dep_start: config.dep_start,
    dep_end: config.dep_end,
    tof_min_days: config.tof_min_days || 100,
    tof_max_days: config.tof_max_days || 400,
    population_size: config.population_size || 30,
    max_iterations: config.max_iterations || 100,
    mode: config.mode || 'pareto',
  };

  try {
    const res = await fetch(`${API}/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);

    const { job_id } = await res.json();
    if (onProgress) onProgress({ status: 'starting', job_id, progress: 0 });

    const ws = new WebSocket(`${WS}/ws/trajectory/${job_id}`);

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (onProgress) onProgress(msg);

      if (msg.status === 'complete' || msg.status === 'failed') {
        if (msg.status === 'complete' && msg.best_departure_jd && msg.best_tof_days) {
          // Auto-draw best result
          drawOptimalTransfer(payload.origin, payload.target, msg.best_departure_jd, msg.best_tof_days);
        }
        if (onComplete) onComplete(msg);
        ws.close();
      }
    };

    ws.onerror = (e) => {
      if (onComplete) onComplete({ status: 'failed', error: 'WS Error' });
    };
  } catch (e) {
    if (onComplete) onComplete({ status: 'failed', error: e.message });
  }
}

// ── Multi-leg Optimizer ───────────────────────────────────────────────────
async function runMultiLegOptimizer(config, onProgress, onComplete) {
  const payload = {
    body_sequence: config.body_sequence,
    dep_start: config.dep_start,
    dep_end: config.dep_end,
    leg_tof_bounds: config.leg_tof_bounds,
    population_size: config.population_size || 40,
    max_iterations: config.max_iterations || 200,
    mode: config.mode || 'pareto',
  };

  try {
    const res = await fetch(`${API}/optimize/multileg`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const { job_id } = await res.json();

    if (onProgress) onProgress({ status: 'starting', job_id, progress: 0 });

    const ws = new WebSocket(`${WS}/ws/trajectory/${job_id}`);

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (onProgress) onProgress(msg);

      if (msg.status === 'complete' || msg.status === 'failed') {
        if (msg.status === 'complete' && msg.best_departure_jd && msg.best_leg_tof_days) {
          fetchAndDrawOptimizedMultiLeg(
            config.body_sequence, msg.best_departure_jd, msg.best_leg_tof_days,
          );
        }
        if (onComplete) onComplete(msg);
        ws.close();
      }
    };

    ws.onerror = () => {
      if (onComplete) onComplete({ status: 'failed', error: 'WS Error' });
    };
  } catch (e) {
    if (onComplete) onComplete({ status: 'failed', error: e.message });
  }
}



// ── Simulation Controls ────────────────────────────────────────────────────
function updateSimSpeed(val) {
  simSpeed = val;
  document.getElementById('sim-speed-val').textContent = val;
  if (ephemerisWs && ephemerisWs.readyState === WebSocket.OPEN) {
    ephemerisWs.send(JSON.stringify({ speed: val }));
  }
}

function togglePause() {
  simPaused = !simPaused;
  const btn = document.getElementById('btn-pause');
  btn.textContent = simPaused ? '\u25B6' : '\u23F8';
  if (ephemerisWs && ephemerisWs.readyState === WebSocket.OPEN) {
    ephemerisWs.send(simPaused ? 'pause' : 'resume');
  }
}

// ── Status Bar ─────────────────────────────────────────────────────────────
function setStatus(state, text) {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('status-text');
  dot.className = 'dot ' + state;
  txt.textContent = text;
  txt.textContent = text;
}

// ── Presets ────────────────────────────────────────────────────────────────
function loadPreset(name) {
  const originSel = document.getElementById('plan-origin');
  const targetSel = document.getElementById('plan-target');

  // Simple mapping for now - just sets origin/target
  // Complex multi-leg presets (vega, cassini) would need more UI support to show intermediate legs,
  // but for now we'll just set the start/end points or defaults.

  let origin = 'earth';
  let target = 'mars';

  switch (name) {
    case 'earth-mars':
      origin = 'earth'; target = 'mars';
      break;
    case 'vega':
      origin = 'earth'; target = 'jupiter'; // Venus-Earth-Gravity-Assist
      break;
    case 'cassini':
      origin = 'earth'; target = 'saturn';
      break;
    case 'messenger':
      origin = 'earth'; target = 'mercury';
      break;
    case 'grand-tour':
      origin = 'earth'; target = 'neptune';
      break;
  }

  if (originSel) originSel.value = origin;
  if (targetSel) targetSel.value = target;

  console.log(`Loaded preset: ${name} (${origin} -> ${target})`);
}

// ── Event Bindings ─────────────────────────────────────────────────────────
function bindEvents() {
  document.getElementById('btn-pause').addEventListener('click', togglePause);
  document.getElementById('sim-speed').addEventListener('input', (e) => {
    updateSimSpeed(parseInt(e.target.value));
  });

  document.getElementById('btn-close-pc').addEventListener('click', () => {
    document.getElementById('porkchop-overlay').classList.add('hidden');
  });

  // Click outside overlay to close
  document.getElementById('porkchop-overlay').addEventListener('click', (e) => {
    if (e.target === document.getElementById('porkchop-overlay')) {
      document.getElementById('porkchop-overlay').classList.add('hidden');
    }
  });

  // Planner Events
  document.getElementById('btn-plan').addEventListener('click', planRoute);
  document.getElementById('plan-any-date').addEventListener('change', (e) => {
    const inputs = document.getElementById('plan-date-inputs');
    if (e.target.checked) inputs.classList.add('hidden');
    else inputs.classList.remove('hidden');
  });

  // Visualize Window button (added dynamically or if present)
  const vizBtn = document.getElementById('btn-viz-window');
  if (vizBtn) vizBtn.addEventListener('click', visualizeWindows);




  // Multi-leg preset button
  const mlBtn = document.getElementById('btn-ml-preset');
  if (mlBtn) {
    mlBtn.addEventListener('click', (e) => {
      const menu = document.getElementById('ml-preset-menu');
      if (menu.classList.contains('hidden')) {
        const rect = e.target.getBoundingClientRect();
        menu.style.left = rect.left + 'px';
        menu.style.top = (rect.bottom + 4) + 'px';
        menu.classList.remove('hidden');
      } else {
        menu.classList.add('hidden');
      }
    });
  }

  // Preset items
  document.querySelectorAll('.preset-item').forEach(item => {
    item.addEventListener('click', () => {
      loadPreset(item.dataset.preset);
      document.getElementById('ml-preset-menu').classList.add('hidden');
    });
  });

  // Close preset menu on click outside
  document.addEventListener('click', (e) => {
    const menu = document.getElementById('ml-preset-menu');
    if (!menu.contains(e.target) && e.target.id !== 'btn-ml-preset') {
      menu.classList.add('hidden');
    }
  });

  const unlockBtn = document.getElementById('btn-unlock-cam');
  if(unlockBtn) {
    unlockBtn.addEventListener('click', unlockCamera);
  }

  window.addEventListener('click', onMouseClick);
  window.addEventListener('mousemove', onMouseMove);

  // Initialize default multi-leg sequence
  loadPreset('earth-mars');
}

// ── Planner Logic ──────────────────────────────────────────────────────────
async function planRoute() {
  const btn = document.getElementById('btn-plan');
  const resultsDiv = document.getElementById('plan-results');
  btn.classList.add('loading');
  btn.disabled = true;
  resultsDiv.innerHTML = '';

  const origin = document.getElementById('plan-origin').value;
  const target = document.getElementById('plan-target').value;
  const mode = document.querySelector('input[name="plan-mode"]:checked').value;
  const anyDate = document.getElementById('plan-any-date').checked;

  const payload = {
    origin, target, mode,
    dep_start: anyDate ? null : document.getElementById('plan-start').value,
    dep_end: anyDate ? null : document.getElementById('plan-end').value,
  };

  try {
    const res = await fetch(`${API}/plan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();

    if (data.routes.length === 0) {
      resultsDiv.innerHTML = '<div class="result-box" style="text-align:center; color:var(--text-dim)">No routes found in this window.</div>';
    } else {
      data.routes.forEach((r, i) => renderRouteCard(r, i, resultsDiv));
      // Auto-view first result
      viewRoute(data.routes[0]);
    }
  } catch (e) {
    resultsDiv.innerHTML = `<div class="result-box" style="color:var(--danger)">${e.message}</div>`;
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

function renderRouteCard(route, index, container) {
  const card = document.createElement('div');
  card.className = `route-card ${index === 0 ? 'active' : ''}`;
  card.dataset.index = index;

  const ratingClass = route.rating || 'moderate';
  const stars = ratingClass === 'excellent' ? '\u2605\u2605\u2605' :
    ratingClass === 'good' ? '\u2605\u2605' :
      ratingClass === 'bad' ? '' : '\u2605';

  const m = massEstimate(route.total_dv_km_s);
  const tofYears = (route.total_tof_days / 365.25).toFixed(1);
  const via = route.type === 'multileg' ?
    route.name.replace('Direct', '').replace(route.origin, '').replace(route.target, '').replace(/\b\w/g, l => l.toUpperCase()) :
    'Direct';

  card.innerHTML = `
    <div class="rc-header">
      <span class="rc-title">${via}</span>
      <span class="rc-rating ${ratingClass}">${stars} ${route.rating.toUpperCase()}</span>
    </div>
    <div class="rc-stats">
      <span>\u0394V: <span class="rc-val">${route.total_dv_km_s.toFixed(2)} km/s</span></span>
      <span>TOF: <span class="rc-val">${route.total_tof_days.toFixed(0)}d</span> (${tofYears}y)</span>
    </div>
    <div class="rc-dates">
      Dep: ${route.departure_iso.split('T')[0]} &nbsp; Arr: ${route.arrival_iso.split('T')[0]}
    </div>
    <div class="rc-actions">
      <button class="rc-btn view-btn">View 3D</button>
      <button class="rc-btn optimize opt-btn">Optimize</button>
    </div>
  `;

  card.querySelector('.view-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    document.querySelectorAll('.route-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    viewRoute(route);
  });

  card.querySelector('.opt-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    optimizeRoute(route);
  });

  card.addEventListener('click', () => {
    card.querySelector('.view-btn').click();
  });

  container.appendChild(card);
}

async function showOrbitInsights(msg, initialElements = null) {
  let box = document.getElementById('orbit-insights-box');
  if (!box) {
    box = document.createElement('div');
    box.id = 'orbit-insights-box';
    box.className = 'panel';
    box.style.position = 'fixed';
    box.style.bottom = '20px';
    box.style.right = '320px';
    box.style.width = '320px';
    box.style.zIndex = '100';
    box.style.background = 'rgba(18, 18, 30, 0.95)';
    box.style.maxHeight = '500px';
    box.style.overflowY = 'auto';
    box.style.borderLeft = '4px solid var(--accent)';
    document.body.appendChild(box);
  }
  
  box.classList.remove('hidden');
  
  const optimizedEl = msg.orbit_elements || (msg.orbit_elements_list ? msg.orbit_elements_list[0] : null);
  const initialEl = initialElements || optimizedEl; // Fallback if no initial
  const insight = msg.insight || "Trajectory refined via NSGA-II.";
  
  let html = `
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border); margin-bottom:8px; padding-bottom:4px">
      <h2 style="margin:0; font-size:10px">Orbit Comparison</h2>
      <button onclick="this.parentElement.parentElement.classList.add('hidden')" style="background:none; border:none; color:var(--text-dim); cursor:pointer">&times;</button>
    </div>
    <div style="font-size:11px; color:var(--accent); margin-bottom:12px; line-height:1.4">
      ${insight}
    </div>
    <div id="orbit-insight-content">Loading orbit details...</div>
  `;
  box.innerHTML = html;

  if (optimizedEl) {
    try {
      // Fetch details for initial
      const resInit = await fetch(`${API}/orbit-info`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          a_km: initialEl.a_km || initialEl.semi_major_axis_km,
          e: initialEl.e || initialEl.eccentricity,
          i_deg: initialEl.i_deg || initialEl.inclination_deg,
          raan_deg: initialEl.raan_deg,
          arg_p_deg: initialEl.arg_p_deg || initialEl.argp_deg
        })
      });
      const initialInfo = await resInit.json();

      // Fetch details for optimized
      const resOpt = await fetch(`${API}/orbit-info`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          a_km: optimizedEl.a_km || optimizedEl.semi_major_axis_km,
          e: optimizedEl.e || optimizedEl.eccentricity,
          i_deg: optimizedEl.i_deg || optimizedEl.inclination_deg,
          raan_deg: optimizedEl.raan_deg,
          arg_p_deg: optimizedEl.arg_p_deg || optimizedEl.argp_deg
        })
      });
      const optimizedInfo = await resOpt.json();

      const row = (label, oldVal, newVal, unit = '', color = '') => {
        const diff = newVal - oldVal;
        const pct = oldVal !== 0 ? (diff / Math.abs(oldVal) * 100) : 0;
        const diffColor = diff < 0 ? 'var(--success)' : 'var(--warn)'; 
        
        return `
          <div class="label" style="${color ? 'color:'+color : ''}">${label}</div>
          <div class="val" style="display:flex; flex-direction:column; align-items:flex-end">
            <span style="color:var(--text-dim); font-size:8px; text-decoration:line-through">${oldVal.toFixed(2)}${unit}</span>
            <span>${newVal.toFixed(2)}${unit}</span>
            ${initialElements ? `<span style="font-size:7px; color:${diffColor}">${diff > 0 ? '+' : ''}${pct.toFixed(1)}%</span>` : ''}
          </div>
        `;
      };

      const content = document.getElementById('orbit-insight-content');
      let detailsHtml = `
        <div class="info-grid" style="font-size:10px; grid-template-columns: 1fr 1fr; row-gap:12px">
          ${row('SMA (a)', (initialEl.a_km || initialEl.semi_major_axis_km) / 1e6, (optimizedEl.a_km || optimizedEl.semi_major_axis_km) / 1e6, 'M km')}
          ${row('Ecc (e)', initialEl.e || initialEl.eccentricity, optimizedEl.e || optimizedEl.eccentricity)}
          ${row('Inc (i)', initialEl.i_deg || initialEl.inclination_deg, optimizedEl.i_deg || optimizedEl.inclination_deg, '°')}
          ${row('Periapsis', initialInfo.periapsis_distance_km / 1e6, optimizedInfo.periapsis_distance_km / 1e6, 'M km', 'var(--success)')}
          ${row('Apoapsis', initialInfo.apoapsis_distance_km / 1e6, optimizedInfo.apoapsis_distance_km / 1e6, 'M km', 'var(--warn)')}
        </div>
      `;

      if (msg.orbit_elements_list && msg.orbit_elements_list.length > 1) {
        detailsHtml += `<div style="font-size:9px; color:var(--text-dim); margin-top:8px">+ ${msg.orbit_elements_list.length - 1} more legs computed</div>`;
      }
      content.innerHTML = detailsHtml;

      // Visuals: Draw Full Orbits
      clearFullOrbits();
      if (initialElements && initialInfo.full_orbit_points_scene) {
        drawFullOrbit(initialInfo.full_orbit_points_scene, 0x555555, true); // Dashed grey for old
      }
      if (optimizedInfo.full_orbit_points_scene) {
        drawFullOrbit(optimizedInfo.full_orbit_points_scene, 0x4ea8de, false); // Solid accent for new
      }

      // Visuals: Apsis Markers
      clearMarkers();
      if (optimizedInfo.periapsis_point_scene) {
        const p = optimizedInfo.periapsis_point_scene;
        addMarker({x: p[0], z: p[1], y: -p[2]}, 0x34d399, 1.5); // Periapsis
      }
      if (optimizedInfo.apoapsis_point_scene) {
        const a = optimizedInfo.apoapsis_point_scene;
        addMarker({x: a[0], z: a[1], y: -a[2]}, 0xf59e0b, 1.5); // Apoapsis
      }

    } catch (e) {
      document.getElementById('orbit-insight-content').innerHTML = `<div style="color:var(--danger)">Error loading details</div>`;
      console.error("Failed to fetch orbit info:", e);
    }
  }
}

let fullOrbitLines = [];
function drawFullOrbit(points, color, dashed = false) {
  const threePts = points.map(p => new THREE.Vector3(p[0], p[2], -p[1]));
  // Close the loop for elliptic
  if (points.length > 2) threePts.push(threePts[0]);

  const geo = new THREE.BufferGeometry().setFromPoints(threePts);
  let mat;
  if (dashed) {
    mat = new THREE.LineDashedMaterial({ color, dashSize: 10, gapSize: 5, transparent: true, opacity: 0.5 });
  } else {
    mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 });
  }
  
  const line = new THREE.Line(geo, mat);
  if (dashed) line.computeLineDistances();
  scene.add(line);
  fullOrbitLines.push(line);
}

function clearFullOrbits() {
  fullOrbitLines.forEach(l => {
    scene.remove(l);
    l.geometry.dispose();
    l.material.dispose();
  });
  fullOrbitLines = [];
}

function viewRoute(route) {
  // Clear scene
  if (transferLine) {
    scene.remove(transferLine);
    transferLine = null;
  }
  clearMultiLegArcs();
  clearMarkers();

  // Draw based on type
  if (route.type === 'direct') {
    drawTransferArc(route.trajectory_points);
  } else {
    // For multileg, we need to construct 'legs' structure if not present or just draw points
    // The backend returns 'legs' with trajectory_points inside
    if (route.legs) {
      drawMultiLegArcs(route.legs);
    }
  }
}

function optimizeRoute(route) {
  // Find card
  const card = document.querySelector(`.route-card[data-index="${route.index}"]`); // Need to add index to route obj or search
  // Actually card is just in DOM. We can find it by active class or just re-render.
  // Better: The caller passes the route object. Let's find the card by some ID or just update the UI state.

  // Actually, let's just use the `card` element if we passed it, but we only passed `route`.
  // We'll search by matching title/dates which is hacky.
  // Let's assume the user clicked the button on the card, so we have context if we change signature.
  // But sticking to signature:

  // We will run the optimizer and show a modal or overlay on the card.
  // For simplicity, let's replace the "Optimize" button with a progress bar.

  // Find the button that triggered this? No event object here.
  // Let's re-render the card with a progress bar.
  // Or simpler: Just find the active card.
  const activeCard = document.querySelector('.route-card.active');
  if (!activeCard) return;

  const actionsDiv = activeCard.querySelector('.rc-actions');
  actionsDiv.innerHTML = `<div class="progress-bar" style="width:100%; margin:4px 0"><div class="fill" style="width:0%"></div></div><div class="rc-status" style="font-size:9px;color:var(--text-dim)">Starting...</div>`;

  let initialElements = null;

  const onProgress = (msg) => {
    const bar = actionsDiv.querySelector('.fill');
    const stat = actionsDiv.querySelector('.rc-status');

    // Store first feasible result as "initial" baseline for comparison
    if (!initialElements && (msg.orbit_elements || msg.orbit_elements_list)) {
      initialElements = msg.orbit_elements || msg.orbit_elements_list[0];
    }

    if (bar && msg.max_iterations) {
      const pct = (msg.iteration / msg.max_iterations) * 100;
      bar.style.width = pct + '%';
    }
    if (stat) {
      if (msg.status === 'running') {
        const insightText = msg.insight ? `<div style="margin-top:2px; font-style:italic; color:var(--accent); font-size:8px">${msg.insight}</div>` : '';
        stat.innerHTML = `Iter ${msg.iteration}: ${msg.best_dv_total ? msg.best_dv_total.toFixed(2) : '...'} km/s${insightText}`;
      } else {
        stat.textContent = msg.status;
      }
    }
  };

  const onComplete = (msg) => {
    if (msg.status === 'complete') {
      const stat = actionsDiv.querySelector('.rc-status');
      stat.textContent = `Done. Best: ${msg.best_dv_total.toFixed(2)} km/s`;

      // Update card stats
      const valEls = activeCard.querySelectorAll('.rc-val');
      if (valEls.length >= 2) {
        valEls[0].textContent = msg.best_dv_total.toFixed(2) + ' km/s';
        let tof = msg.best_total_tof_days || msg.best_tof_days;
        if (tof) valEls[1].textContent = tof.toFixed(0) + 'd';
      }

      if (msg.orbit_elements || msg.orbit_elements_list) {
        showOrbitInsights(msg, initialElements);
      }
    }
  };

  if (route.type === 'direct') {
    // Set window +/- 6 months around best date
    const d = new Date(route.departure_iso);
    const start = new Date(d); start.setMonth(d.getMonth() - 6);
    const end = new Date(d); end.setMonth(d.getMonth() + 6);
    const clamp = (dt) => dt.toISOString().split('T')[0];
    
    const selectedMode = document.querySelector('input[name="plan-mode"]:checked').value;

    runOptimizer({
      origin: route.origin,
      target: route.target,
      dep_start: clamp(start),
      dep_end: clamp(end),
      mode: selectedMode,
    }, onProgress, onComplete);

  } else {
    // Multi-leg
    // Departure window +/- 90 days
    const d = new Date(route.departure_iso);
    const start = new Date(d); start.setDate(d.getDate() - 90);
    const end = new Date(d); end.setDate(d.getDate() + 90);
    const clamp = (dt) => dt.toISOString().split('T')[0];

    // Build bounds from legs in result
    let legBounds = [];
    if (route.legs) {
      legBounds = route.legs.map(l => [l.tof_days * 0.5, l.tof_days * 1.5]);
    } else {
      console.error("Cannot optimize multi-leg without leg data");
      return;
    }
    
    const selectedMode = document.querySelector('input[name="plan-mode"]:checked').value;

    runMultiLegOptimizer({
      body_sequence: route.body_sequence,
      dep_start: clamp(start),
      dep_end: clamp(end),
      leg_tof_bounds: legBounds,
      mode: selectedMode,
    }, onProgress, onComplete);
  }
}

async function visualizeWindows() {
  const origin = document.getElementById('plan-origin').value;
  const target = document.getElementById('plan-target').value;
  const anyDate = document.getElementById('plan-any-date').checked;
  let start = '2028-01-01', end = '2030-01-01'; // Defaults

  if (!anyDate) {
    start = document.getElementById('plan-start').value;
    end = document.getElementById('plan-end').value;
  } else {
    // Default 2 year window from now? Or next synodic window?
    // Just pick 2028-2030 for now as a demo window, or use current planner logic.
    // Better: Use the date range from the date inputs if set, else defaults.
    // If "Any date" is checked, inputs are hidden but have values.
    // We'll just use the hidden values if they are reasonable, or force a 4-year window.
    start = '2028-01-01';
    end = '2032-01-01';
  }

  await generatePorkchop({
    origin, target, dep_start: start, dep_end: end
  });
}

function toggleAdvanced() {
  // Deleted
}

// ── Multi-Leg Trajectory ───────────────────────────────────────────────────
// (Manual functions removed, kept only formatting helpers if needed)

// ── Multi-leg Optimizer ───────────────────────────────────────────────────
// (Merged into runMultiLegOptimizer above)


async function fetchAndDrawOptimizedMultiLeg(bodies, depJd, legTofs) {
  try {
    const res = await fetch(`${API}/multileg`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        body_sequence: bodies,
        departure_date: jdToIso(depJd),
        leg_tof_days: legTofs,
      }),
    });
    if (res.ok) {
      const data = await res.json();
      drawMultiLegArcs(data.legs);
    }
  } catch (_) { }
}

// ── Render Loop ────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);

  if (focusedBodyId && bodyMeshes[focusedBodyId]) {
    const mesh = bodyMeshes[focusedBodyId];
    const currentBodyPosition = mesh.position;

    // 1. Calculate how much the planet moved since last frame
    const delta = new THREE.Vector3().subVectors(currentBodyPosition, previousBodyPosition);

    // 2. Add that movement to the camera's position
    camera.position.add(delta);

    // 3. Update the controls target to look at the new planet position
    controls.target.copy(currentBodyPosition);

    // 4. Update previous position for the next frame
    previousBodyPosition.copy(currentBodyPosition);
  }

  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}

// ── Boot ───────────────────────────────────────────────────────────────────
init();
