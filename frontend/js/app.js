// ═══════════════════════════════════════════════════════════════════════════
//  TARS Frontend — Interplanetary Trajectory Designer
//  Three.js 3D scene + API client + UI
// ═══════════════════════════════════════════════════════════════════════════

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

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
let orbitLines = {};                // naif_id -> THREE.Line
let transferLine = null;            // current Lambert arc
let multiLegLines = [];             // multi-leg arc lines
let scene, camera, renderer, controls;
let ephemerisWs = null;             // WebSocket for streaming
let simPaused = false;
let simSpeed = 10;
let epochRange = { start: '2025-01-01', end: '2059-12-31', start_jd: null, end_jd: null }; // updated from backend

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
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
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
      epochRange.start_jd = data.start_jd;
      epochRange.end_jd = data.end_jd;
    }
  } catch (_) { }
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
  // Fetch orbit paths for planets over the full ephemeris range
  const planets = bodies.filter(b => b.parent_id === null);
  for (const body of planets) {
    try {
      const res = await fetch(
        `${API}/bodies/${body.name.toLowerCase()}/ephemeris?start=${epochRange.start}&end=${epochRange.end}&step_days=10&scene_units=true`
      );
      const data = await res.json();
      drawOrbit(body, data.points);
    } catch (e) {
      console.warn(`Failed to fetch orbit for ${body.name}:`, e);
    }
  }
}

function drawOrbit(body, points) {
  const positions = [];
  const colors = [];
  const base = new THREE.Color(body.color);
  const n = points.length;
  const isSun = body.naif_id === 10;

  for (let i = 0; i < n; i++) {
    positions.push(points[i].x, points[i].z, -points[i].y);
    // Trail fading: dim at start (old positions), bright at end (recent)
    const t = n > 1 ? i / (n - 1) : 1;
    const brightness = 0.12 + 0.78 * t;
    colors.push(base.r * brightness, base.g * brightness, base.b * brightness);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const mat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: isSun ? 0 : 0.55,
  });
  const line = new THREE.Line(geo, mat);
  scene.add(line);
  orbitLines[body.naif_id] = line;
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

  // Name label sprite
  const label = createTextSprite(body.name, body.color);
  label.position.set(0, radius + 2, 0);
  label.scale.set(20, 10, 1);
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
    try { ephemerisWs.close(); } catch (_) { }
  }

  ephemerisWs = new WebSocket(`${WS}/ws/ephemeris/stream`);

  ephemerisWs.onopen = () => {
    setStatus('ok', 'Streaming');
    // Send config — only planets (no moons for now to keep it clean)
    const planetIds = bodies.filter(b => b.parent_id === null).map(b => b.naif_id);
    // Use the epoch range fetched from the backend
    const startJd = epochRange.start_jd || 2460676.5;
    ephemerisWs.send(JSON.stringify({
      body_ids: planetIds,
      start_jd: startJd,
      speed: simSpeed,
      fps: 30,
      scene_units: true,
    }));
  };

  ephemerisWs.onmessage = (event) => {
    const snap = JSON.parse(event.data);
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
  }
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

  const onProgress = (msg) => {
    const bar = actionsDiv.querySelector('.fill');
    const stat = actionsDiv.querySelector('.rc-status');
    if (bar && msg.max_iterations) {
      const pct = (msg.iteration / msg.max_iterations) * 100;
      bar.style.width = pct + '%';
    }
    if (stat) {
      if (msg.status === 'running') {
        stat.textContent = `Iter ${msg.iteration}: ${msg.best_dv_total ? msg.best_dv_total.toFixed(2) : '...'} km/s`;
      } else {
        stat.textContent = msg.status;
      }
    }
  };

  const onComplete = (msg) => {
    // Re-render card with new stats?
    // Or just update values.
    if (msg.status === 'complete') {
      const stat = actionsDiv.querySelector('.rc-status');
      stat.textContent = `Done. Best: ${msg.best_dv_total.toFixed(2)} km/s`;

      // Update card stats
      const valEls = activeCard.querySelectorAll('.rc-val');
      if (valEls.length >= 2) {
        valEls[0].textContent = msg.best_dv_total.toFixed(2) + ' km/s';
        // Update TOF if available (multi-leg vs single)
        // msg structure differs slightly
        let tof = msg.best_total_tof_days || msg.best_tof_days;
        if (tof) valEls[1].textContent = tof.toFixed(0) + 'd';
      }
    }
  };

  if (route.type === 'direct') {
    // Set window +/- 6 months around best date
    const d = new Date(route.departure_iso);
    const start = new Date(d); start.setMonth(d.getMonth() - 6);
    const end = new Date(d); end.setMonth(d.getMonth() + 6);
    const clamp = (dt) => dt.toISOString().split('T')[0];

    runOptimizer({
      origin: route.origin,
      target: route.target,
      dep_start: clamp(start),
      dep_end: clamp(end),
      // Use defaults for others
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
      // Fallback using route.legs_ratio and total TOF if we had that, 
      // but we only have total.
      // The catalog route has 'legs_ratio'.
      // This is getting complex.
      // If we are optimizing a planner result, it SHOULD have legs.
      // If not, we abort.
      console.error("Cannot optimize multi-leg without leg data");
      return;
    }

    runMultiLegOptimizer({
      body_sequence: route.body_sequence,
      dep_start: clamp(start),
      dep_end: clamp(end),
      leg_tof_bounds: legBounds,
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
function jdToIso(jd) {
  const ms = (jd - 2440587.5) * 86400000;
  return new Date(ms).toISOString().split('T')[0];
}

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
  controls.update();
  renderer.render(scene, camera);
}

// ── Boot ───────────────────────────────────────────────────────────────────
init();
