// ═══════════════════════════════════════════════════════════════════════════
//  TARS Frontend — Interplanetary Trajectory Designer
//  Three.js 3D scene + API client + UI
// ═══════════════════════════════════════════════════════════════════════════

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const API = 'http://localhost:8000';
const WS  = 'ws://localhost:8000';

// ── Viridis colormap (reversed: low=yellow/bright, high=purple/dark) ──────
const VIRIDIS = [
  [0.993,0.906,0.144],[0.741,0.873,0.150],[0.478,0.821,0.318],
  [0.267,0.749,0.441],[0.135,0.659,0.518],[0.128,0.567,0.551],
  [0.164,0.471,0.558],[0.207,0.372,0.553],[0.253,0.265,0.530],
  [0.282,0.141,0.458],[0.267,0.004,0.329],
];

// ── Mass modeling defaults ────────────────────────────────────────────────
const MASS_M0  = 2000;           // wet mass, kg
const MASS_ISP = 320;            // bipropellant Isp, seconds
const G0_KMS   = 9.80665e-3;     // gravitational accel, km/s²

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
let epochRange = { start: '2025-01-01', end: '2059-12-31' }; // updated from backend

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
    'lambert-origin', 'lambert-target',
    'pc-origin', 'pc-target',
    'opt-origin', 'opt-target',
  ];
  for (const id of selectors) {
    const sel = document.getElementById(id);
    sel.innerHTML = '';
    for (const p of planets) {
      const opt = document.createElement('option');
      opt.value = p.name.toLowerCase();
      opt.textContent = p.name;
      sel.appendChild(opt);
    }
  }
  // Defaults
  document.getElementById('lambert-origin').value = 'earth';
  document.getElementById('lambert-target').value = 'mars';
  document.getElementById('pc-origin').value = 'earth';
  document.getElementById('pc-target').value = 'mars';
  document.getElementById('opt-origin').value = 'earth';
  document.getElementById('opt-target').value = 'mars';
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
async function computeLambert() {
  const btn = document.getElementById('btn-lambert');
  const resultBox = document.getElementById('lambert-result');
  btn.classList.add('loading');
  btn.disabled = true;
  resultBox.classList.add('hidden');

  const payload = {
    origin: document.getElementById('lambert-origin').value,
    target: document.getElementById('lambert-target').value,
    departure_date: document.getElementById('lambert-dep').value,
    tof_days: parseFloat(document.getElementById('lambert-tof').value),
  };

  try {
    const res = await fetch(`${API}/lambert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }

    const data = await res.json();

    // Show result
    resultBox.innerHTML = `
<span class="label">Departure:</span> ${data.departure_iso.split('T')[0]}
<span class="label">Arrival:</span>   ${data.arrival_iso.split('T')[0]}
<span class="label">TOF:</span>       ${data.tof_days} days
<span class="label">dV depart:</span> <span class="dv">${data.dv_departure_km_s.toFixed(3)} km/s</span>
<span class="label">dV arrive:</span> <span class="dv">${data.dv_arrival_km_s.toFixed(3)} km/s</span>
<span class="label">dV total:</span>  <span class="dv">${data.dv_total_km_s.toFixed(3)} km/s</span>
<span class="label">C3 dep:</span>    ${(data.dv_departure_km_s ** 2).toFixed(2)} km\u00B2/s\u00B2
<span class="label">C3 arr:</span>    ${(data.dv_arrival_km_s ** 2).toFixed(2)} km\u00B2/s\u00B2
<span class="label">Mass est</span> <span class="label">(${MASS_M0}kg, Isp=${MASS_ISP}s):</span>
  prop: ${massEstimate(data.dv_total_km_s).propellant.toFixed(0)} kg  final: ${massEstimate(data.dv_total_km_s).final.toFixed(0)} kg`;
    resultBox.classList.remove('hidden');

    // Draw trajectory arc in 3D
    drawTransferArc(data.trajectory_points);
  } catch (e) {
    resultBox.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
    resultBox.classList.remove('hidden');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
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

// ── Pork-Chop Plot ─────────────────────────────────────────────────────────
async function generatePorkchop() {
  const btn = document.getElementById('btn-porkchop');
  btn.classList.add('loading');
  btn.disabled = true;

  const payload = {
    origin: document.getElementById('pc-origin').value,
    target: document.getElementById('pc-target').value,
    dep_start: document.getElementById('pc-dep-start').value,
    dep_end: document.getElementById('pc-dep-end').value,
    tof_min_days: parseFloat(document.getElementById('pc-tof-min').value),
    tof_max_days: parseFloat(document.getElementById('pc-tof-max').value),
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
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
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

function lerp(a, b, t) { return a + (b - a) * t; }

// ── Optimizer ──────────────────────────────────────────────────────────────
async function runOptimizer() {
  const btn = document.getElementById('btn-optimize');
  const box = document.getElementById('opt-progress');
  btn.classList.add('loading');
  btn.disabled = true;
  box.classList.remove('hidden');
  box.innerHTML = '<span class="label">Submitting...</span>';

  const payload = {
    origin: document.getElementById('opt-origin').value,
    target: document.getElementById('opt-target').value,
    dep_start: document.getElementById('opt-dep-start').value,
    dep_end: document.getElementById('opt-dep-end').value,
    tof_min_days: 100,
    tof_max_days: 400,
    population_size: parseInt(document.getElementById('opt-pop').value),
    max_iterations: parseInt(document.getElementById('opt-iters').value),
  };

  try {
    const res = await fetch(`${API}/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);

    const { job_id } = await res.json();
    box.innerHTML = `<span class="label">Job:</span> ${job_id.slice(0, 8)}...\n<div class="progress-bar"><div class="fill" id="opt-fill" style="width:0%"></div></div>`;

    // Connect WebSocket for progress
    const ws = new WebSocket(`${WS}/ws/trajectory/${job_id}`);

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const pct = msg.max_iterations > 0
        ? Math.round((msg.iteration / msg.max_iterations) * 100)
        : 0;
      const fill = document.getElementById('opt-fill');
      if (fill) fill.style.width = pct + '%';

      const dvStr = msg.best_dv_total != null ? msg.best_dv_total.toFixed(3) : '...';
      box.innerHTML = `
<span class="label">Job:</span> ${job_id.slice(0, 8)}...  <span class="label">Status:</span> ${msg.status}
<div class="progress-bar"><div class="fill" id="opt-fill" style="width:${pct}%"></div></div>
<span class="label">Iteration:</span> ${msg.iteration} / ${msg.max_iterations}
<span class="label">Best ΔV:</span>  <span class="dv">${dvStr} km/s</span>
<span class="label">Dep JD:</span>   ${msg.best_departure_jd ? msg.best_departure_jd.toFixed(2) : '—'}
<span class="label">TOF:</span>      ${msg.best_tof_days ? msg.best_tof_days.toFixed(1) + ' days' : '—'}`;

      if (msg.status === 'complete' || msg.status === 'failed') {
        btn.classList.remove('loading');
        btn.disabled = false;

        // If complete, also compute and draw the best Lambert transfer
        if (msg.status === 'complete' && msg.best_departure_jd && msg.best_tof_days) {
          drawOptimalTransfer(payload.origin, payload.target, msg.best_departure_jd, msg.best_tof_days);
        }
      }
    };

    ws.onerror = () => {
      // Fall back to polling
      pollOptStatus(job_id, box, btn, payload);
    };

    ws.onclose = () => {};
  } catch (e) {
    box.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

async function pollOptStatus(jobId, box, btn, payload) {
  const poll = async () => {
    try {
      const res = await fetch(`${API}/optimize/${jobId}/status`);
      const data = await res.json();
      if (data.status === 'complete' || data.status === 'failed') {
        const r = data.result || {};
        box.innerHTML = `
<span class="label">Status:</span> ${data.status}
<span class="label">Best ΔV:</span> <span class="dv">${r.best_dv_total ? r.best_dv_total.toFixed(3) + ' km/s' : '—'}</span>
<span class="label">TOF:</span> ${r.best_tof_days ? r.best_tof_days.toFixed(1) + ' days' : '—'}`;
        btn.classList.remove('loading');
        btn.disabled = false;

        if (data.status === 'complete' && r.best_departure_jd && r.best_tof_days) {
          drawOptimalTransfer(payload.origin, payload.target, r.best_departure_jd, r.best_tof_days);
        }
        return;
      }
      setTimeout(poll, 1000);
    } catch (_) {
      setTimeout(poll, 2000);
    }
  };
  poll();
}

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
  } catch (_) {}
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
}

// ── Event Bindings ─────────────────────────────────────────────────────────
function bindEvents() {
  document.getElementById('btn-lambert').addEventListener('click', computeLambert);
  document.getElementById('btn-porkchop').addEventListener('click', generatePorkchop);
  document.getElementById('btn-optimize').addEventListener('click', runOptimizer);
  document.getElementById('btn-pause').addEventListener('click', togglePause);
  document.getElementById('btn-multileg').addEventListener('click', computeMultiLeg);
  document.getElementById('btn-ml-add').addEventListener('click', mlAddLeg);
  document.getElementById('btn-ml-optimize').addEventListener('click', runMultiLegOptimizer);

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

  // Multi-leg preset button
  document.getElementById('btn-ml-preset').addEventListener('click', (e) => {
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

// ── Multi-Leg Trajectory ───────────────────────────────────────────────────

// Preset trajectories with approximate TOFs
const ML_PRESETS = {
  'earth-mars': {
    bodies: ['earth', 'mars'],
    tofs: [259],
    departure: '2028-11-15',
  },
  'vega': {
    bodies: ['earth', 'venus', 'earth', 'jupiter'],
    tofs: [140, 340, 580],
    departure: '2030-01-15',
  },
  'cassini': {
    bodies: ['earth', 'venus', 'venus', 'earth', 'jupiter', 'saturn'],
    tofs: [120, 350, 350, 600, 1100],
    departure: '2029-06-01',
  },
  'messenger': {
    bodies: ['earth', 'venus', 'venus', 'mercury'],
    tofs: [110, 225, 90],
    departure: '2028-09-15',
  },
  'grand-tour': {
    bodies: ['earth', 'jupiter', 'saturn', 'uranus', 'neptune'],
    tofs: [700, 1050, 1800, 2500],
    departure: '2030-06-01',
  },
};

let mlBodies = ['earth', 'mars'];
let mlTofs = [250];

function loadPreset(name) {
  const preset = ML_PRESETS[name];
  if (!preset) return;
  mlBodies = [...preset.bodies];
  mlTofs = [...preset.tofs];
  document.getElementById('ml-departure').value = preset.departure;
  renderMLLegs();
}

function renderMLLegs() {
  const container = document.getElementById('multileg-legs');
  container.innerHTML = '';
  const planets = getPlanets();

  for (let i = 0; i < mlBodies.length; i++) {
    const row = document.createElement('div');
    row.className = 'leg-row';

    // Body number
    const num = document.createElement('span');
    num.className = 'leg-num';
    num.textContent = i === 0 ? 'DEP' : i === mlBodies.length - 1 ? 'ARR' : `F${i}`;
    row.appendChild(num);

    // Body selector
    const sel = document.createElement('select');
    sel.className = 'ml-body-select';
    sel.dataset.index = i;
    for (const p of planets) {
      const opt = document.createElement('option');
      opt.value = p.name.toLowerCase();
      opt.textContent = p.name;
      sel.appendChild(opt);
    }
    sel.value = mlBodies[i];
    sel.addEventListener('change', (e) => {
      mlBodies[parseInt(e.target.dataset.index)] = e.target.value;
    });
    row.appendChild(sel);

    // TOF input (for each leg except last body)
    if (i < mlBodies.length - 1) {
      const arrow = document.createElement('span');
      arrow.className = 'leg-arrow';
      arrow.textContent = '\u2192';
      row.appendChild(arrow);

      const tofInput = document.createElement('input');
      tofInput.type = 'number';
      tofInput.min = '10';
      tofInput.max = '20000';
      tofInput.value = mlTofs[i] || 200;
      tofInput.title = 'TOF (days)';
      tofInput.dataset.index = i;
      tofInput.addEventListener('change', (e) => {
        mlTofs[parseInt(e.target.dataset.index)] = parseFloat(e.target.value);
      });
      row.appendChild(tofInput);

      const dLabel = document.createElement('span');
      dLabel.className = 'leg-arrow';
      dLabel.textContent = 'd';
      row.appendChild(dLabel);
    }

    // Remove button (not for first or if only 2 bodies)
    if (i > 0 && mlBodies.length > 2) {
      const removeBtn = document.createElement('button');
      removeBtn.className = 'btn-remove-leg';
      removeBtn.textContent = '\u00D7';
      removeBtn.title = 'Remove';
      removeBtn.dataset.index = i;
      removeBtn.addEventListener('click', (e) => {
        const idx = parseInt(e.target.dataset.index);
        mlBodies.splice(idx, 1);
        // Adjust TOFs: merge the two adjacent TOFs
        if (idx < mlTofs.length) {
          mlTofs.splice(idx > 0 ? idx - 1 : idx, 1);
        } else if (mlTofs.length > mlBodies.length - 1) {
          mlTofs.pop();
        }
        renderMLLegs();
      });
      row.appendChild(removeBtn);
    }

    container.appendChild(row);
  }
}

function mlAddLeg() {
  // Insert a new body before the last one (add a flyby)
  const lastBody = mlBodies[mlBodies.length - 1];
  mlBodies.splice(mlBodies.length - 1, 0, 'venus');
  // Add a default TOF for the new leg
  mlTofs.push(200);
  renderMLLegs();
}

// Leg colors for multi-leg arcs
const LEG_COLORS = [
  0x4ea8de, // blue
  0x34d399, // green
  0xf59e0b, // amber
  0xef4444, // red
  0x7b2ff7, // purple
  0xe879f9, // pink
  0x06b6d4, // cyan
  0xfbbf24, // yellow
];

function clearMultiLegArcs() {
  for (const obj of multiLegLines) {
    scene.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) obj.material.dispose();
  }
  multiLegLines.length = 0;
}

function drawMultiLegArcs(legs) {
  clearMultiLegArcs();
  clearMarkers();

  for (let i = 0; i < legs.length; i++) {
    const leg = legs[i];
    const color = LEG_COLORS[i % LEG_COLORS.length];
    const positions = [];

    for (const p of leg.trajectory_points) {
      positions.push(p.x, p.z, -p.y);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
    const line = new THREE.Line(geo, mat);
    scene.add(line);
    multiLegLines.push(line);

    // Departure marker for first leg
    if (i === 0 && leg.trajectory_points.length > 0) {
      addMarker(leg.trajectory_points[0], 0x34d399, 2.5); // green = departure
    }

    // Flyby marker at each intermediate body
    if (i < legs.length - 1 && leg.trajectory_points.length > 0) {
      const lastPt = leg.trajectory_points[leg.trajectory_points.length - 1];
      addMarker(lastPt, 0xf59e0b, 2); // amber = flyby
    }

    // Arrival marker for last leg
    if (i === legs.length - 1 && leg.trajectory_points.length > 0) {
      const lastPt = leg.trajectory_points[leg.trajectory_points.length - 1];
      addMarker(lastPt, 0xef4444, 2.5); // red = arrival
    }
  }
}

async function computeMultiLeg() {
  const btn = document.getElementById('btn-multileg');
  const resultBox = document.getElementById('ml-result');
  btn.classList.add('loading');
  btn.disabled = true;
  resultBox.classList.add('hidden');

  const payload = {
    body_sequence: [...mlBodies],
    departure_date: document.getElementById('ml-departure').value,
    leg_tof_days: mlTofs.slice(0, mlBodies.length - 1).map(Number),
  };

  try {
    const res = await fetch(`${API}/multileg`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }

    const data = await res.json();

    // Build result display
    resultBox.innerHTML = formatMultiLegResult(data);
    resultBox.classList.remove('hidden');

    // Draw in 3D
    drawMultiLegArcs(data.legs);

  } catch (e) {
    resultBox.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
    resultBox.classList.remove('hidden');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

// ── Multi-leg result formatting (shared by compute + optimizer) ────────────
function formatMultiLegResult(data) {
  const m = massEstimate(data.total_dv_km_s);
  let html = '';
  html += `<span class="label">Route:</span> ${data.body_sequence.join(' \u2192 ')}\n`;
  html += `<span class="label">Depart:</span> ${data.departure_iso.split('T')[0]}\n`;
  html += `<span class="label">Arrive:</span> ${data.arrival_iso.split('T')[0]}\n`;
  html += `<span class="label">Total TOF:</span> ${data.total_tof_days.toFixed(0)} days (${(data.total_tof_days / 365.25).toFixed(1)} years)\n`;
  html += `<span class="label">Total \u0394V:</span> <span class="dv">${data.total_dv_km_s.toFixed(3)} km/s</span>\n`;
  html += `<span class="label">  Departure:</span> <span class="dv">${data.departure_dv_km_s.toFixed(3)} km/s</span>\n`;
  html += `<span class="label">  Arrival:</span> <span class="dv">${data.arrival_dv_km_s.toFixed(3)} km/s</span>\n`;
  if (data.flyby_dv_km_s > 0) {
    html += `<span class="label">  Flyby \u0394V:</span> <span class="dv">${data.flyby_dv_km_s.toFixed(3)} km/s</span>\n`;
  }
  // C3 + mass estimate
  html += `<span class="label">C3 dep:</span>  ${(data.departure_dv_km_s ** 2).toFixed(2)} km\u00B2/s\u00B2\n`;
  html += `<span class="label">C3 arr:</span>  ${(data.arrival_dv_km_s ** 2).toFixed(2)} km\u00B2/s\u00B2\n`;
  html += `<span class="label">Mass est</span> <span class="label">(${MASS_M0}kg, Isp=${MASS_ISP}s):</span>\n`;
  html += `  prop: ${m.propellant.toFixed(0)} kg  final: ${m.final.toFixed(0)} kg\n`;

  // Flyby details
  for (const fb of data.flybys) {
    html += `\n<div class="flyby-card">`;
    html += `<div class="fb-header">${fb.body} flyby (${fb.epoch_iso.split('T')[0]})</div>`;
    html += `V\u221E in: ${fb.v_inf_in_km_s.toFixed(2)} km/s  out: ${fb.v_inf_out_km_s.toFixed(2)} km/s\n`;
    html += `Turn: ${fb.turning_angle_deg.toFixed(1)}\u00B0  Alt: ${fb.flyby_altitude_km.toFixed(0)} km\n`;
    if (fb.feasible_unpowered) {
      html += `<span class="fb-feasible">Unpowered flyby feasible</span>`;
    } else {
      html += `<span class="fb-powered">Powered: ${fb.powered_dv_km_s.toFixed(3)} km/s needed</span>`;
    }
    html += `</div>`;
  }

  // Leg summary
  html += `\n<span class="label">Legs:</span>`;
  for (const leg of data.legs) {
    const conv = leg.converged ? '' : ' [!]';
    html += `\n  ${leg.origin} \u2192 ${leg.target}: ${leg.tof_days.toFixed(0)}d, \u0394V dep=${leg.dv_departure_km_s.toFixed(2)} arr=${leg.dv_arrival_km_s.toFixed(2)} km/s${conv}`;
  }
  return html;
}

// ── Multi-leg Optimizer ───────────────────────────────────────────────────
function jdToIso(jd) {
  const ms = (jd - 2440587.5) * 86400000;
  return new Date(ms).toISOString().split('T')[0];
}

async function runMultiLegOptimizer() {
  const btn = document.getElementById('btn-ml-optimize');
  const box = document.getElementById('ml-opt-progress');
  btn.classList.add('loading');
  btn.disabled = true;
  box.classList.remove('hidden');
  box.innerHTML = '<span class="label">Submitting...</span>';

  const departure = document.getElementById('ml-departure').value;
  const depDate = new Date(departure);

  // Departure window: current departure +/- 90 days, clamped to epoch range
  const depStart = new Date(depDate);
  depStart.setDate(depStart.getDate() - 90);
  const depEnd = new Date(depDate);
  depEnd.setDate(depEnd.getDate() + 90);

  const clampIso = (d) => {
    const iso = d.toISOString().split('T')[0];
    if (iso < epochRange.start) return epochRange.start;
    if (iso > epochRange.end) return epochRange.end;
    return iso;
  };

  // Build TOF bounds: each TOF +/- 50%, min 30 days
  const legTofBounds = mlTofs.slice(0, mlBodies.length - 1).map(tof => {
    const t = Number(tof);
    return [Math.max(30, Math.round(t * 0.5)), Math.round(t * 1.5)];
  });

  const payload = {
    body_sequence: [...mlBodies],
    dep_start: clampIso(depStart),
    dep_end: clampIso(depEnd),
    leg_tof_bounds: legTofBounds,
    population_size: 40,
    max_iterations: 200,
  };

  try {
    const res = await fetch(`${API}/optimize/multileg`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const { job_id } = await res.json();

    box.innerHTML = `<span class="label">Job:</span> ${job_id.slice(0, 8)}...\n<div class="progress-bar"><div class="fill" id="ml-opt-fill" style="width:0%"></div></div>`;

    // Stream progress via WebSocket
    const ws = new WebSocket(`${WS}/ws/trajectory/${job_id}`);

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const pct = msg.max_iterations > 0
        ? Math.round((msg.iteration / msg.max_iterations) * 100) : 0;
      const dvStr = msg.best_dv_total != null ? msg.best_dv_total.toFixed(3) : '...';
      const tofStr = msg.best_leg_tof_days
        ? msg.best_leg_tof_days.map(t => t.toFixed(0) + 'd').join(', ') : '...';
      const route = msg.body_sequence
        ? msg.body_sequence.join(' \u2192 ') : mlBodies.join(' \u2192 ');

      box.innerHTML = `<span class="label">Job:</span> ${job_id.slice(0, 8)}...  <span class="label">Status:</span> ${msg.status}
<div class="progress-bar"><div class="fill" id="ml-opt-fill" style="width:${pct}%"></div></div>
<span class="label">Iter:</span> ${msg.iteration} / ${msg.max_iterations}
<span class="label">Best \u0394V:</span> <span class="dv">${dvStr} km/s</span>
<span class="label">Route:</span> ${route}
<span class="label">Leg TOFs:</span> ${tofStr}
<span class="label">\u0394V dep:</span> ${msg.best_dv_departure != null ? msg.best_dv_departure.toFixed(3) : '...'} km/s
<span class="label">\u0394V arr:</span> ${msg.best_dv_arrival != null ? msg.best_dv_arrival.toFixed(3) : '...'} km/s
<span class="label">\u0394V flyby:</span> ${msg.best_dv_flyby != null ? msg.best_dv_flyby.toFixed(3) : '...'} km/s`;

      if (msg.status === 'complete' || msg.status === 'failed') {
        btn.classList.remove('loading');
        btn.disabled = false;
        if (msg.status === 'complete' && msg.best_departure_jd && msg.best_leg_tof_days) {
          // Update UI with optimized parameters
          document.getElementById('ml-departure').value = jdToIso(msg.best_departure_jd);
          mlTofs = [...msg.best_leg_tof_days];
          renderMLLegs();
          // Fetch full trajectory and draw
          fetchAndDrawOptimizedMultiLeg(
            [...mlBodies], msg.best_departure_jd, msg.best_leg_tof_days,
          );
        }
      }
    };

    ws.onerror = () => {
      box.innerHTML += '\n<span style="color:var(--warn)">WS error</span>';
      btn.classList.remove('loading');
      btn.disabled = false;
    };
    ws.onclose = () => {};
  } catch (e) {
    box.innerHTML = `<span style="color:var(--danger)">${e.message}</span>`;
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

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
      // Show full result in ml-result box
      const resultBox = document.getElementById('ml-result');
      resultBox.innerHTML = formatMultiLegResult(data);
      resultBox.classList.remove('hidden');
    }
  } catch (_) {}
}

// ── Render Loop ────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ── Boot ───────────────────────────────────────────────────────────────────
init();
