/**
 * Flight-data access layer: loads the scenario manifest and per-scenario rollouts exported by
 * `src/viz/export_flight_data.py`, converts them to typed arrays and provides allocation-free
 * interpolation at arbitrary simulation time.
 *
 * World frame (identical to scene.glb): +X east, +Y up, +Z south; runway threshold at the origin,
 * runway along -Z. Quaternions are [x, y, z, w] for the exported aircraft model (nose +X, up +Y).
 */
const BASE = import.meta.env.BASE_URL;

export const MPS_TO_KT = 1.943844;
export const M_TO_FT = 3.280839895;
export const DEG = Math.PI / 180;

const rawCache = new Map();
const episodeCache = new Map();

export async function loadManifest() {
  const res = await fetch(`${BASE}data/flight_data.json`);
  if (!res.ok) throw new Error(`flight_data.json: HTTP ${res.status}`);
  const manifest = await res.json();
  if (manifest.schema !== "alma-autoland-flight-data") throw new Error("flight_data.json: unexpected schema");
  if (manifest.default_episode) {
    rawCache.set(manifest.default_episode.id, manifest.default_episode);
  }
  return manifest;
}

export async function loadRig() {
  const res = await fetch(`${BASE}models/scene.rig.json`);
  if (!res.ok) throw new Error(`scene.rig.json: HTTP ${res.status}`);
  return res.json();
}

export async function loadEpisode(manifest, id) {
  if (episodeCache.has(id)) return episodeCache.get(id);
  let raw = rawCache.get(id);
  if (!raw) {
    const entry = manifest.scenarios.find((s) => s.id === id);
    if (!entry) throw new Error(`unknown scenario ${id}`);
    const res = await fetch(`${BASE}data/${entry.file}`);
    if (!res.ok) throw new Error(`${entry.file}: HTTP ${res.status}`);
    raw = await res.json();
    rawCache.set(id, raw);
  }
  const ep = prepareEpisode(raw);
  episodeCache.set(id, ep);
  return ep;
}

export function rawEpisode(id) {
  return rawCache.get(id) ?? null;
}

const flat = (rows, width) => {
  const out = new Float32Array(rows.length * width);
  rows.forEach((r, i) => {
    for (let k = 0; k < width; k++) out[i * width + k] = r[k] ?? 0;
  });
  return out;
};

const POSE_SIGMA_POS = 2; // samples (0.1 s at 20 Hz)
const POSE_SIGMA_QUAT = 5; // samples (0.25 s)

/** Zero-phase Gaussian low-pass of a flat array of `width`-component rows, ends held. */
function gaussianSmooth(src, width, sigma) {
  const radius = Math.ceil(sigma * 3);
  const kernel = Array.from({ length: radius * 2 + 1 }, (_, i) => Math.exp(-0.5 * ((i - radius) / sigma) ** 2));
  const norm = kernel.reduce((a, b) => a + b, 0);
  const n = src.length / width;
  const out = new Float32Array(src.length);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < width; k++) {
      let acc = 0;
      for (let j = -radius; j <= radius; j++) acc += kernel[j + radius] * src[Math.min(Math.max(i + j, 0), n - 1) * width + k];
      out[i * width + k] = acc / norm;
    }
  }
  return out;
}

/** Remove q / -q sign flips so that the components can be filtered. */
function alignQuaternions(src) {
  const out = Float32Array.from(src);
  for (let i = 4; i < out.length; i += 4) {
    let dot = 0;
    for (let k = 0; k < 4; k++) dot += out[i + k] * out[i - 4 + k];
    if (dot < 0) for (let k = 0; k < 4; k++) out[i + k] = -out[i + k];
  }
  return out;
}

export function prepareEpisode(raw) {
  const c = raw.columns;
  const n = c.t.length;
  const f = (a) => Float32Array.from(a, (v) => v ?? 0);
  const ep = {
    id: raw.id,
    controller: raw.controller,
    crosswindKt: raw.crosswind_kt,
    gustPct: raw.gust_pct,
    glidepathDeg: raw.glidepath_deg,
    metrics: raw.metrics,
    touchdown: raw.touchdown,
    n,
    t: Float64Array.from(c.t),
    pos: flat(c.pos, 3),
    quat: flat(c.quat, 4),
    euler: flat(c.euler_deg, 3),
    rates: flat(c.rates_dps, 3),
    action: flat(c.action, 4),
    gear: flat(c.gear_n, 3),
    wind: flat(c.wind_mps, 3),
    x: f(c.x_m),
    y: f(c.y_m),
    h: f(c.h_m),
    vz: f(c.vz_mps),
    gs: f(c.gs_mps),
    ias: f(c.ias_mps),
    alpha: f(c.alpha_deg),
    beta: f(c.beta_deg),
    elev: f(c.elevator_deg),
    ail: f(c.aileron_deg),
    rud: f(c.rudder_deg),
    thr: f(c.throttle),
    turb: f(c.turb_mps),
    reward: f(c.reward),
    // Share of control authority held by the learned policy (1 = PPO, 0 = classical fallback)
    // and the runtime-assurance monitor that is active (index into rtaReasonCodes).
    authority: c.authority ? f(c.authority) : new Float32Array(n).fill(raw.controller === "pid" ? 0 : 1),
    rtaReason: c.rta_reason ? Uint8Array.from(c.rta_reason) : new Uint8Array(n),
    rtaReasonCodes: raw.rta_reason_codes ?? ["none"],
    controllerLabel: raw.controller_label ?? raw.controller,
  };
  ep.tEnd = ep.t[n - 1];
  // Render pose only (instruments keep the raw data): a 70 t airframe does not twitch at 20 Hz.
  ep.visPos = gaussianSmooth(ep.pos, 3, POSE_SIGMA_POS);
  ep.visQuat = gaussianSmooth(alignQuaternions(ep.quat), 4, POSE_SIGMA_QUAT);
  // Ground track from position differences (for drift angle and flight-path vector).
  ep.track = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - 1);
    const b = Math.min(n - 1, i + 1);
    const dx = ep.pos[b * 3] - ep.pos[a * 3];
    const dz = ep.pos[b * 3 + 2] - ep.pos[a * 3 + 2];
    // Heading convention of the simulator: 0 = runway direction (-Z), positive clockwise (towards +X).
    ep.track[i] = Math.hypot(dx, dz) > 1e-3 ? Math.atan2(dx, -dz) / DEG : ep.euler[i * 3 + 2];
  }
  // First contact time of each gear leg (nose, main left, main right) from the recorded loads.
  ep.contacts = [0, 1, 2].map((g) => {
    for (let i = 0; i < n; i++) {
      if (ep.gear[i * 3 + g] > 0) return { index: i, t: ep.t[i], vz: ep.vz[Math.max(0, i - 1)] };
    }
    return null;
  });
  ep.cumulativeReward = new Float32Array(n);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    acc += ep.reward[i];
    ep.cumulativeReward[i] = acc;
  }
  return ep;
}

export function createSample() {
  return {
    i: 0,
    u: 0,
    t: 0,
    pos: [0, 0, 0],
    quat: [0, 0, 0, 1],
    visPos: [0, 0, 0],
    visQuat: [0, 0, 0, 1],
    euler: [0, 0, 0],
    rates: [0, 0, 0],
    action: [0, 0, 0, 0],
    gear: [0, 0, 0],
    wind: [0, 0, 0],
    x: 0,
    y: 0,
    h: 0,
    vz: 0,
    gs: 0,
    ias: 0,
    alpha: 0,
    beta: 0,
    elev: 0,
    ail: 0,
    rud: 0,
    thr: 0,
    turb: 0,
    reward: 0,
    cumReward: 0,
    track: 0,
    authority: 0,
    rtaReason: "none",
  };
}

/** Index i such that t[i] <= time < t[i+1] (clamped). */
export function findIndex(ep, time) {
  const t = ep.t;
  if (time <= t[0]) return 0;
  if (time >= t[ep.n - 1]) return ep.n - 2;
  let lo = 0;
  let hi = ep.n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= time) lo = mid;
    else hi = mid;
  }
  return lo;
}

const lerp = (a, b, u) => a + (b - a) * u;

/** Uniform Catmull-Rom (cubic Hermite) segment between p1 and p2; C1-continuous across samples. */
function catmullRom(p0, p1, p2, p3, u) {
  const u2 = u * u;
  const u3 = u2 * u;
  return 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (3 * p1 - p0 - 3 * p2 + p3) * u3);
}

const wrap180 = (d) => ((((d + 180) % 360) + 360) % 360) - 180;

/** Catmull-Rom for angles in degrees: neighbours are unwrapped around the segment start. */
function catmullRomAngle(a0, a1, a2, a3, u) {
  const b0 = a1 + wrap180(a0 - a1);
  const b2 = a1 + wrap180(a2 - a1);
  const b3 = b2 + wrap180(a3 - a2);
  return wrap180(catmullRom(b0, a1, b2, b3, u));
}

/** Smooth quaternion: sign-aligned componentwise Catmull-Rom, renormalised (C1, no slerp kinks). */
function catmullRomQuat(q, i0, i1, i2, i3, u, out) {
  let len = 0;
  const dot = (a, b) => q[a] * q[b] + q[a + 1] * q[b + 1] + q[a + 2] * q[b + 2] + q[a + 3] * q[b + 3];
  const s0 = dot(i0, i1) < 0 ? -1 : 1;
  const s2 = dot(i2, i1) < 0 ? -1 : 1;
  const s3 = s2 * (dot(i3, i2) < 0 ? -1 : 1);
  for (let k = 0; k < 4; k++) {
    out[k] = catmullRom(s0 * q[i0 + k], q[i1 + k], s2 * q[i2 + k], s3 * q[i3 + k], u);
    len += out[k] * out[k];
  }
  len = Math.sqrt(len) || 1;
  for (let k = 0; k < 4; k++) out[k] /= len;
}

// Channels interpolated with cubic splines; bounded or event-like channels stay linear so that
// spline overshoot can never produce, e.g., negative gear loads or authority above 100 %.
const CUBIC_KEYS = ["x", "y", "h", "vz", "gs", "ias", "alpha", "beta", "elev", "ail", "rud", "turb"];
const LINEAR_KEYS = ["thr", "reward", "authority"];

/**
 * Interpolate every channel at `time` into `out`. The 20 Hz simulation samples are joined with
 * uniform Catmull-Rom splines (position, attitude quaternion, Euler angles, rates, air data, control
 * surfaces), giving continuous velocity and angular rate at the 60 Hz display refresh.
 */
export function sampleEpisode(ep, time, out) {
  const i = findIndex(ep, time);
  const j = i + 1;
  const i0 = Math.max(i - 1, 0);
  const i3 = Math.min(j + 1, ep.n - 1);
  const span = ep.t[j] - ep.t[i];
  const u = span > 0 ? Math.min(Math.max((time - ep.t[i]) / span, 0), 1) : 0;
  out.i = i;
  out.u = u;
  out.t = Math.min(Math.max(time, ep.t[0]), ep.tEnd);
  for (let k = 0; k < 3; k++) {
    out.pos[k] = catmullRom(ep.pos[i0 * 3 + k], ep.pos[i * 3 + k], ep.pos[j * 3 + k], ep.pos[i3 * 3 + k], u);
    out.euler[k] = catmullRomAngle(ep.euler[i0 * 3 + k], ep.euler[i * 3 + k], ep.euler[j * 3 + k], ep.euler[i3 * 3 + k], u);
    out.rates[k] = catmullRom(ep.rates[i0 * 3 + k], ep.rates[i * 3 + k], ep.rates[j * 3 + k], ep.rates[i3 * 3 + k], u);
    out.gear[k] = Math.max(lerp(ep.gear[i * 3 + k], ep.gear[j * 3 + k], u), 0);
    out.wind[k] = catmullRom(ep.wind[i0 * 3 + k], ep.wind[i * 3 + k], ep.wind[j * 3 + k], ep.wind[i3 * 3 + k], u);
  }
  for (let k = 0; k < 4; k++) out.action[k] = lerp(ep.action[i * 4 + k], ep.action[j * 4 + k], u);
  catmullRomQuat(ep.quat, i0 * 4, i * 4, j * 4, i3 * 4, u, out.quat);
  catmullRomQuat(ep.visQuat, i0 * 4, i * 4, j * 4, i3 * 4, u, out.visQuat);
  for (let k = 0; k < 3; k++) out.visPos[k] = catmullRom(ep.visPos[i0 * 3 + k], ep.visPos[i * 3 + k], ep.visPos[j * 3 + k], ep.visPos[i3 * 3 + k], u);
  for (const key of CUBIC_KEYS) {
    const a = ep[key];
    out[key] = catmullRom(a[i0], a[i], a[j], a[i3], u);
  }
  for (const key of LINEAR_KEYS) out[key] = lerp(ep[key][i], ep[key][j], u);
  out.thr = Math.min(Math.max(out.thr, 0), 1);
  out.track = catmullRomAngle(ep.track[i0], ep.track[i], ep.track[j], ep.track[i3], u);
  out.cumReward = lerp(ep.cumulativeReward[i], ep.cumulativeReward[j], u);
  out.rtaReason = ep.rtaReasonCodes[ep.rtaReason[u < 0.5 ? i : j]] ?? "none";
  return out;
}

/** Snap free slider values to the nearest computed grid point and return the scenario entry. */
export function snapScenario(manifest, controller, crosswindKt, gustPct, glidepathDeg) {
  const nearest = (arr, v) => arr.reduce((best, x) => (Math.abs(x - v) < Math.abs(best - v) ? x : best), arr[0]);
  const g = manifest.grid;
  const xw = nearest(g.crosswind_kt, crosswindKt);
  const gu = nearest(g.gust_pct, gustPct);
  const gp = nearest(g.glidepath_deg, glidepathDeg);
  const entry = manifest.scenarios.find(
    (s) => s.controller === controller && s.crosswind_kt === xw && s.gust_pct === gu && Math.abs(s.glidepath_deg - gp) < 1e-6,
  );
  return { entry, snapped: { crosswindKt: xw, gustPct: gu, glidepathDeg: gp } };
}

export const CONTROLLERS = [
  { id: "ppo", short: "PPO + RTA", hint: "PPO policy under Simplex runtime assurance (classical autoland fallback)" },
  { id: "ppo_raw", short: "PPO RAW", hint: "PPO policy without protection" },
  { id: "pid", short: "PID AUTOLAND", hint: "Classical crosswind autoland (baseline / fallback)" },
];

export function outcomeLabel(metrics) {
  if (!metrics) return { text: "—", tone: "mute" };
  if (metrics.success) return { text: "LANDED", tone: "ok" };
  const reason = (metrics.crash_reason || "timeout").replaceAll("_", " ").toUpperCase();
  return { text: reason, tone: "warn" };
}
