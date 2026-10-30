import { startTransition, useEffect, useReducer, useSyncExternalStore } from "react";

/**
 * Two channels of state:
 *  - `ui`: low-frequency mission configuration (scenario, controller, camera, playback flags);
 *    React components subscribe through `useUI(selector)`.
 *  - `clock`: a mutable object advanced once per WebGL frame by the 3D engine (simulation time,
 *    interpolated telemetry sample, render statistics). Overlays poll it with `useClock(hz)` so
 *    60 FPS rendering never forces 60 Hz React reconciliation of the whole UI; the 3D engine
 *    never calls into React.
 */
function createStore(initial) {
  let state = initial;
  const listeners = new Set();
  return {
    get: () => state,
    set(patch) {
      const next = typeof patch === "function" ? patch(state) : patch;
      state = { ...state, ...next };
      listeners.forEach((l) => l());
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export const CAMERAS = [
  { id: "CAM-01", name: "CHASE", hint: "Lag-followed tail camera" },
  { id: "CAM-02", name: "TDZ TRACK", hint: "Runway touchdown-zone tracking" },
  { id: "CAM-03", name: "COCKPIT HUD", hint: "Pilot eye point, head-up display" },
  { id: "CAM-04", name: "FREE ORBIT", hint: "Interactive orbit around the airframe" },
];

/**
 * Render-quality presets. BALANCED is the default and targets a solid 60 FPS on integrated and
 * mid-range GPUs; CINEMATIC re-enables depth of field and MSAA.
 *
 * `dpr` is a fixed cap on the device pixel ratio, never scaled at runtime: the viewport always
 * renders at min(devicePixelRatio, dpr), at least 1.0, so it never drops below native and blurs.
 * The sun's shadow box is 96 m across to hold the airliner, so 2048^2 gives about 5 cm per texel.
 */
export const QUALITY = {
  PERFORMANCE: { label: "PERF", dpr: 1.0, post: false, dof: false, msaa: 0, smaa: false, shadowMap: 1024 },
  BALANCED: { label: "BAL", dpr: 1.5, post: true, dof: false, msaa: 4, smaa: false, shadowMap: 2048 },
  CINEMATIC: { label: "CINE", dpr: 2.0, post: true, dof: true, msaa: 4, smaa: false, shadowMap: 2048 },
};

/** Fixed render pixel ratio for a preset on this display. */
export function presetDpr(quality) {
  const device = Math.min(window.devicePixelRatio || 1, 2);
  return Math.max(1, Math.min(device, QUALITY[quality].dpr));
}

function storedQuality() {
  try {
    const q = window.localStorage.getItem("alma.quality");
    return q && QUALITY[q] ? q : "BALANCED";
  } catch {
    return "BALANCED";
  }
}

export function setQuality(q) {
  if (!QUALITY[q]) return;
  ui.set({ quality: q });
  try {
    window.localStorage.setItem("alma.quality", q);
  } catch {
    /* storage unavailable (private mode): the choice simply is not remembered */
  }
}

export const ui = createStore({
  quality: storedQuality(),
  manifest: null,
  episode: null,
  scenarioId: null,
  loading: true,
  error: null,
  controller: "ppo", // "ppo" | "pid"
  crosswindKt: 10,
  gustPct: 50,
  glidepathDeg: 3.0,
  autopilot: "ENGAGED", // "ENGAGED" | "MANUAL"
  playing: true,
  speed: 1,
  camera: "CAM-01",
  surfaceGain: 1,
  showFlow: true,
  showLeft: true,
  showRight: true,
});

export function useUI(selector) {
  return useSyncExternalStore(ui.subscribe, () => selector(ui.get()));
}

export const clock = {
  t: 0, // simulation time [s]
  frame: 0, // rendered frames since load
  fps: 0,
  frameMs: 0, // CPU time spent in the per-frame telemetry pipeline
  latencyMs: 0, // age of the newest telemetry sample used for the interpolated pose
  sample: null, // interpolated telemetry sample (see flightData.sampleEpisode)
  index: 0, // telemetry sample index
  render: { calls: 0, triangles: 0, api: "WEBGL", passes: [], dpr: 1 },
  seekTo: null, // pending seek requested by the UI, consumed by the engine
};

/**
 * One shared UI ticker for every overlay. Subscribers are grouped by rate, so each group
 * re-renders in a single React batch per tick, not one rAF loop and one render per component.
 *
 * Rates below URGENT_HZ (charts, analytics readouts, timeline, status bar) are flushed as
 * transitions: React renders them time-sliced and yields to the browser, so they can never block
 * a WebGL frame. Only the flight instruments (PFD, cockpit HUD) render as urgent 60 Hz updates.
 */
const URGENT_HZ = 30;
const groups = new Map(); // hz -> { period, last, subs }
let raf = 0;

function tick(ts) {
  for (const [hz, g] of groups) {
    // Phase-locked with a small tolerance: on a 60 Hz display rAF intervals jitter around
    // 16.6 ms, and a strict `>= period` test would skip every other frame (30 Hz judder).
    if (ts - g.last < g.period - 1.5) continue;
    g.last = Math.max(g.last + g.period, ts - g.period);
    const notify = () => g.subs.forEach((f) => f());
    if (hz >= URGENT_HZ) notify();
    else startTransition(notify);
  }
  raf = groups.size ? requestAnimationFrame(tick) : 0;
}

/** Re-render the calling component `hz` times per second while mounted. */
export function useClock(hz = 15) {
  const [, force] = useReducer((x) => x + 1, 0);
  useEffect(() => {
    if (!groups.has(hz)) groups.set(hz, { period: 1000 / hz, last: -Infinity, subs: new Set() });
    const g = groups.get(hz);
    g.subs.add(force);
    if (!raf) raf = requestAnimationFrame(tick);
    return () => {
      g.subs.delete(force);
      if (!g.subs.size) groups.delete(hz);
    };
  }, [hz]);
  return clock;
}

export function seek(t) {
  clock.seekTo = t;
}
