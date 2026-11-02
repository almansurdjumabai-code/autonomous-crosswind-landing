import { useEffect, useState } from "react";
import Canvas3D from "./components/Canvas3D.jsx";
import Dashboard from "./components/Dashboard.jsx";
import StatusBar from "./components/StatusBar.jsx";
import TopBar from "./components/TopBar.jsx";
import { CAMERAS, clock, QUALITY, seek, setQuality, ui, useUI } from "./lib/store.js";
import { AIRCRAFT, availableControllers, loadEpisode, loadManifest, loadRig, snapScenario } from "./lib/flightData.js";

/**
 * Shareable deep links: ?ac=atr72|cessna172&ctrl=ppo|ppo_raw|pid&xw=15&gust=50&gp=3&t=42.5&cam=3&paused=1&panels=0
 * (t is the simulation time to open at; cam is 1-4).
 */
let pendingSeek = null;
function deepLink() {
  const p = new URLSearchParams(window.location.search);
  const num = (k) => (p.has(k) && Number.isFinite(Number(p.get(k))) ? Number(p.get(k)) : null);
  const cam = num("cam");
  return {
    ac: AIRCRAFT.some((a) => a.id === p.get("ac")) ? p.get("ac") : null,
    ctrl: ["ppo", "ppo_raw", "pid"].includes(p.get("ctrl")) ? p.get("ctrl") : null,
    xw: num("xw"),
    gust: num("gust"),
    gp: num("gp"),
    t: num("t"),
    cam: cam >= 1 && cam <= 4 ? CAMERAS[cam - 1].id : null,
    paused: p.get("paused") === "1",
    panels: p.get("panels") === "0" ? false : null,
  };
}

/** Load the 3D rig once and the data set of the selected aircraft whenever it changes. */
function useBootstrap() {
  const [rig, setRig] = useState(null);
  const aircraft = useUI((s) => s.aircraft);
  useEffect(() => {
    const q = deepLink();
    if (q.ac && q.ac !== ui.get().aircraft) ui.set({ aircraft: q.ac });
    let cancelled = false;
    loadRig()
      .then((rigDoc) => !cancelled && setRig(rigDoc))
      .catch((err) => ui.set({ error: err.message, loading: false }));
    return () => {
      cancelled = true;
    };
  }, []);
  useEffect(() => {
    let cancelled = false;
    const first = ui.get().manifest == null;
    const q = first ? deepLink() : {};
    if (first && q.ac && q.ac !== aircraft) return undefined; // the deep link is applied first
    ui.set({ loading: true, error: null });
    loadManifest(aircraft)
      .then((manifest) => {
        if (cancelled) return;
        const st = ui.get();
        const d = manifest.default_episode;
        const controllers = availableControllers(manifest).map((c) => c.id);
        const wanted = q.ctrl ?? (first ? d?.controller : st.controller) ?? d?.controller;
        const narrow = window.innerWidth < 1100;
        ui.set({
          manifest,
          episode: null,
          scenarioId: null,
          controller: controllers.includes(wanted) ? wanted : (d?.controller ?? controllers[0] ?? "pid"),
          crosswindKt: q.xw ?? (first ? d?.crosswind_kt : st.crosswindKt) ?? 10,
          gustPct: q.gust ?? (first ? d?.gust_pct : st.gustPct) ?? 50,
          glidepathDeg: q.gp ?? (first ? d?.glidepath_deg : st.glidepathDeg) ?? 3,
          ...(first
            ? {
                camera: q.cam ?? "CAM-01",
                playing: !q.paused,
                showLeft: q.panels === false ? false : !narrow,
                showRight: q.panels === false ? false : !narrow,
              }
            : {}),
        });
        if (first && q.t != null) pendingSeek = q.t;
      })
      .catch((err) => ui.set({ error: err.message, loading: false }));
    return () => {
      cancelled = true;
    };
  }, [aircraft]);
  return rig;
}

/** Load the rollout nearest to the requested conditions whenever they change (debounced). */
function useScenarioSync() {
  const manifest = useUI((s) => s.manifest);
  const controller = useUI((s) => s.controller);
  const xw = useUI((s) => s.crosswindKt);
  const gust = useUI((s) => s.gustPct);
  const gp = useUI((s) => s.glidepathDeg);
  useEffect(() => {
    if (!manifest) return undefined;
    const { entry } = snapScenario(manifest, controller, xw, gust, gp);
    if (!entry || entry.id === ui.get().scenarioId) return undefined;
    let cancelled = false;
    const id = setTimeout(() => {
      ui.set({ loading: ui.get().episode == null });
      const progress = clock.sample && ui.get().episode ? (clock.t - ui.get().episode.t[0]) / (ui.get().episode.tEnd - ui.get().episode.t[0]) : 0;
      loadEpisode(manifest, entry.id)
        .then((ep) => {
          if (cancelled) return;
          ui.set({ episode: ep, scenarioId: entry.id, loading: false, error: null });
          if (pendingSeek != null) {
            seek(pendingSeek);
            pendingSeek = null;
          } else {
            // Keep the relative position in the approach when switching between comparable rollouts.
            seek(ep.t[0] + Math.min(Math.max(progress, 0), 1) * (ep.tEnd - ep.t[0]));
          }
        })
        .catch((err) => !cancelled && ui.set({ error: err.message, loading: false }));
    }, 120);
    return () => {
      cancelled = true;
      clearTimeout(id);
    };
  }, [manifest, controller, xw, gust, gp]);
}

function useKeyboard() {
  useEffect(() => {
    const onKey = (e) => {
      if (e.target instanceof HTMLInputElement && e.target.type !== "range") return;
      const st = ui.get();
      if (e.code === "Space") {
        e.preventDefault();
        if (st.autopilot === "MANUAL") ui.set({ autopilot: "ENGAGED", playing: true });
        else ui.set({ playing: !st.playing });
      } else if (["Digit1", "Digit2", "Digit3", "Digit4"].includes(e.code)) {
        ui.set({ camera: CAMERAS[Number(e.code.slice(-1)) - 1].id });
      } else if (e.code === "ArrowRight" || e.code === "ArrowLeft") {
        if (e.target instanceof HTMLInputElement) return;
        const step = (e.shiftKey ? 5 : 0.5) * (e.code === "ArrowRight" ? 1 : -1);
        seek(clock.t + step);
      } else if (e.code === "KeyF") {
        ui.set({ showFlow: !st.showFlow });
      } else if (e.code === "KeyG") {
        const ids = Object.keys(QUALITY);
        setQuality(ids[(ids.indexOf(st.quality) + 1) % ids.length]);
      } else if (e.code === "KeyR") {
        seek(0);
      } else if (e.code === "KeyM") {
        ui.set({ autopilot: st.autopilot === "MANUAL" ? "ENGAGED" : "MANUAL", playing: st.autopilot === "MANUAL" });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

export default function App() {
  const rig = useBootstrap();
  useScenarioSync();
  useKeyboard();
  return (
    <main className="relative h-full w-full overflow-hidden bg-obsidian-950">
      <div className="absolute inset-0">
        <Canvas3D rig={rig} />
      </div>
      <TopBar />
      <Dashboard />
      <StatusBar />
    </main>
  );
}
