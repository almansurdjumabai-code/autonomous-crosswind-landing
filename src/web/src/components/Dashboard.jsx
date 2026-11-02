import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Crosshair, Download, Hand, Pause, Play, Plane, Power, RotateCcw, Video } from "lucide-react";
import { useProgress } from "@react-three/drei";
import { CAMERAS, clock, QUALITY, seek, setQuality, ui, useClock, useUI } from "../lib/store.js";
import { AIRCRAFT, aircraftLimits, availableControllers, M_TO_FT, MPS_TO_KT, outcomeLabel, rawEpisode, snapScenario } from "../lib/flightData.js";
import { AoAGauge, AttitudeIndicator, CockpitHUD, HeadingTape, SlipIndicator, VerticalSpeed, VerticalTape } from "./avionics.jsx";
import TelemetryCharts from "./Charts.jsx";

const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : "—");
const signed = (v, d = 1) => (Number.isFinite(v) ? `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}` : "—");

/* ---------------------------------------------------------------------------------------------- */
function Section({ title, children, aside }) {
  return (
    <section className="border-b border-line px-3 py-3 last:border-b-0">
      <div className="mb-2.5 flex items-center justify-between">
        <h2 className="label text-ink-dim">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

function Slider({ id, label, unit, min, max, step, value, onChange, snapped, decimals = 0 }) {
  const fill = ((value - min) / (max - min)) * 100;
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="label">{label}</label>
        <span className="readout text-[13px] text-ink">
          {value.toFixed(decimals)} <span className="text-[10px] text-ink-mute">{unit}</span>
        </span>
      </div>
      <input
        id={id}
        type="range"
        className="precision"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ "--fill": `${fill}%` }}
      />
      <div className="readout flex justify-between text-[9px] text-ink-mute">
        <span>{min}</span>
        <span>
          GRID <span className={snapped === value ? "text-ink-dim" : "text-data"}>{snapped?.toFixed(decimals)} {unit}</span>
        </span>
        <span>{max}</span>
      </div>
    </div>
  );
}

function CommandButton({ icon: Icon, label, onClick, tone = "neutral", active = false, hint }) {
  const tones = {
    primary: active ? "border-hud bg-hud/15 text-hud" : "border-hud/40 text-hud hover:bg-hud/10",
    caution: active ? "border-caution bg-caution/15 text-caution" : "border-caution/40 text-caution hover:bg-caution/10",
    neutral: "border-line-strong text-ink-dim hover:border-ink-mute hover:text-ink",
  };
  return (
    <button
      type="button"
      onClick={onClick}
      title={hint}
      aria-pressed={tone !== "neutral" ? active : undefined}
      className={`readout flex min-h-11 items-center gap-2 border px-2.5 py-2 text-left text-[10.5px] font-medium tracking-[0.1em] transition-colors focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-data ${tones[tone]}`}
    >
      <Icon size={14} strokeWidth={1.6} className="shrink-0" />
      <span className="leading-tight">{label}</span>
    </button>
  );
}

/* ---------------------------------------------------------------------------------------------- */
function exportTelemetry() {
  const st = ui.get();
  const raw = st.scenarioId ? rawEpisode(st.manifest, st.scenarioId) : null;
  if (!raw) return;
  const doc = {
    exported_utc: new Date().toISOString(),
    project: "AUTONOMOUS-CROSSWIND-LANDING",
    exported_by: "crosswind autoland mission console",
    aircraft: st.manifest?.aircraft ?? null,
    scenario: { controller: st.controller, crosswind_kt: raw.crosswind_kt, gust_pct: raw.gust_pct, glidepath_deg: raw.glidepath_deg },
    playback_time_s: clock.t,
    frame: "three.js / glTF: +X east, +Y up, +Z south; runway threshold at origin",
    episode: raw,
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(doc)], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `${st.manifest?.aircraftId ?? "rollout"}_${raw.id}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function ControlPanel() {
  const st = useUI((s) => s);
  const snap = st.manifest ? snapScenario(st.manifest, st.controller, st.crosswindKt, st.gustPct, st.glidepathDeg) : null;
  const outcome = outcomeLabel(st.episode?.metrics);
  const m = st.episode?.metrics;
  const lim = aircraftLimits(st.manifest);
  const controllers = availableControllers(st.manifest);
  const learned = controllers.some((c) => c.id === "ppo");
  const xwMax = Math.max(30, ...(st.manifest?.grid?.crosswind_kt ?? []));
  const engage = () => {
    if (clock.t >= (st.episode?.tEnd ?? Infinity) - 0.05) seek(0);
    ui.set({ controller: learned ? "ppo" : "pid", autopilot: "ENGAGED", playing: true });
  };
  return (
    <aside className="glass pointer-events-auto flex h-full flex-col overflow-y-auto [scrollbar-width:thin]" aria-label="Mission controls">
      <Section title="Aircraft" aside={<span className="readout text-[10px] text-ink-mute">V<sub>REF</sub> {lim.vrefKt.toFixed(0)} KT · {(lim.massKg / 1000).toFixed(1)} T</span>}>
        <div className="grid grid-cols-2 border border-line" role="radiogroup" aria-label="Aircraft">
          {AIRCRAFT.map((a) => (
            <button
              key={a.id}
              type="button"
              role="radio"
              title={a.hint}
              aria-checked={st.aircraft === a.id}
              onClick={() => ui.set({ aircraft: a.id })}
              className={`readout flex min-h-9 items-center justify-center gap-1.5 px-1 text-[10px] tracking-[0.08em] ${st.aircraft === a.id ? "bg-data/15 text-data" : "text-ink-mute hover:text-ink"}`}
            >
              <Plane size={12} strokeWidth={1.6} /> {a.short}
            </button>
          ))}
        </div>
        <p className="mt-2 text-[11px] leading-snug text-ink-mute">
          {lim.name}: rollouts of the 6DoF simulator with this airframe's data. The 3D airframe is a generic airliner model shown at true scale.
        </p>
      </Section>

      <Section title="Guidance agent" aside={<span className="readout text-[10px] text-ink-mute">20 Hz POLICY</span>}>
        <div className="mb-3 grid border border-line" style={{ gridTemplateColumns: `repeat(${Math.max(controllers.length, 1)}, minmax(0, 1fr))` }} role="radiogroup" aria-label="Controller">
          {controllers.map((c) => (
            <button
              key={c.id}
              type="button"
              role="radio"
              title={c.hint}
              aria-checked={st.controller === c.id}
              onClick={() => ui.set({ controller: c.id })}
              className={`readout min-h-9 px-1 text-[10px] tracking-[0.08em] ${st.controller === c.id ? "bg-data/15 text-data" : "text-ink-mute hover:text-ink"}`}
            >
              {c.short}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          <CommandButton icon={Power} label={learned ? "ENGAGE PPO AUTOPILOT" : "ENGAGE AUTOLAND"} tone="primary" active={st.autopilot === "ENGAGED" && st.controller === (learned ? "ppo" : "pid")} onClick={engage} hint={learned ? "Replay the PPO policy rollout for the selected conditions" : "Replay the classical autoland rollout for the selected conditions (no trained policy for this aircraft yet)"} />
          <CommandButton icon={Hand} label="MANUAL OVERRIDE" tone="caution" active={st.autopilot === "MANUAL"} onClick={() => ui.set({ autopilot: "MANUAL", playing: false })} hint="Disengage: the timeline is under operator control (← → keys, scrubber)" />
          <CommandButton icon={RotateCcw} label="RESET TRAJECTORY" onClick={() => { seek(0); ui.set({ playing: st.autopilot === "ENGAGED" }); }} hint="Return to the start of the approach" />
          <CommandButton icon={Download} label="EXPORT TELEMETRY JSON" onClick={exportTelemetry} hint="Download the full 20 Hz telemetry of this rollout" />
        </div>
      </Section>

      <Section title="Environment" aside={<span className="readout text-[10px] text-ink-mute">{st.manifest?.scenarios.length ?? 0} ROLLOUTS</span>}>
        <Slider id="xw" label="Crosswind velocity" unit="kt" min={0} max={xwMax} step={1} value={st.crosswindKt} onChange={(v) => ui.set({ crosswindKt: v })} snapped={snap?.snapped.crosswindKt} />
        <Slider id="gust" label="Gust turbulence intensity" unit="%" min={0} max={100} step={1} value={st.gustPct} onChange={(v) => ui.set({ gustPct: v })} snapped={snap?.snapped.gustPct} />
        <Slider id="gp" label="Glidepath angle" unit="°" min={2.5} max={3.5} step={0.1} decimals={1} value={st.glidepathDeg} onChange={(v) => ui.set({ glidepathDeg: v })} snapped={snap?.snapped.glidepathDeg} />
        <p className="mt-1 text-[11px] leading-snug text-ink-mute">
          Each grid point is a full 6DoF rollout (Dryden turbulence, 1-cosine gusts, wind shear). Sliders select the nearest computed rollout.
        </p>
      </Section>

      <Section title="Rollout outcome">
        <div className="mb-2 flex items-center justify-between">
          <span className="readout truncate text-[10.5px] text-ink-dim" title={st.scenarioId ?? ""}>{st.scenarioId ?? "—"}</span>
          <span className={`readout border px-1.5 py-0.5 text-[10px] tracking-[0.12em] ${outcome.tone === "ok" ? "border-hud/50 text-hud" : outcome.tone === "warn" ? "border-warning/60 text-warning" : "border-line text-ink-mute"}`}>
            {outcome.text}
          </span>
        </div>
        <dl className="readout grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
          <dt className="text-ink-mute">TD SINK</dt>
          <dd className={`text-right ${m?.touchdown_sink_rate_mps < lim.crashSink ? "text-warning" : "text-ink"}`}>{fmt(m?.touchdown_sink_rate_mps, 2)} m/s</dd>
          <dt className="text-ink-mute">TD DIST THR</dt>
          <dd className="text-right text-ink">{fmt(m?.touchdown_x_m, 0)} m</dd>
          <dt className="text-ink-mute">TD LATERAL</dt>
          <dd className="text-right text-ink">{signed(m?.touchdown_y_m, 2)} m</dd>
          <dt className="text-ink-mute">TD CRAB</dt>
          <dd className="text-right text-ink">{signed(m?.touchdown_crab_deg, 1)}°</dd>
          <dt className="text-ink-mute">EPISODE RETURN</dt>
          <dd className="text-right text-ink">{fmt(m?.episode_return, 1)}</dd>
          <dt className="text-ink-mute">PPO AUTHORITY</dt>
          <dd className="text-right text-ink">{m?.policy_authority_mean != null ? `${(m.policy_authority_mean * 100).toFixed(0)} %` : "—"}</dd>
        </dl>
      </Section>

      <Section title="Camera">
        <div className="grid grid-cols-2 gap-1.5">
          {CAMERAS.map((c, i) => (
            <button
              key={c.id}
              type="button"
              title={c.hint}
              aria-pressed={st.camera === c.id}
              onClick={() => ui.set({ camera: c.id })}
              className={`readout flex min-h-11 flex-col items-start justify-center border px-2.5 py-1.5 text-left ${st.camera === c.id ? "border-data bg-data/10 text-data" : "border-line text-ink-dim hover:border-line-strong hover:text-ink"}`}
            >
              <span className="text-[10px] tracking-[0.14em]">[{c.id}] <span className="text-ink-mute">{i + 1}</span></span>
              <span className="text-[10.5px] font-medium tracking-[0.1em]">{c.name}</span>
            </button>
          ))}
        </div>
        <div className="mt-3">
          <div className="mb-1.5 flex items-baseline justify-between">
            <span className="label">Graphics quality</span>
            <span className="readout text-[9px] text-ink-mute">G to cycle</span>
          </div>
          <div className="mb-3 grid grid-cols-3 border border-line" role="radiogroup" aria-label="Graphics quality">
            {Object.entries(QUALITY).map(([id, q]) => (
              <button
                key={id}
                type="button"
                role="radio"
                aria-checked={st.quality === id}
                title={id === "PERFORMANCE" ? "No post-processing, native resolution" : id === "BALANCED" ? "Half-res bloom + MSAA x4, no depth of field (default)" : "Depth of field, MSAA x4, up to 2x pixel ratio"}
                onClick={() => setQuality(id)}
                className={`readout min-h-9 text-[10px] tracking-[0.1em] ${st.quality === id ? "bg-data/15 text-data" : "text-ink-mute hover:text-ink"}`}
              >
                {q.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            aria-pressed={st.showFlow}
            onClick={() => ui.set({ showFlow: !st.showFlow })}
            className={`readout mb-3 flex min-h-9 w-full items-center justify-between border px-2.5 text-[10.5px] tracking-[0.1em] ${st.showFlow ? "border-data bg-data/10 text-data" : "border-line text-ink-dim hover:border-line-strong hover:text-ink"}`}
          >
            <span>WIND &amp; AIRFLOW</span>
            <span className="text-[9px] text-ink-mute">F · {st.showFlow ? "ON" : "OFF"}</span>
          </button>
          <Slider id="gain" label="Surface deflection visual gain" unit="×" min={1} max={3} step={0.5} decimals={1} value={st.surfaceGain} onChange={(v) => ui.set({ surfaceGain: v })} snapped={st.surfaceGain} />
        </div>
      </Section>
    </aside>
  );
}

/* ---------------------------------------------------------------------------------------------- */
const RTA_TEXT = {
  bank: "BANK LIMIT", pitch: "PITCH LIMIT", alpha: "AOA LIMIT", airspeed: "LOW SPEED", sink_rate: "SINK RATE",
  glideslope: "G/S DEVIATION", localizer: "LOC DEVIATION", flare_gate: "FLARE GATE", locked_out: "PRIMARY LOCKED OUT",
};

function AuthorityAnnunciator({ s }) {
  const controller = useUI((st) => st.controller);
  if (controller !== "ppo") return null;
  const auth = s.authority;
  const fallback = auth < 0.5;
  return (
    <div className={`mb-1.5 flex items-center gap-2 border px-2 py-1 ${fallback ? "border-caution/50" : "border-hud/40"}`} role="status">
      <span className={`readout text-[10px] font-bold tracking-[0.14em] ${fallback ? "text-caution" : "text-hud"}`}>
        {fallback ? "RTA FALLBACK" : "PPO IN COMMAND"}
      </span>
      <div className="relative h-1 flex-1 bg-line" aria-label={`PPO authority ${(auth * 100).toFixed(0)} percent`}>
        <div className="absolute inset-0 origin-left bg-hud transition-transform duration-100" style={{ transform: `scaleX(${auth})` }} />
      </div>
      <span className="readout w-10 text-right text-[10px] text-ink">{(auth * 100).toFixed(0)}%</span>
      {s.rtaReason !== "none" && <span className="readout text-[10px] text-caution">{RTA_TEXT[s.rtaReason] ?? s.rtaReason.toUpperCase()}</span>}
    </div>
  );
}

/* ---------------------------------------------------------------------------------------------- */
function PrimaryFlightDisplay() {
  const clk = useClock(60);
  const s = clk.sample;
  const gp = useUI((st) => st.episode?.glidepathDeg ?? 3);
  const lim = aircraftLimits(useUI((st) => st.manifest));
  if (!s) return null;
  const [roll, pitch, hdg] = s.euler;
  const gamma = (Math.atan2(s.vz, Math.max(s.gs, 1)) * 180) / Math.PI;
  let drift = s.track - hdg;
  if (drift > 180) drift -= 360;
  if (drift < -180) drift += 360;
  const gsErr = s.h - Math.max(lim.aimPointM - s.x, 0) * Math.tan((gp * Math.PI) / 180);
  const xw = -s.wind[0] * MPS_TO_KT;
  const hw = s.wind[2] * MPS_TO_KT;
  return (
    <div className="glass scanlines pointer-events-auto px-3 pb-2 pt-2" aria-label="Primary flight display">
      <AuthorityAnnunciator s={s} />
      <div className="mb-1 flex items-center justify-between">
        <span className="label">PFD · glass cockpit</span>
        <span className="readout text-[10px] text-ink-mute">
          XW {xw >= 0 ? "R" : "L"} {Math.abs(xw).toFixed(1)} KT · HW {hw.toFixed(1)} KT · TURB {s.turb.toFixed(1)} M/S
        </span>
      </div>
      <div className="flex items-start gap-1.5">
        <VerticalTape label="IAS" unit="kt" value={s.ias * MPS_TO_KT} pxPerUnit={3.2} step={5} labelEvery={10} side="left"
          bugs={[{ value: lim.vrefKt, color: "#7cffb2" }]} tone={s.ias < lim.minAirspeedMps * 1.1 ? "#ff4d4d" : "#e8eef3"} />
        <div className="flex flex-col items-center">
          <AttitudeIndicator pitch={pitch} roll={roll} size={210} fpv={{ x: drift, y: gamma - pitch }} />
          <HeadingTape heading={hdg} track={s.track} width={210} />
        </div>
        <VerticalTape label="ALT" unit="ft" value={s.h * M_TO_FT} pxPerUnit={1.25} step={10} labelEvery={50} side="right"
          bugs={[{ value: 0, color: "#ffb547" }]} tone={s.h < 15 ? "#ffb547" : "#e8eef3"} />
        <VerticalSpeed vz={s.vz} crash={lim.crashSink} />
      </div>
      <div className="mt-1 flex items-center justify-between border-t border-line pt-1.5">
        <AoAGauge alpha={s.alpha} warn={lim.alphaWarnDeg} stall={lim.alphaStallDeg} />
        <SlipIndicator beta={s.beta} />
        <dl className="readout grid grid-cols-2 gap-x-2 gap-y-0.5 text-[10.5px]">
          <dt className="text-ink-mute">GS</dt>
          <dd className="text-right text-ink">{(s.gs * MPS_TO_KT).toFixed(0)} kt</dd>
          <dt className="text-ink-mute">G/S DEV</dt>
          <dd className={`text-right ${Math.abs(gsErr) > 15 ? "text-caution" : "text-ink"}`}>{signed(gsErr, 1)} m</dd>
          <dt className="text-ink-mute">LOC DEV</dt>
          <dd className={`text-right ${Math.abs(s.y) > 10 ? "text-caution" : "text-ink"}`}>{signed(s.y, 1)} m</dd>
          <dt className="text-ink-mute">THR</dt>
          <dd className="text-right text-ink">{(s.thr * 100).toFixed(0)} %</dd>
        </dl>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------------------------- */
function Readout({ label, value, unit, tone = "text-ink" }) {
  return (
    <div className="flex items-baseline justify-between gap-2 border-b border-line/60 py-1">
      <span className="label text-[9px]">{label}</span>
      <span className={`readout text-[12px] ${tone}`}>
        {value} <span className="text-[9.5px] text-ink-mute">{unit}</span>
      </span>
    </div>
  );
}

function GearBars({ loads, max = 14000 }) {
  return (
    <div className="grid grid-cols-3 gap-2">
      {["NOSE", "MAIN L", "MAIN R"].map((name, i) => (
        <div key={name}>
          <div className="mb-1 h-1.5 bg-line">
            <div className="h-full origin-left bg-data transition-transform duration-75" style={{ transform: `scaleX(${Math.min(loads[i] / max, 1)})` }} />
          </div>
          <div className="readout flex justify-between text-[9.5px]">
            <span className="text-ink-mute">{name}</span>
            <span className="text-ink">{(loads[i] / 1000).toFixed(1)}k</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function AnalyticsPanel() {
  const clk = useClock(12);
  const s = clk.sample;
  const lim = aircraftLimits(useUI((st) => st.manifest));
  return (
    <aside className="glass pointer-events-auto flex h-full flex-col overflow-y-auto px-3 py-3 [scrollbar-width:thin]" aria-label="Telemetry analytics">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="label text-ink-dim">High-frequency telemetry</h2>
        <span className="readout text-[10px] text-ink-mute">WINDOW 15 S</span>
      </div>
      <TelemetryCharts />
      {s && (
        <>
          <h2 className="label mb-1 mt-3 text-ink-dim">State vector</h2>
          <div className="grid grid-cols-2 gap-x-4">
            <Readout label="φ ROLL" value={signed(s.euler[0], 1)} unit="°" tone={Math.abs(s.euler[0]) > 25 ? "text-caution" : "text-ink"} />
            <Readout label="p" value={signed(s.rates[0], 1)} unit="°/s" />
            <Readout label="θ PITCH" value={signed(s.euler[1], 1)} unit="°" />
            <Readout label="q" value={signed(s.rates[1], 1)} unit="°/s" />
            <Readout label="ψ HDG" value={signed(s.euler[2], 1)} unit="°" />
            <Readout label="r" value={signed(s.rates[2], 1)} unit="°/s" />
            <Readout label="δe" value={signed(s.elev, 1)} unit="°" />
            <Readout label="δa" value={signed(s.ail, 1)} unit="°" />
            <Readout label="δr" value={signed(s.rud, 1)} unit="°" />
            <Readout label="X THR" value={fmt(s.x, 0)} unit="m" />
            <Readout label="Vz" value={signed(s.vz, 2)} unit="m/s" tone={s.vz < lim.crashSink ? "text-warning" : "text-ink"} />
            <Readout label="Σ REWARD" value={fmt(s.cumReward, 1)} unit="" />
          </div>
          <h2 className="label mb-1.5 mt-3 text-ink-dim">Gear normal load [kN]</h2>
          <GearBars loads={s.gear} max={lim.maxGearLoadN} />
          <h2 className="label mb-1 mt-3 text-ink-dim">Policy action vector</h2>
          <div className="grid grid-cols-4 gap-1.5">
            {["ELEV", "AIL", "RUD", "THR"].map((n, i) => (
              <div key={n} className="border border-line px-1.5 py-1 text-center">
                <div className="label text-[8.5px]">{n}</div>
                <div className="readout text-[11px] text-ink">{signed(s.action[i], 2)}</div>
                <div className="relative mt-1 h-1 bg-line">
                  <div className="absolute top-0 h-full bg-data" style={{ left: `${50 + Math.min(s.action[i], 0) * 50}%`, width: `${Math.abs(s.action[i]) * 50}%` }} />
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </aside>
  );
}

/* ---------------------------------------------------------------------------------------------- */
function Timeline() {
  const clk = useClock(15);
  const ep = useUI((s) => s.episode);
  const playing = useUI((s) => s.playing);
  const speed = useUI((s) => s.speed);
  const autopilot = useUI((s) => s.autopilot);
  if (!ep) return null;
  const t0 = ep.t[0];
  const span = ep.tEnd - t0;
  const pct = (t) => ((t - t0) / span) * 100;
  const td = ep.touchdown?.time_s;
  return (
    <div className="glass pointer-events-auto flex items-center gap-3 px-3 py-2">
      <button
        type="button"
        onClick={() => {
          // Play re-engages the autopilot replay (leaving manual override); pause just holds the clock.
          const resume = !playing || autopilot === "MANUAL";
          if (resume && clk.t >= ep.tEnd - 0.05) seek(0);
          ui.set({ playing: resume, autopilot: resume ? "ENGAGED" : autopilot });
        }}
        className="grid h-8 w-8 shrink-0 place-items-center border border-line-strong text-ink hover:border-data hover:text-data"
        aria-label={playing ? "Pause playback" : "Play"}
      >
        {playing ? <Pause size={14} /> : <Play size={14} />}
      </button>
      <div className="relative flex-1">
        <input
          type="range"
          className="precision"
          min={t0}
          max={ep.tEnd}
          step={0.05}
          value={clk.t}
          onChange={(e) => seek(Number(e.target.value))}
          aria-label="Mission timeline"
          style={{ "--fill": `${pct(clk.t)}%` }}
        />
        {td != null && (
          <div className="pointer-events-none absolute -top-1 h-1.5 w-px bg-caution" style={{ left: `${pct(td)}%` }} title="Touchdown">
            <span className="readout absolute -top-3 left-1/2 -translate-x-1/2 text-[9px] text-caution">TD</span>
          </div>
        )}
        <div className="readout pointer-events-none flex justify-between text-[9px] text-ink-mute">
          <span>T+{t0.toFixed(0)}</span>
          
          <span>T+{ep.tEnd.toFixed(1)}</span>
        </div>
      </div>
      <div className="flex shrink-0 border border-line" role="radiogroup" aria-label="Playback speed">
        {[0.25, 0.5, 1, 2, 4].map((v) => (
          <button
            key={v}
            type="button"
            role="radio"
            aria-checked={speed === v}
            onClick={() => ui.set({ speed: v })}
            className={`readout min-h-8 px-1.5 text-[10px] ${speed === v ? "bg-data/15 text-data" : "text-ink-mute hover:text-ink"}`}
          >
            {v}×
          </button>
        ))}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------------------------- */
function TouchdownAnnunciator() {
  const clk = useClock(10);
  const ep = useUI((s) => s.episode);
  const lim = aircraftLimits(useUI((s) => s.manifest));
  const [shown, setShown] = useState(null);
  const last = useRef(0);
  const t = clk.t;
  useEffect(() => {
    const td = ep?.touchdown?.time_s;
    if (td == null) return;
    if (last.current < td && t >= td && t - td < 0.5) setShown({ ...ep.touchdown, metrics: ep.metrics });
    else if (t < td) setShown(null);
    last.current = t;
  }, [t, ep]);
  useEffect(() => {
    if (!shown) return undefined;
    const id = setTimeout(() => setShown(null), 4500);
    return () => clearTimeout(id);
  }, [shown]);
  const hard = shown && shown.sink_rate_mps < lim.crashSink;
  return (
    <AnimatePresence>
      {shown && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.25 }}
          className={`glass pointer-events-none px-4 py-2.5 ${hard ? "border-warning/60" : "border-hud/50"}`}
          role="alert"
        >
          <div className={`readout text-[12px] font-bold tracking-[0.2em] ${hard ? "text-warning" : "text-hud"}`}>
            {hard ? "HARD LANDING" : "TOUCHDOWN"}
          </div>
          <div className="readout mt-1 text-[11px] text-ink">
            Vz {shown.sink_rate_mps.toFixed(2)} m/s · X {shown.x_m.toFixed(0)} m · Y {signed(shown.y_m, 2)} m · CRAB {signed((shown.heading_error_rad * 180) / Math.PI, 1)}°
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function CockpitHUDLayer() {
  const gp = useUI((s) => s.episode?.glidepathDeg ?? 3);
  const lim = aircraftLimits(useUI((s) => s.manifest));
  const clk = useClock(60);
  const [size, setSize] = useState({ w: window.innerWidth, h: window.innerHeight });
  useEffect(() => {
    const on = () => setSize({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return <CockpitHUD s={clk.sample} width={size.w} height={size.h} glidepathDeg={gp} crashSink={lim.crashSink} />;
}

function HUDLayer() {
  const camera = useUI((s) => s.camera);
  return camera === "CAM-03" ? <CockpitHUDLayer /> : null;
}

function LoadingOverlay() {
  const { progress, active } = useProgress();
  const loading = useUI((s) => s.loading);
  const error = useUI((s) => s.error);
  const visible = loading || active || error;
  return (
    <AnimatePresence>
      {visible && (
        <motion.div initial={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.6 }}
          className="absolute inset-0 z-40 grid place-items-center bg-obsidian-950">
          <div className="w-[320px]">
            <div className="readout text-[12px] tracking-[0.24em] text-ink">CROSSWIND AUTOLAND</div>
            <div className="label mt-1">{error ? "Telemetry link fault" : "Acquiring telemetry link · loading scene.glb"}</div>
            <div className="mt-4 h-px bg-line">
              <div className="h-px bg-data transition-[width]" style={{ width: `${error ? 100 : Math.max(progress, loading ? 8 : 100)}%` }} />
            </div>
            <div className={`readout mt-2 text-[10px] ${error ? "text-warning" : "text-ink-mute"}`}>
              {error ? String(error) : `${progress.toFixed(0)} %`}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/* ---------------------------------------------------------------------------------------------- */
export default function Dashboard() {
  const showLeft = useUI((s) => s.showLeft);
  const showRight = useUI((s) => s.showRight);
  const camera = useUI((s) => s.camera);
  const cockpit = camera === "CAM-03";
  return (
    <div className="pointer-events-none absolute inset-0 z-20">
      <HUDLayer />
      <AnimatePresence>
        {showLeft && (
          <motion.div key="left" initial={{ x: -24, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: -24, opacity: 0 }} transition={{ duration: 0.22 }}
            className="absolute bottom-9 left-2 top-14 w-[292px] max-w-[calc(100vw-1rem)]">
            <ControlPanel />
          </motion.div>
        )}
        {showRight && (
          <motion.div key="right" initial={{ x: 24, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 24, opacity: 0 }} transition={{ duration: 0.22 }}
            className="absolute bottom-9 right-2 top-14 hidden w-[340px] lg:block">
            <AnalyticsPanel />
          </motion.div>
        )}
      </AnimatePresence>
      <div className={`absolute bottom-9 flex flex-col items-center gap-2 ${showLeft ? "left-[302px]" : "left-2"} ${showRight ? "lg:right-[350px]" : "right-2"} right-2`}>
        <div className="absolute -top-14 left-1/2 -translate-x-1/2">
          <TouchdownAnnunciator />
        </div>
        {!cockpit && (
          <div className="hidden md:block">
            <PrimaryFlightDisplay />
          </div>
        )}
        <div className="w-full max-w-[760px]">
          <Timeline />
        </div>
      </div>
      <div className="pointer-events-none absolute left-1/2 top-14 -translate-x-1/2">
        <div className="readout flex items-center gap-1.5 border border-line bg-obsidian-950/60 px-2 py-1 text-[10px] tracking-[0.14em] text-ink-dim">
          <Video size={11} /> {CAMERAS.find((c) => c.id === camera)?.id} · {CAMERAS.find((c) => c.id === camera)?.name}
          {cockpit && <Crosshair size={11} className="text-hud" />}
        </div>
      </div>
      <LoadingOverlay />
    </div>
  );
}
