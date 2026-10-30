import { memo, useEffect, useRef } from "react";
import { clock, useUI } from "../lib/store.js";
import { findIndex } from "../lib/flightData.js";

// Categorical slots (validated dark-mode palette, fixed order).
export const SERIES = ["#3987e5", "#d95926", "#199e70"];
const GRID = "rgba(148, 175, 196, 0.10)";
const ZERO = "rgba(148, 175, 196, 0.30)";
const AXIS = "#5d6b77";
const WINDOW_S = 15;
const HEIGHT = 86;
const LEFT = 30;
const BOTTOM = 14;
// Canvas strokes cost a fraction of a millisecond, so the strips can redraw this often; the SVG
// chart library they replace stalled the WebGL frame by about 120 ms on every update.
const CHART_HZ = 20;

const Y_SERIES = [{ channel: (ep, i) => ep.y[i], name: "Y_err", color: SERIES[0] }];
const Y_LIMITS = [15, -15];
const PHI_SERIES = [{ channel: (ep, i) => ep.euler[i * 3], name: "φ", color: SERIES[0] }];
const SURFACE_SERIES = [
  { channel: (ep, i) => ep.elev[i], name: "δe", color: SERIES[0] },
  { channel: (ep, i) => ep.ail[i], name: "δa", color: SERIES[1] },
  { channel: (ep, i) => ep.rud[i], name: "δr", color: SERIES[2] },
];
const LAMBDA_SERIES = [{ channel: (ep, i) => ep.authority[i] * 100, name: "λ", color: SERIES[0] }];
const PERCENT = [0, 100];
const NO_REFS = [];

/** Nice round tick step covering `span` in about three intervals. */
function tickStep(span) {
  const raw = span / 3 || 1;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
}

function drawChart(canvas, ep, t, series, domain, refLines, hoverX) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth;
  if (canvas.width !== Math.round(w * dpr)) canvas.width = Math.round(w * dpr);
  if (canvas.height !== Math.round(HEIGHT * dpr)) canvas.height = Math.round(HEIGHT * dpr);
  const c = canvas.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, HEIGHT);
  const t0 = Math.max(0, t - WINDOW_S);
  const span = Math.max(t - t0, 1e-3);
  const i0 = findIndex(ep, t0);
  const i1 = Math.min(findIndex(ep, t) + 1, ep.n - 1);

  let lo = domain ? domain[0] : Infinity;
  let hi = domain ? domain[1] : -Infinity;
  if (!domain) {
    for (const s of series) {
      for (let i = i0; i <= i1; i++) {
        const v = s.channel(ep, i);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (hi - lo < 0.5) {
      const mid = (hi + lo) / 2;
      lo = mid - 0.25;
      hi = mid + 0.25;
    }
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;
  }
  const plotW = w - LEFT - 4;
  const plotH = HEIGHT - BOTTOM - 4;
  const X = (time) => LEFT + ((time - t0) / span) * plotW;
  const Y = (v) => 4 + (1 - (v - lo) / (hi - lo)) * plotH;

  c.font = '9px "JetBrains Mono", monospace';
  c.lineWidth = 1;
  c.textBaseline = "middle";
  c.textAlign = "right";
  const dy = tickStep(hi - lo);
  for (let v = Math.ceil(lo / dy) * dy; v <= hi; v += dy) {
    const y = Math.round(Y(v)) + 0.5;
    c.strokeStyle = GRID;
    c.beginPath();
    c.moveTo(LEFT, y);
    c.lineTo(w - 4, y);
    c.stroke();
    c.fillStyle = AXIS;
    c.fillText(Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1), LEFT - 4, y);
  }
  c.textAlign = "center";
  c.textBaseline = "alphabetic";
  const dx = tickStep(span);
  for (let v = Math.ceil(t0 / dx) * dx; v <= t; v += dx) c.fillText(v.toFixed(0), Math.min(X(v), w - 10), HEIGHT - 2);
  if (lo < 0 && hi > 0) {
    c.strokeStyle = ZERO;
    c.beginPath();
    c.moveTo(LEFT, Y(0));
    c.lineTo(w - 4, Y(0));
    c.stroke();
  }
  c.strokeStyle = "#ff4d4d";
  c.setLineDash([3, 3]);
  for (const r of refLines) {
    if (r <= lo || r >= hi) continue;
    c.beginPath();
    c.moveTo(LEFT, Y(r));
    c.lineTo(w - 4, Y(r));
    c.stroke();
  }
  c.setLineDash([]);

  c.save();
  c.beginPath();
  c.rect(LEFT, 0, plotW, HEIGHT - BOTTOM);
  c.clip();
  c.lineWidth = 1.5;
  c.lineJoin = "round";
  for (const s of series) {
    c.strokeStyle = s.color;
    c.beginPath();
    for (let i = i0; i <= i1; i++) {
      const x = X(ep.t[i]);
      const y = Y(s.channel(ep, i));
      if (i === i0) c.moveTo(x, y);
      else c.lineTo(x, y);
    }
    c.stroke();
  }
  c.restore();

  if (hoverX != null && hoverX >= LEFT) {
    const time = t0 + ((hoverX - LEFT) / plotW) * span;
    const i = Math.min(findIndex(ep, time) + 1, ep.n - 1);
    c.strokeStyle = "rgba(148, 175, 196, 0.35)";
    c.beginPath();
    c.moveTo(hoverX, 0);
    c.lineTo(hoverX, HEIGHT - BOTTOM);
    c.stroke();
    c.textAlign = hoverX > w / 2 ? "right" : "left";
    c.textBaseline = "top";
    const tx = hoverX + (hoverX > w / 2 ? -5 : 5);
    c.fillStyle = "#e8eef3";
    c.fillText(`T+${ep.t[i].toFixed(1)}`, tx, 3);
    series.forEach((s, k) => {
      c.fillStyle = s.color;
      c.fillText(`${s.name} ${s.channel(ep, i).toFixed(2)}`, tx, 14 + k * 11);
    });
  }
}

const Panel = memo(function Panel({ title, ep, series, domain = null, refLines = NO_REFS }) {
  const canvas = useRef(null);
  const hover = useRef(null);
  const last = useRef(-1);
  useEffect(() => {
    let raf = 0;
    let next = 0;
    last.current = -1;
    const tick = (now) => {
      raf = requestAnimationFrame(tick);
      if (now < next) return;
      next = now + 1000 / CHART_HZ;
      // Paused and not hovered: nothing changed, so nothing is redrawn.
      if (last.current === clock.t && hover.current === last.hover) return;
      last.current = clock.t;
      last.hover = hover.current;
      if (canvas.current) drawChart(canvas.current, ep, Math.max(clock.t, ep.t[1]), series, domain, refLines, hover.current);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [ep, series, domain, refLines]);
  return (
    <div className="border-t border-line pt-2">
      <div className="mb-1 flex items-center justify-between">
        <span className="label">{title}</span>
        {series.length > 1 && (
          <div className="flex gap-3">
            {series.map((s) => (
              <span key={s.name} className="readout flex items-center gap-1 text-[10px] text-ink-dim">
                <span className="inline-block h-[2px] w-3" style={{ background: s.color }} />
                {s.name}
              </span>
            ))}
          </div>
        )}
      </div>
      <canvas
        ref={canvas}
        className="block w-full"
        style={{ height: HEIGHT }}
        onMouseMove={(e) => (hover.current = e.nativeEvent.offsetX)}
        onMouseLeave={() => (hover.current = null)}
      />
    </div>
  );
});

/** Memoised with no props: the analytics panel's own readout ticks never redraw the charts. */
const TelemetryCharts = memo(function TelemetryCharts() {
  const ep = useUI((s) => s.episode);
  const controller = useUI((s) => s.controller);
  if (!ep) return null;
  return (
    <div className="space-y-2">
      <Panel title="Lateral offset Y_err [m]" ep={ep} series={Y_SERIES} refLines={Y_LIMITS} />
      <Panel title="Roll correction φ [deg]" ep={ep} series={PHI_SERIES} />
      <Panel title="Control surfaces δ [deg]" ep={ep} series={SURFACE_SERIES} />
      {controller === "ppo" && <Panel title="PPO control authority λ [%]" ep={ep} series={LAMBDA_SERIES} domain={PERCENT} />}
    </div>
  );
});

export default TelemetryCharts;
