/* Vector avionics instruments (SVG). All angles in degrees, speeds in kt, altitude in ft. */

const HUD = "#7cffb2";
const INK = "#e8eef3";
const DIM = "#5d6b77";
const CAUTION = "#ffb547";
const WARNING = "#ff4d4d";
const DATA = "#5cc8ff";

const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const fmt = (v, d = 0) => (Number.isFinite(v) ? v.toFixed(d) : "---");

/* ---------------------------------------------------------------------------------------------- */
export function AttitudeIndicator({ pitch, roll, size = 210, fpv = null }) {
  const r = size / 2;
  const ppd = size / 50; // pixels per degree of pitch (±25 deg visible)
  const ladder = [];
  for (let p = -30; p <= 30; p += 2.5) {
    if (p === 0) continue;
    const major = p % 10 === 0;
    const mid = p % 5 === 0;
    const w = major ? 46 : mid ? 26 : 12;
    const y = -p * ppd;
    ladder.push(
      <g key={p}>
        <line x1={-w / 2} x2={w / 2} y1={y} y2={y} stroke={INK} strokeWidth={major ? 1.2 : 0.9} opacity={major ? 0.95 : 0.7} />
        {major && (
          <>
            <text x={-w / 2 - 6} y={y + 3.5} fill={INK} fontSize="9" textAnchor="end" className="readout">{Math.abs(p)}</text>
            <text x={w / 2 + 6} y={y + 3.5} fill={INK} fontSize="9" className="readout">{Math.abs(p)}</text>
          </>
        )}
      </g>,
    );
  }
  const rollTicks = [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60];
  const rollR = r - 8;
  return (
    <svg width={size} height={size} viewBox={`${-r} ${-r} ${size} ${size}`} role="img" aria-label={`Attitude: pitch ${fmt(pitch, 1)} degrees, roll ${fmt(roll, 1)} degrees`}>
      <defs>
        <clipPath id="adi-clip">
          <circle r={r - 2} />
        </clipPath>
        <linearGradient id="adi-sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#0d2a47" />
          <stop offset="1" stopColor="#2b5c86" />
        </linearGradient>
        <linearGradient id="adi-gnd" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#5a3d22" />
          <stop offset="1" stopColor="#2a1c10" />
        </linearGradient>
      </defs>
      <g clipPath="url(#adi-clip)">
        <g transform={`rotate(${-roll}) translate(0 ${pitch * ppd})`}>
          <rect x={-size} y={-size * 2} width={size * 2} height={size * 2} fill="url(#adi-sky)" />
          <rect x={-size} y={0} width={size * 2} height={size * 2} fill="url(#adi-gnd)" />
          <line x1={-size} x2={size} y1={0} y2={0} stroke={INK} strokeWidth="1.4" />
          {ladder}
        </g>
        {fpv && (
          <g transform={`translate(${fpv.x * ppd} ${-fpv.y * ppd})`} stroke={HUD} strokeWidth="1.4" fill="none">
            <circle r="5" />
            <line x1="-13" x2="-5" y1="0" y2="0" />
            <line x1="5" x2="13" y1="0" y2="0" />
            <line x1="0" x2="0" y1="-5" y2="-10" />
          </g>
        )}
      </g>
      {/* Roll scale */}
      <g>
        {rollTicks.map((a) => {
          const big = a % 30 === 0;
          const rad = ((a - 90) * Math.PI) / 180;
          const r1 = rollR;
          const r2 = rollR - (big ? 10 : 6);
          return (
            <line key={a} x1={Math.cos(rad) * r1} y1={Math.sin(rad) * r1} x2={Math.cos(rad) * r2} y2={Math.sin(rad) * r2} stroke={INK} strokeWidth={big ? 1.4 : 1} />
          );
        })}
        <path d={`M0 ${-rollR + 1} l-5 -8 h10 z`} fill={INK} />
        <g transform={`rotate(${-roll})`}>
          <path d={`M0 ${-rollR + 12} l-6 9 h12 z`} fill="none" stroke={CAUTION} strokeWidth="1.4" />
        </g>
      </g>
      {/* Fixed aircraft reference */}
      <g stroke={CAUTION} strokeWidth="3" fill="none" strokeLinecap="square">
        <path d="M-58 0 h28 l8 8" />
        <path d="M58 0 h-28 l-8 8" />
        <rect x="-2" y="-2" width="4" height="4" fill={CAUTION} stroke="none" />
      </g>
      <circle r={r - 1.5} fill="none" stroke="rgb(148 175 196 / 0.35)" />
    </svg>
  );
}

/* ---------------------------------------------------------------------------------------------- */
export function VerticalTape({ value, label, unit, height = 210, width = 64, pxPerUnit, step, labelEvery, side = "left", bugs = [], decimals = 0, tone = INK }) {
  const h2 = height / 2;
  const ticks = [];
  const range = h2 / pxPerUnit;
  const start = Math.floor((value - range) / step) * step;
  for (let v = start; v <= value + range; v += step) {
    const y = -(v - value) * pxPerUnit;
    // Suppress labels hidden behind the current-value box.
    const isLabel = Math.abs(v / labelEvery - Math.round(v / labelEvery)) < 1e-6 && Math.abs(y) > 15;
    const tickX = side === "left" ? width / 2 : -width / 2;
    const dir = side === "left" ? -1 : 1;
    ticks.push(
      <g key={v.toFixed(3)}>
        <line x1={tickX} x2={tickX + dir * (isLabel ? 10 : 5)} y1={y} y2={y} stroke={INK} strokeWidth="1" opacity="0.8" />
        {isLabel && (
          <text x={tickX + dir * 14} y={y + 3.5} fill={INK} fontSize="10" textAnchor={side === "left" ? "end" : "start"} className="readout" opacity="0.85">
            {Math.round(v)}
          </text>
        )}
      </g>,
    );
  }
  return (
    <svg width={width} height={height + 28} viewBox={`${-width / 2} ${-h2 - 18} ${width} ${height + 28}`} role="img" aria-label={`${label} ${fmt(value, decimals)} ${unit}`}>
      <defs>
        <clipPath id={`tape-${label}`}>
          <rect x={-width / 2} y={-h2} width={width} height={height} />
        </clipPath>
      </defs>
      <text x="0" y={-h2 - 6} fill={DIM} fontSize="9" textAnchor="middle" className="label" style={{ fontSize: 9 }}>
        {label} <tspan fill={DIM}>{unit}</tspan>
      </text>
      <rect x={-width / 2} y={-h2} width={width} height={height} fill="rgb(4 6 9 / 0.55)" stroke="rgb(148 175 196 / 0.2)" />
      <g clipPath={`url(#tape-${label})`}>
        {ticks}
        {bugs.map((b, i) => {
          const y = -(b.value - value) * pxPerUnit;
          return <rect key={i} x={side === "left" ? width / 2 - 4 : -width / 2} y={y - 1} width="4" height="2" fill={b.color} />;
        })}
      </g>
      <g>
        <path
          d={side === "left"
            ? `M${-width / 2 + 2} -11 h${width - 14} l8 11 l-8 11 h${-(width - 14)} z`
            : `M${width / 2 - 2} -11 h${-(width - 14)} l-8 11 l8 11 h${width - 14} z`}
          fill="#040609"
          stroke={tone}
          strokeWidth="1.2"
        />
        <text x={side === "left" ? 2 : -2} y="4.5" fill={tone} fontSize="13" fontWeight="700" textAnchor="middle" className="readout">
          {fmt(value, decimals)}
        </text>
      </g>
    </svg>
  );
}

/* ---------------------------------------------------------------------------------------------- */
export function VerticalSpeed({ vz, height = 210, crash = -1.5 }) {
  const h2 = height / 2;
  const scale = (v) => -clamp(v, -6, 4) * (h2 / 6);
  const tone = vz < crash ? WARNING : vz < crash * 0.66 ? CAUTION : HUD;
  const marks = [-6, -4, -2, -1, 0, 1, 2, 4];
  return (
    <svg width="48" height={height + 28} viewBox={`-24 ${-h2 - 18} 48 ${height + 28}`} role="img" aria-label={`Vertical speed ${fmt(vz, 2)} metres per second`}>
      <text x="0" y={-h2 - 6} fill={DIM} fontSize="9" textAnchor="middle" className="label" style={{ fontSize: 9 }}>VSI m/s</text>
      <rect x="-6" y={-h2} width="12" height={height} fill="rgb(4 6 9 / 0.55)" stroke="rgb(148 175 196 / 0.2)" />
      {marks.map((m) => (
        <g key={m}>
          <line x1="-6" x2="-11" y1={scale(m)} y2={scale(m)} stroke={INK} opacity="0.7" />
          <text x="-13" y={scale(m) + 3} fill={INK} opacity="0.75" fontSize="8.5" textAnchor="end" className="readout">{m}</text>
        </g>
      ))}
      <line x1="-6" x2="10" y1={scale(crash)} y2={scale(crash)} stroke={WARNING} strokeWidth="1.4" strokeDasharray="3 2" />
      <rect x="-5" y={Math.min(0, scale(vz))} width="10" height={Math.abs(scale(vz))} fill={tone} opacity="0.85" />
      <text x="10" y={h2 + 8} fill={tone} fontSize="11" fontWeight="700" textAnchor="end" className="readout">{fmt(vz, 2)}</text>
    </svg>
  );
}

/* ---------------------------------------------------------------------------------------------- */
export function AoAGauge({ alpha, warn = 12, stall = 16 }) {
  const r = 34;
  const ang = (a) => (-200 + (clamp(a, -4, 20) + 4) * (220 / 24)) * (Math.PI / 180);
  const pt = (a, rr) => [Math.cos(ang(a)) * rr, Math.sin(ang(a)) * rr];
  const arc = (a0, a1, rr) => {
    const [x0, y0] = pt(a0, rr);
    const [x1, y1] = pt(a1, rr);
    return `M${x0} ${y0} A${rr} ${rr} 0 0 1 ${x1} ${y1}`;
  };
  const tone = alpha >= stall ? WARNING : alpha >= warn ? CAUTION : INK;
  const [nx, ny] = pt(alpha, r - 4);
  return (
    <svg width="96" height="70" viewBox="-48 -42 96 70" role="img" aria-label={`Angle of attack ${fmt(alpha, 1)} degrees`}>
      <path d={arc(-4, warn, r)} stroke="rgb(148 175 196 / 0.45)" strokeWidth="3" fill="none" />
      <path d={arc(warn, stall, r)} stroke={CAUTION} strokeWidth="3" fill="none" />
      <path d={arc(stall, 20, r)} stroke={WARNING} strokeWidth="3" fill="none" />
      <line x1="0" y1="0" x2={nx} y2={ny} stroke={tone} strokeWidth="2" />
      <circle r="2.5" fill={tone} />
      <text x="0" y="22" fill={tone} fontSize="12" fontWeight="700" textAnchor="middle" className="readout">{fmt(alpha, 1)}°</text>
      <text x="-44" y="22" fill={DIM} fontSize="9" className="readout">AoA</text>
    </svg>
  );
}

export function SlipIndicator({ beta, width = 150 }) {
  const w2 = width / 2;
  const x = (clamp(beta, -12, 12) / 12) * (w2 - 10);
  const tone = Math.abs(beta) > 8 ? CAUTION : DATA;
  return (
    <svg width={width} height="34" viewBox={`${-w2} -12 ${width} 34`} role="img" aria-label={`Sideslip ${fmt(beta, 1)} degrees`}>
      <line x1={-w2 + 6} x2={w2 - 6} y1="0" y2="0" stroke="rgb(148 175 196 / 0.4)" />
      {[-10, -5, 0, 5, 10].map((b) => (
        <line key={b} x1={(b / 12) * (w2 - 10)} x2={(b / 12) * (w2 - 10)} y1="-5" y2="5" stroke={INK} opacity={b === 0 ? 0.9 : 0.5} />
      ))}
      <path d={`M${x} -2 l-6 -8 h12 z`} fill={tone} />
      <text x="0" y="18" fill={tone} fontSize="11" fontWeight="700" textAnchor="middle" className="readout">β {fmt(beta, 1)}°</text>
    </svg>
  );
}

export function HeadingTape({ heading, track, width = 300 }) {
  const w2 = width / 2;
  const ppd = 4;
  const ticks = [];
  const start = Math.floor((heading - w2 / ppd) / 5) * 5;
  for (let d = start; d <= heading + w2 / ppd; d += 5) {
    const x = (d - heading) * ppd;
    const hdg = ((d % 360) + 360) % 360;
    const label = hdg % 10 === 0;
    ticks.push(
      <g key={d}>
        <line x1={x} x2={x} y1="0" y2={label ? 8 : 4} stroke={INK} opacity="0.75" />
        {label && <text x={x} y="19" fill={INK} opacity="0.85" fontSize="9" textAnchor="middle" className="readout">{String(hdg).padStart(3, "0")}</text>}
      </g>,
    );
  }
  const tx = clamp((track - heading) * ppd, -w2 + 6, w2 - 6);
  return (
    <svg width={width} height="34" viewBox={`${-w2} -6 ${width} 34`} role="img" aria-label={`Heading ${fmt(heading)} degrees, track ${fmt(track)} degrees`}>
      <clipPath id="hdg-clip"><rect x={-w2} y="-6" width={width} height="34" /></clipPath>
      <g clipPath="url(#hdg-clip)">{ticks}</g>
      <path d={`M${tx} 0 l-4 -6 h8 z`} fill={HUD} />
      <rect x="-19" y="-6" width="38" height="13" fill="#040609" stroke={INK} />
      <text x="0" y="4.5" fill={INK} fontSize="10" fontWeight="700" textAnchor="middle" className="readout">{String(Math.round(((heading % 360) + 360) % 360)).padStart(3, "0")}</text>
    </svg>
  );
}

/* ---------------------------------------------------------------------------------------------- */
/* Full-screen head-up display for the cockpit camera (conformal with a 62 deg vertical FOV).      */
/* ---------------------------------------------------------------------------------------------- */
export function CockpitHUD({ s, width, height, vfovDeg = 62, glidepathDeg = 3, crashSink = -1.5 }) {
  if (!s) return null;
  const cx = width / 2;
  const cy = height / 2;
  const f = cy / Math.tan(((vfovDeg / 2) * Math.PI) / 180);
  const px = (deg) => f * Math.tan((deg * Math.PI) / 180);
  const [rollDeg, pitchDeg, hdgDeg] = s.euler;
  const gamma = (Math.atan2(s.vz, Math.max(s.gs, 1)) * 180) / Math.PI;
  let drift = s.track - hdgDeg;
  if (drift > 180) drift -= 360;
  if (drift < -180) drift += 360;
  const fpvX = px(drift);
  const fpvY = -px(gamma - pitchDeg);
  const ias = s.ias * 1.943844;
  const altFt = s.h * 3.28084;
  const ladder = [];
  for (let p = -20; p <= 20; p += 5) {
    const y = -px(p - pitchDeg) + 0;
    if (p === 0) {
      ladder.push(<line key="hz" x1={-width} x2={width} y1={y} y2={y} stroke={HUD} strokeWidth="1.5" />);
      continue;
    }
    const neg = p < 0;
    ladder.push(
      <g key={p}>
        <path d={`M-120 ${y} h80 ${neg ? "" : "v8"}`} stroke={HUD} strokeWidth="1.3" fill="none" strokeDasharray={neg ? "10 6" : undefined} />
        <path d={`M120 ${y} h-80 ${neg ? "" : "v8"}`} stroke={HUD} strokeWidth="1.3" fill="none" strokeDasharray={neg ? "10 6" : undefined} />
        <text x="-128" y={y + 4} fill={HUD} fontSize="12" textAnchor="end" className="readout">{Math.abs(p)}</text>
        <text x="128" y={y + 4} fill={HUD} fontSize="12" className="readout">{Math.abs(p)}</text>
      </g>,
    );
  }
  const gpLine = -px(-glidepathDeg - pitchDeg);
  return (
    <svg className="pointer-events-none absolute inset-0 hud-glow" width={width} height={height} aria-hidden="true">
      <g transform={`translate(${cx} ${cy})`} fontFamily="JetBrains Mono, monospace">
        <g transform={`rotate(${-rollDeg})`}>
          {ladder}
          <line x1="-60" x2="60" y1={gpLine} y2={gpLine} stroke={HUD} strokeWidth="1" strokeDasharray="2 5" opacity="0.8" />
          <text x="66" y={gpLine + 4} fill={HUD} fontSize="11" className="readout">-{glidepathDeg.toFixed(1)}</text>
        </g>
        {/* Boresight */}
        <path d="M-18 0 h10 l4 6 l4 -6 l4 6 l4 -6 h10" stroke={HUD} strokeWidth="1.5" fill="none" />
        {/* Flight path vector */}
        <g transform={`translate(${fpvX} ${fpvY})`} stroke={HUD} strokeWidth="1.8" fill="none">
          <circle r="8" />
          <line x1="-22" x2="-8" y1="0" y2="0" />
          <line x1="8" x2="22" y1="0" y2="0" />
          <line x1="0" x2="0" y1="-8" y2="-16" />
        </g>
        {/* Roll scale */}
        <g transform={`translate(0 ${-cy + 70})`}>
          {[-45, -30, -20, -10, 0, 10, 20, 30, 45].map((a) => {
            const r = 160;
            const rad = ((a + 90) * Math.PI) / 180;
            return <line key={a} x1={-Math.cos(rad) * r} y1={Math.sin(rad) * r - r} x2={-Math.cos(rad) * (r - (a % 30 === 0 ? 14 : 8))} y2={Math.sin(rad) * (r - (a % 30 === 0 ? 14 : 8)) - r} stroke={HUD} strokeWidth="1.4" />;
          })}
          <g transform={`rotate(${-rollDeg} 0 160)`}>
            <path d="M0 18 l-7 11 h14 z" fill="none" stroke={HUD} strokeWidth="1.5" />
          </g>
        </g>
        {/* Airspeed / altitude boxes */}
        <g transform={`translate(${-Math.min(cx - 90, 360)} 0)`}>
          <rect x="-46" y="-16" width="92" height="32" fill="rgb(0 0 0 / 0.25)" stroke={HUD} strokeWidth="1.4" />
          <text x="0" y="7" fill={HUD} fontSize="20" fontWeight="700" textAnchor="middle" className="readout">{ias.toFixed(0)}</text>
          <text x="0" y="34" fill={HUD} fontSize="11" textAnchor="middle">IAS KT</text>
        </g>
        <g transform={`translate(${Math.min(cx - 90, 360)} 0)`}>
          <rect x="-56" y="-16" width="112" height="32" fill="rgb(0 0 0 / 0.25)" stroke={HUD} strokeWidth="1.4" />
          <text x="0" y="7" fill={HUD} fontSize="20" fontWeight="700" textAnchor="middle" className="readout">{altFt.toFixed(0)}</text>
          <text x="0" y="34" fill={HUD} fontSize="11" textAnchor="middle">RA FT</text>
          <text x="0" y="56" fill={s.vz < crashSink ? WARNING : HUD} fontSize="13" textAnchor="middle" className="readout">VS {s.vz.toFixed(1)}</text>
        </g>
        <g transform={`translate(0 ${cy - 80})`}>
          <text x="0" y="0" fill={HUD} fontSize="14" textAnchor="middle" className="readout">
            HDG {String(Math.round(((hdgDeg % 360) + 360) % 360)).padStart(3, "0")}   TRK {String(Math.round(((s.track % 360) + 360) % 360)).padStart(3, "0")}
          </text>
          <text x="0" y="20" fill={HUD} fontSize="12" textAnchor="middle" className="readout">
            AOA {s.alpha.toFixed(1)}   β {s.beta.toFixed(1)}   LOC {s.y >= 0 ? "R" : "L"}{Math.abs(s.y).toFixed(1)}M
          </text>
        </g>
      </g>
    </svg>
  );
}
