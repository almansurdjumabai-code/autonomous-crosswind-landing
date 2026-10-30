import { PanelLeft, PanelRight } from "lucide-react";
import { ui, useClock, useUI } from "../lib/store.js";

function Glyph() {
  return (
    <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true">
      <rect x="0.5" y="0.5" width="31" height="31" fill="none" stroke="rgb(148 175 196 / 0.35)" />
      <path d="M4 22h24" stroke="#7cffb2" strokeWidth="1.5" />
      <path d="M8 22l8-13 8 13" stroke="#e8eef3" strokeWidth="1.3" fill="none" />
      <path d="M16 9v13" stroke="#e8eef3" strokeWidth="1" strokeDasharray="1.5 2" />
      <circle cx="16" cy="9" r="1.6" fill="#7cffb2" />
    </svg>
  );
}

function SystemStatus() {
  const controller = useUI((s) => s.controller);
  const autopilot = useUI((s) => s.autopilot);
  const loading = useUI((s) => s.loading);
  const error = useUI((s) => s.error);
  const clk = useClock(4);
  const name = { ppo: "PPO MODEL", ppo_raw: "PPO MODEL (UNPROTECTED)", pid: "PID AUTOLAND" }[controller] ?? "PPO MODEL";
  let dot = "bg-hud";
  let text = `LIVE TELEMETRY // ${name}: ACTIVE`;
  if (controller === "ppo" && clk.sample && clk.sample.authority < 0.5) {
    dot = "bg-caution";
    text = "LIVE TELEMETRY // PPO MODEL: ACTIVE // RTA FALLBACK ENGAGED";
  }
  if (error) {
    dot = "bg-warning";
    text = "TELEMETRY LINK FAULT";
  } else if (loading) {
    dot = "bg-caution";
    text = "ACQUIRING TELEMETRY LINK";
  } else if (autopilot === "MANUAL") {
    dot = "bg-caution";
    text = `MANUAL OVERRIDE // ${name}: STANDBY`;
  }
  return (
    <div className="glass flex items-center gap-2.5 px-3 py-1.5" role="status" aria-live="polite">
      <span className={`pulse-dot inline-block h-1.5 w-1.5 ${dot}`} />
      <span className="readout text-[11px] tracking-[0.14em] text-ink">{text}</span>
    </div>
  );
}

export default function TopBar() {
  const showLeft = useUI((s) => s.showLeft);
  const showRight = useUI((s) => s.showRight);
  return (
    <header className="pointer-events-auto absolute inset-x-0 top-0 z-30 flex h-12 items-center gap-3 border-b border-line bg-obsidian-950/85 px-3">
      <button
        type="button"
        onClick={() => ui.set({ showLeft: !showLeft })}
        className="grid h-7 w-7 place-items-center border border-line text-ink-dim hover:border-line-strong hover:text-ink"
        aria-label={showLeft ? "Hide control panel" : "Show control panel"}
        aria-pressed={showLeft}
      >
        <PanelLeft size={14} />
      </button>
      <div className="flex items-center gap-2.5">
        <Glyph />
        <div className="leading-none">
          <div className="readout text-[13px] font-bold tracking-[0.2em] text-ink">
            ALMA-AUTOLAND <span className="text-ink-mute">|</span> <span className="font-medium text-ink-dim">6DoF FLIGHT DYNAMICS</span>
          </div>
          <div className="label mt-1 hidden text-[9px] md:block">Autonomous crosswind landing dynamics &amp; guidance framework</div>
        </div>
      </div>
      <div className="mx-auto hidden lg:block">
        <SystemStatus />
      </div>
      <div className="ml-auto flex items-center gap-2 lg:ml-0">
        <div className="lg:hidden">
          <SystemStatus />
        </div>
        <div className="hidden items-center gap-2 border border-line px-2.5 py-1.5 sm:flex">
          <span className="inline-block h-2.5 w-3.5 bg-[#00afca]" aria-hidden="true" />
          <span className="readout text-[11px] tracking-[0.14em] text-ink-dim">
            CHIEF ARCHITECT: <span className="text-ink">JUMABAY ALMANSUR</span> <span className="text-data">[KZ]</span>
          </span>
        </div>
        <button
          type="button"
          onClick={() => ui.set({ showRight: !showRight })}
          className="grid h-7 w-7 place-items-center border border-line text-ink-dim hover:border-line-strong hover:text-ink"
          aria-label={showRight ? "Hide analytics panel" : "Show analytics panel"}
          aria-pressed={showRight}
        >
          <PanelRight size={14} />
        </button>
      </div>
    </header>
  );
}
