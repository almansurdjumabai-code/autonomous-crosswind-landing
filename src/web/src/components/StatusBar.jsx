import { useClock, useUI } from "../lib/store.js";

/* global __APP_VERSION__, __BUILD_TIME__ */

function Cell({ label, children, tone = "text-ink" }) {
  return (
    <div className="flex shrink-0 items-center gap-2 whitespace-nowrap border-r border-line px-3 last:border-r-0">
      <span className="label text-[9px]">{label}</span>
      <span className={`readout text-[11px] ${tone}`}>{children}</span>
    </div>
  );
}

const clockText = (t) => {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `T+${String(m).padStart(2, "0")}:${s.toFixed(3).padStart(6, "0")}`;
};

const compact = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n));

export default function StatusBar() {
  const clk = useClock(15);
  const ep = useUI((s) => s.episode);
  const manifest = useUI((s) => s.manifest);
  const n = ep?.n ?? 0;
  const passes = clk.render.passes.join("›");
  const lat = clk.latencyMs;
  const ckpt = manifest?.provenance?.ppo_checkpoint;
  const quality = useUI((s) => s.quality);
  const built = __BUILD_TIME__.replace("T", " ").slice(0, 16) + "Z";
  return (
    <footer className="pointer-events-auto absolute inset-x-0 bottom-0 z-30 flex h-7 items-center overflow-x-auto border-t border-line bg-obsidian-950/90 [scrollbar-width:none]">
      <Cell label="Mission clock">{clockText(clk.t)}</Cell>
      <Cell label="Frame">
        {String(clk.index + 1).padStart(4, "0")}/{String(n).padStart(4, "0")} <span className="text-ink-mute">@20 Hz</span>
      </Cell>
      <Cell label="WebGL" tone={clk.fps > 50 ? "text-hud" : clk.fps > 30 ? "text-caution" : "text-warning"}>
        {clk.render.api} {passes} <span className="text-ink-dim">OK</span>
      </Cell>
      <Cell label="FPS" tone={clk.fps > 55 ? "text-hud" : clk.fps > 40 ? "text-caution" : "text-warning"}>
        {clk.fps.toFixed(0)} <span className="text-ink-mute">{quality} · DPR {clk.render.dpr.toFixed(2)}</span>
      </Cell>
      <Cell label="DC / TRI">{clk.render.calls} / {compact(clk.render.triangles)}</Cell>
      <Cell label="Latency" tone={lat < 50 ? "text-hud" : "text-caution"}>
        {lat.toFixed(1)} ms <span className="text-ink-mute">sample age</span> · {clk.frameMs.toFixed(2)} ms <span className="text-ink-mute">pipeline</span>
      </Cell>
      <div className="ml-auto flex shrink-0 items-center">
        <Cell label="Build">
          v{__APP_VERSION__} · {built} · {manifest ? `${manifest.scenarios.length} SCN` : "—"}
          {ckpt ? ` · CKPT ${(ckpt.num_timesteps / 1e6).toFixed(2)}M ${ckpt.sha256_12}` : ""}
        </Cell>
      </div>
    </footer>
  );
}
