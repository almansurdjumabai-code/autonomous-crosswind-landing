"""Evaluation sweeps and publication-quality figures.

Loads a trained PPO checkpoint (with its VecNormalize statistics) or the
classical baseline autopilot, flies Monte-Carlo landings across a sweep of
crosswind speeds, writes per-episode metrics (CSV + LaTeX table), optional
telemetry JSON, and produces IEEE-formatted vector figures:

* ``reward_convergence``      training-episode return (rolling mean +/- std) and
                              deterministic evaluation return vs. timesteps
* ``trajectories_3d``         3-D approach trajectories coloured by crosswind
* ``trajectory_profiles``     vertical profile vs. glideslope and lateral track
* ``touchdown_dispersion``    touchdown points with 2-sigma ellipses on the runway
* ``control_effort``          histograms of surface deflections and throttle
* ``sink_rate_vs_crosswind``  touchdown vertical speed distribution per crosswind

Usage::

    python -m src.viz.evaluate_and_plot --model runs/ppo_crosswind/best/best_model.zip \
        --run-dir runs/ppo_crosswind --crosswinds 0 2.5 5 7.5 10 --episodes 20 --out-dir results/ppo
    python -m src.viz.evaluate_and_plot --policy baseline --out-dir results/baseline
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

# Allow running as a plain script (python src/.../file.py) as well as with -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from src.agents.baseline_autopilot import ClassicalAutopilot  # noqa: E402
from src.agents.reward_shaping import CRASH_SINK_RATE_MPS  # noqa: E402
from src.env.aircraft_env import CrosswindLandingEnv, EnvConfig  # noqa: E402
from src.utils.telemetry import TelemetryRecorder, export_json, trajectory_arrays  # noqa: E402

# ----------------------------------------------------------------------------- style
IEEE_COLUMN_IN = 3.5
IEEE_TEXT_IN = 7.16
SERIES_1 = "#2a78d6"  # categorical slot 1 (blue)
SERIES_2 = "#eb6834"  # categorical slot 2 (orange)
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d8d4"
RUNWAY_FILL = "#e7e6e2"
CRITICAL = "#e34948"
# Sequential single-hue ramp (light -> dark blue) for crosswind magnitude.
CROSSWIND_CMAP = LinearSegmentedColormap.from_list("xwind", ["#9cc3ef", "#2a78d6", "#0f3b70"])


def apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,  # embed TrueType (IEEE PDF eXpress requirement)
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.dpi": 600,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, name: str, formats: tuple[str, ...]) -> list[Path]:
    paths = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- policies
class Policy(Protocol):
    name: str

    def reset(self, env: CrosswindLandingEnv) -> None: ...

    def act(self, obs: np.ndarray) -> np.ndarray: ...


class BaselinePolicy:
    name = "baseline"
    action_mode = "absolute"
    residual_scale = 0.3

    def __init__(self) -> None:
        self._ap: Optional[ClassicalAutopilot] = None

    def reset(self, env: CrosswindLandingEnv) -> None:
        self._ap = ClassicalAutopilot(env)

    def act(self, obs: np.ndarray) -> np.ndarray:
        assert self._ap is not None
        return self._ap.act()


class PPOPolicy:
    """Deterministic SB3 PPO policy with frozen VecNormalize observation statistics."""

    name = "ppo"

    def __init__(self, model_path: str, vecnormalize_path: Optional[str], device: str = "cpu") -> None:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from src.agents.train_ppo import find_vecnormalize, load_ppo, run_action_mode

        self.model = load_ppo(model_path, device=device)
        self.action_mode, self.residual_scale = run_action_mode(model_path)
        stats = Path(vecnormalize_path) if vecnormalize_path else find_vecnormalize(model_path)
        self.vecnorm = VecNormalize.load(str(stats), DummyVecEnv([CrosswindLandingEnv]))
        self.vecnorm.training = False
        self.vecnorm.norm_reward = False

    def reset(self, env: CrosswindLandingEnv) -> None:
        return None

    def act(self, obs: np.ndarray) -> np.ndarray:
        norm = self.vecnorm.normalize_obs(obs[None, :].astype(np.float32))
        action, _ = self.model.predict(norm, deterministic=True)
        return np.asarray(action[0], dtype=np.float32)


# ----------------------------------------------------------------------------- evaluation
@dataclass
class EpisodeResult:
    crosswind_mps: float
    seed: int
    metrics: dict
    arrays: dict[str, np.ndarray]


def run_episode(policy: Policy, crosswind: float, seed: int, telemetry_dir: Optional[Path]) -> EpisodeResult:
    cfg = EnvConfig(action_mode=policy.action_mode, residual_scale=policy.residual_scale)
    env = TelemetryRecorder(CrosswindLandingEnv(cfg), extra_metadata={"policy": policy.name, "action_mode": policy.action_mode})
    obs, _ = env.reset(seed=seed, options={"wind_speed_mps": abs(crosswind), "wind_from_deg": 90.0 if crosswind >= 0 else -90.0})
    policy.reset(env.unwrapped)
    while True:
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        if terminated or truncated:
            break
    doc = env.last_episode
    assert doc is not None
    if telemetry_dir is not None:
        export_json(doc, telemetry_dir / f"{policy.name}_xw{crosswind:05.1f}_seed{seed:04d}.json")
    return EpisodeResult(crosswind, seed, dict(info["episode_metrics"]), trajectory_arrays(doc))


def evaluate(
    policy: Policy, crosswinds: list[float], episodes: int, base_seed: int, telemetry_dir: Optional[Path]
) -> list[EpisodeResult]:
    results = []
    for i, xw in enumerate(crosswinds):
        for k in range(episodes):
            seed = base_seed + 1000 * i + k
            r = run_episode(policy, xw, seed, telemetry_dir)
            results.append(r)
            m = r.metrics
            print(
                f"[{policy.name}] xw={xw:5.1f} m/s seed={seed}: "
                f"{'SUCCESS' if m['success'] else (m['crash_reason'] or 'timeout'):>22s} "
                f"sink={m['touchdown_sink_rate_mps']:6.2f} x_td={m['touchdown_x_m']:7.1f} y_td={m['touchdown_y_m']:6.2f}"
            )
    return results


def metrics_frame(results: list[EpisodeResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"crosswind_mps": r.crosswind_mps, "seed": r.seed}
        row.update(r.metrics)
        a = r.arrays["actions"]
        row["mean_abs_action"] = float(np.nanmean(np.abs(a))) if a.size else float("nan")
        row["control_rate_rms"] = float(np.sqrt(np.nanmean(np.diff(a, axis=0) ** 2))) if len(a) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("crosswind_mps")
    return pd.DataFrame(
        {
            "episodes": g.size(),
            "success_rate": g["success"].mean(),
            "crash_rate": g["crashed"].mean(),
            "sink_mean_mps": g["touchdown_sink_rate_mps"].mean(),
            "sink_std_mps": g["touchdown_sink_rate_mps"].std(),
            "x_td_mean_m": g["touchdown_x_m"].mean(),
            "x_td_std_m": g["touchdown_x_m"].std(),
            "abs_y_td_mean_m": g["touchdown_y_m"].apply(lambda s: s.abs().mean()),
            "abs_crab_mean_deg": g["touchdown_crab_deg"].apply(lambda s: s.abs().mean()),
            "control_rate_rms": g["control_rate_rms"].mean(),
        }
    )


def write_latex_table(summary: pd.DataFrame, path: Path, caption: str) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\begin{tabular}{rrrrrr}",
        r"\hline",
        r"$V_{xw}$ [m/s] & Success & $\dot h_{td}$ [m/s] & $x_{td}$ [m] & $|y_{td}|$ [m] & $|\psi_{crab}|$ [deg] \\",
        r"\hline",
    ]

    def fmt(v: float, nd: int) -> str:
        return "--" if not np.isfinite(v) else f"{v:.{nd}f}"

    for xw, r in summary.iterrows():
        lines.append(
            f"{xw:.1f} & {100 * r.success_rate:.0f}\\% & {fmt(r.sink_mean_mps, 2)} $\\pm$ {fmt(r.sink_std_mps, 2)} & "
            f"{fmt(r.x_td_mean_m, 0)} $\\pm$ {fmt(r.x_td_std_m, 0)} & {fmt(r.abs_y_td_mean_m, 2)} & "
            f"{fmt(r.abs_crab_mean_deg, 1)} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- figures
def plot_reward_convergence(run_dir: Path, out_dir: Path, formats: tuple[str, ...], window: int = 100) -> bool:
    """Training return (from Monitor CSVs) and evaluation return (EvalCallback npz)."""
    monitor_files = sorted(glob.glob(str(run_dir / "monitor" / "*.monitor.csv")))
    eval_file = run_dir / "eval" / "evaluations.npz"
    if not monitor_files and not eval_file.exists():
        print(f"[plot] no training logs under {run_dir}; skipping reward convergence")
        return False

    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_IN, 2.2))
    if monitor_files:
        frames = [pd.read_csv(f, skiprows=1) for f in monitor_files]
        mon = pd.concat(frames, ignore_index=True).sort_values("t")
        steps = mon["l"].cumsum().to_numpy()
        roll = mon["r"].rolling(window, min_periods=max(window // 5, 1))
        mean, std = roll.mean().to_numpy(), roll.std().to_numpy()
        ax.plot(steps, mean, color=SERIES_1, label=f"Training (rolling {window} ep.)")
        ax.fill_between(steps, mean - std, mean + std, color=SERIES_1, alpha=0.18, linewidth=0)
    if eval_file.exists():
        ev = np.load(eval_file)
        ts, res = ev["timesteps"], ev["results"]
        mu, sd = res.mean(axis=1), res.std(axis=1)
        ax.plot(ts, mu, color=SERIES_2, marker="o", markersize=2.5, label="Evaluation (deterministic)")
        ax.fill_between(ts, mu - sd, mu + sd, color=SERIES_2, alpha=0.18, linewidth=0)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.legend(loc="lower right")
    save_figure(fig, out_dir, "reward_convergence", formats)
    return True


def _crosswind_colors(crosswinds: list[float]):
    norm = Normalize(vmin=min(crosswinds), vmax=max(max(crosswinds), min(crosswinds) + 1e-6))
    return norm, (lambda xw: CROSSWIND_CMAP(norm(xw)))


def plot_trajectories_3d(results: list[EpisodeResult], env_cfg: EnvConfig, out_dir: Path, formats: tuple[str, ...]) -> None:
    xws = sorted({r.crosswind_mps for r in results})
    norm, color = _crosswind_colors(xws)
    fig = plt.figure(figsize=(IEEE_TEXT_IN * 0.7, 3.4))
    ax = fig.add_subplot(projection="3d")
    rw = env_cfg.runway
    hw = 0.5 * rw.width_m
    ax.plot([0, rw.length_m, rw.length_m, 0, 0], [-hw, -hw, hw, hw, -hw], [0, 0, 0, 0, 0], color=INK_MUTED, lw=0.8)
    xs = np.linspace(-2200.0, rw.aim_point_m, 50)
    ax.plot(xs, np.zeros_like(xs), (rw.aim_point_m - xs) * math.tan(rw.glideslope_rad), color=INK_MUTED, lw=0.6, ls="--")
    for r in results:
        p = r.arrays["position_ned"]
        ax.plot(p[:, 0], p[:, 1], -p[:, 2], color=color(r.crosswind_mps), lw=0.6, alpha=0.85)
        if r.metrics["touched_down"]:
            ax.scatter([r.metrics["touchdown_x_m"]], [r.metrics["touchdown_y_m"]], [0.0], s=4, color=color(r.crosswind_mps))
    ax.set_xlabel("Along-track $x$ [m]", labelpad=4)
    ax.set_ylabel("Cross-track $y$ [m]", labelpad=8)
    ax.set_zlabel("Height $h$ [m]", labelpad=2)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.view_init(elev=22, azim=-60)
    ax.set_box_aspect((3.0, 1.0, 0.8))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CROSSWIND_CMAP)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.45, pad=0.0, aspect=30)
    cb.set_label("Crosswind at 20 ft [m/s]")
    save_figure(fig, out_dir, "trajectories_3d", formats)


def plot_trajectory_profiles(results: list[EpisodeResult], env_cfg: EnvConfig, out_dir: Path, formats: tuple[str, ...]) -> None:
    xws = sorted({r.crosswind_mps for r in results})
    norm, color = _crosswind_colors(xws)
    rw = env_cfg.runway
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(IEEE_COLUMN_IN, 3.4), sharex=True)
    xs = np.linspace(-2300.0, rw.aim_point_m, 50)
    ax1.plot(xs, (rw.aim_point_m - xs) * math.tan(rw.glideslope_rad), color=INK, lw=0.8, ls="--", label="3° glideslope")
    for r in results:
        p = r.arrays["position_ned"]
        ax1.plot(p[:, 0], -p[:, 2], color=color(r.crosswind_mps), lw=0.5, alpha=0.8)
        ax2.plot(p[:, 0], p[:, 1], color=color(r.crosswind_mps), lw=0.5, alpha=0.8)
    ax1.set_ylabel("Height $h$ [m]")
    ax1.legend(loc="upper right")
    hw = 0.5 * rw.width_m
    ax2.add_patch(Rectangle((0, -hw), rw.length_m, rw.width_m, color=RUNWAY_FILL, zorder=0, lw=0))
    ax2.axhline(0.0, color=INK_MUTED, lw=0.5, ls="--")
    ax2.set_ylabel("Cross-track $y$ [m]")
    ax2.set_xlabel("Along-track distance from threshold $x$ [m]")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CROSSWIND_CMAP)
    cb = fig.colorbar(sm, ax=(ax1, ax2), shrink=0.8, pad=0.02)
    cb.set_label("Crosswind [m/s]")
    save_figure(fig, out_dir, "trajectory_profiles", formats)


def _cov_ellipse(x: np.ndarray, y: np.ndarray, n_sigma: float, **kw) -> Optional[Ellipse]:
    if len(x) < 3:
        return None
    cov = np.cov(np.vstack([x, y]))
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    angle = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    return Ellipse(
        (float(x.mean()), float(y.mean())),
        2 * n_sigma * math.sqrt(vals[1]),
        2 * n_sigma * math.sqrt(vals[0]),
        angle=angle,
        fill=False,
        **kw,
    )


def plot_touchdown_dispersion(df: pd.DataFrame, env_cfg: EnvConfig, out_dir: Path, formats: tuple[str, ...]) -> None:
    rw = env_cfg.runway
    hw = 0.5 * rw.width_m
    td = df[np.isfinite(df["touchdown_x_m"])]
    xws = sorted(df["crosswind_mps"].unique())
    norm, color = _crosswind_colors(xws)
    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_IN, 2.0))
    ax.add_patch(Rectangle((0, -hw), rw.length_m, rw.width_m, color=RUNWAY_FILL, zorder=0, lw=0))
    ax.axhline(0.0, color=INK_MUTED, lw=0.5, ls="--", zorder=1)
    ax.axvline(rw.aim_point_m, color=INK_MUTED, lw=0.5, ls=":", zorder=1)
    ax.text(rw.aim_point_m, hw + 1.0, "aim point", ha="center", va="bottom", fontsize=6, color=INK_MUTED)
    for xw in xws:
        sub = td[td["crosswind_mps"] == xw]
        if sub.empty:
            continue
        crashed = sub["crashed"] > 0.5
        c = color(xw)
        ax.scatter(sub.loc[~crashed, "touchdown_x_m"], sub.loc[~crashed, "touchdown_y_m"], s=9, color=c,
                   edgecolors="white", linewidths=0.4, label=f"{xw:g} m/s", zorder=3)
        ax.scatter(sub.loc[crashed, "touchdown_x_m"], sub.loc[crashed, "touchdown_y_m"], s=12, marker="x",
                   color=c, linewidths=0.8, zorder=3)
        e = _cov_ellipse(sub["touchdown_x_m"].to_numpy(), sub["touchdown_y_m"].to_numpy(), 2.0, edgecolor=c, lw=0.8)
        if e is not None:
            ax.add_patch(e)
    if td.empty:
        ax.set_xlim(0, rw.length_m)
    else:
        lo = min(0.0, td["touchdown_x_m"].min() - 30.0)
        hi = max(2 * rw.aim_point_m, td["touchdown_x_m"].max() + 30.0)
        ax.set_xlim(lo, hi)
    ax.set_ylim(-hw * 1.3, hw * 1.3)
    ax.set_xlabel("Touchdown distance from threshold $x_{td}$ [m]")
    ax.set_ylabel("Lateral offset $y_{td}$ [m]")
    ax.legend(title="Crosswind (× = crash)", ncol=3, loc="lower right", fontsize=6, title_fontsize=6)
    save_figure(fig, out_dir, "touchdown_dispersion", formats)


def plot_control_effort(results: list[EpisodeResult], env_cfg: EnvConfig, out_dir: Path, formats: tuple[str, ...]) -> None:
    xws = sorted({r.crosswind_mps for r in results})
    _, color = _crosswind_colors(xws)
    ac = env_cfg.aircraft
    labels = ("Elevator $\\delta_e$ [deg]", "Aileron $\\delta_a$ [deg]", "Rudder $\\delta_r$ [deg]", "Throttle $\\delta_t$ [-]")
    ranges = (
        math.degrees(ac.elevator_limit_rad),
        math.degrees(ac.aileron_limit_rad),
        math.degrees(ac.rudder_limit_rad),
    )
    fig, axes = plt.subplots(2, 2, figsize=(IEEE_COLUMN_IN, 2.8))
    for j, ax in enumerate(axes.flat):
        bins = np.linspace(0.0, 1.0, 31) if j == 3 else np.linspace(-ranges[j], ranges[j], 31)
        for xw in xws:
            data = np.concatenate([r.arrays["controls"][:, j] for r in results if r.crosswind_mps == xw])
            data = data[np.isfinite(data)]
            ax.hist(data, bins=bins, density=True, histtype="step", color=color(xw), lw=0.9, label=f"{xw:g} m/s")
        ax.set_xlabel(labels[j])
        ax.set_yticks([])
        ax.grid(False)
    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density")
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=len(xws), bbox_to_anchor=(0.5, 1.04), fontsize=6)
    fig.tight_layout()
    save_figure(fig, out_dir, "control_effort", formats)


def plot_sink_rate(df: pd.DataFrame, out_dir: Path, formats: tuple[str, ...]) -> None:
    xws = sorted(df["crosswind_mps"].unique())
    data = [df.loc[df["crosswind_mps"] == xw, "touchdown_sink_rate_mps"].dropna().to_numpy() for xw in xws]
    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_IN, 1.9))
    keep = [i for i, d in enumerate(data) if d.size]
    if keep:
        ax.boxplot([data[i] for i in keep], positions=[xws[i] for i in keep], widths=0.6 * (np.diff(xws).min() if len(xws) > 1 else 1.0),
                   patch_artist=True, boxprops=dict(facecolor="#cfe0f6", edgecolor=SERIES_1, lw=0.7),
                   medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(color=SERIES_1, lw=0.7),
                   capprops=dict(color=SERIES_1, lw=0.7), flierprops=dict(marker=".", markersize=3, markeredgecolor=SERIES_1))
    ax.axhline(CRASH_SINK_RATE_MPS, color=CRITICAL, lw=0.8, ls="--")
    ax.text(xws[0], CRASH_SINK_RATE_MPS, " crash limit", va="bottom", ha="left", fontsize=6, color=INK_MUTED)
    ax.set_xticks(xws)
    ax.set_xticklabels([f"{x:g}" for x in xws])
    ax.set_xlabel("Crosswind at 20 ft [m/s]")
    ax.set_ylabel("Touchdown $\\dot h$ [m/s]")
    save_figure(fig, out_dir, "sink_rate_vs_crosswind", formats)


# ----------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a landing policy and generate paper figures.")
    p.add_argument("--model", help="Path to a PPO .zip checkpoint")
    p.add_argument("--vecnormalize", help="Path to VecNormalize .pkl (auto-detected if omitted)")
    p.add_argument("--policy", choices=("ppo", "baseline"), default=None)
    p.add_argument("--run-dir", help="Training run directory (for reward convergence curves)")
    p.add_argument("--crosswinds", type=float, nargs="+", default=[0.0, 2.5, 5.0, 7.5, 10.0])
    p.add_argument("--episodes", type=int, default=20, help="episodes per crosswind level")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--telemetry", action="store_true", help="export a telemetry JSON per episode")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)

    kind = a.policy or ("ppo" if a.model else "baseline")
    if kind == "ppo" and not a.model:
        p.error("--model is required for the PPO policy")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(a.formats)
    apply_ieee_style()

    if a.run_dir:
        plot_reward_convergence(Path(a.run_dir), out_dir, formats)

    policy: Policy = PPOPolicy(a.model, a.vecnormalize, a.device) if kind == "ppo" else BaselinePolicy()
    telemetry_dir = out_dir / "telemetry" if a.telemetry else None
    results = evaluate(policy, a.crosswinds, a.episodes, a.seed, telemetry_dir)

    df = metrics_frame(results)
    df.to_csv(out_dir / f"episodes_{policy.name}.csv", index=False)
    summary = summary_table(df)
    summary.to_csv(out_dir / f"summary_{policy.name}.csv")
    write_latex_table(summary, out_dir / f"summary_{policy.name}.tex",
                      f"Landing performance of the {policy.name.upper()} controller vs. crosswind.")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.round(3))

    env_cfg = EnvConfig()
    plot_trajectories_3d(results, env_cfg, out_dir, formats)
    plot_trajectory_profiles(results, env_cfg, out_dir, formats)
    plot_touchdown_dispersion(df, env_cfg, out_dir, formats)
    plot_control_effort(results, env_cfg, out_dir, formats)
    plot_sink_rate(df, out_dir, formats)
    print(f"Figures and tables written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
