"""Figures and summary statistics for the report, from report/data and runs/*/eval.

    python -m src.viz.report_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("report/data")
FIG = Path("report/figures")
COLORS = {"pid": "#d95926", "ppo_rta": "#1f6fc4", "ppo_raw": "#199e70"}
LABELS = {"pid": "Classical autoland", "ppo_rta": "Residual PPO + RTA", "ppo_raw": "Residual PPO, no RTA"}
RECOVERY_BAND_M = 5.0
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "savefig.dpi": 300})


def num(rows, key):
    return np.array([r[key] for r in rows if r[key] is not None and np.isfinite(r[key])], dtype=float)


def sweep_summary(sweep) -> list[dict]:
    out = []
    for c in COLORS:
        for w in sorted({r["wind_mps"] for r in sweep}):
            rows = [r for r in sweep if r["controller"] == c and r["wind_mps"] == w]
            if not rows:
                continue
            ok = [r for r in rows if r["success"]]
            td = [r for r in rows if r["touched_down"]]
            reasons: dict[str, int] = {}
            for r in rows:
                if r["crash_reason"]:
                    reasons[r["crash_reason"]] = reasons.get(r["crash_reason"], 0) + 1
            out.append({
                "controller": c, "wind_mps": w, "n": len(rows), "success": len(ok), "touched": len(td),
                "y_abs_mean": float(np.mean(np.abs(num(td, "y_td_m")))) if td else None,
                "y_abs_p95": float(np.percentile(np.abs(num(td, "y_td_m")), 95)) if td else None,
                "y_rms_mean": float(np.mean(num(rows, "y_rms_final_m"))) if len(num(rows, "y_rms_final_m")) else None,
                "sink_mean": float(np.mean(num(td, "sink_mps"))) if td else None,
                "load_mean": float(np.mean(num(td, "lateral_load_g"))) if td else None,
                "load_p95": float(np.percentile(num(td, "lateral_load_g"), 95)) if td else None,
                "authority": float(np.mean([r["authority_mean"] for r in rows])), "reasons": reasons,
            })
    return out


def gust_summary(gust) -> list[dict]:
    out = []
    for c in COLORS:
        rows = [r for r in gust if r["controller"] == c]
        recov, peak_y, peak_phi, reaction = [], [], [], []
        for r in rows:
            a = np.array(r["rows"])
            t = a[:, 0] - r["t_gust"]
            post = t >= 0
            peak_y.append(float(np.abs(a[post, 4]).max()))
            peak_phi.append(float(np.abs(a[post, 1]).max()))
            # Recovery: time after onset of the last excursion outside the band, while airborne.
            air = post & (a[:, 5] > 2.0)
            outside = np.where(air & (np.abs(a[:, 4]) > RECOVERY_BAND_M))[0]
            recov.append(float(t[outside[-1]]) if len(outside) else 0.0)
            # Reaction: first time bank has moved 2 deg from its value at onset.
            i0 = int(np.argmax(post))
            moved = np.where(post & (np.abs(a[:, 1] - a[i0, 1]) > 2.0))[0]
            reaction.append(float(t[moved[0]]) if len(moved) else float("nan"))
        out.append({"controller": c, "n": len(rows), "success": sum(r["success"] for r in rows),
                    "touched": sum(r["touched_down"] for r in rows),
                    "reasons": {k: sum(r["crash_reason"] == k for r in rows) for k in {r["crash_reason"] for r in rows} if k},
                    "peak_y_median": float(np.median(peak_y)), "peak_y_max": float(np.max(peak_y)),
                    "peak_phi_median": float(np.median(peak_phi)), "recovery_median": float(np.median(recov)),
                    "recovery_p90": float(np.percentile(recov, 90)), "reaction_median": float(np.nanmedian(reaction)),
                    "duration": float(np.mean([r["gust_duration"] for r in rows]))})
    return out


def fig_gust(gust) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.5), sharex=True)
    channels = [(1, "Roll $\\phi$ [deg]"), (2, "Pitch $\\theta$ [deg]"), (3, "Heading error $\\psi$ [deg]"), (4, "Lateral offset $y$ [m]")]
    grid = np.arange(-3.0, 16.0, 0.05)
    for ax, (col, label) in zip(axes.ravel(), channels):
        for c in ("pid", "ppo_rta"):
            curves = []
            for r in (r for r in gust if r["controller"] == c):
                a = np.array(r["rows"])
                t = a[:, 0] - r["t_gust"]
                curves.append(np.interp(grid, t, a[:, col], left=np.nan, right=np.nan))
            q25, q50, q75 = np.nanpercentile(np.array(curves), [25, 50, 75], axis=0)
            ax.fill_between(grid, q25, q75, color=COLORS[c], alpha=0.2, lw=0)
            ax.plot(grid, q50, color=COLORS[c], lw=1.2, label=LABELS[c])
        d = np.mean([r["gust_duration"] for r in gust])
        ax.axvspan(0, d, color="0.5", alpha=0.12, lw=0)
        ax.set_ylabel(label)
    axes[0, 0].legend(loc="best")
    for ax in axes[1]:
        ax.set_xlabel("Time since gust onset [s]")
    fig.tight_layout()
    fig.savefig(FIG / "gust_response.png")


def fig_sweep(summary) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1))
    for c in COLORS:
        rows = [s for s in summary if s["controller"] == c]
        w = [s["wind_mps"] for s in rows]
        axes[0].plot(w, [100 * s["success"] / s["n"] for s in rows], "-o", ms=3, color=COLORS[c], label=LABELS[c])
        have = [s for s in rows if s["y_abs_mean"] is not None]
        axes[1].plot([s["wind_mps"] for s in have], [s["y_abs_mean"] for s in have], "-o", ms=3, color=COLORS[c])
        axes[2].plot([s["wind_mps"] for s in have], [s["load_mean"] for s in have], "-o", ms=3, color=COLORS[c])
    axes[0].set_ylabel("Successful landings [%]")
    axes[0].set_ylim(-3, 103)
    axes[1].set_ylabel("Mean $|y_{td}|$ [m]")
    axes[2].set_ylabel("Mean peak $|n_y|$ [g]")
    for ax in axes:
        ax.set_xlabel("Crosswind $W_{20}$ [m/s]")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "monte_carlo.png")


def fig_training() -> dict:
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    info = {}
    series = {"Absolute PPO (ppo_crosswind)": (["runs/ppo_crosswind"], "#7a7a7a"),
              "Residual PPO (ppo_residual)": (["runs/ppo_residual", "runs/ppo_residual_resume1"], "#1f6fc4")}
    for label, (dirs, color) in series.items():
        ts, med = [], []
        for d in dirs:
            e = np.load(f"{d}/eval/evaluations.npz")
            ts += list(e["timesteps"])
            med += list(np.median(e["results"], axis=1))
        ts, med = np.array(ts), np.array(med)
        # The first residual run also evaluated past its last checkpoint; the resumed run replaces those steps.
        if "Residual" in label:
            first = len(np.load(f"{dirs[0]}/eval/evaluations.npz")["timesteps"])
            keep = np.arange(len(ts)) >= first
            keep |= ts <= 250_000
            ts, med = ts[keep], med[keep]
        order = np.argsort(ts)
        ax.plot(ts[order] / 1e6, med[order], "-o", ms=2, lw=1, color=color, label=label)
        info[label] = {"n": int(len(ts)), "last_step": int(ts.max()), "last_median": float(med[order][-1])}
    ax.set_yscale("symlog", linthresh=200)
    ax.set_xlabel("Environment steps [million]")
    ax.set_ylabel("Median evaluation return")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "training.png")
    return info


def main() -> None:
    sweep = json.loads((DATA / "sweep.json").read_text())
    gust = json.loads((DATA / "gust.json").read_text())
    s = sweep_summary(sweep)
    g = gust_summary(gust)
    fig_sweep(s)
    fig_gust(gust)
    t = fig_training()
    (DATA / "summary.json").write_text(json.dumps({"sweep": s, "gust": g, "training": t}, indent=1))
    for row in s:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    for row in g:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    print(t)
    both = {c: [r for r in sweep if r["controller"] == c and r["wind_mps"] <= 10] for c in ("pid", "ppo_rta")}
    a = {(r["wind_mps"], r["seed"]): r["success"] for r in both["pid"]}
    b = {(r["wind_mps"], r["seed"]): r["success"] for r in both["ppo_rta"]}
    print("paired <=10 m/s: pid only", sum(a[k] and not b[k] for k in a), "rta only", sum(b[k] and not a[k] for k in a), "both", sum(a[k] and b[k] for k in a), "n", len(a))


if __name__ == "__main__":
    main()
