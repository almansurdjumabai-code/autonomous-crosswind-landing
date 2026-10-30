"""Export simulator rollouts as compact JSON for the WebGL mission-control app.

Runs the 6DoF environment over a scenario grid (controller x crosswind x gust
intensity x glidepath), records every 20 Hz control step with
:class:`TelemetryRecorder`, converts poses into the Three.js / glTF frame used by
``public/models/scene.glb`` and writes:

* ``<out>/flight_data.json``         manifest: scenario grid, per-scenario outcome metrics,
                                      runway geometry, units, provenance + the default episode inline
* ``<out>/episodes/<scenario>.json``  one column-oriented episode per scenario (fetched lazily)

Coordinate frame (Three.js, matches the GLB): +X east, +Y up, +Z south; the runway
runs from the threshold at the origin towards -Z. Quaternions are [x, y, z, w] and
rotate the exported aircraft model (nose +X, up +Y, left wing -Z) into the world.

Usage::

    python -m src.viz.export_flight_data --model runs/ppo_crosswind/best/best_model.zip \
        --out src/web/public/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.agents.baseline_autopilot import ClassicalAutopilot
from src.agents.safety_supervisor import SafetySupervisor
from src.env.aircraft_env import CrosswindLandingEnv, EnvConfig, KT_TO_MPS
from src.utils.telemetry import TelemetryRecorder

SCHEMA = "alma-autoland-flight-data"
SCHEMA_VERSION = "1.0.0"
CROSSWINDS_KT = (0, 5, 10, 15, 20, 25, 30)
GUST_LEVELS_PCT = (0, 50, 100)
GLIDEPATHS_DEG = (2.5, 3.0, 3.5)


def scenario_id(controller: str, xw_kt: float, gust_pct: int, gp_deg: float) -> str:
    return f"{controller}_xw{int(xw_kt):02d}_g{gust_pct:03d}_gp{int(round(gp_deg * 10)):02d}"


CONTROLLER_LABELS = {
    "ppo": "PPO policy + runtime assurance (Simplex)",
    "ppo_raw": "PPO policy, unprotected",
    "pid": "Classical crosswind autoland (PID)",
}
RTA_REASONS = ("bank", "pitch", "alpha", "airspeed", "sink_rate", "glideslope", "localizer", "flare_gate")


def env_config(gust_pct: int, gp_deg: float, action_mode: str = "absolute", residual_scale: float = 0.3) -> EnvConfig:
    cfg = EnvConfig(action_mode=action_mode, residual_scale=residual_scale)
    g = gust_pct / 100.0
    cfg.wind = replace(
        cfg.wind,
        turbulence_enabled=g > 0.0,
        turbulence_scale=2.0 * g,
        gusts_enabled=g > 0.0,
        gust_rate_hz=0.03 + 0.12 * g,
        gust_amplitude_fraction=0.3 + 0.5 * g,
    )
    cfg.runway = replace(cfg.runway, glideslope_rad=math.radians(gp_deg))
    return cfg


class Controller:
    """Uniform interface: ``act(obs)`` plus ``authority`` (share of control held by the learned
    policy, 0..1) and ``reason`` (active runtime-assurance monitor) for telemetry."""

    authority: float = 0.0
    reason: Optional[str] = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PIDController(Controller):
    def __init__(self, env: CrosswindLandingEnv) -> None:
        self.ap = ClassicalAutopilot(env)

    def act(self, obs):
        return self.ap.act()


class RawPolicyController(Controller):
    authority = 1.0

    def __init__(self, policy: Callable[[np.ndarray], np.ndarray]) -> None:
        self.policy = policy

    def act(self, obs):
        return self.policy(obs)


class SupervisedController(Controller):
    def __init__(self, env: CrosswindLandingEnv, policy: Callable[[np.ndarray], np.ndarray]) -> None:
        self.rta = SafetySupervisor(env, policy)
        self.authority = 1.0

    def act(self, obs):
        a = self.rta.act(obs)
        self.authority, self.reason = self.rta.authority, self.rta.reason
        return a


class PPOPolicy:
    """Deterministic PPO policy with frozen observation normalisation."""

    def __init__(self, model_path: str) -> None:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from src.agents.train_ppo import find_vecnormalize, load_ppo, run_action_mode

        self.model = load_ppo(model_path, device="cpu")
        self.vecnorm = VecNormalize.load(str(find_vecnormalize(model_path)), DummyVecEnv([CrosswindLandingEnv]))
        self.vecnorm.training = False
        self.model_path = model_path
        self.action_mode, self.residual_scale = run_action_mode(model_path)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        a, _ = self.model.predict(self.vecnorm.normalize_obs(obs[None, :].astype(np.float32)), deterministic=True)
        return np.asarray(a[0], dtype=np.float32)


def _r(a, nd: int) -> list:
    return np.round(np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0), nd).tolist()


def to_web_episode(doc: dict[str, Any], sid: str, controller: str, xw_kt: float, gust_pct: int, gp_deg: float) -> dict:
    fr = doc["frames"]
    loc_b = np.array([f["blender"]["location"] for f in fr], dtype=np.float64)
    q_b = np.array([f["blender"]["rotation_quaternion"] for f in fr], dtype=np.float64)  # w x y z
    # Blender (Z-up) -> glTF/Three (Y-up): (x, y, z) -> (x, z, -y), applied to world and model alike.
    pos = np.stack([loc_b[:, 0], loc_b[:, 2], -loc_b[:, 1]], axis=1)
    quat = np.stack([q_b[:, 1], q_b[:, 3], -q_b[:, 2], q_b[:, 0]], axis=1)  # x y z w
    for i in range(1, len(quat)):
        if float(np.dot(quat[i], quat[i - 1])) < 0.0:
            quat[i] = -quat[i]
    w_b = np.array([f["wind_blender"] for f in fr], dtype=np.float64)
    wind = np.stack([w_b[:, 0], w_b[:, 2], -w_b[:, 1]], axis=1)
    turb = np.array([np.add(f["wind_ned"]["turbulence"], f["wind_ned"]["gust"]) for f in fr], dtype=np.float64)
    ned = np.array([f["ned"]["position"] for f in fr], dtype=np.float64)
    vel = np.array([f["ned"]["velocity"] for f in fr], dtype=np.float64)
    m = doc["episode"]["metrics"]
    td = doc["episode"]["touchdown"]
    return {
        "id": sid,
        "controller": controller,
        "crosswind_kt": xw_kt,
        "gust_pct": gust_pct,
        "glidepath_deg": gp_deg,
        "dt": doc["episode"]["control_dt_s"],
        "n": len(fr),
        "touchdown": td,
        "metrics": m,
        "columns": {
            "t": _r([f["t"] for f in fr], 3),
            "pos": _r(pos, 3),
            "quat": _r(quat, 5),
            "euler_deg": _r([f["ned"]["euler_deg"] for f in fr], 2),
            "rates_dps": _r([f["ned"]["body_rates_dps"] for f in fr], 2),
            "x_m": _r(ned[:, 0], 2),
            "y_m": _r(ned[:, 1], 3),
            "h_m": _r(-ned[:, 2], 3),
            "vz_mps": _r(-vel[:, 2], 3),
            "gs_mps": _r(np.hypot(vel[:, 0], vel[:, 1]), 2),
            "ias_mps": _r([f["airdata"]["airspeed_mps"] for f in fr], 2),
            "alpha_deg": _r([f["airdata"]["alpha_deg"] for f in fr], 2),
            "beta_deg": _r([f["airdata"]["beta_deg"] for f in fr], 2),
            "elevator_deg": _r([f["controls"]["elevator_deg"] for f in fr], 2),
            "aileron_deg": _r([f["controls"]["aileron_deg"] for f in fr], 2),
            "rudder_deg": _r([f["controls"]["rudder_deg"] for f in fr], 2),
            "throttle": _r([f["controls"]["throttle"] for f in fr], 3),
            "action": _r([f["controls"]["action"] for f in fr], 3),
            "gear_n": _r([f["gear_loads_n"] for f in fr], 0),
            "wind_mps": _r(wind, 2),
            "turb_mps": _r(np.linalg.norm(turb, axis=1), 2),
            "reward": _r([f["reward"] for f in fr], 3),
        },
    }


def run_episode(controller_name: str, factory, xw_kt: float, gust_pct: int, gp_deg: float, seed: int,
                action_mode: str = "absolute", residual_scale: float = 0.3) -> dict:
    env = TelemetryRecorder(CrosswindLandingEnv(env_config(gust_pct, gp_deg, action_mode, residual_scale)))
    obs, _ = env.reset(seed=seed, options={"wind_speed_mps": xw_kt * KT_TO_MPS, "wind_from_deg": 90.0, "start_y_m": 0.0})
    ctrl: Controller = factory(env.unwrapped)
    authority = [ctrl.authority]
    reasons = [0]
    while True:
        obs, _, term, trunc, _ = env.step(ctrl.act(obs))
        authority.append(ctrl.authority)
        reasons.append(RTA_REASONS.index(ctrl.reason) + 1 if ctrl.reason in RTA_REASONS else 0)
        if term or trunc:
            break
    sid = scenario_id(controller_name, xw_kt, gust_pct, gp_deg)
    ep = to_web_episode(env.last_episode, sid, controller_name, xw_kt, gust_pct, gp_deg)
    ep["controller_label"] = CONTROLLER_LABELS[controller_name]
    ep["columns"]["authority"] = _r(authority, 3)
    ep["columns"]["rta_reason"] = reasons
    ep["rta_reason_codes"] = ["none", *RTA_REASONS]
    ep["metrics"]["policy_authority_mean"] = float(np.mean(authority))
    return ep


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="runs/ppo_crosswind/best/best_model.zip")
    p.add_argument("--out", default="src/web/public/data")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--crosswinds", type=float, nargs="+", default=list(CROSSWINDS_KT))
    p.add_argument("--gusts", type=int, nargs="+", default=list(GUST_LEVELS_PCT))
    p.add_argument("--glidepaths", type=float, nargs="+", default=list(GLIDEPATHS_DEG))
    p.add_argument("--controllers", nargs="+", default=["ppo", "ppo_raw", "pid"], choices=("ppo", "ppo_raw", "pid"))
    p.add_argument("--default", default=None, help="scenario id embedded in flight_data.json")
    a = p.parse_args(argv)

    out = Path(a.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    factories: dict[str, Any] = {}
    provenance: dict[str, Any] = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": a.seed}
    action_mode, residual_scale = "absolute", 0.3
    if {"ppo", "ppo_raw"} & set(a.controllers):
        ppo = PPOPolicy(a.model)
        action_mode, residual_scale = ppo.action_mode, ppo.residual_scale
        if "ppo" in a.controllers:
            factories["ppo"] = lambda env: SupervisedController(env, ppo)
        if "ppo_raw" in a.controllers:
            factories["ppo_raw"] = lambda env: RawPolicyController(ppo)
        digest = hashlib.sha256(Path(a.model).read_bytes()).hexdigest()[:12]
        provenance["ppo_checkpoint"] = {"path": str(Path(a.model)).replace("\\", "/"), "sha256_12": digest,
                                        "num_timesteps": int(getattr(ppo.model, "num_timesteps", 0)),
                                        "action_mode": action_mode, "residual_scale": residual_scale}
    if "pid" in a.controllers:
        factories["pid"] = PIDController

    scenarios = []
    episodes: dict[str, dict] = {}
    t0 = time.time()
    for ctrl, factory in factories.items():
        for xw in a.crosswinds:
            for g in a.gusts:
                for gp in a.glidepaths:
                    mode = action_mode if ctrl != "pid" else "absolute"
                    ep = run_episode(ctrl, factory, xw, g, gp, a.seed, mode, residual_scale)
                    path = out / "episodes" / f"{ep['id']}.json"
                    path.write_text(json.dumps(ep, separators=(",", ":")), encoding="utf-8")
                    m = ep["metrics"]
                    scenarios.append({
                        "id": ep["id"], "controller": ctrl, "crosswind_kt": xw, "gust_pct": g, "glidepath_deg": gp,
                        "file": f"episodes/{ep['id']}.json", "frames": ep["n"],
                        "success": bool(m["success"]), "crash_reason": m["crash_reason"],
                        "touchdown_sink_mps": m["touchdown_sink_rate_mps"], "touchdown_x_m": m["touchdown_x_m"],
                        "touchdown_y_m": m["touchdown_y_m"], "episode_return": m["episode_return"],
                        "policy_authority_mean": m["policy_authority_mean"],
                    })
                    episodes[ep["id"]] = ep
                    sink = m["touchdown_sink_rate_mps"]
                    sink_txt = f"{sink:6.2f}" if isinstance(sink, (int, float)) else "   n/a"
                    print(f"{ep['id']:28s} {'LANDED' if m['success'] else (m['crash_reason'] or 'timeout'):>22s} "
                          f"sink={sink_txt} frames={ep['n']}  ({time.time() - t0:5.1f}s)", flush=True)

    default_id = a.default or scenario_id(a.controllers[0], 10 if 10 in a.crosswinds else a.crosswinds[0],
                                          50 if 50 in a.gusts else a.gusts[0], 3.0 if 3.0 in a.glidepaths else a.glidepaths[0])
    cfg = EnvConfig()
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "project": "ALMA-AUTOLAND",
        "frame": "three.js / glTF: +X east, +Y up, +Z south; runway threshold at origin, runway along -Z",
        "units": {"length": "m", "speed": "m/s", "angle": "deg", "force": "N", "time": "s"},
        "runway": {"length_m": cfg.runway.length_m, "width_m": cfg.runway.width_m, "aim_point_m": cfg.runway.aim_point_m},
        "control_limits_deg": {"elevator": 25.0, "aileron": 20.0, "rudder": 25.0},
        "crash_sink_rate_mps": -1.5,
        "controller_labels": {k: CONTROLLER_LABELS[k] for k in factories},
        "grid": {"controllers": list(factories), "crosswind_kt": list(a.crosswinds), "gust_pct": list(a.gusts),
                 "glidepath_deg": list(a.glidepaths)},
        "provenance": provenance,
        "scenarios": scenarios,
        "default_scenario": default_id,
        "default_episode": episodes.get(default_id),
    }
    (out / "flight_data.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    ok = sum(s["success"] for s in scenarios)
    print(f"wrote {len(scenarios)} scenarios ({ok} landed) to {out.resolve()} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
