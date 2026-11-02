"""Measurements behind report/Almansur_Crosswind_Project.tex: every number and plot comes from here.

Experiments (all on the 6DoF simulator, fixed seeds, the same seeds for every controller):

* ``sweep``   Monte-Carlo landings per mean crosswind, random initial lateral offset.
* ``gust``    a scripted 20 m/s peak lateral 1-cosine gust on short final, with attitude histories.
* ``crab``    steady crab angle in calm-turbulence air against the analytic arcsin(W_y / V_a).
* ``latency`` wall-clock cost of one control decision per controller.

Usage::

    python -m src.viz.report_experiment --model report/model_snapshot/best/best_model.zip

The aircraft preset is read from the checkpoint's run configuration (``--aircraft`` overrides it;
the report's snapshot was trained on the Cessna-172-class preset).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.env.aircraft_env import AIRCRAFT_PRESETS, CrosswindLandingEnv, EnvConfig, make_env_config
from src.viz.export_flight_data import PIDController, PPOPolicy, RawPolicyController, SupervisedController
from src.viz.export_flight_data import env_config as _env_config

CONTROLLERS = ("pid", "ppo_rta", "ppo_raw")
CROSSWINDS_MPS = (0.0, 5.0, 10.0, 15.0, 20.0)
SWEEP_EPISODES = 30
GUST_EPISODES = 20
GUST_PEAK_MPS = 20.0
GUST_LENGTH_M = 120.0
GUST_TRIGGER_X_M = -800.0
GUST_MEAN_WIND_MPS = 5.0
RECOVERY_BAND_M = 5.0
LOAD_WINDOW_S = 1.0
_policy = None
_aircraft = "cessna172"


def env_config(gust_pct: int, gp_deg: float) -> EnvConfig:
    return _env_config(gust_pct, gp_deg, aircraft=_aircraft)


def _init_worker(model_path: str, aircraft: str) -> None:
    global _policy, _aircraft
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)
    _aircraft = aircraft
    _policy = PPOPolicy(model_path)


def make_env(controller: str, cfg: EnvConfig) -> CrosswindLandingEnv:
    if controller != "pid":
        cfg = replace(cfg, action_mode=_policy.action_mode, residual_scale=_policy.residual_scale)
    return CrosswindLandingEnv(cfg)


def make_controller(controller: str, env: CrosswindLandingEnv):
    if controller == "pid":
        return PIDController(env)
    if controller == "ppo_rta":
        return SupervisedController(env, _policy)
    return RawPolicyController(_policy)


class ScriptedLateralGust:
    """One 1-cosine gust blowing from the right, triggered when the aircraft crosses a station."""

    def __init__(self, env: CrosswindLandingEnv) -> None:
        self.env = env
        self.reset()

    def reset(self) -> None:
        self.t0 = None
        self.duration = None

    def step(self, t, airspeed_mps, w20_mps, dt, rng) -> np.ndarray:
        if self.t0 is None and float(self.env.state[0]) >= GUST_TRIGGER_X_M:
            self.t0 = t
            self.duration = GUST_LENGTH_M / max(airspeed_mps, 10.0)
        if self.t0 is None or t - self.t0 > self.duration:
            return np.zeros(3)
        shape = 0.5 * (1.0 - math.cos(2.0 * math.pi * (t - self.t0) / self.duration))
        return np.array([0.0, -GUST_PEAK_MPS * shape, 0.0])


def _lateral_load_g(env: CrosswindLandingEnv) -> float:
    """Body-axis lateral specific force (aerodynamic + gear, gravity excluded) in g."""
    aux = env._aux
    return float((aux["aero_force_b"][1] + aux["gear_force_b"][1]) / (env.mass * 9.80665))


def fly(controller: str, cfg: EnvConfig, seed: int, wind_mps: float, options: dict | None = None, scripted_gust: bool = False) -> dict:
    env = make_env(controller, cfg)
    if scripted_gust:
        env.gusts = ScriptedLateralGust(env)
    obs, _ = env.reset(seed=seed, options={"wind_speed_mps": wind_mps, "wind_from_deg": 90.0, **(options or {})})
    ctrl = make_controller(controller, env)
    rows, peak_load, y_final, authority = [], 0.0, [], []
    while True:
        obs, _, term, trunc, info = env.step(ctrl.act(obs))
        k = env._kinematics()
        authority.append(ctrl.authority)
        if env.touchdown is not None and env.time_s - env.touchdown.time_s <= LOAD_WINDOW_S:
            peak_load = max(peak_load, abs(_lateral_load_g(env)))
        if k["h"] < 30.0 and not env.touched_down:
            y_final.append(k["y"])
        rows.append((env.time_s, math.degrees(k["phi"]), math.degrees(k["theta"]), math.degrees(k["heading_error"]), k["y"], k["h"],
                     float(env._aux["airdata"][0])))
        if term or trunc:
            break
    m = info["episode_metrics"]
    out = {
        "controller": controller, "seed": seed, "wind_mps": wind_mps, "success": bool(m["success"]), "crash_reason": m["crash_reason"],
        "touched_down": bool(m["touched_down"]), "sink_mps": m["touchdown_sink_rate_mps"], "y_td_m": m["touchdown_y_m"], "x_td_m": m["touchdown_x_m"],
        "crab_td_deg": m["touchdown_crab_deg"], "roll_td_deg": m["touchdown_roll_deg"], "lateral_load_g": peak_load if m["touched_down"] else None,
        "y_rms_final_m": float(np.sqrt(np.mean(np.square(y_final)))) if y_final else None, "authority_mean": float(np.mean(authority)),
    }
    if scripted_gust:
        out["t_gust"] = env.gusts.t0
        out["gust_duration"] = env.gusts.duration
        out["rows"] = rows
    return out


def _sweep_task(args: tuple) -> dict:
    controller, wind, seed = args
    return fly(controller, env_config(50, 3.0), seed, wind)


def _gust_task(args: tuple) -> dict:
    controller, seed = args
    cfg = env_config(50, 3.0)
    return fly(controller, cfg, seed, GUST_MEAN_WIND_MPS, {"start_y_m": 0.0, "start_x_m": -1800.0}, scripted_gust=True)


def _crab_task(args: tuple) -> dict:
    wind, seed = args
    cfg = make_env_config(_aircraft)
    env = CrosswindLandingEnv(cfg)
    obs, _ = env.reset(seed=seed, options={"wind_speed_mps": wind, "wind_from_deg": 90.0, "turbulence": False, "gusts": False, "start_y_m": 0.0, "start_x_m": -1800.0})
    ctrl = PIDController(env)
    measured, predicted = [], []
    while True:
        obs, _, term, trunc, _ = env.step(ctrl.act(obs))
        k = env._kinematics()
        if 30.0 < k["h"] < 50.0:
            va = float(env._aux["airdata"][0])
            w_y = float(env.mean_wind.velocity_ned(k["h"])[1])
            measured.append(abs(math.degrees(k["heading_error"])))
            predicted.append(math.degrees(math.asin(min(abs(w_y) / va, 1.0))))
        if term or trunc or k["h"] < 30.0:
            break
    return {"wind_mps": wind, "measured_deg": float(np.mean(measured)), "predicted_deg": float(np.mean(predicted))}


def _latency_task(controller: str) -> dict:
    env = make_env(controller, env_config(50, 3.0))
    obs, _ = env.reset(seed=1, options={"wind_speed_mps": 5.0, "wind_from_deg": 90.0})
    ctrl = make_controller(controller, env)
    decide, step = [], []
    for _ in range(600):
        t0 = time.perf_counter()
        a = ctrl.act(obs)
        t1 = time.perf_counter()
        obs, _, term, trunc, _ = env.step(a)
        t2 = time.perf_counter()
        decide.append((t1 - t0) * 1000)
        step.append((t2 - t1) * 1000)
        if term or trunc:
            break
    return {"controller": controller, "decide_ms_mean": float(np.mean(decide)), "decide_ms_p99": float(np.percentile(decide, 99)),
            "env_step_ms_mean": float(np.mean(step)), "samples": len(decide)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--aircraft", choices=tuple(AIRCRAFT_PRESETS), default=None,
                   help="airframe preset (default: the one the checkpoint was trained on)")
    p.add_argument("--out", default="report/data")
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from src.agents.train_ppo import run_settings

    aircraft = a.aircraft or run_settings(a.model)["aircraft"]
    print(f"aircraft preset: {aircraft}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(a.model, aircraft)) as pool:
        crab = list(pool.map(_crab_task, [(w, 1000) for w in (2.5, 5.0, 7.5, 10.0, 12.5, 15.0)]))
        (out / "crab.json").write_text(json.dumps(crab, indent=1), encoding="utf-8")
        print("crab done", time.time() - t0, flush=True)
        latency = [pool.submit(_latency_task, c).result() for c in CONTROLLERS]
        (out / "latency.json").write_text(json.dumps(latency, indent=1), encoding="utf-8")
        print("latency done", time.time() - t0, flush=True)
        gust = list(pool.map(_gust_task, [(c, 2000 + s) for c in CONTROLLERS for s in range(GUST_EPISODES)]))
        (out / "gust.json").write_text(json.dumps(gust), encoding="utf-8")
        print("gust done", time.time() - t0, flush=True)
        tasks = [(c, w, 3000 + s) for c in CONTROLLERS for w in CROSSWINDS_MPS for s in range(SWEEP_EPISODES)]
        sweep = []
        for i, row in enumerate(pool.map(_sweep_task, tasks), 1):
            sweep.append(row)
            if i % 30 == 0:
                print(f"sweep {i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
                (out / "sweep.json").write_text(json.dumps(sweep), encoding="utf-8")
        (out / "sweep.json").write_text(json.dumps(sweep), encoding="utf-8")
    print(f"finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
