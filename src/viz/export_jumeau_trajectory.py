"""Export an ATR 72 approach flown by the simulator for the 3D FFS digital twin (``simulateur_3d_depannage``).

The twin's ``interface/vol.js`` plays a hand-written approach (``Vol.lancer('approche')``) with
ATR 72 orders of magnitude. This script records a real rollout of the 6DoF simulator with the
ATR 72-600 preset and writes it in the twin's conventions, so the visual, the motion cues and the
cockpit levers can follow a physically simulated crosswind landing instead of a scripted one:

* world frame: X along the runway from the threshold (m), Y to the right (m), ``h`` height of the
  wheels above the runway (m), the aim point 300 m past the threshold, as in ``vol.js``
  (``VISEE``, ``PISTE``);
* ``psi`` heading relative to the runway, ``theta`` pitch, ``phi`` bank, in radians;
* ``V`` airspeed (m/s), ``vz`` vertical speed (m/s, positive up), ``gamma`` flight-path angle;
* ``sol`` true once the wheels are on the ground;
* cockpit commands as the twin's ``cmd`` object expects them, in [-1, 1] for the control column
  (``mancheT`` pull positive, ``mancheR`` right positive) and the pedals (``palonnier``, right
  positive), and the power lever ``pl`` in [0, 1] (0 = flight idle, 1 = full power; the twin maps
  it on its notches).

Usage::

    python -m src.viz.export_jumeau_trajectory --crosswind-kt 15 --gust-pct 50 \
        --out docs/jumeau/approche_atr72_xw15_g050.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.baseline_autopilot import ClassicalAutopilot
from src.env.aircraft_env import CrosswindLandingEnv, KT_TO_MPS, aircraft_key
from src.viz.export_flight_data import env_config

SCHEMA = "atr72-approche-jumeau"
SCHEMA_VERSION = "1.0.0"


def record(crosswind_kt: float, gust_pct: int, glidepath_deg: float, seed: int, start_distance_m: float,
           aircraft: str = "atr72") -> dict:
    cfg = env_config(gust_pct, glidepath_deg, aircraft=aircraft)
    env = CrosswindLandingEnv(cfg)
    obs, _ = env.reset(seed=seed, options={"wind_speed_mps": crosswind_kt * KT_TO_MPS, "wind_from_deg": 90.0,
                                           "start_y_m": 0.0, "start_x_m": -abs(start_distance_m)})
    ap = ClassicalAutopilot(env)
    ac = env.cfg.aircraft
    frames = []

    def frame() -> dict:
        k = env._kinematics()
        va, alpha, beta, _ = env._aux["airdata"]
        gs = math.hypot(k["x_dot"], k["y_dot"])
        a = env.command_to_action(env.actuators)  # normalised surface positions
        return {
            "t": round(env.time_s, 3),
            "x": round(k["x"], 3), "y": round(k["y"], 3), "h": round(max(k["h"] - ac.wheel_height_m, 0.0), 3),
            "h_cg": round(k["h"], 3),
            "psi": round(k["heading_error"], 5), "theta": round(k["theta"], 5), "phi": round(k["phi"], 5),
            "V": round(float(va), 3), "Vsol": round(gs, 3), "vz": round(k["h_dot"], 3),
            "gamma": round(math.atan2(k["h_dot"], max(gs, 1.0)), 5),
            "alpha": round(float(alpha), 5), "beta": round(float(beta), 5),
            "sol": bool(env.touched_down),
            "cmd": {
                # Elevator trailing edge down (positive) pitches the nose down: the column is pulled when it is negative.
                "mancheT": round(-float(a[0]), 4),
                "mancheR": round(float(a[1]), 4),
                # Rudder trailing edge left (positive) yaws the nose left: left pedal, so right pedal is negative.
                "palonnier": round(-float(a[2]), 4),
                "pl": round(float(env.actuators[3]), 4),
                "volets": 1.0, "train": 1.0,
            },
            "vent": {"total_mps": [round(float(v), 3) for v in env.wind_at(k["h"])]},
        }

    frames.append(frame())
    while True:
        obs, _, term, trunc, info = env.step(ap.act())
        frames.append(frame())
        if term or trunc:
            break
    m = info["episode_metrics"]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "autonomous-crosswind-landing: 6DoF simulator, classical crosswind autoland",
        "aircraft": {"preset": aircraft_key(ac), "name": ac.name, "mass_kg": ac.mass_kg, "vapp_mps": cfg.approach_speed_mps,
                     "wheel_height_m": ac.wheel_height_m},
        "piste": {"longueur_m": cfg.runway.length_m, "largeur_m": cfg.runway.width_m, "visee_m": cfg.runway.aim_point_m,
                  "pente_deg": glidepath_deg},
        "conditions": {"vent_travers_kt": crosswind_kt, "vent_de_deg": 90.0, "rafales_pct": gust_pct, "graine": seed},
        "repere": "X le long de la piste depuis le seuil (m), Y a droite (m), h hauteur des roues (m), angles en radians",
        "dt": cfg.control_dt,
        "resultat": {"pose": bool(m["success"]), "raison": m["crash_reason"], "vz_toucher_mps": m["touchdown_sink_rate_mps"],
                     "x_toucher_m": m["touchdown_x_m"], "y_toucher_m": m["touchdown_y_m"], "crabe_toucher_deg": m["touchdown_crab_deg"]},
        "frames": frames,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--crosswind-kt", type=float, default=15.0)
    p.add_argument("--gust-pct", type=int, default=50)
    p.add_argument("--glidepath-deg", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--start-distance-m", type=float, default=3500.0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    doc = record(a.crosswind_kt, a.gust_pct, a.glidepath_deg, a.seed, a.start_distance_m)
    out = Path(a.out) if a.out else Path("results/jumeau") / f"approche_atr72_xw{int(a.crosswind_kt):02d}_g{a.gust_pct:03d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    r = doc["resultat"]
    print(f"{len(doc['frames'])} frames -> {out}  {'POSE' if r['pose'] else r['raison']} vz={r['vz_toucher_mps']:.2f} m/s "
          f"x={r['x_toucher_m']:.0f} m crabe={r['crabe_toucher_deg']:.1f} deg")


if __name__ == "__main__":
    main()
