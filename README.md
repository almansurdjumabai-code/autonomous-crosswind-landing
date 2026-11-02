# Autonomous Crosswind Landing — ATR 72

**Nonlinear 6-DoF flight simulation, classical autoland, residual reinforcement learning with runtime assurance, and a WebGL mission console, set up for the ATR 72-600 regional turboprop.**

This repository is a fork of *Autonomous Crosswind Landing* by Almansur Jumabay (11th grade, Dostyq School, Astana), adapted to the **ATR use case**: the aircraft flown by the ATR 72 full-flight simulators (FFS2 · ATR 72-500 Thales, FFS6 and FFS10 · ATR 72-600 CAE) that the companion 3D digital twin [`simulateur_3d_depannage`](https://github.com/satoutahhaithem/simulateur_3d_depannage) represents. The original Cessna-172-class aircraft is kept as a preset so the original experiments stay reproducible.

📄 [Original technical report (PDF, Cessna 172 experiments)](Almansur_Crosswind_Project.pdf) · 📘 [Full technical documentation](docs/TECHNICAL.md)

![WebGL mission console replaying simulator telemetry](report/figures/console_chase.png)

> **Project status.** All results come from simulation. The ATR 72 airframe is a landing-configuration model built from published geometry and certified speeds, not from ATR flight-test data. The classical autoland has been re-tuned and validated on it; **no PPO policy has been trained on the ATR preset yet** (the training pipeline is ready, see below). The console shows a generic airliner 3D model at true scale, driven by the ATR flight data. Hardware-in-the-loop integration remains a proposal.

## The problem

In a crosswind an aircraft approaches with its nose pointed into the wind, then must align with the runway within seconds of touchdown while gusts and wind shear disturb it. On the ATR 72 the margins are tight: the pitch at touchdown is bounded by the tail-strike attitude (about 7°), the bank by the propeller clearance (about 15°), and the demonstrated crosswind limit is 35 kt. This project asks whether a learned policy can improve on a competent classical autoland without giving up safety.

## What is in the repository

| Component | Description | Location |
|---|---|---|
| Flight simulator | 13-state quaternion rigid body, RK4 at 100 Hz, stall and ground effect, landing gear contact, wind shear and veer, Dryden turbulence, 1-cosine gusts | `src/env/aircraft_env.py` |
| Aircraft presets | `atr72` (ATR 72-600, flaps 30, 21 t, V_REF 107 kt, 2500 m × 45 m runway) and `cessna172` (original data). Airframe, operating point, envelope, reward tolerances | `AIRCRAFT_PRESETS`, `make_env_config(aircraft=...)` |
| Classical autoland | Cascaded PID with wind estimation, continuous-entry exponential flare, pitch-rate limit, angle-of-attack protection and limited de-crab; one gain set per aircraft | `src/agents/baseline_autopilot.py` |
| Learned policy | PPO (Stable-Baselines3 / PyTorch) that adds bounded corrections to the autoland; `--aircraft` selects the preset | `src/agents/train_ppo.py` |
| Runtime assurance | Simplex monitor with a 1 s look-ahead, altitude-scheduled limits per aircraft, flare gate and primary lock-out after repeated reversions | `src/agents/safety_supervisor.py` |
| Mission console | React and Three.js replay of recorded flights, one data set per aircraft, instruments scaled from the manifest (V_REF bug, AoA limits, hard-landing threshold) | `src/web/` |
| Digital-twin bridge | Exports a simulated ATR approach in the conventions of the FFS twin's `interface/vol.js` | `src/viz/export_jumeau_trajectory.py`, `docs/jumeau/` |
| Blender pipeline | Procedural airfield and airliner model, glTF export, cinematic renderer | `src/viz/` |
| Report experiments | Scripts behind every number and plot in the original report | `src/viz/report_*.py`, `report/` |
| Tests | Integrator order, conservation laws, turbulence statistics, gust stability, reward properties, aircraft presets and strike geometry | `tests/` |

## The ATR 72-600 preset

| Quantity | Value | Source / rationale |
|---|---|---|
| Landing mass | 21 000 kg | typical landing mass (MLW 22 350 kg) |
| Wing | S = 61.0 m², b = 27.05 m, AR = 12, high wing 1.7 m above the CG | published geometry |
| Approach speed | 55 m/s (107 kt), flaps 30 | V_REF at 21 t; V_REF / V_S ≈ 1.22 with the modelled C_Lmax ≈ 2.7 |
| Trim on a 3° path | α ≈ 6.3°, θ ≈ 3.3°, δe ≈ −8.8°, 33 % thrust | solved by the simulator |
| Propulsion | two PW127M, 56 kN static, about 40 kN at V_REF | linear thrust model |
| Gear | wheelbase 10.75 m, track 4.10 m, CG 2.55 m above the wheels, 0.22 m static compression | published geometry |
| Strike points | propeller discs (1.25 m clearance), wing tips (4.3 m), aft fuselage (tail strike at ≈ 7° compressed) | geometry |
| Runway | 2500 m × 45 m, aim point 300 m, 3° glideslope | the FFS visual's test runway (`vol.js`) |
| Hard landing | touchdown sink below −2.5 m/s (≈ 500 ft/min) | between the airline report level and the 10 ft/s certification limit |

The stability derivatives are a landing-configuration estimate (large C_L0 and nose-down C_m0 from flaps 30, large fin and rudder, high-wing dihedral effect). They reproduce the certified speeds and a plausible approach attitude; they are documented in `AircraftParameters` and are meant to be replaced by better data when available.

## Results (classical autoland, ATR 72-600)

Same seeds for every condition, crosswind from the right at 20 ft, 3° glideslope unless stated.

| Conditions | Landed | Notes |
|---|---:|---|
| Steady crosswind 0–35 kt, no turbulence | 8 / 8 | touchdown sink −0.8 to −1.6 m/s, residual crab ≤ 11° at 30–35 kt |
| Gusty: 0/10/20/30 kt × 50/100 % gusts × 2.5°/3.5° | 12 / 16 | failures are hard landings at 20–30 kt with gusts |
| Monte-Carlo: 0/10/20/30 kt, 50 % gusts, 10 random offsets each | 33–34 / 40 | 10/10 at 0 kt, 9/10 at 10 kt, 7/10 at 20 kt, 7–8/10 at 30 kt |
| Console grid: 0–35 kt × 0/50/100 % gusts × 2.5/3.0/3.5° (`src/web/public/data/atr72`) | 62 / 72 | 7 of the 10 failures are at 30–35 kt with 100 % gusts; the rest are sinks of −2.5 to −2.8 m/s |

What limits the ATR autoland is the flare in turbulence: with 2.9 m/s of approach sink and a 6.5° pitch cap, a late gust cannot always be arrested within the 2.5 m/s limit. This is exactly where a residual policy has room to help.

**Legacy results (Cessna-172-class preset, original report).** Successful landings out of 30 per crosswind with turbulence and gusts: 30/19/14/4 for the autoland and 30/19/13/5 for residual PPO + RTA at 5/10/15/20 m/s. The re-tuned flare logic (continuous entry, wings level after touchdown) also applies to the Cessna preset and lands 35/40 of the same Monte-Carlo set; the numbers in the PDF report were measured before that change.

## Quick start

Requires Python 3.12+ and Node.js.

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                            # 59 tests, both presets
```

Fly the ATR autoland and look at one approach:

```python
from src.env.aircraft_env import CrosswindLandingEnv, make_env_config, KT_TO_MPS
from src.agents.baseline_autopilot import ClassicalAutopilot

env = CrosswindLandingEnv(make_env_config("atr72"))          # "cessna172" for the original aircraft
obs, _ = env.reset(seed=7, options={"wind_speed_mps": 20 * KT_TO_MPS, "wind_from_deg": 90})
ap = ClassicalAutopilot(env)                                  # gains follow the preset
while True:
    obs, r, term, trunc, info = env.step(ap.act())
    if term or trunc:
        break
print(info["episode_metrics"])
```

Run the console, which replays the rollouts included in this repository (`src/web/public/data/atr72`, `.../cessna172`):

```bash
cd src/web
npm install
npm run dev                       # http://127.0.0.1:5173
```

Keys: `Space` play/pause, `1`–`4` cameras, `F` airflow overlay, `G` graphics quality. Deep links take `?ac=atr72|cessna172&ctrl=pid|ppo&xw=20&gust=50&gp=3&t=60&cam=3`.

Regenerate the console data, train a policy on the ATR, or reproduce the report's measurements:

```bash
python -m src.viz.export_flight_data --aircraft atr72 --controllers pid            # 72 autoland rollouts
python -m src.agents.train_ppo --aircraft atr72 --total-timesteps 5000000 --n-envs 8 --run-dir runs/ppo_atr72
python -m src.viz.export_flight_data --aircraft atr72 --model runs/ppo_atr72/best/best_model.zip
python -m src.viz.report_experiment --model report/model_snapshot/best/best_model.zip   # Cessna snapshot
```

The aircraft preset is written to each run's `config.json`; evaluation and export rebuild the same environment and refuse a checkpoint trained on another preset. The chain has been exercised end to end on the ATR preset with a short CPU smoke run (6k steps, residual mode): training, checkpoint and VecNormalize saving, export of the three controllers to the console format, the preset-mismatch refusal, and the evaluation figures. A real training run (millions of steps) still has to be done on a GPU machine.

## Bridge to the FFS digital twin

`simulateur_3d_depannage` plays a hand-written approach in `interface/vol.js` (V_APP 107 kt, 3° path, aim point 300 m). `src/viz/export_jumeau_trajectory.py` records a simulated ATR 72 approach in the same frame (X along the runway from the threshold, Y right, wheel height, radians) with the cockpit commands the twin animates (`mancheT`, `mancheR`, `palonnier`, `pl`) and the wind at each step, so the visual, the motion cues and the levers can follow a physically simulated crosswind landing:

```bash
python -m src.viz.export_jumeau_trajectory --crosswind-kt 15 --gust-pct 50 --seed 2029 --out docs/jumeau/approche_atr72_xw15_g050.json
```

`docs/jumeau/approche_atr72_xw15_g050.json` is one such approach (15 kt crosswind, 50 % gusts, landed at −0.9 m/s, 301 m past the threshold, 2.2° of residual crab).

## Roadmap

- Train the residual PPO policy on the ATR preset (the pipeline, export and console are ready; the console shows the autoland alone until a checkpoint exists).
- Replace the estimated stability derivatives by ATR data when available, and the airliner 3D model by an ATR 72 model.
- Feed the exported approaches into the twin's `vol.js` replay.
- Build the proposed hardware bench and measure real sensor and serial latency.

## Acknowledgement

The simulator, the original controllers, the console and the report are the work of Almansur Jumabay; this fork adapts them to the ATR 72. Parts of the software, visualisation and documentation were prepared with the assistance of an AI coding tool (Claude Code).
