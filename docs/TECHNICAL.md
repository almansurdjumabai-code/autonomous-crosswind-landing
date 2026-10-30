# Technical documentation

A reinforcement-learning codebase for training and evaluating autonomous landing
controllers for a Cessna-172-class aircraft in crosswind, wind shear and turbulence.
It includes:

- a non-linear six-degree-of-freedom rigid-body simulator wrapped as a Gymnasium environment,
- a reward function that combines potential-based shaping with sparse touchdown terms,
- a Stable-Baselines3 PPO training pipeline,
- a classical cascaded-PID autopilot as a baseline,
- telemetry export for Blender and Three.js,
- scripts that produce IEEE-formatted figures.

```
aero_project/
├── src/
│   ├── env/aircraft_env.py          6DoF dynamics, atmosphere, landing gear, Gymnasium env
│   ├── agents/reward_shaping.py     potential-based + dense + sparse reward
│   ├── agents/train_ppo.py          PPO pipeline (SB3), schedules, curriculum, checkpoints
│   ├── agents/baseline_autopilot.py classical ILS-coupled PID autopilot (baseline)
│   ├── utils/telemetry.py           trajectory recorder -> JSON for Blender / Three.js
│   ├── viz/evaluate_and_plot.py     Monte-Carlo crosswind sweeps + IEEE figures
│   ├── viz/blender_cinematic_renderer.py  telemetry-driven 3-shot cinematic teaser (Blender bpy)
│   ├── viz/export_scene_to_gltf.py  Blender scene -> src/web/public/models/scene.glb (+ rig manifest)
│   ├── viz/export_flight_data.py    simulator rollout grid -> src/web/public/data (web telemetry)
│   └── web/                         React + Three.js mission-control console (Vite)
├── tests/test_physics.py            integrator / conservation / turbulence / gust stability
├── tests/test_telemetry_reward.py   frame conversions, strict JSON, reward properties
├── requirements.txt
└── pytest.ini
```

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

pytest                             # 40 tests

# Train (8 parallel workers, 5 M steps). Monitor progress with TensorBoard.
python -m src.agents.train_ppo --total-timesteps 5000000 --n-envs 8 --run-dir runs/ppo_xwind
tensorboard --logdir runs

# Evaluate the best checkpoint across crosswinds and make the paper figures
python -m src.viz.evaluate_and_plot --model runs/ppo_xwind/best/best_model.zip \
    --run-dir runs/ppo_xwind --crosswinds 0 2.5 5 7.5 10 --episodes 20 \
    --out-dir results/ppo --telemetry

# Same sweep with the classical autopilot as a baseline
python -m src.viz.evaluate_and_plot --policy baseline --out-dir results/baseline
```

Use the environment directly:

```python
import gymnasium as gym
import src.env  # registers CrosswindLanding-v0

env = gym.make("CrosswindLanding-v0")
obs, info = env.reset(seed=0, options={"wind_speed_mps": 7.5, "wind_from_deg": 90})
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`reset` accepts these `options`: `wind_speed_mps`, `wind_from_deg` (relative to the
runway; +90 is a direct crosswind from the right), `turbulence` (bool), `gusts` (bool),
`start_x_m` and `start_y_m`.

---

## 1. Flight dynamics

### 1.1 Reference frames

- **World frame:** flat-Earth North-East-Down (NED) with its origin at the runway
  threshold on the centreline. The runway lies along $+x$. Altitude is $h=-z$.
- **Body frame:** Forward-Right-Down (FRD), with its origin at the centre of gravity.
- **Attitude:** the unit quaternion $\mathbf q=[q_0,q_1,q_2,q_3]^\top$ (scalar first,
  Hamilton convention) rotates body vectors into NED. The ZYX Euler angles
  $(\phi,\theta,\psi)$ are computed from it. Quaternion integration has no singularity
  at $\theta=\pm90^\circ$ (see `test_no_gimbal_lock_at_vertical_pitch`).

The body-to-NED rotation matrix is

$$
R_{nb}(\mathbf q)=\begin{bmatrix}
q_0^2+q_1^2-q_2^2-q_3^2 & 2(q_1q_2-q_0q_3) & 2(q_1q_3+q_0q_2)\\
2(q_1q_2+q_0q_3) & q_0^2-q_1^2+q_2^2-q_3^2 & 2(q_2q_3-q_0q_1)\\
2(q_1q_3-q_0q_2) & 2(q_2q_3+q_0q_1) & q_0^2-q_1^2-q_2^2+q_3^2
\end{bmatrix}.
$$

### 1.2 Equations of motion

The integrated state is
$\mathbf x=[\,\mathbf p^n,\ \mathbf v^b,\ \mathbf q,\ \boldsymbol\omega^b\,]\in\mathbb R^{13}$,
where $\mathbf v^b=[u,v,w]^\top$ and $\boldsymbol\omega^b=[p,q,r]^\top$:

$$
\begin{aligned}
\dot{\mathbf p}^n &= R_{nb}\,\mathbf v^b \\
\dot{\mathbf v}^b &= \frac{1}{m}\left(\mathbf F_A+\mathbf F_T+\mathbf F_G+R_{nb}^\top
\begin{bmatrix}0\\0\\mg\end{bmatrix}\right)-\boldsymbol\omega^b\times\mathbf v^b \\
\dot{\mathbf q} &= \tfrac12\,\mathbf q\otimes\begin{bmatrix}0\\\boldsymbol\omega^b\end{bmatrix}
+k\left(1-\lVert\mathbf q\rVert^2\right)\mathbf q \\
\dot{\boldsymbol\omega}^b &= I^{-1}\left(\mathbf M_A+\mathbf M_G-\boldsymbol\omega^b\times I\boldsymbol\omega^b\right)
\end{aligned}
$$

$I$ is the full $3\times3$ inertia tensor, including products of inertia. The last term in
$\dot{\mathbf q}$ is a Baumgarte constraint stabiliser. After each step the quaternion is
also renormalised. In component form, the translational equations are the familiar ones:

$$
\dot u = rv-qw+\tfrac{F_x}{m},\qquad \dot v = pw-ru+\tfrac{F_y}{m},\qquad \dot w = qu-pv+\tfrac{F_z}{m}.
$$

### 1.3 Numerical integration

The equations are integrated with classical fourth-order Runge–Kutta at
$\Delta t_{\text{phys}}=0.01\,$s. The agent acts at $\Delta t_{\text{ctrl}}=0.05\,$s
(20 Hz, 5 physics sub-steps per action):

$$
\mathbf x_{k+1}=\mathbf x_k+\tfrac{\Delta t}{6}\left(\mathbf k_1+2\mathbf k_2+2\mathbf k_3+\mathbf k_4\right).
$$

Within each sub-step, turbulence and gusts are held constant (zero-order hold). The mean
wind is re-evaluated at each Runge–Kutta stage altitude, so wind shear is integrated
consistently.

### 1.4 Aerodynamics

The relative-wind velocity in body axes is
$\mathbf v_r=\mathbf v^b-R_{nb}^\top\mathbf w^n$, where $\mathbf w^n$ is the total wind
(mean, turbulence and gusts). From it:

$$
V_a=\lVert\mathbf v_r\rVert,\qquad \alpha=\operatorname{atan2}(w_r,u_r),\qquad
\beta=\arcsin\!\left(\frac{v_r}{V_a}\right),\qquad \bar q=\tfrac12\rho(h)V_a^2 .
$$

Wind therefore acts on the aircraft only through $\alpha$, $\beta$ and $V_a$. The
non-dimensional rates are $\hat p=\tfrac{pb}{2V_a}$, $\hat q=\tfrac{qc}{2V_a}$ and
$\hat r=\tfrac{rb}{2V_a}$.

**Lift with a smooth stall model** (Beard & McLain). The sigmoid $\sigma$ blends the
linear lift curve into flat-plate lift:

$$
\sigma(\alpha)=\frac{1+e^{-M(\alpha-\alpha_0)}+e^{M(\alpha+\alpha_0)}}
{\left(1+e^{-M(\alpha-\alpha_0)}\right)\left(1+e^{M(\alpha+\alpha_0)}\right)},
\qquad
C_L=(1-\sigma)(C_{L_0}+C_{L_\alpha}\alpha)+\sigma\,2\,\mathrm{sgn}(\alpha)\sin^2\alpha\cos\alpha
+C_{L_q}\hat q+C_{L_{\delta_e}}\delta_e .
$$

**Drag with ground effect.** Induced drag is scaled by McCormick's factor:

$$
\Phi_{ge}=\frac{(16h_w/b)^2}{1+(16h_w/b)^2},\qquad
C_D=(1-\sigma)\left(C_{D_0}+\Phi_{ge}\frac{C_{L,\text{lin}}^2}{\pi eA\!R}\right)+\sigma\left(C_{D_0}+2\sin^2\alpha\right)
+C_{D_\beta}\beta^2+C_{D_{\delta_e}}|\delta_e| .
$$

**Lateral-directional and moment coefficients.** These are linear in the stability
derivatives:

$$
\begin{aligned}
C_Y &= C_{Y_\beta}\beta+C_{Y_p}\hat p+C_{Y_r}\hat r+C_{Y_{\delta_r}}\delta_r\\
C_l &= C_{l_\beta}\beta+C_{l_p}\hat p+C_{l_r}\hat r+C_{l_{\delta_a}}\delta_a+C_{l_{\delta_r}}\delta_r\\
C_m &= C_{m_0}+C_{m_\alpha}\alpha+C_{m_q}\hat q+C_{m_{\delta_e}}\delta_e\\
C_n &= C_{n_\beta}\beta+C_{n_p}\hat p+C_{n_r}\hat r+C_{n_{\delta_a}}\delta_a+C_{n_{\delta_r}}\delta_r
\end{aligned}
$$

**Body-axis forces and moments.** Lift and drag are resolved from stability axes:

$$
\mathbf F_A=\bar qS\begin{bmatrix}-C_D\cos\alpha+C_L\sin\alpha\\ C_Y\\ -C_D\sin\alpha-C_L\cos\alpha\end{bmatrix},
\qquad
\mathbf M_A=\bar qS\begin{bmatrix}b\,C_l\\ c\,C_m\\ b\,C_n\end{bmatrix}.
$$

**Propulsion.** A fixed-pitch propeller with thrust along the body $x$-axis through the CG:
$T=\delta_t\,T_{\text{static}}\,\tfrac{\rho}{\rho_0}\max\left(0,1-V_a/V_{T=0}\right)$.

**Actuators.** Each actuator is a first-order lag (exact discretisation) with a rate
limit. The time constants are 0.06 s for the control surfaces and 0.4 s for the engine.
The surface rate limit is $80^\circ$/s.

The aircraft data (Cessna 172 class, Roskam / Stevens & Lewis) are set in `AircraftParameters`:
$m=1043$ kg, $S=16.2$ m², $b=10.92$ m, $c=1.49$ m,
$I_{xx},I_{yy},I_{zz}=1285,\,1825,\,2667$ kg m².

### 1.5 Atmosphere and wind

**Density.** ISA troposphere: $\rho(h)=1.225\,(1-2.25577\times10^{-5}h)^{4.2559}$.

**Mean wind with shear and directional veer.** The wind speed follows a power law, and
the wind direction turns linearly with altitude. $W_{20}$ is the reference speed at
20 ft (6.096 m):

$$
W(h)=W_{20}\left(\frac{h}{6.096}\right)^{1/7},\qquad
\chi(h)=\chi_{20}+k_{\text{veer}}\frac{h-6.096}{100},\qquad
\mathbf w^n_{\text{mean}}=-W(h)\begin{bmatrix}\cos\chi\\\sin\chi\\0\end{bmatrix}.
$$

**Dryden turbulence (MIL-F-8785C, low altitude).** Turbulence is modelled as a frozen
field (Taylor hypothesis) with $V=V_a$. The shaping filters are:

$$
H_u(s)\propto\frac{1}{1+\tau_u s},\qquad
H_{v,w}(s)\propto\frac{1+\sqrt3\,\tau_{v,w}s}{(1+\tau_{v,w}s)^2},\qquad \tau_i=\frac{L_i}{V}.
$$

The scale lengths and intensities are (with $h$ in ft, clamped to $[10,1000]$):

$$
L_w=h,\quad L_u=L_v=\frac{h}{(0.177+0.000823h)^{1.2}},\quad
\sigma_w=0.1\,W_{20},\quad \sigma_u=\sigma_v=\frac{\sigma_w}{(0.177+0.000823h)^{0.4}} .
$$

The filters are discretised exactly:

- The $u$ channel is an exact Gauss–Markov recursion,
  $x_{k+1}=e^{-\Delta t/\tau}x_k+\sigma\sqrt{1-e^{-2\Delta t/\tau}}\,\eta_k$.
- The $v$ and $w$ channels use the closed-form transition matrix of the repeated-pole system,
  $\Phi=e^{-\Delta t/\tau}\begin{bmatrix}1+\Delta t/\tau & \Delta t\\ -\Delta t/\tau^2 & 1-\Delta t/\tau\end{bmatrix}$.
  Their process-noise covariance
  $Q_d=q\int_0^{\Delta t}e^{As}BB^\top e^{A^\top s}ds$ is computed with 16-point
  Gauss–Legendre quadrature. The test suite checks it against Van Loan's method.
- The noise intensities are chosen so that the stationary variance equals $\sigma_i^2$
  exactly. For the first-order filter $q=2\sigma^2\tau$; for the second-order filters
  $q=\sigma^2\tau$.

Turbulence is generated in the mean-wind frame (along-wind, cross-wind, vertical) and
rotated into NED.

**Discrete gusts.** "1-cosine" gusts arrive as a Poisson process (rate $\lambda$). Each
gust has a random 3-D direction and a random amplitude and length. A gust starting at
$t_0$ contributes

$$
\mathbf w_g(t)=\frac{\mathbf A}{2}\left(1-\cos\frac{2\pi(t-t_0)}{T}\right),\qquad 0\le t-t_0\le T=\frac{d_m}{V_a}.
$$

### 1.6 Ground interaction

There are three landing-gear legs: nose (steerable, coupled to the rudder) and left and
right mains. Each is a unilateral spring–damper. For leg $i$ at body position
$\mathbf r_i$, the penetration depth is $\delta_i=z_{\text{cg}}+(R_{nb}\mathbf r_i)_z$,
and the contact-point velocity is $\mathbf v_{c,i}=R_{nb}(\mathbf v^b+\boldsymbol\omega\times\mathbf r_i)$.
The leg forces are:

$$
N_i=\max\left(0,\ k_i\delta_i+c_i\dot\delta_i\right),\qquad
F_{\text{long}}=-\mu_r N_i\tanh\frac{v_{\text{long}}}{\varepsilon},\qquad
F_{\text{lat}}=-\mu_{\text{lat}}N_i\tanh\frac{v_{\text{lat}}}{\varepsilon}.
$$

The forces are rotated into body axes and contribute moments $\mathbf r_i\times\mathbf F_i$.
The smooth $\tanh$ Coulomb model keeps the system Lipschitz-continuous, which keeps RK4
well behaved. Structural contact points (wingtips and tail) and a minimum fuselage height
are used to detect strikes.

### 1.7 Trim

The initial state is a wings-level, steady $3^\circ$ glide at $V_a=33$ m/s. It is found by
solving $\dot u=\dot w=\dot q=0$ for $(\alpha,\delta_e,\delta_t)$ by bounded nonlinear
least squares. The solution is
$\alpha\approx7.6^\circ$, $\theta\approx4.6^\circ$, $\delta_e\approx-6.0^\circ$ and
$\delta_t\approx0.29$, with a residual below $10^{-13}$.

---

## 2. MDP specification

### 2.1 State (simulator)

| Symbol | Meaning | Units |
|---|---|---|
| $x, y, z$ | NED position relative to the threshold (runway along $+x$, $h=-z$) | m |
| $u, v, w$ | body-axis inertial velocity | m/s |
| $\phi, \theta, \psi$ | roll, pitch, yaw (ZYX), derived from the integrated quaternion | rad |
| $p, q, r$ | body angular rates | rad/s |

### 2.2 Action space — `Box([-1, 1]^4)`

| Index | Channel | Mapping (`trim_referenced_actions=True`, default) |
|---|---|---|
| 0 | elevator $\delta_e$ | $\delta_{e,\text{trim}}+a_0\cdot25^\circ$, clipped to $\pm25^\circ$ |
| 1 | aileron $\delta_a$ | $a_1\cdot20^\circ$ |
| 2 | rudder $\delta_r$ | $a_2\cdot25^\circ$ (also steers the nose wheel, $\pm10^\circ$) |
| 3 | throttle $\delta_t$ | piecewise linear: $-1\mapsto0$, $0\mapsto\delta_{t,\text{trim}}$, $+1\mapsto1$ |

With the trim-referenced mapping, a zero action is the trimmed glide, so the initial
policy (mean action zero) starts close to the trimmed glide. Set
`EnvConfig.trim_referenced_actions=False` for absolute mappings.

### 2.3 Observation — `Box(-10, 10, (23,))`

$x/2000,\ y/100,\ h/100,\ u/40,\ v/10,\ w/10,\ \phi/(\pi/4),\ \theta/(\pi/6),\ \psi/(\pi/4),\ p,\ q,\ r,$
$(V_a-V_{\text{ref}})/10,\ \alpha/0.2,\ \beta/0.2,\ e_{gs}/20,\ \dot h/5,\ \dot y/10,$ the four
normalised actuator positions, and a weight-on-wheels flag.

The observation contains air data ($V_a$, $\alpha$, $\beta$), as from a pitot-static
system and vanes, but not the wind vector itself, so the agent must infer the wind.
Training also applies `VecNormalize` running normalisation.

### 2.4 Episode

- **Start:** the aircraft starts 1500–2200 m before the threshold, on the glideslope
  ($\sigma_h=6$ m), with lateral offset $\pm40$ m, heading $\pm8^\circ$, bank $\pm5^\circ$ and
  airspeed $\pm2$ m/s. The wind is $W_{20}\sim\mathcal U(0,W_{\max}\cdot\text{severity})$ from
  $\pm[30^\circ,150^\circ]$, with turbulence and gusts enabled.
- **Touchdown:** detected as the first gear contact, at 100 Hz resolution. It is a crash
  if any of these hold:
  - $\dot h<-1.5$ m/s,
  - the nose gear touches first,
  - the aircraft touches down before the threshold,
  - the aircraft touches down outside the runway edges.
- **Success:** 3 s of rollout after touchdown without a crash.
- **Other crashes:**
  - a wingtip, tail or fuselage strike,
  - $|\phi|>70^\circ$ or $|\theta|>40^\circ$,
  - $V_a<18$ m/s (stall) or $V_a>70$ m/s,
  - $|y|>400$ m or $h>500$ m,
  - overflying the runway end while airborne,
  - a runway excursion or overrun during rollout,
  - a non-finite state.
- **Truncation:** at 150 s.

---

## 3. Reward (`src/agents/reward_shaping.py`)

The per-step reward is $r_t=F_t+D_t+S_t$, made of a shaping term, dense penalties and
sparse terms.

**Potential-based shaping** (Ng et al., 1999) leaves the optimal policy unchanged. The
potential $\Phi$ is set to 0 on terminal transitions and after touchdown:

$$
F_t=\gamma\,\Phi(s_{t+1})-\Phi(s_t),\qquad
\Phi(s)=-w_{gs}\,\rho\!\left(\frac{e_{gs}}{\tau_{gs}(d)}\right)-w_{loc}\,\rho\!\left(\frac{y}{\tau_{loc}(d)}\right)-w_{prog}\frac{|d|}{1000},
$$

where:

- $\rho(e)=\sqrt{1+e^2}-1$ is a pseudo-Huber loss,
- $e_{gs}=h-\max(d,0)\tan3^\circ$ is the glideslope error,
- $d$ is the distance to the aim point,
- $\tau(d)=\tau_0+k\,d$ is a tolerance that widens with distance, mimicking the angular
  ILS beam.

**Dense penalties** $D_t$:

| Term | Expression |
|---|---|
| Attitude envelope | $-w\,[(\lvert\phi\rvert-25^\circ)_+^2+\text{dist}(\theta,[-10^\circ,15^\circ])^2]/(10^\circ)^2$ |
| Wings level near ground | $-w\,(1-h/15)\,(\phi/5^\circ)^2$ for $h<15$ m |
| Body rates | $-w\,(p^2+q^2+r^2)$ |
| Airspeed | $-w\,((V_a-V_{\text{ref}})/4)^2$ (airborne) |
| Stall protection | $-w\,((\alpha-12^\circ)_+/2^\circ)^2$ |
| Control effort / rate | $-w_u\lVert a_t\rVert^2-w_{\Delta u}\lVert a_t-a_{t-1}\rVert^2$ |
| De-crab in flare | $-w\,(1-h/15)\left[(\psi/5^\circ)^2+(\dot y/2)^2\right]$ for $h<15$ m |
| Rollout centreline | $-w\,(y/(W_{rwy}/2))^2$ |

The sum of these penalties is bounded below by $-5$ per step. This keeps value
regression well conditioned in deep-stall or tumbling states.

**Sparse terms** $S_t$. The touchdown reward is

$$
R_{td}=100\left[0.4\,e^{-\left(\frac{\dot h_{td}+0.3}{0.5}\right)^2}
+0.3\,e^{-\left(\frac{x_{td}-x_{aim}}{150}\right)^2}
+0.2\,e^{-\left(\frac{y_{td}}{W/8}\right)^2}
+0.1\,e^{-\left(\frac{\psi_{td}}{5^\circ}\right)^2}\right].
$$

A touchdown with $\dot h_{td}<-1.5$ m/s instead receives
$R=-100-20\,(-1.5-\dot h_{td})$. An undershoot or a runway-edge touchdown receives $-100$.
Other crashes receive $-100$. Completing the rollout earns $+50$.

---

## 4. Training (`src/agents/train_ppo.py`)

| Hyper-parameter | Value |
|---|---|
| Algorithm | PPO (SB3), `MlpPolicy`, separate $\pi$ / $V$ networks $[256,256]$, tanh |
| Initialisation | orthogonal: gain $\sqrt2$ hidden layers, $0.01$ policy head, $1$ value head; $\log\sigma_0=-0.7$ |
| Learning rate | cosine annealing $3\cdot10^{-4}\to10^{-5}$ with 2 % linear warm-up |
| Clip range | linear $0.2\to0.08$ |
| Rollout | 8 envs × 1024 steps, minibatch 512, 10 epochs, target KL 0.03 |
| $\gamma$, $\lambda_{GAE}$ | 0.99, 0.95 (the shaping $\gamma$ is set to PPO's $\gamma$) |
| Normalisation | `VecNormalize` (observations and returns, clip 10) |
| Curriculum | wind severity ramps linearly from 0.2 to 1.0 over 2 M steps |
| Checkpoints | every 250 k steps (model and VecNormalize) |
| Best model | `EvalCallback` every 50 k steps, 20 deterministic episodes at full wind severity; saves `best/best_model.zip` and `best/best_vecnormalize.pkl` |

TensorBoard logs the standard PPO diagnostics and these extra groups:

- `landing/*`: success rate, crash rate, touchdown sink rate, touchdown position, crab angle.
- `crash_reasons/*`: the share of episodes ending with each crash reason.
- `curriculum/wind_severity`.

To resume a run: `--resume-from runs/.../checkpoints/ppo_landing_1000000_steps.zip`. The
matching VecNormalize statistics are found automatically.

The local environment needed a version workaround. With torch 2.14 and SB3 2.9.2,
`PPO.load` fails with `PytorchStreamReader failed reading file`, because torch cannot read
the nested archive from the stream SB3 passes it. `train_ppo.load_ppo` catches that error
and repeats SB3's loading steps with each tensor read into memory first.

---

## 4b. Guidance and control stack

Three controllers fly the same simulator. The web console can switch between all three.

| Controller | What it is | Grid result (63 rollouts) |
|---|---|---|
| `pid` | Classical crosswind autoland (`src/agents/baseline_autopilot.py`) | 55/63 landed |
| `ppo` | PPO policy under Simplex runtime assurance (`src/agents/safety_supervisor.py`) | 50/63 landed, PPO authority 34 % on average |
| `ppo_raw` | PPO policy without protection | 0/63 (the 1.45M-step checkpoint was trained before the fixes below) |

The grid is crosswind 0–30 kt × gust intensity 0/50/100 % × glidepath 2.5/3.0/3.5°.

**Classical autoland.** It mirrors airline technique:

- **Wind estimation:** inertial velocity minus the air-velocity vector built from Va, α and β.
- **Speed:** the approach speed is gust-adjusted, Vref + min(0.5·gust + 0.12·crosswind, 6 m/s), and power is held through the flare.
- **Flare:** a dedicated faster sink-rate loop.
- **Stall protection:** an angle-of-attack limit caps the pitch command at α ≤ 12°. Before this, gusts could drive the aircraft into a 45° deep stall that hit the ground at −27 m/s.
- **De-crab:** a partial de-crab limits sideslip to 12°, so the aircraft keeps a small residual crab instead of losing lift. The old full de-crab needed 25–30° of sideslip at 25 kt.
- **Lateral:** wing-low to hold the centreline.

All steady crosswinds up to 30 kt now land. The remaining failures are in severe turbulence (100 % gusts at 15 kt and above), against the −1.5 m/s crash criterion.

**Runtime assurance (Simplex).** PPO flies while an envelope monitor checks the state 1 s ahead:

- bank and sink rate, with limits scheduled by altitude;
- pitch;
- angle of attack;
- airspeed;
- glideslope and localizer deviation.

When it predicts a violation, authority blends to the autoland at 2.5/s and stays latched for 3 s. Inside the 15 m flare gate it never hands authority back. Every rollout records the per-frame authority λ and the monitor that triggered.

**Residual training (default).** `python -m src.agents.train_ppo` now trains in residual mode: `a = a_autoland + 0.3·a_policy`, with log σ₀ = −1.6. The policy starts from a competent controller and learns bounded corrections. The flare reward now includes a dense sink-rate term. Use `--action-mode absolute` for the old formulation. Evaluation and export read the action mode from the run's `config.json`.

## 5. Telemetry (`src/utils/telemetry.py`)

`TelemetryRecorder` is a Gymnasium wrapper. It records every control step and writes a
strict-JSON document (`NaN` is written as `null`) with the schema
`crosswind-landing-telemetry` v1.0.0. Each frame contains:

- NED position, velocity, quaternion, Euler angles and body rates,
- Blender pose (`location`, `rotation_quaternion` in w-x-y-z order),
- Three.js pose (`position`, `quaternion` in x-y-z-w order),
- air data,
- control-surface deflections and normalised actions,
- the wind split into mean, turbulence, gust and total, in all three frames,
- landing-gear loads,
- the per-term reward.

The poses are converted as $R_{\text{target}}=W\,R_{nb}\,B$. The conversion matrices are
stored in the document:

| Target | World frame $W$ | Model axes $B$ |
|---|---|---|
| Blender | East-North-Up (Z-up) | nose $+X$, left wing $+Y$, up $+Z$ |
| Three.js | X east, Y up, Z south | glTF: nose $+Z$, up $+Y$, left wing $+X$ |

Quaternion signs are made continuous between frames, which gives smooth keyframe
interpolation.

Blender import:

```python
import bpy, json
doc = json.load(open("episode_00000.json"))
obj = bpy.data.objects["Aircraft"]; obj.rotation_mode = "QUATERNION"
fps = bpy.context.scene.render.fps
for f in doc["frames"]:
    frame = round(f["t"] * fps)
    obj.location = f["blender"]["location"];            obj.keyframe_insert("location", frame=frame)
    obj.rotation_quaternion = f["blender"]["rotation_quaternion"]; obj.keyframe_insert("rotation_quaternion", frame=frame)
```

Three.js playback:

```js
const f = doc.frames[i];
plane.position.fromArray(f.threejs.position);
plane.quaternion.set(...f.threejs.quaternion);   // x, y, z, w
```

---

## 6. Evaluation and figures (`src/viz/evaluate_and_plot.py`)

The script runs Monte-Carlo landings for each crosswind level, with a direct crosswind
(90°), Dryden turbulence and discrete gusts. It writes:

- `episodes_<policy>.csv` (one row per episode),
- `summary_<policy>.csv`,
- `summary_<policy>.tex`, an IEEE table,
- optionally, one telemetry JSON per episode.

It also produces these figures in PDF (TrueType fonts embedded) and 600-dpi PNG, at
IEEE column width:

| File | Content |
|---|---|
| `reward_convergence` | training return (rolling mean ± std, from Monitor logs) and deterministic eval return vs. steps |
| `trajectories_3d` | 3-D approach paths over the runway, coloured by crosswind (single-hue sequential scale) |
| `trajectory_profiles` | vertical profile vs. the 3° glideslope, and lateral track vs. the runway |
| `touchdown_dispersion` | touchdown points with 2σ covariance ellipses on the runway; crashes marked × |
| `control_effort` | density histograms of elevator, aileron, rudder and throttle per crosswind |
| `sink_rate_vs_crosswind` | box plots of touchdown $\dot h$ with the −1.5 m/s crash limit |

Reference run of the classical baseline autopilot (6 episodes per level, seed 12345):

| Crosswind [m/s] | 0 | 2.5 | 5 | 7.5 | 10 |
|---|---|---|---|---|---|
| Success rate | 100 % | 100 % | 100 % | 83 % | 50 % |
| Mean touchdown $\dot h$ [m/s] | −0.85 | −0.91 | −1.02 | −1.16 | −2.43 |

These results show that the task gets much harder at the top of the crosswind range, and
they are the baseline the learned policy should be compared against. Six episodes per
level is a small sample, so run `--episodes 20` or more before quoting numbers in a paper.

---

## 7. Cinematic teaser (`src/viz/blender_cinematic_renderer.py`)

This script builds a complete Blender scene from one telemetry episode and renders a
three-shot teaser. It needs Blender 4.2 or later and was developed and verified on
Blender 5.2 LTS. Everything is generated procedurally, so it needs no assets.

```bash
# 4K, 30 fps, lossless QuickTime Animation (QTRLE) + Remotion timeline sidecar
blender --background --python src/viz/blender_cinematic_renderer.py -- \
    --telemetry results/telemetry_demo/baseline_xw7.5_s1_00000.json --output renders/crosswind_teaser.mov

# fast look-dev pass: 720p, 16 samples, H.264, every 2nd frame
blender -b --python src/viz/blender_cinematic_renderer.py -- --telemetry ep.json --preview

# build only, save a .blend to inspect or render it by hand
blender -b --python src/viz/blender_cinematic_renderer.py -- --telemetry ep.json --no-render --save-blend teaser.blend
```

| Option | Default | Meaning |
|---|---|---|
| `--codec` | `qtrle` | `qtrle` (lossless RGB .mov), `prores` (ProRes 4444 XQ), `png` (lossless PNG-in-MOV), `ffv1` (lossless .mkv), `h264` (lossy proxy), `png-sequence` (16-bit PNG frames) |
| `--mood` | `golden` | `golden` (sun at 6.5° elevation, broken cloud deck, damp runway) or `storm` (overcast, low cloud deck, soaked reflective runway) |
| `--engine` | `EEVEE` | `CYCLES` also works; it uses the GPU when one is available |
| `--surface-gain` | `1.0` | multiplies control-surface deflections for readability (1 = physical) |
| `--surface-source` | `deflection` | `deflection` uses the actuator state; `action` uses the raw policy output × limits |
| `--pre-roll` / `--post-roll` | 14 / 3 s | how much of the episode before and after touchdown is used |
| `--frames START END` | full cut | renders a sub-range, for example to split work across machines |

**What comes from the telemetry**

- **Airframe pose:** keyframed at every 20 Hz simulation sample, with auto-clamped Bézier
  handles. The quaternions are made hemisphere-continuous first.
- **Control surfaces:** the ailerons, elevator and rudder rotate about their true hinge
  lines (axis-angle rotation) by the recorded deflections. The signs follow the
  simulator's conventions.
- **Landing gear:**
  - Each leg's coil spring and strut stretch between a fixed attach point and a hub. The
    hub moves by the geometric penetration of that wheel's contact point below the runway,
    computed from the recorded pose. This is the same contact model the simulator uses.
  - The wheels spin up from rest at their first contact.
  - The nose wheel steers with the rudder.
- **Propeller:** its speed follows the throttle. A faint disc stands in for the
  motion-blurred arc.
- **Anti-collision strobes:** double-flash pattern.
- **Tire smoke:** 36 volumetric puffs per main wheel are spawned at the interpolated
  contact instant. They expand and fade, and drift with the recorded near-ground wind,
  so the crosswind is visible.
- **Camera shake:** scales with the recorded Dryden turbulence plus discrete-gust
  magnitude. A decaying impulse is added at touchdown.

**Shots**

The three shots are bound to timeline markers. The camera sides are chosen from the wind
direction.

1. **Low-angle tracking dolly.** The camera is on the downwind side, below the aircraft
   and arcing in toward the nose. The crab and bank against the sky are clearly visible.
2. **Airframe-mounted close-up.** The camera looks forward past the elevator and rudder to
   the aileron, with depth of field focused on the tail.
3. **Lagging chase cam through touchdown.** The camera is on the upwind side, so the
   crosswind carries the tire smoke across frame and away from the lens.

**Look**

- **Sky:** physical (multiple-scattering) sky, matched to a warm sun lamp.
- **Atmosphere:** bounded volumetric haze with a 450 m scale height, and a volumetric
  cloud deck.
- **Materials:**
  - polished aluminium airframe with clear-coated navy/orange cheat-lines and panel seams;
  - asphalt with aggregate, rubber deposits in the touchdown zone and wet puddles;
  - weathered, eroded paint for the ICAO markings and the "36" designator.
- **Lights:** edge, threshold, end and approach lights, plus a PAPI on the aim point.
- **Terrain:** procedural terrain with a tree line.
- **Render settings:**
  - EEVEE ray-traced reflections (the replacement for screen-space reflections);
  - horizon-scan AO / GI;
  - volumetric shadows (god rays);
  - motion blur;
  - depth of field.
- **Compositor:** bloom (Glare), chromatic aberration, teal/orange grade and a vignette,
  using the node-group compositor of Blender 5 with a fallback for older versions.

**Remotion integration**

Next to the video the script writes `<name>.timeline.json`. It contains:

- `fps`, `width` and `height`;
- `duration_in_frames`;
- per-shot `from_frame` / `to_frame` and the camera name;
- `touchdown_frame` and `main_gear_contact_frames`;
- the episode metrics.

You can feed this straight into a `<Composition>` and use it to time overlays.

**Storage**

QTRLE at 4K is about 20 MB per frame, so the full 511-frame cut is roughly 10 GB. Use
`--codec prores` for files about 4× smaller, or `--codec png-sequence` if you will
transcode in Remotion.

**EEVEE (Blender 4.2+) gotchas the script handles**

- A *world* volume is treated as an infinite atmosphere that absorbs all sky light, so the
  haze is a bounded mesh volume instead.
- The volume froxels are distributed over the camera clip range, so the chase camera uses
  a 1.2 km far clip to keep the small smoke puffs resolved.
- Bloom moved from the render settings to the compositor Glare node.

## 8. WebGL mission control (`src/web/`)

This is Stage 4. It connects the Blender scene (Stage 3) to an interactive console built
with React 19, three.js, @react-three/fiber/drei/postprocessing, Tailwind 4, Framer
Motion and canvas strip charts. The airframe is a procedural 737-800-class airliner with a fictional
livery (`src/viz/airliner_model.py`), rendered at true scale on the recorded light-aircraft telemetry;
press `F` for the wind and airflow overlay (streamlines computed in Blender, tracers, tip vortices).

```bash
# 1) Package the Blender scene as glTF. Builds the scene if needed. Writes scene.glb + scene.rig.json.
blender -b --python src/viz/export_scene_to_gltf.py -- \
    --telemetry results/telemetry_demo/baseline_xw7.5_s1_00000.json --output src/web/public/models/scene.glb

# 2) Roll out the scenario grid on the 6DoF simulator.
#    2 controllers × 7 crosswinds × 3 gust levels × 3 glidepaths = 126 rollouts.
python -m src.viz.export_flight_data --model runs/ppo_crosswind/best/best_model.zip --out src/web/public/data

# 3) Run the console
cd src/web && npm install && npm run dev        # http://127.0.0.1:5173
```

**What the GLB export does**

- **Materials:** glTF cannot store procedural shaders. The asphalt, grass and runway paint
  get seamless PBR texture sets (albedo, ORM, normal) made from FFT-filtered 1/f noise.
  Every other material becomes a constant metallic-roughness material; emissive lights
  use `KHR_materials_emissive_strength`.
- **Geometry:** curves and text become meshes, and the terrain and trees are decimated.
- **Rest pose:** the aircraft is exported at the origin with zero deflections. The node
  names `Aileron_L`, `Aileron_R`, `Elevator`, `Rudder`, `Hub_*`, `Strut_*`, `Spring_*`,
  `Wheel_*`, `Nose_Steer` and `Propeller` are kept.
- **Rig manifest:** `scene.rig.json` holds the hinge axes, deflection signs and gear rest
  geometry in the glTF frame.

**How the console uses the data**

- **Playback:** the console replays real simulator rollouts. It does not re-simulate in
  the browser. Every 20 Hz sample is interpolated per display frame: linear for channels,
  slerp for attitude. That one interpolated sample drives the airframe pose, the control
  surfaces (about their true hinge axes), the gear (struts and coil springs re-stretched
  to the geometric wheel penetration), the wheel spin-up, the propeller speed and the
  nose-wheel steering.
- **Sliders:** they select the nearest computed rollout. The snapped grid value is shown
  under each slider.
- **Tire smoke:** emitted at each gear's recorded contact, with burst size, particle size
  and opacity scaled by the touchdown sink rate Vz. It drifts with the recorded wind, and
  is rebuilt deterministically when you scrub the timeline.
- **Cameras:** CAM-01 runway-referenced chase (on the extended centreline, fixed attitude), CAM-02 touchdown-zone tracking (focal length
  follows distance), CAM-03 cockpit with a conformal HUD (the flight-path vector shows the
  crab angle), and CAM-04 free orbit. Switching between them is an eased 1.25 s
  interpolation of pose and field of view.
- **Post-processing:** bloom, depth of field focused on the airframe, vignette and AgX
  tone mapping. Visibility drops with gust intensity and crosswind through exponential fog.
- **Controls:**
  - `ENGAGE PPO AUTOPILOT` replays the PPO rollout for the selected conditions.
  - `MANUAL OVERRIDE` disengages playback and gives you the timeline (scrubber, ← → keys).
  - `RESET TRAJECTORY` returns to the start of the approach.
  - `EXPORT TELEMETRY JSON` downloads the full rollout.
  - The PPO / PID toggle compares against the baseline.
- **Keyboard:** `Space` play/pause, `1`–`4` cameras, `M` manual override, `R` reset.
- **Deep links:** `?ctrl=pid&xw=15&gust=50&gp=3&t=58.6&cam=1&paused=1&panels=0`.

**Airfield environment** (`src/viz/airfield_assets.py`). The GLB export uses a minimal WebGL profile by default:

- parallel taxiway and connectors, with yellow centrelines, hold-short markings and blue edge lights;
- a concrete apron (exported as a single flat slab);
- approach lights on masts, a windsock, an ILS localizer and a glideslope antenna;
- a ridged-noise mountain range with forest, rock and snow colouring.

The Blender scene also has three arched hangars, an MRO building, fuel tanks, floodlight masts and a chain-link perimeter fence. The WebGL profile leaves them out to cut clutter and cost during low-altitude flight. The cinematic renderer still uses them, and `--full-scene` exports them, with the light-baked apron, as before. The profile also joins world meshes that share an untextured material into one `Static_<material>` node each. That takes the world from 49 draw calls to 22, and from 217k triangles to 148k. Grass and runway-marking textures are exported at full resolution (1024²).

At runtime:

- The sequenced approach flashers run at 2 Hz.
- Each PAPI unit is red or white based on the aircraft's actual elevation angle.
- Runway light intensity steps up as visibility drops.
- The asphalt uses a physical material with a wet clear-coat.

**Texture export fix.** Blender 5.2's glTF exporter writes black pixels when it re-encodes generated in-memory images. Every procedural texture is therefore encoded to JPEG/PNG first and embedded as-is.

**Rendering performance.** The quality presets are selected in the Camera panel (or with `G`) and remembered per browser:

| Preset | Post-processing | Shadows | Pixel ratio (fixed, capped by the display) |
|---|---|---|---|
| `PERF` | none (renderer AgX tone mapping) | 1024² | 1.0 |
| `BAL` (default) | half-res bloom (5 mips), MSAA ×4, vignette, AgX | 2048² | up to 1.5 |
| `CINE` | bloom, depth of field, MSAA ×4, vignette, AgX | 2048² | up to 2.0 |

The pixel ratio is never scaled at runtime and never drops below 1.0, so the viewport stays sharp. The sun's 96 m shadow box is snapped to whole shadow texels as it follows the aircraft, so shadow edges do not crawl. Textures and shader programs are uploaded and compiled when the scene loads, so a camera switch never stalls on a first-time compile. The full-scene ContactShadows pass was removed; the sun's shadow map already draws the aircraft shadow. The footer shows the preset, the live pixel ratio and FPS.

**Ground and apron.**

- **Ground shader:** the asphalt, grass and concrete use `src/lib/groundShader.js`. It adds near-field detail (albedo and normal re-sampled at 5–8× and faded by distance) and a far-field macro layer (re-sampled at 0.05–0.1×) that hides tiling. Both are normalised by the texture's mean colour.

**60 Hz playback.** The 20 Hz telemetry is joined with uniform Catmull-Rom splines for position, attitude quaternion (sign-aligned, renormalised), Euler angles, rates, air data and control surfaces. It is exact at the samples and C¹-continuous across them. On real telemetry the velocity step at sample boundaries drops from 0.38 m/s (linear) to 0.0008 m/s. Bounded channels (gear loads, throttle, RTA authority) stay linear so the spline cannot overshoot them. The PFD and cockpit HUD re-render at 60 Hz. All overlays share one UI ticker. Strip charts (10 Hz), analytics readouts (12 Hz), the timeline and the status bar (15 Hz) render as React transitions, so React can pause them to let a WebGL frame through.

**Refreshing the PPO rollouts.** After training, re-run step 2. The footer shows the checkpoint's step count and hash.

## 9. Tests

Run `pytest`. All 40 tests pass. They check:

- **Kinematics:** the Euler–quaternion round trip, that $R_{nb}$ is orthonormal and matches
  the ZYX product, and quaternion integration through $\theta=90^\circ$.
- **Conservation:** for a torque-free asymmetric rigid body with products of inertia, the
  inertial angular momentum is conserved to $10^{-6}$ and the kinetic energy to $10^{-7}$
  over 20 s. The quaternion norm stays bounded without projection.
- **Convergence order:** the measured RK4 error ratio is about 16 when the step is
  halved (4th order); for forward Euler it is about 2 (1st order). A constant-force case
  is checked against the analytic solution.
- **Turbulence:** the Dryden discretisation matches `expm` and Van Loan's $Q_d$ and is
  stationary under the Lyapunov equation. Sampled standard deviations match the
  MIL-F-8785C $\sigma_{u,v,w}$ within 12 %.
- **Extreme gusts:**
  - The environment stays finite (unit quaternion, observations inside the observation
    space) under 25 m/s wind, 3× turbulence and 1 Hz gusts of at least 10 m/s with full
    vertical components, with random actions.
  - A 30 m/s vertical 1-cosine gust gives step-size-converged results.
  - The derivatives stay finite at $V_a=0$, with reversed flow and in 140 m/s gusts.
- **Ground:** the parked aircraft settles with the summed gear load equal to its weight
  within 1 %, and the load split satisfies the static moment balance. A 3 m/s sink-rate
  touchdown is classified as `hard_landing`.
- **Telemetry and reward:**
  - The frame matrices are proper rotations, and the Blender and Three.js poses are
    correct for known attitudes.
  - The exported JSON is strict.
  - The shaping term telescopes to $\Phi(s_T)-\Phi(s_0)$.
  - The touchdown reward orders landings correctly and applies the crash threshold.
  - The dense penalty stays within its bound.
