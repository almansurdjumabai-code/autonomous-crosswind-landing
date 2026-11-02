"""Numerical-integrity tests for the 6DoF flight dynamics.

Covers kinematic identities, conservation laws of the rigid-body equations,
the convergence order of the RK4 / Euler integrators, statistical correctness
of the Dryden turbulence filters, trim, ground-reaction equilibrium, and
numerical stability of the full environment under extreme wind gusts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.env.aircraft_env import (
    AIRCRAFT_PRESETS,
    G0,
    IDX_OMEGA,
    IDX_QUAT,
    IDX_VEL,
    STATE_DIM,
    CrosswindLandingEnv,
    DrydenTurbulence,
    EnvConfig,
    MeanWindProfile,
    euler_step,
    euler_to_quat,
    make_env_config,
    normalize_quaternion,
    quat_to_dcm,
    quat_to_euler,
    rigid_body_derivatives,
    rk4_step,
)

# An asymmetric inertia tensor with products of inertia (kg m^2), positive definite.
INERTIA = np.array([[1285.3, 0.0, -45.0], [0.0, 1824.9, 0.0], [-45.0, 0.0, 2666.9]])
INERTIA_INV = np.linalg.inv(INERTIA)
MASS = 1043.0


def _torque_free_state(omega=(1.2, -0.7, 2.1)) -> np.ndarray:
    x = np.zeros(STATE_DIM)
    x[IDX_VEL] = [30.0, 1.0, -2.0]
    x[IDX_QUAT] = euler_to_quat(0.3, -0.2, 1.0)
    x[IDX_OMEGA] = omega
    return x


def _free_dynamics(x: np.ndarray) -> np.ndarray:
    return rigid_body_derivatives(x, np.zeros(3), np.zeros(3), MASS, INERTIA, INERTIA_INV)


def _integrate(step, x0: np.ndarray, dt: float, t_end: float) -> np.ndarray:
    x = x0.copy()
    for _ in range(int(round(t_end / dt))):
        x = step(_free_dynamics, x, dt)
    return x


# ----------------------------------------------------------------------------- kinematics
@pytest.mark.parametrize("seed", range(5))
def test_euler_quaternion_roundtrip(seed):
    rng = np.random.default_rng(seed)
    for _ in range(200):
        phi = rng.uniform(-math.pi, math.pi)
        theta = rng.uniform(-1.5, 1.5)
        psi = rng.uniform(-math.pi, math.pi)
        q = euler_to_quat(phi, theta, psi)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)
        out = quat_to_euler(q)
        for a, b in zip((phi, theta, psi), out):
            assert abs(math.atan2(math.sin(a - b), math.cos(a - b))) < 1e-9


def test_dcm_is_proper_rotation_and_matches_euler_sequence():
    phi, theta, psi = 0.4, -0.3, 2.0
    r = quat_to_dcm(euler_to_quat(phi, theta, psi))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-12)
    rz = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])
    ry = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    rx = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]])
    assert np.allclose(r, rz @ ry @ rx, atol=1e-12)


def test_no_gimbal_lock_at_vertical_pitch():
    """Quaternion integration must pass through theta = 90 deg without blowing up."""
    x = np.zeros(STATE_DIM)
    x[IDX_QUAT] = euler_to_quat(0.0, math.radians(80.0), 0.0)
    x[IDX_OMEGA] = [0.0, 0.5, 0.0]  # pitch through the vertical
    for _ in range(200):
        x = normalize_quaternion(rk4_step(_free_dynamics, x, 0.01))
    assert np.all(np.isfinite(x))
    # 80 deg + 0.5 rad/s * 2 s = 137.3 deg pitch rotation about body y.
    r = quat_to_dcm(x[IDX_QUAT])
    angle = math.atan2(-r[2, 0], r[0, 0])
    assert math.isclose(angle, math.radians(80.0) + 1.0, abs_tol=1e-6)


# ----------------------------------------------------------------------------- conservation
def test_torque_free_body_conserves_angular_momentum_and_energy():
    x0 = _torque_free_state()
    t_end, dt = 20.0, 0.01

    def invariants(x):
        omega = x[IDX_OMEGA]
        h_inertial = quat_to_dcm(x[IDX_QUAT]) @ (INERTIA @ omega)
        energy = 0.5 * omega @ INERTIA @ omega
        return h_inertial, energy

    h0, e0 = invariants(x0)
    x = x0.copy()
    for _ in range(int(t_end / dt)):
        x = normalize_quaternion(rk4_step(_free_dynamics, x, dt))
    h1, e1 = invariants(x)
    assert np.linalg.norm(h1 - h0) / np.linalg.norm(h0) < 1e-6
    assert abs(e1 - e0) / e0 < 1e-7
    assert abs(np.linalg.norm(x[IDX_QUAT]) - 1.0) < 1e-12


def test_quaternion_norm_drift_is_bounded_without_projection():
    """The Baumgarte term alone must keep |q| near 1 over a long horizon."""
    x = _torque_free_state()
    for _ in range(5000):
        x = rk4_step(_free_dynamics, x, 0.01)
    assert abs(np.linalg.norm(x[IDX_QUAT]) - 1.0) < 1e-6


def test_constant_body_force_without_rotation_matches_analytic_solution():
    x0 = np.zeros(STATE_DIM)
    x0[IDX_QUAT] = euler_to_quat(0.0, 0.0, 0.0)
    x0[IDX_VEL] = [20.0, 0.0, 0.0]
    force = np.array([MASS * 2.0, -MASS * 1.0, MASS * 0.5])

    def f(x):
        return rigid_body_derivatives(x, force, np.zeros(3), MASS, INERTIA, INERTIA_INV)

    x = x0.copy()
    t = 3.0
    for _ in range(300):
        x = rk4_step(f, x, 0.01)
    a = force / MASS
    expected_pos = x0[IDX_VEL] * t + 0.5 * a * t * t
    assert np.allclose(x[0:3], expected_pos, atol=1e-9)
    assert np.allclose(x[IDX_VEL], x0[IDX_VEL] + a * t, atol=1e-10)


# ----------------------------------------------------------------------------- convergence
def test_rk4_is_fourth_order_and_euler_first_order():
    x0 = _torque_free_state()
    t_end = 2.0
    reference = _integrate(rk4_step, x0, 1e-4, t_end)

    def err(step, dt):
        return np.linalg.norm(_integrate(step, x0, dt, t_end)[IDX_OMEGA] - reference[IDX_OMEGA])

    rk_ratio = err(rk4_step, 0.04) / err(rk4_step, 0.02)
    eu_ratio = err(euler_step, 0.004) / err(euler_step, 0.002)
    assert 13.0 < rk_ratio < 19.0, rk_ratio  # 2^4 = 16
    assert 1.7 < eu_ratio < 2.3, eu_ratio  # 2^1 = 2


# ----------------------------------------------------------------------------- atmosphere
def test_wind_shear_profile():
    w = MeanWindProfile(10.0, math.radians(90.0), 1.0 / 7.0, math.radians(10.0))
    assert math.isclose(w.speed(MeanWindProfile.REF_HEIGHT_M), 10.0)
    assert w.speed(100.0) > w.speed(10.0) > w.speed(1.0) > 0.0
    v = w.velocity_ned(MeanWindProfile.REF_HEIGHT_M)
    # Wind from 90 deg (from the right of the runway) blows towards -y.
    assert np.allclose(v, [0.0, -10.0, 0.0], atol=1e-12)
    assert w.direction_from(106.096) - w.direction_from(6.096) == pytest.approx(math.radians(10.0))


def test_dryden_second_order_discretization_properties():
    tau, dt = 0.8, 0.05
    phi, qd = DrydenTurbulence.second_order_discretization(tau, dt)
    a = np.array([[0.0, 1.0], [-1.0 / tau**2, -2.0 / tau]])
    from scipy.linalg import expm, solve_continuous_lyapunov

    assert np.allclose(phi, expm(a * dt), atol=1e-12)
    # Van Loan's method for the exact discrete process-noise covariance.
    b = np.array([[0.0], [1.0]])
    m = np.block([[-a, b @ b.T], [np.zeros((2, 2)), a.T]]) * dt
    em = expm(m)
    qd_exact = em[2:, 2:].T @ em[:2, 2:]
    assert np.allclose(qd, qd_exact, rtol=1e-10, atol=1e-14)
    # Stationary covariance P = phi P phi^T + Qd must match the Lyapunov solution.
    p_cont = solve_continuous_lyapunov(a, -b @ b.T)
    assert np.allclose(phi @ p_cont @ phi.T + qd, p_cont, atol=1e-12)


def test_dryden_turbulence_has_specified_variance():
    rng = np.random.default_rng(0)
    turb = DrydenTurbulence()
    h, v, w20, dt, n = 20.0, 30.0, 10.0, 0.1, 60000
    _, _, _, s_u, s_v, s_w = DrydenTurbulence.parameters(h, w20)
    samples = np.empty((n, 3))
    for i in range(n):
        samples[i] = turb.step(h, v, w20, 0.0, dt, rng)  # wind from north -> u_g along -x
    samples = samples[2000:]
    std = samples.std(axis=0)
    assert std[0] == pytest.approx(s_u, rel=0.12)
    assert std[1] == pytest.approx(s_v, rel=0.12)
    assert std[2] == pytest.approx(s_w, rel=0.12)
    assert np.all(np.abs(samples.mean(axis=0)) < 0.2 * np.array([s_u, s_v, s_w]))


# ----------------------------------------------------------------------------- aircraft model
PRESETS = tuple(AIRCRAFT_PRESETS)


@pytest.fixture(scope="module", params=PRESETS)
def env(request) -> CrosswindLandingEnv:
    return CrosswindLandingEnv(make_env_config(request.param))


def test_trim_is_an_equilibrium(env):
    trim = env.trim
    assert trim["residual_norm"] < 1e-8
    assert 0.0 < trim["throttle"] < 1.0
    assert abs(trim["elevator"]) < env.cfg.aircraft.elevator_limit_rad
    assert 0.0 < trim["alpha"] < env.cfg.aircraft.alpha_stall_rad


def test_derivatives_finite_in_pathological_air_data(env):
    """Zero airspeed, reversed flow and hurricane-force wind must not produce NaN."""
    env.reset(seed=0, options={"wind_speed_mps": 0.0})
    cases = [
        (np.zeros(3), np.zeros(3)),  # aircraft moving with the air mass: Va = 0
        (np.array([-30.0, 0.0, 0.0]), np.zeros(3)),  # tail-first flow, alpha = 180 deg
        (np.array([30.0, 0.0, 0.0]), np.array([0.0, 120.0, -80.0])),  # extreme gust
    ]
    for vel_b, wind in cases:
        x = env.state.copy()
        x[IDX_VEL] = vel_b
        env._gust_turb_ned = wind
        d = env.derivatives(x, np.array([0.3, -0.3, 0.4, 1.0]))
        assert np.all(np.isfinite(d))
    env._gust_turb_ned = np.zeros(3)


def test_landing_gear_static_equilibrium(env):
    """Parked aircraft settles on its gear with the summed normal load equal to the weight."""
    env.reset(seed=0, options={"wind_speed_mps": 0.0, "turbulence": False, "gusts": False})
    ac = env.cfg.aircraft
    x = np.zeros(STATE_DIM)
    x[2] = -ac.wheel_height_m  # main gear struts just touching
    x[IDX_QUAT] = euler_to_quat(0.0, 0.0, 0.0)
    env.mean_wind = MeanWindProfile(0.0, 0.0, 1.0 / 7.0, 0.0)
    env._gust_turb_ned = np.zeros(3)
    ctrl = np.array([0.0, 0.0, 0.0, 0.0])
    for _ in range(1500):
        x = normalize_quaternion(rk4_step(lambda s: env.derivatives(s, ctrl), x, 0.01))
    assert np.all(np.isfinite(x))
    assert np.linalg.norm(x[IDX_VEL]) < 1e-3
    assert np.linalg.norm(x[IDX_OMEGA]) < 1e-3
    _, _, aux = env.total_forces(x, ctrl, np.zeros(3))
    total_load = sum(aux["gear_loads"])
    assert total_load == pytest.approx(ac.mass_kg * G0, rel=0.01)
    # Load split follows the static moment balance about the CG.
    nose, left, right = aux["gear_loads"]
    assert left == pytest.approx(right, rel=1e-6)
    assert nose < left


# ----------------------------------------------------------------------------- extreme gusts
def _extreme_gust_config() -> EnvConfig:
    return make_env_config(
        max_wind_speed_mps=25.0,
        turbulence_scale=3.0,
        gust_rate_hz=1.0,
        gust_amplitude_fraction=1.5,
        gust_min_amplitude_mps=10.0,
        gust_length_range_m=(10.0, 40.0),
        gust_vertical_fraction=1.0,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_environment_numerically_stable_under_extreme_gusts(seed):
    env = CrosswindLandingEnv(_extreme_gust_config())
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed, options={"wind_speed_mps": 25.0, "wind_from_deg": 90.0})
    steps = 0
    while steps < 1500:
        obs, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 4).astype(np.float32))
        steps += 1
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert np.all(np.isfinite(env.state))
        assert abs(np.linalg.norm(env.state[IDX_QUAT]) - 1.0) < 1e-9
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            assert info["crash_reason"] != "numerical_divergence"
            obs, _ = env.reset(seed=seed + 100 + steps, options={"wind_speed_mps": 25.0, "wind_from_deg": 90.0})


def test_integration_step_converges_during_extreme_discrete_gust():
    """A 30 m/s vertical gust is resolved: halving the physics step barely changes the state."""
    cfg = EnvConfig()
    env = CrosswindLandingEnv(cfg)
    env.reset(seed=3, options={"wind_speed_mps": 15.0, "turbulence": False, "gusts": False})
    x0 = env.state.copy()
    ctrl = env.actuators.copy()

    def simulate(dt: float) -> np.ndarray:
        x = x0.copy()
        t = 0.0
        while t < 2.0 - 1e-12:
            # 1-cosine gust, 30 m/s peak upward (negative NED z), 1.5 s duration.
            g = 15.0 * (1.0 - math.cos(2.0 * math.pi * min(t, 1.5) / 1.5))
            env._gust_turb_ned = np.array([0.0, 8.0, -g])
            x = normalize_quaternion(rk4_step(lambda s: env.derivatives(s, ctrl), x, dt))
            t += dt
        return x

    coarse = simulate(0.01)
    fine = simulate(0.005)
    env._gust_turb_ned = np.zeros(3)
    assert np.all(np.isfinite(coarse))
    # Zero-order-hold gust sampling differs between the two step sizes, so the
    # tolerance reflects the gust discretisation (first order), not RK4 error.
    assert np.linalg.norm(coarse[IDX_VEL] - fine[IDX_VEL]) < 0.5
    assert np.linalg.norm(coarse[IDX_OMEGA] - fine[IDX_OMEGA]) < 0.1


@pytest.mark.parametrize("aircraft", PRESETS)
def test_hard_landing_is_detected_as_crash(aircraft):
    env = CrosswindLandingEnv(make_env_config(aircraft))
    env.reset(seed=0, options={"wind_speed_mps": 0.0, "turbulence": False, "gusts": False})
    ac = env.cfg.aircraft
    env.state[0:3] = [300.0, 0.0, -(ac.wheel_height_m + 0.3)]  # wheels 30 cm above the aim point
    env.state[IDX_QUAT] = euler_to_quat(0.0, math.radians(3.0), 0.0)
    r = quat_to_dcm(env.state[IDX_QUAT])
    v = env.cfg.approach_speed_mps
    env.state[IDX_VEL] = r.T @ np.array([v, 0.0, 4.0])  # 4 m/s sink: beyond every preset's limit
    terminated = False
    for _ in range(40):
        _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
        if terminated:
            break
    assert terminated
    assert info["crash_reason"] == "hard_landing"
    assert info["episode_metrics"]["touchdown_sink_rate_mps"] < env.cfg.reward.crash_sink_rate_mps
    assert reward <= env.cfg.reward.crash_penalty


# ----------------------------------------------------------------------------- regressions
def test_dryden_output_stays_bounded_when_tau_changes_abruptly():
    """Regression: raw (non-normalised) filter states amplified the output by (tau_old/tau_new)^2
    whenever the advection speed changed, which diverged when V included the turbulence."""
    rng = np.random.default_rng(3)
    turb = DrydenTurbulence()
    h, w20, dt = 6.0, 10.0, 0.01
    _, _, _, s_u, s_v, s_w = DrydenTurbulence.parameters(h, w20)
    out = []
    for k in range(40000):
        v = 20.0 if (k // 50) % 2 == 0 else 110.0  # tau jumps by 5.5x every 0.5 s
        out.append(turb.step(h, v, w20, 0.0, dt, rng))
    out = np.array(out)
    assert np.all(np.isfinite(out))
    assert out.std(axis=0)[2] == pytest.approx(s_w, rel=0.25)
    assert np.abs(out).max() < 8.0 * max(s_u, s_v, s_w)


def test_severe_gust_scenario_that_previously_overflowed_completes():
    """Regression for the 20 kt / 100 % gust / 3.5 deg glidepath rollout that overflowed."""
    import warnings
    from dataclasses import replace as dc_replace

    from src.agents.baseline_autopilot import ClassicalAutopilot
    from src.env.aircraft_env import KT_TO_MPS

    cfg = EnvConfig()
    cfg.wind = dc_replace(cfg.wind, turbulence_scale=2.0, gust_rate_hz=0.15, gust_amplitude_fraction=0.8)
    cfg.runway = dc_replace(cfg.runway, glideslope_rad=math.radians(3.5))
    env = CrosswindLandingEnv(cfg)
    env.reset(seed=2026, options={"wind_speed_mps": 20 * KT_TO_MPS, "wind_from_deg": 90.0, "start_y_m": 0.0})
    ap = ClassicalAutopilot(env)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for _ in range(4000):
            _, reward, term, trunc, info = env.step(ap.act())
            assert np.isfinite(reward) and abs(reward) < 1000.0
            assert env._state_is_physical() or info["crash_reason"] == "numerical_divergence"
            if term or trunc:
                break
    assert term or trunc


def test_hard_landing_penalty_is_capped():
    from src.agents.reward_shaping import LandingRewardShaper, TouchdownEvent

    shaper = LandingRewardShaper()
    r_extreme, _ = shaper.touchdown_reward(TouchdownEvent(0, 300, 0, -5000.0, 0, 0, 0, 30, 0), 15.0, 300.0)
    c = shaper.config
    assert r_extreme == pytest.approx(c.crash_penalty - c.crash_excess_sink_scale * c.crash_excess_sink_cap_mps)


# ----------------------------------------------------------------------------- control stack
def _fly(env, controller, max_steps=4000):
    obs, _ = env.reset(seed=7, options={"wind_speed_mps": 12 * 0.514444, "wind_from_deg": 90.0, "start_y_m": 0.0})
    info = {}
    for _ in range(max_steps):
        obs, _, term, trunc, info = env.step(controller(obs))
        if term or trunc:
            break
    return info["episode_metrics"]


@pytest.mark.parametrize("aircraft", PRESETS)
def test_classical_autoland_lands_in_steady_crosswind_with_residual_crab(aircraft):
    from src.agents.baseline_autopilot import ClassicalAutopilot

    env = CrosswindLandingEnv(make_env_config(aircraft, turbulence_enabled=False, gusts_enabled=False))
    ap = None

    def ctrl(obs):
        nonlocal ap
        if ap is None or ap.env is not env:
            ap = ClassicalAutopilot(env)
        return ap.act()

    m = _fly(env, ctrl)
    assert m["success"] == 1.0
    assert env.cfg.reward.crash_sink_rate_mps < m["touchdown_sink_rate_mps"] < 0.0
    assert abs(m["touchdown_y_m"]) < 5.0
    assert abs(m["touchdown_crab_deg"]) < 10.0


@pytest.mark.parametrize("aircraft", PRESETS)
def test_safety_supervisor_recovers_an_adversarial_policy(aircraft):
    """A policy that commands full right roll and full nose-down must be overridden."""
    from src.agents.safety_supervisor import SafetySupervisor

    env = CrosswindLandingEnv(make_env_config(aircraft, turbulence_enabled=False, gusts_enabled=False))
    bad = lambda obs: np.array([1.0, 1.0, 0.0, -1.0], dtype=np.float32)  # noqa: E731
    holder = {}

    def ctrl(obs):
        if holder.get("env_t", 1e9) > env.time_s:  # new episode
            holder["rta"] = SafetySupervisor(env, bad)
        holder["env_t"] = env.time_s
        a = holder["rta"].act(obs)
        holder.setdefault("min_auth", 1.0)
        holder["min_auth"] = min(holder["min_auth"], holder["rta"].authority)
        return a

    m = _fly(env, ctrl)
    assert holder["min_auth"] == 0.0  # the supervisor took full authority
    assert m["crashed"] == 0.0
    assert m["success"] == 1.0


@pytest.mark.parametrize("aircraft", PRESETS)
def test_safety_supervisor_leaves_a_safe_policy_in_command(aircraft):
    from src.agents.baseline_autopilot import ClassicalAutopilot
    from src.agents.safety_supervisor import SafetySupervisor

    env = CrosswindLandingEnv(make_env_config(aircraft, turbulence_enabled=False, gusts_enabled=False))
    env.reset(seed=7, options={"wind_speed_mps": 0.0, "start_y_m": 0.0})
    good = ClassicalAutopilot(env)
    rta = SafetySupervisor(env, lambda obs: good.act())
    for _ in range(200):  # 10 s of stable approach, far above the flare gate
        env.step(rta.act(None))
    assert rta.authority == 1.0 and rta.reason is None


@pytest.mark.parametrize("aircraft", PRESETS)
def test_residual_mode_with_zero_action_reproduces_the_autoland(aircraft):
    env = CrosswindLandingEnv(make_env_config(aircraft, action_mode="residual", turbulence_enabled=False, gusts_enabled=False))
    m = _fly(env, lambda obs: np.zeros(4, dtype=np.float32))
    assert m["success"] == 1.0
