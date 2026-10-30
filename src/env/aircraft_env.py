"""6DoF crosswind landing environment (Gymnasium).

Conventions
-----------
* World frame: flat-Earth North-East-Down (NED) with origin at the runway
  threshold on the centreline. The runway is aligned with +x (heading 0), so
  ``x`` is the along-track distance past the threshold, ``y`` the lateral offset
  (positive right of centreline) and ``z`` is positive *down* (altitude
  ``h = -z``).
* Body frame: Forward-Right-Down (FRD) at the centre of gravity.
* Attitude is integrated as a unit quaternion (scalar first, Hamilton product,
  body -> NED) to avoid the Euler-angle singularity at theta = +/-90 deg; the
  Euler angles (phi, theta, psi, ZYX sequence) are derived from it.
* Control surface sign conventions follow the standard aerodynamic convention:
  positive elevator (trailing edge down) pitches nose down, positive aileron
  rolls right wing down, positive rudder (trailing edge left) yaws nose left.

Physical model
--------------
* Non-linear rigid-body equations of motion with full inertia tensor,
  integrated by classical 4th-order Runge-Kutta at ``physics_dt``.
* Aerodynamics: Cessna-172-class stability-derivative model with a smooth
  sigmoid blend to a flat-plate post-stall model, McCormick ground effect on
  induced drag, and a fixed-pitch-propeller thrust model.
* Atmosphere: ISA density, power-law mean wind with directional shear (veer)
  with altitude, MIL-F-8785C low-altitude Dryden turbulence (exactly
  discretised shaping filters) and Poisson-triggered "1-cosine" discrete gusts.
* Ground: three spring-damper landing-gear legs with smooth Coulomb rolling and
  cornering friction, nose-wheel steering, plus structural contact points
  (wingtips, tail) used for crash detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.optimize import least_squares

from src.agents.reward_shaping import (
    FlightSnapshot,
    LandingRewardShaper,
    RewardConfig,
    TouchdownEvent,
)

G0: float = 9.80665
RHO0: float = 1.225
FT_TO_M: float = 0.3048
KT_TO_MPS: float = 0.514444

STATE_DIM: int = 13  # x y z | u v w | q0 q1 q2 q3 | p q r
IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_QUAT = slice(6, 10)
IDX_OMEGA = slice(10, 13)


# =============================================================================
# Kinematic utilities
# =============================================================================
def euler_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZYX Euler angles -> unit quaternion [q0, q1, q2, q3] (body -> NED)."""
    cr, sr = math.cos(0.5 * phi), math.sin(0.5 * phi)
    cp, sp = math.cos(0.5 * theta), math.sin(0.5 * theta)
    cy, sy = math.cos(0.5 * psi), math.sin(0.5 * psi)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Unit quaternion -> ZYX Euler angles (phi, theta, psi)."""
    q0, q1, q2, q3 = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    phi = math.atan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
    s = max(-1.0, min(1.0, 2.0 * (q0 * q2 - q1 * q3)))
    theta = math.asin(s)
    psi = math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
    return phi, theta, psi


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Direction cosine matrix R_nb mapping body-frame vectors to NED."""
    q0, q1, q2, q3 = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
            [2.0 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2.0 * (q2 * q3 - q0 * q1)],
            [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
        ],
        dtype=np.float64,
    )


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3-vector cross product (scalar implementation; ~10x faster than np.cross for size 3)."""
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def isa_density(altitude_m: float) -> float:
    """ISA troposphere density [kg/m^3] at geopotential altitude (valid < 11 km)."""
    h = min(max(altitude_m, -500.0), 11000.0)
    return RHO0 * (1.0 - 2.25577e-5 * h) ** 4.2559


# =============================================================================
# Rigid-body dynamics and integration
# =============================================================================
def rigid_body_derivatives(
    state: np.ndarray,
    force_body: np.ndarray,
    moment_body: np.ndarray,
    mass: float,
    inertia: np.ndarray,
    inertia_inv: np.ndarray,
    quat_stabilization_gain: float = 1.0,
) -> np.ndarray:
    """Time derivative of the 13-element rigid-body state.

    Translational (body axes):   v_dot = F/m - omega x v
    Rotational (Euler):          omega_dot = I^-1 (M - omega x I omega)
    Attitude (quaternion):       q_dot = 1/2 q (x) [0, omega] + k (1 - |q|^2) q
    Navigation:                  p_dot = R_nb v

    ``force_body`` must already contain gravity. The last quaternion term is a
    Baumgarte-type constraint stabiliser that drives |q| -> 1 without altering the
    attitude dynamics on the constraint manifold.
    """
    vel = state[IDX_VEL]
    q = state[IDX_QUAT]
    omega = state[IDX_OMEGA]
    p, qr, r = float(omega[0]), float(omega[1]), float(omega[2])
    q0, q1, q2, q3 = float(q[0]), float(q[1]), float(q[2]), float(q[3])

    out = np.empty(STATE_DIM, dtype=np.float64)
    out[IDX_POS] = quat_to_dcm(q) @ vel
    out[IDX_VEL] = force_body / mass - cross3(omega, vel)
    norm_err = 1.0 - (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    k = quat_stabilization_gain * norm_err
    out[6] = 0.5 * (-q1 * p - q2 * qr - q3 * r) + k * q0
    out[7] = 0.5 * (q0 * p + q2 * r - q3 * qr) + k * q1
    out[8] = 0.5 * (q0 * qr - q1 * r + q3 * p) + k * q2
    out[9] = 0.5 * (q0 * r + q1 * qr - q2 * p) + k * q3
    out[IDX_OMEGA] = inertia_inv @ (moment_body - cross3(omega, inertia @ omega))
    return out


def rk4_step(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dt: float) -> np.ndarray:
    """One classical 4th-order Runge-Kutta step of x_dot = f(x)."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_step(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dt: float) -> np.ndarray:
    """Explicit forward-Euler step (provided for convergence-order comparisons)."""
    return x + dt * f(x)


def normalize_quaternion(state: np.ndarray) -> np.ndarray:
    """Project the quaternion part of ``state`` back onto the unit sphere (in place)."""
    q = state[IDX_QUAT]
    n = math.sqrt(float(q @ q))
    if n > 0.0 and math.isfinite(n):
        state[IDX_QUAT] = q / n
    return state


# =============================================================================
# Aircraft definition
# =============================================================================
@dataclass(frozen=True)
class GearLeg:
    """Spring-damper landing gear leg attached at ``position_body`` (FRD, m)."""

    name: str
    position_body: tuple[float, float, float]
    stiffness_n_per_m: float
    damping_ns_per_m: float
    is_main: bool
    steerable: bool = False


@dataclass(frozen=True)
class AircraftParameters:
    """Cessna-172-class general-aviation aircraft (Roskam / Stevens & Lewis data)."""

    mass_kg: float = 1043.0
    inertia_kgm2: tuple[tuple[float, float, float], ...] = (
        (1285.3, 0.0, 0.0),
        (0.0, 1824.9, 0.0),
        (0.0, 0.0, 2666.9),
    )
    wing_area_m2: float = 16.2
    span_m: float = 10.92
    chord_m: float = 1.49
    oswald_efficiency: float = 0.80
    wing_height_above_cg_m: float = 1.0

    # Longitudinal aerodynamics.
    CL0: float = 0.31
    CL_alpha: float = 5.143
    CL_q: float = 3.9
    CL_de: float = 0.43
    CD0: float = 0.031
    CD_beta: float = 0.17
    CD_de: float = 0.06
    alpha_stall_rad: float = 0.28
    stall_blend_rate: float = 50.0
    Cm0: float = -0.015
    Cm_alpha: float = -0.89
    Cm_q: float = -12.4
    Cm_de: float = -1.28

    # Lateral-directional aerodynamics.
    CY_beta: float = -0.31
    CY_p: float = -0.037
    CY_r: float = 0.21
    CY_dr: float = 0.187
    Cl_beta: float = -0.089
    Cl_p: float = -0.47
    Cl_r: float = 0.096
    Cl_da: float = 0.178
    Cl_dr: float = 0.0147
    Cn_beta: float = 0.065
    Cn_p: float = -0.03
    Cn_r: float = -0.099
    Cn_da: float = -0.053
    Cn_dr: float = -0.0657

    # Propulsion: T = dt * T_static * (rho/rho0) * max(0, 1 - Va / V_zero_thrust).
    thrust_static_n: float = 2200.0
    prop_zero_thrust_speed_mps: float = 110.0

    # Actuators: limits, first-order time constants, rate limits.
    elevator_limit_rad: float = math.radians(25.0)
    aileron_limit_rad: float = math.radians(20.0)
    rudder_limit_rad: float = math.radians(25.0)
    surface_time_constant_s: float = 0.06
    throttle_time_constant_s: float = 0.40
    surface_rate_limit_rps: float = math.radians(80.0)
    throttle_rate_limit_ps: float = 1.0

    # Ground handling.
    gear: tuple[GearLeg, ...] = (
        GearLeg("nose", (1.30, 0.0, 0.90), 30000.0, 3000.0, is_main=False, steerable=True),
        GearLeg("main_left", (-0.30, -1.20, 1.00), 40000.0, 4000.0, is_main=True),
        GearLeg("main_right", (-0.30, 1.20, 1.00), 40000.0, 4000.0, is_main=True),
    )
    structural_points: tuple[tuple[str, tuple[float, float, float]], ...] = (
        ("wingtip_left", (-0.30, -5.46, -0.60)),
        ("wingtip_right", (-0.30, 5.46, -0.60)),
        ("tail", (-4.50, 0.0, 0.0)),
    )
    rolling_friction: float = 0.03
    cornering_friction: float = 0.70
    friction_velocity_scale_mps: float = 0.30
    nose_steer_limit_rad: float = math.radians(10.0)

    @property
    def inertia(self) -> np.ndarray:
        return np.array(self.inertia_kgm2, dtype=np.float64)

    @property
    def aspect_ratio(self) -> float:
        return self.span_m**2 / self.wing_area_m2

    @property
    def control_limits(self) -> np.ndarray:
        return np.array(
            [self.elevator_limit_rad, self.aileron_limit_rad, self.rudder_limit_rad, 1.0], dtype=np.float64
        )


# =============================================================================
# Atmosphere: mean wind with shear, Dryden turbulence, discrete gusts
# =============================================================================
@dataclass
class WindConfig:
    """Wind / turbulence configuration. Speeds are referenced to 20 ft AGL (MIL-F-8785C)."""

    max_wind_speed_mps: float = 10.0
    wind_from_min_rad: float = math.radians(30.0)
    wind_from_max_rad: float = math.radians(150.0)
    shear_exponent: float = 1.0 / 7.0
    veer_rad_per_100m: float = math.radians(8.0)
    turbulence_enabled: bool = True
    turbulence_scale: float = 1.0
    gusts_enabled: bool = True
    gust_rate_hz: float = 0.05
    gust_amplitude_fraction: float = 0.6
    gust_min_amplitude_mps: float = 1.0
    gust_length_range_m: tuple[float, float] = (30.0, 120.0)
    gust_vertical_fraction: float = 0.3


class MeanWindProfile:
    """Power-law wind speed profile with linear directional veer with altitude.

        W(h)   = W20 * (max(h, h_min) / 6.096)^(1/7)
        chi(h) = chi20 + k_veer * (h - 6.096) / 100

    ``chi`` is the direction the wind blows *from*, measured clockwise from the
    runway heading. The returned vector is the air-mass velocity in NED.
    """

    REF_HEIGHT_M = 20.0 * FT_TO_M
    MIN_HEIGHT_M = 0.5

    def __init__(self, w20_mps: float, from_rad: float, exponent: float, veer_rad_per_100m: float) -> None:
        self.w20 = w20_mps
        self.from_rad = from_rad
        self.exponent = exponent
        self.veer = veer_rad_per_100m

    def speed(self, h: float) -> float:
        return self.w20 * (max(h, self.MIN_HEIGHT_M) / self.REF_HEIGHT_M) ** self.exponent

    def direction_from(self, h: float) -> float:
        return self.from_rad + self.veer * (max(h, 0.0) - self.REF_HEIGHT_M) / 100.0

    def velocity_ned(self, h: float) -> np.ndarray:
        w = self.speed(h)
        chi = self.direction_from(h)
        return np.array([-w * math.cos(chi), -w * math.sin(chi), 0.0], dtype=np.float64)


class DrydenTurbulence:
    """MIL-F-8785C low-altitude Dryden turbulence with exactly discretised filters.

    Spectra (frozen-field, Taylor hypothesis, V = airspeed relative to the *mean* wind):

        H_u(s) ~ 1 / (1 + tau_u s)
        H_v(s) ~ (1 + sqrt(3) tau_v s) / (1 + tau_v s)^2,     tau_i = L_i / V
        H_w(s) ~ (1 + sqrt(3) tau_w s) / (1 + tau_w s)^2

    Scale lengths and intensities (h in ft, clamped to [10, 1000] ft):

        L_w = h,  L_u = L_v = h / (0.177 + 0.000823 h)^1.2
        sigma_w = 0.1 W20,  sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h)^0.4

    The white-noise intensity of each filter is chosen so that the stationary
    output variance equals sigma_i^2 exactly: for the first-order filter the
    exact Gauss-Markov recursion is used; for the second-order filters the
    transition matrix uses the closed-form exponential of the repeated-pole
    system and the process-noise covariance is integrated by 16-point
    Gauss-Legendre quadrature. Components are generated in the mean-wind frame
    (u along the wind, v across, w down) and rotated into NED.

    Because tau = L / V varies continuously with altitude and airspeed, the
    second-order filter states are stored in tau-normalised coordinates
    s = (x1 / tau^2, x2 / tau), whose stationary covariance is independent of
    tau, so the output y = s1 + sqrt(3) s2 stays variance-consistent when tau
    changes. (Storing raw states would scale the output by (tau_old / tau_new)^2
    on every change of tau, which -- with V including the turbulence itself --
    forms a positive feedback loop.) V must therefore exclude the turbulence:
    callers pass the airspeed relative to the mean wind.
    """

    _GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)

    def __init__(self) -> None:
        self.x_u = 0.0
        self.x_v = np.zeros(2)
        self.x_w = np.zeros(2)

    def reset(self) -> None:
        self.x_u = 0.0
        self.x_v[:] = 0.0
        self.x_w[:] = 0.0

    @staticmethod
    def parameters(h_m: float, w20_mps: float) -> tuple[float, float, float, float, float, float]:
        """Return (L_u, L_v, L_w, sigma_u, sigma_v, sigma_w) in SI units."""
        h_ft = min(max(h_m / FT_TO_M, 10.0), 1000.0)
        denom = 0.177 + 0.000823 * h_ft
        l_w = h_ft * FT_TO_M
        l_uv = h_ft / denom**1.2 * FT_TO_M
        sigma_w = 0.1 * w20_mps
        sigma_uv = sigma_w / denom**0.4
        return l_uv, l_uv, l_w, sigma_uv, sigma_uv, sigma_w

    @classmethod
    def second_order_discretization(cls, tau: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Transition matrix and unit-intensity noise covariance of
        x_dot = [[0, 1], [-1/tau^2, -2/tau]] x + [0, 1]^T eta over ``dt``."""
        a = 1.0 / tau

        def expm_b(s: np.ndarray) -> np.ndarray:  # e^{A s} B for each s, shape (2, n)
            e = np.exp(-a * s)
            return np.vstack((e * s, e * (1.0 - a * s)))

        e = math.exp(-a * dt)
        phi = e * np.array([[1.0 + a * dt, dt], [-a * a * dt, 1.0 - a * dt]])
        s = 0.5 * dt * (cls._GL_NODES + 1.0)
        g = expm_b(s)
        qd = 0.5 * dt * (g * cls._GL_WEIGHTS) @ g.T
        return phi, qd

    def _step_second_order(self, s: np.ndarray, tau: float, sigma: float, dt: float, rng: np.random.Generator) -> float:
        # Output y = (1/tau^2) x1 + (sqrt(3)/tau) x2 has variance q / tau for
        # noise intensity q, hence q = sigma^2 * tau. ``s`` holds the normalised
        # states (x1 / tau^2, x2 / tau); convert with the current tau, propagate, convert back.
        x = np.array([s[0] * tau * tau, s[1] * tau])
        phi, qd = self.second_order_discretization(tau, dt)
        q = sigma * sigma * tau
        cov = q * qd
        # Cholesky of a 2x2 SPD matrix with a tiny diagonal floor for robustness.
        l11 = math.sqrt(max(cov[0, 0], 1e-300))
        l21 = cov[1, 0] / l11
        l22 = math.sqrt(max(cov[1, 1] - l21 * l21, 0.0))
        n1, n2 = rng.standard_normal(2)
        x_new = phi @ x + np.array([l11 * n1, l21 * n1 + l22 * n2])
        s[0] = x_new[0] / (tau * tau)
        s[1] = x_new[1] / tau
        return s[0] + math.sqrt(3.0) * s[1]

    def step(
        self, h_m: float, airspeed_mps: float, w20_mps: float, wind_from_rad: float, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Advance the filters by ``dt`` and return the turbulence velocity in NED."""
        if w20_mps <= 0.0:
            self.reset()
            return np.zeros(3)
        v = min(max(airspeed_mps, 5.0), 120.0)
        l_u, l_v, l_w, s_u, s_v, s_w = self.parameters(h_m, w20_mps)

        tau_u = l_u / v
        rho = math.exp(-dt / tau_u)
        self.x_u = rho * self.x_u + s_u * math.sqrt(1.0 - rho * rho) * rng.standard_normal()
        u_g = self.x_u
        v_g = self._step_second_order(self.x_v, l_v / v, s_v, dt, rng)
        w_g = self._step_second_order(self.x_w, l_w / v, s_w, dt, rng)

        # Mean-wind frame -> NED. The wind blows *towards* chi + pi.
        dx, dy = -math.cos(wind_from_rad), -math.sin(wind_from_rad)
        return np.array([u_g * dx - v_g * dy, u_g * dy + v_g * dx, w_g])


@dataclass
class _Gust:
    start_s: float
    duration_s: float
    amplitude_ned: np.ndarray


class DiscreteGustGenerator:
    """Poisson-triggered MIL-F-8785C '1-cosine' discrete gusts.

    Each gust contributes v(t) = (A/2) (1 - cos(2 pi (t - t0) / T)) for
    t0 <= t <= t0 + T, with T = d_m / V and a random 3-D direction.
    """

    def __init__(self, config: WindConfig) -> None:
        self.config = config
        self.active: list[_Gust] = []

    def reset(self) -> None:
        self.active.clear()

    def step(self, t: float, airspeed_mps: float, w20_mps: float, dt: float, rng: np.random.Generator) -> np.ndarray:
        c = self.config
        if c.gust_rate_hz > 0.0 and rng.random() < 1.0 - math.exp(-c.gust_rate_hz * dt):
            amp_max = max(c.gust_min_amplitude_mps, c.gust_amplitude_fraction * w20_mps)
            amp = rng.uniform(0.3, 1.0) * amp_max
            az = rng.uniform(-math.pi, math.pi)
            vert = rng.uniform(-c.gust_vertical_fraction, c.gust_vertical_fraction)
            direction = np.array([math.cos(az), math.sin(az), vert])
            direction /= np.linalg.norm(direction)
            length = rng.uniform(*c.gust_length_range_m)
            self.active.append(_Gust(t, length / max(airspeed_mps, 10.0), amp * direction))

        total = np.zeros(3)
        alive: list[_Gust] = []
        for g in self.active:
            tau = t - g.start_s
            if tau <= g.duration_s:
                total += 0.5 * (1.0 - math.cos(2.0 * math.pi * tau / g.duration_s)) * g.amplitude_ned
                alive.append(g)
        self.active = alive
        return total


# =============================================================================
# Environment configuration
# =============================================================================
@dataclass
class RunwayConfig:
    length_m: float = 1600.0
    width_m: float = 30.0
    aim_point_m: float = 300.0
    glideslope_rad: float = math.radians(3.0)
    field_elevation_m: float = 0.0


@dataclass
class EnvConfig:
    aircraft: AircraftParameters = field(default_factory=AircraftParameters)
    runway: RunwayConfig = field(default_factory=RunwayConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    physics_dt: float = 0.01
    control_dt: float = 0.05
    max_episode_s: float = 150.0
    rollout_time_s: float = 3.0

    approach_speed_mps: float = 33.0
    start_distance_range_m: tuple[float, float] = (1500.0, 2200.0)
    start_lateral_range_m: tuple[float, float] = (-40.0, 40.0)
    start_height_sigma_m: float = 6.0
    start_heading_range_rad: tuple[float, float] = (math.radians(-8.0), math.radians(8.0))
    start_roll_range_rad: tuple[float, float] = (math.radians(-5.0), math.radians(5.0))
    start_airspeed_range_mps: tuple[float, float] = (-2.0, 2.0)

    # Actions are offsets from the trimmed glide controls when True.
    trim_referenced_actions: bool = True
    # "absolute": the policy commands the surfaces directly.
    # "residual": the policy adds a bounded correction to the classical crosswind autoland,
    #   a_total = clip(a_autoland + residual_scale * a_policy), so learning starts from a
    #   competent controller and the action scale matches the size of useful corrections.
    action_mode: str = "absolute"
    residual_scale: float = 0.3

    # Flight-envelope limits that terminate the episode as a crash.
    max_bank_rad: float = math.radians(70.0)
    max_pitch_rad: float = math.radians(40.0)
    min_airspeed_mps: float = 18.0
    max_airspeed_mps: float = 70.0
    max_lateral_m: float = 400.0
    max_height_m: float = 500.0
    min_fuselage_height_m: float = 0.25


# =============================================================================
# Environment
# =============================================================================
class CrosswindLandingEnv(gym.Env):
    """Autonomous crosswind landing with 6DoF flight dynamics.

    Action (Box[-1, 1]^4): [elevator, aileron, rudder, throttle].
    Observation (Box, 23): normalised position, velocity, attitude, rates, air
    data (Va, alpha, beta), glideslope error, vertical and lateral speed,
    actuator positions and weight-on-wheels flag (see ``OBS_LABELS``).
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 20}

    OBS_LABELS: tuple[str, ...] = (
        "x/2000", "y/100", "h/100",
        "u/40", "v/10", "w/10",
        "phi/(pi/4)", "theta/(pi/6)", "psi_err/(pi/4)",
        "p", "q", "r",
        "(Va-Vref)/10", "alpha/0.2", "beta/0.2",
        "gs_err/20", "h_dot/5", "y_dot/10",
        "elevator", "aileron", "rudder", "throttle",
        "weight_on_wheels",
    )

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        ratio = self.cfg.control_dt / self.cfg.physics_dt
        self.n_substeps = int(round(ratio))
        if self.n_substeps < 1 or abs(ratio - self.n_substeps) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of physics_dt")

        ac = self.cfg.aircraft
        self.mass = ac.mass_kg
        self.inertia = ac.inertia
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.limits = ac.control_limits
        self._gear_r = [np.array(g.position_body, dtype=np.float64) for g in ac.gear]
        self._struct_r = [(n, np.array(r, dtype=np.float64)) for n, r in ac.structural_points]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(len(self.OBS_LABELS),), dtype=np.float32)

        self.reward_shaper = LandingRewardShaper(self.cfg.reward)
        self.turbulence = DrydenTurbulence()
        self.gusts = DiscreteGustGenerator(self.cfg.wind)
        self.wind_severity: float = 1.0

        self.trim = self.compute_trim(self.cfg.approach_speed_mps, -self.cfg.runway.glideslope_rad, 100.0)

        # Episode state (initialised in reset()).
        self.state = np.zeros(STATE_DIM)
        self.actuators = np.zeros(4)
        self.command = np.zeros(4)
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.time_s = 0.0
        self.mean_wind = MeanWindProfile(0.0, 0.0, self.cfg.wind.shear_exponent, self.cfg.wind.veer_rad_per_100m)
        self._gust_turb_ned = np.zeros(3)
        self._turb_ned = np.zeros(3)
        self._gust_ned = np.zeros(3)
        self.touchdown: Optional[TouchdownEvent] = None
        self.touched_down = False
        self.crashed = False
        self.crash_reason: Optional[str] = None
        self.success = False
        self._td_reported = False
        self._touchdown_time = 0.0
        self._residual_ap = None
        self._aux: dict[str, Any] = {}
        self.episode_return = 0.0

    # ------------------------------------------------------------------ public
    def set_wind_severity(self, severity: float) -> None:
        """Curriculum hook: scales the maximum sampled wind speed (0..1)."""
        self.wind_severity = float(np.clip(severity, 0.0, 1.0))

    def get_wind_severity(self) -> float:
        return self.wind_severity

    # --------------------------------------------------------------- physics
    def _aero_propulsion(
        self, vel_b: np.ndarray, omega: np.ndarray, r_nb: np.ndarray, ctrl: np.ndarray, wind_ned: np.ndarray, h: float
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
        """Aerodynamic + propulsive force and moment in body axes.

        Returns (F_b, M_b, (Va, alpha, beta, qbar))."""
        ac = self.cfg.aircraft
        v_rel = vel_b - r_nb.T @ wind_ned
        ur, vr, wr = float(v_rel[0]), float(v_rel[1]), float(v_rel[2])
        va = math.sqrt(ur * ur + vr * vr + wr * wr)
        va_s = max(va, 1.0)
        alpha = math.atan2(wr, ur)
        beta = math.asin(max(-1.0, min(1.0, vr / va_s)))
        rho = isa_density(h + self.cfg.runway.field_elevation_m)
        qbar = 0.5 * rho * va * va

        de, da, dr, dt = float(ctrl[0]), float(ctrl[1]), float(ctrl[2]), float(ctrl[3])
        p, q, r = float(omega[0]), float(omega[1]), float(omega[2])
        b, c, s = ac.span_m, ac.chord_m, ac.wing_area_m2
        p_hat = p * b / (2.0 * va_s)
        q_hat = q * c / (2.0 * va_s)
        r_hat = r * b / (2.0 * va_s)

        # Smooth stall blending (Beard & McLain, 2012).
        m, a0 = ac.stall_blend_rate, ac.alpha_stall_rad
        e1 = math.exp(max(-60.0, min(60.0, -m * (alpha - a0))))
        e2 = math.exp(max(-60.0, min(60.0, m * (alpha + a0))))
        sigma = (1.0 + e1 + e2) / ((1.0 + e1) * (1.0 + e2))
        sa, ca = math.sin(alpha), math.cos(alpha)
        cl_lin = ac.CL0 + ac.CL_alpha * alpha
        cl_alpha = (1.0 - sigma) * cl_lin + sigma * (2.0 * math.copysign(1.0, alpha) * sa * sa * ca)
        cl = cl_alpha + ac.CL_q * q_hat + ac.CL_de * de

        # Induced drag with McCormick ground-effect factor.
        h_wing = max(h + ac.wing_height_above_cg_m, 0.01)
        g = (16.0 * h_wing / b) ** 2
        ground_effect = g / (1.0 + g)
        cd_attached = ac.CD0 + ground_effect * cl_lin * cl_lin / (math.pi * ac.oswald_efficiency * ac.aspect_ratio)
        cd = (1.0 - sigma) * cd_attached + sigma * (ac.CD0 + 2.0 * sa * sa)
        cd += ac.CD_beta * beta * beta + ac.CD_de * abs(de)

        cy = ac.CY_beta * beta + ac.CY_p * p_hat + ac.CY_r * r_hat + ac.CY_dr * dr
        cl_roll = ac.Cl_beta * beta + ac.Cl_p * p_hat + ac.Cl_r * r_hat + ac.Cl_da * da + ac.Cl_dr * dr
        cm = ac.Cm0 + ac.Cm_alpha * alpha + ac.Cm_q * q_hat + ac.Cm_de * de
        cn = ac.Cn_beta * beta + ac.Cn_p * p_hat + ac.Cn_r * r_hat + ac.Cn_da * da + ac.Cn_dr * dr

        qs = qbar * s
        thrust = dt * ac.thrust_static_n * (rho / RHO0) * max(0.0, 1.0 - va / ac.prop_zero_thrust_speed_mps)
        force = np.array(
            [
                qs * (-cd * ca + cl * sa) + thrust,
                qs * cy,
                qs * (-cd * sa - cl * ca),
            ]
        )
        moment = np.array([qs * b * cl_roll, qs * c * cm, qs * b * cn])
        return force, moment, (va, alpha, beta, qbar)

    def _gear_forces(
        self, pos: np.ndarray, vel_b: np.ndarray, omega: np.ndarray, r_nb: np.ndarray, rudder: float
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Spring-damper gear reactions with smooth Coulomb friction. Returns (F_b, M_b, normal loads)."""
        ac = self.cfg.aircraft
        force = np.zeros(3)
        moment = np.zeros(3)
        loads: list[float] = []
        hx, hy = float(r_nb[0, 0]), float(r_nb[1, 0])
        hn = math.hypot(hx, hy)
        if hn < 1e-9:
            hx, hy, hn = 1.0, 0.0, 1.0
        hx, hy = hx / hn, hy / hn
        for leg, r in zip(ac.gear, self._gear_r):
            penetration = float(pos[2] + r_nb[2] @ r)
            if penetration <= 0.0:
                loads.append(0.0)
                continue
            v_c_ned = r_nb @ (vel_b + cross3(omega, r))
            f_n = leg.stiffness_n_per_m * penetration + leg.damping_ns_per_m * float(v_c_ned[2])
            if f_n <= 0.0:
                loads.append(0.0)
                continue
            wx, wy = hx, hy
            if leg.steerable:
                steer = -ac.nose_steer_limit_rad * rudder / ac.rudder_limit_rad
                cs, ss = math.cos(steer), math.sin(steer)
                wx, wy = hx * cs - hy * ss, hx * ss + hy * cs
            lx, ly = -wy, wx
            v_long = float(v_c_ned[0]) * wx + float(v_c_ned[1]) * wy
            v_lat = float(v_c_ned[0]) * lx + float(v_c_ned[1]) * ly
            eps = ac.friction_velocity_scale_mps
            f_long = -ac.rolling_friction * f_n * math.tanh(v_long / eps)
            f_lat = -ac.cornering_friction * f_n * math.tanh(v_lat / eps)
            f_b = r_nb.T @ np.array([f_long * wx + f_lat * lx, f_long * wy + f_lat * ly, -f_n])
            force += f_b
            moment += cross3(r, f_b)
            loads.append(f_n)
        return force, moment, loads

    def total_forces(
        self, x: np.ndarray, ctrl: np.ndarray, wind_ned: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """All external forces/moments (body axes) acting on the airframe at state ``x``."""
        r_nb = quat_to_dcm(x[IDX_QUAT])
        vel_b, omega = x[IDX_VEL], x[IDX_OMEGA]
        h = -float(x[2])
        f_a, m_a, air = self._aero_propulsion(vel_b, omega, r_nb, ctrl, wind_ned, h)
        f_g, m_g, loads = self._gear_forces(x[IDX_POS], vel_b, omega, r_nb, float(ctrl[2]))
        gravity_b = r_nb.T @ np.array([0.0, 0.0, self.mass * G0])
        aux = {"airdata": air, "gear_loads": loads, "aero_force_b": f_a, "gear_force_b": f_g}
        return f_a + f_g + gravity_b, m_a + m_g, aux

    def wind_at(self, h: float) -> np.ndarray:
        """Total wind (mean + held gust/turbulence) at altitude ``h``."""
        return self.mean_wind.velocity_ned(h) + self._gust_turb_ned

    def derivatives(self, x: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
        f, m, _ = self.total_forces(x, ctrl, self.wind_at(-float(x[2])))
        return rigid_body_derivatives(x, f, m, self.mass, self.inertia, self.inertia_inv)

    # -------------------------------------------------------------------- trim
    def compute_trim(self, airspeed: float, gamma: float, h: float) -> dict[str, float]:
        """Wings-level steady glide trim in still air: solves u_dot = w_dot = q_dot = 0
        for (alpha, elevator, throttle) at flight-path angle ``gamma``."""
        ac = self.cfg.aircraft
        saved = getattr(self, "_gust_turb_ned", None)
        self._gust_turb_ned = np.zeros(3)
        self.mean_wind = MeanWindProfile(0.0, 0.0, 0.0, 0.0)

        def residual(z: np.ndarray) -> np.ndarray:
            alpha, de, dt = z
            x = np.zeros(STATE_DIM)
            x[2] = -h
            x[3:6] = [airspeed * math.cos(alpha), 0.0, airspeed * math.sin(alpha)]
            x[6:10] = euler_to_quat(0.0, alpha + gamma, 0.0)
            d = self.derivatives(x, np.array([de, 0.0, 0.0, dt]))
            return np.array([d[3], d[5], d[11] * 10.0])

        sol = least_squares(
            residual,
            x0=np.array([0.08, -0.05, 0.3]),
            bounds=([-0.1, -ac.elevator_limit_rad, 0.0], [0.25, ac.elevator_limit_rad, 1.0]),
            xtol=1e-12,
            ftol=1e-12,
        )
        if saved is not None:
            self._gust_turb_ned = saved
        alpha, de, dt = (float(v) for v in sol.x)
        return {
            "alpha": alpha,
            "theta": alpha + gamma,
            "elevator": de,
            "throttle": dt,
            "airspeed": airspeed,
            "gamma": gamma,
            "residual_norm": float(np.linalg.norm(sol.fun)),
        }

    # ------------------------------------------------------------ action map
    def action_to_command(self, action: np.ndarray) -> np.ndarray:
        """Map a normalised action in [-1, 1]^4 to physical actuator commands."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.cfg.trim_referenced_actions:
            cmd = np.array(
                [
                    self.trim["elevator"] + a[0] * self.limits[0],
                    a[1] * self.limits[1],
                    a[2] * self.limits[2],
                    self._throttle_from_action(float(a[3])),
                ]
            )
        else:
            cmd = np.array([a[0] * self.limits[0], a[1] * self.limits[1], a[2] * self.limits[2], 0.5 * (a[3] + 1.0)])
        cmd[:3] = np.clip(cmd[:3], -self.limits[:3], self.limits[:3])
        cmd[3] = min(max(cmd[3], 0.0), 1.0)
        return cmd

    def command_to_action(self, cmd: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`action_to_command` (used by baseline controllers)."""
        if self.cfg.trim_referenced_actions:
            a = np.array(
                [
                    (cmd[0] - self.trim["elevator"]) / self.limits[0],
                    cmd[1] / self.limits[1],
                    cmd[2] / self.limits[2],
                    self._action_from_throttle(float(cmd[3])),
                ]
            )
        else:
            a = np.array([cmd[0] / self.limits[0], cmd[1] / self.limits[1], cmd[2] / self.limits[2], 2.0 * cmd[3] - 1.0])
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def _throttle_from_action(self, a: float) -> float:
        """Piecewise-linear map with a=0 -> trim throttle, a=-1 -> 0, a=+1 -> 1."""
        t = self.trim["throttle"]
        return t + a * (1.0 - t) if a >= 0.0 else t + a * t

    def _action_from_throttle(self, throttle: float) -> float:
        t = self.trim["throttle"]
        if throttle >= t:
            return (throttle - t) / max(1.0 - t, 1e-9)
        return (throttle - t) / max(t, 1e-9)

    def _update_actuators(self, dt: float) -> None:
        ac = self.cfg.aircraft
        taus = (ac.surface_time_constant_s,) * 3 + (ac.throttle_time_constant_s,)
        rates = (ac.surface_rate_limit_rps,) * 3 + (ac.throttle_rate_limit_ps,)
        for i in range(4):
            delta = (self.command[i] - self.actuators[i]) * (1.0 - math.exp(-dt / taus[i]))
            max_d = rates[i] * dt
            self.actuators[i] += max(-max_d, min(max_d, delta))

    # ---------------------------------------------------------- derived data
    def _kinematics(self) -> dict[str, float]:
        x = self.state
        phi, theta, psi = quat_to_euler(x[IDX_QUAT])
        r_nb = quat_to_dcm(x[IDX_QUAT])
        v_ned = r_nb @ x[IDX_VEL]
        rw = self.cfg.runway
        h = -float(x[2])
        dist_to_aim = rw.aim_point_m - float(x[0])
        h_ref = max(dist_to_aim, 0.0) * math.tan(rw.glideslope_rad)
        return {
            "x": float(x[0]),
            "y": float(x[1]),
            "h": h,
            "h_dot": -float(v_ned[2]),
            "x_dot": float(v_ned[0]),
            "y_dot": float(v_ned[1]),
            "phi": phi,
            "theta": theta,
            "psi": psi,
            "heading_error": wrap_angle(psi),
            "gs_error": h - h_ref,
            "dist_to_aim": dist_to_aim,
        }

    def _main_gear_contact(self) -> tuple[bool, bool]:
        r_nb = quat_to_dcm(self.state[IDX_QUAT])
        z = float(self.state[2])
        main = any(z + float(r_nb[2] @ r) > 0.0 for leg, r in zip(self.cfg.aircraft.gear, self._gear_r) if leg.is_main)
        nose = any(
            z + float(r_nb[2] @ r) > 0.0 for leg, r in zip(self.cfg.aircraft.gear, self._gear_r) if not leg.is_main
        )
        return main, nose

    def _structural_contact(self) -> Optional[str]:
        r_nb = quat_to_dcm(self.state[IDX_QUAT])
        z = float(self.state[2])
        for name, r in self._struct_r:
            if z + float(r_nb[2] @ r) > 0.0:
                return f"{name}_strike"
        if -z < self.cfg.min_fuselage_height_m:
            return "fuselage_strike"
        return None

    # Bounds far outside any reachable flight condition of this airframe; exceeding them
    # means the integration has diverged even if every number is still finite.
    MAX_PHYSICAL_SPEED_MPS = 250.0
    MAX_PHYSICAL_RATE_RPS = 40.0
    MAX_PHYSICAL_ALTITUDE_M = 5000.0

    def _state_is_physical(self) -> bool:
        x = self.state
        if not np.all(np.isfinite(x)):
            return False
        return (
            float(np.linalg.norm(x[IDX_VEL])) < self.MAX_PHYSICAL_SPEED_MPS
            and float(np.max(np.abs(x[IDX_OMEGA]))) < self.MAX_PHYSICAL_RATE_RPS
            and abs(float(x[2])) < self.MAX_PHYSICAL_ALTITUDE_M
        )

    def _check_envelope(self, k: dict[str, float]) -> Optional[str]:
        c = self.cfg
        if not self._state_is_physical():
            return "numerical_divergence"
        va = self._aux.get("airdata", (c.approach_speed_mps,))[0]
        if abs(k["phi"]) > c.max_bank_rad:
            return "bank_limit"
        if abs(k["theta"]) > c.max_pitch_rad:
            return "pitch_limit"
        if not self.touched_down:
            if va < c.min_airspeed_mps:
                return "stall"
            if va > c.max_airspeed_mps:
                return "overspeed"
            if abs(k["y"]) > c.max_lateral_m:
                return "lateral_out_of_bounds"
            if k["h"] > c.max_height_m:
                return "altitude_out_of_bounds"
            if k["x"] > c.runway.length_m:
                return "long_landing"
        else:
            if abs(k["y"]) > 0.5 * c.runway.width_m:
                return "runway_excursion"
            if k["x"] > c.runway.length_m:
                return "runway_overrun"
        return None

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        c = self.cfg
        rng = self.np_random

        # Wind sampling (options allow deterministic evaluation sweeps).
        if "wind_speed_mps" in options:
            w20 = float(options["wind_speed_mps"])
        else:
            w20 = float(rng.uniform(0.0, c.wind.max_wind_speed_mps * self.wind_severity))
        if "wind_from_deg" in options:
            wind_from = math.radians(float(options["wind_from_deg"]))
        else:
            wind_from = float(rng.uniform(c.wind.wind_from_min_rad, c.wind.wind_from_max_rad))
            wind_from *= 1.0 if rng.random() < 0.5 else -1.0
        self.turbulence_enabled = bool(options.get("turbulence", c.wind.turbulence_enabled))
        self.gusts_enabled = bool(options.get("gusts", c.wind.gusts_enabled))
        self.mean_wind = MeanWindProfile(w20, wind_from, c.wind.shear_exponent, c.wind.veer_rad_per_100m)
        self.turbulence.reset()
        self.gusts.reset()
        self._gust_turb_ned = np.zeros(3)
        self._turb_ned = np.zeros(3)
        self._gust_ned = np.zeros(3)

        # Initial condition: on (perturbed) glideslope, trimmed, air-relative.
        x0 = -float(rng.uniform(*c.start_distance_range_m))
        x0 = float(options.get("start_x_m", x0))
        h0 = (c.runway.aim_point_m - x0) * math.tan(c.runway.glideslope_rad)
        h0 += float(np.clip(rng.normal(0.0, c.start_height_sigma_m), -3 * c.start_height_sigma_m, 3 * c.start_height_sigma_m))
        y0 = float(options.get("start_y_m", rng.uniform(*c.start_lateral_range_m)))
        psi0 = float(rng.uniform(*c.start_heading_range_rad))
        phi0 = float(rng.uniform(*c.start_roll_range_rad))
        va0 = c.approach_speed_mps + float(rng.uniform(*c.start_airspeed_range_mps))
        alpha0 = self.trim["alpha"]
        theta0 = self.trim["theta"]
        q0 = euler_to_quat(phi0, theta0, psi0)
        r_nb = quat_to_dcm(q0)
        v_air_b = np.array([va0 * math.cos(alpha0), 0.0, va0 * math.sin(alpha0)])
        v_b = v_air_b + r_nb.T @ self.mean_wind.velocity_ned(h0)

        self.state = np.zeros(STATE_DIM)
        self.state[IDX_POS] = [x0, y0, -h0]
        self.state[IDX_VEL] = v_b
        self.state[IDX_QUAT] = q0
        self.actuators = np.array([self.trim["elevator"], 0.0, 0.0, self.trim["throttle"]])
        self.command = self.actuators.copy()
        self.prev_action = self.command_to_action(self.command)
        self.time_s = 0.0
        self.touchdown = None
        self.touched_down = False
        self.crashed = False
        self.crash_reason = None
        self.success = False
        self._td_reported = False
        self._touchdown_time = 0.0
        self.episode_return = 0.0
        _, _, self._aux = self.total_forces(self.state, self.actuators, self.wind_at(h0))

        k = self._kinematics()
        self._residual_ap = None
        if c.action_mode == "residual":
            from src.agents.baseline_autopilot import ClassicalAutopilot  # local: avoids an import cycle

            self._residual_ap = ClassicalAutopilot(self)
        elif c.action_mode != "absolute":
            raise ValueError(f"unknown action_mode {c.action_mode!r}")
        self.reward_shaper.reset(self._snapshot(k, self.prev_action))
        return self._observation(k), self._info(k, None)

    def step(self, action: np.ndarray):
        c = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self._residual_ap is not None:
            base = np.asarray(self._residual_ap.act(c.control_dt), dtype=np.float32)
            self.command = self.action_to_command(np.clip(base + c.residual_scale * action, -1.0, 1.0))
        else:
            self.command = self.action_to_command(action)
        dt = c.physics_dt
        rng = self.np_random

        for _ in range(self.n_substeps):
            self._update_actuators(dt)
            h = -float(self.state[2])
            # Frozen-turbulence advection speed: airspeed relative to the mean wind only.
            r_nb = quat_to_dcm(self.state[IDX_QUAT])
            va_mean = float(np.linalg.norm(self.state[IDX_VEL] - r_nb.T @ self.mean_wind.velocity_ned(h)))
            self._turb_ned = (
                c.wind.turbulence_scale
                * self.turbulence.step(h, va_mean, self.mean_wind.w20, self.mean_wind.direction_from(h), dt, rng)
                if self.turbulence_enabled
                else np.zeros(3)
            )
            self._gust_ned = (
                self.gusts.step(self.time_s, va_mean, self.mean_wind.w20, dt, rng) if self.gusts_enabled else np.zeros(3)
            )
            self._gust_turb_ned = self._turb_ned + self._gust_ned
            ctrl = self.actuators.copy()
            self.state = normalize_quaternion(rk4_step(lambda s: self.derivatives(s, ctrl), self.state, dt))
            self.time_s += dt
            if not self._state_is_physical():
                self.crashed, self.crash_reason = True, "numerical_divergence"
                break
            _, _, self._aux = self.total_forces(self.state, ctrl, self.wind_at(-float(self.state[2])))

            main, nose = self._main_gear_contact()
            if not self.touched_down and (main or nose):
                k = self._kinematics()
                self.touched_down = True
                self.touchdown = TouchdownEvent(
                    time_s=self.time_s,
                    x_m=k["x"],
                    y_m=k["y"],
                    sink_rate_mps=k["h_dot"],
                    heading_error_rad=k["heading_error"],
                    roll_rad=k["phi"],
                    pitch_rad=k["theta"],
                    airspeed_mps=float(self._aux["airdata"][0]),
                    lateral_speed_mps=k["y_dot"],
                )
                self._touchdown_time = self.time_s
                td = self.touchdown
                if td.sink_rate_mps < c.reward.crash_sink_rate_mps:
                    self.crashed, self.crash_reason = True, "hard_landing"
                elif nose and not main:
                    self.crashed, self.crash_reason = True, "nose_gear_first"
                elif td.x_m < 0.0:
                    self.crashed, self.crash_reason = True, "undershoot"
                elif abs(td.y_m) > 0.5 * c.runway.width_m:
                    self.crashed, self.crash_reason = True, "runway_edge_touchdown"
            if not self.crashed:
                reason = self._structural_contact()
                if reason is not None:
                    self.crashed, self.crash_reason = True, reason
            if self.crashed:
                break

        k = self._kinematics()
        if not self.crashed:
            reason = self._check_envelope(k)
            if reason is not None:
                self.crashed, self.crash_reason = True, reason
        if self.touched_down and not self.crashed and self.time_s - self._touchdown_time >= c.rollout_time_s:
            self.success = True

        terminated = self.crashed or self.success
        truncated = (not terminated) and self.time_s >= c.max_episode_s
        new_touchdown = self.touchdown if (self.touchdown is not None and not self._td_reported) else None
        if new_touchdown is not None:
            self._td_reported = True
        snap = self._snapshot(k, action, touchdown=new_touchdown)
        breakdown = self.reward_shaper.compute(snap, terminated)
        reward = float(breakdown.total)
        self.episode_return += reward
        self.prev_action = action

        obs = self._observation(k)
        info = self._info(k, breakdown)
        if terminated or truncated:
            info["episode_metrics"] = self.episode_metrics()
        return obs, reward, terminated, truncated, info

    def episode_metrics(self) -> dict[str, Any]:
        td = self.touchdown
        return {
            "success": float(self.success),
            "crashed": float(self.crashed),
            "crash_reason": self.crash_reason or "",
            "touched_down": float(self.touched_down),
            "touchdown_sink_rate_mps": td.sink_rate_mps if td else float("nan"),
            "touchdown_x_m": td.x_m if td else float("nan"),
            "touchdown_y_m": td.y_m if td else float("nan"),
            "touchdown_crab_deg": math.degrees(td.heading_error_rad) if td else float("nan"),
            "touchdown_roll_deg": math.degrees(td.roll_rad) if td else float("nan"),
            "wind_speed_20ft_mps": self.mean_wind.w20,
            "crosswind_20ft_mps": self.mean_wind.w20 * math.sin(self.mean_wind.from_rad),
            "episode_return": self.episode_return,
            "episode_time_s": self.time_s,
        }

    # ----------------------------------------------------------- observation
    def _observation(self, k: dict[str, float]) -> np.ndarray:
        x = self.state
        va, alpha, beta, _ = self._aux["airdata"]
        main, nose = self._main_gear_contact()
        act_norm = self.command_to_action(self.actuators)
        obs = np.array(
            [
                k["x"] / 2000.0,
                k["y"] / 100.0,
                k["h"] / 100.0,
                x[3] / 40.0,
                x[4] / 10.0,
                x[5] / 10.0,
                k["phi"] / (math.pi / 4),
                k["theta"] / (math.pi / 6),
                k["heading_error"] / (math.pi / 4),
                x[10],
                x[11],
                x[12],
                (va - self.cfg.approach_speed_mps) / 10.0,
                alpha / 0.2,
                beta / 0.2,
                k["gs_error"] / 20.0,
                k["h_dot"] / 5.0,
                k["y_dot"] / 10.0,
                *act_norm,
                1.0 if (main or nose) else 0.0,
            ],
            dtype=np.float64,
        )
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _snapshot(self, k: dict[str, float], action: np.ndarray, touchdown: Optional[TouchdownEvent] = None) -> FlightSnapshot:
        va, alpha, beta, _ = self._aux["airdata"]
        main, nose = self._main_gear_contact()
        return FlightSnapshot(
            x_m=k["x"],
            y_m=k["y"],
            h_m=k["h"],
            h_dot_mps=k["h_dot"],
            lateral_speed_mps=k["y_dot"],
            glideslope_error_m=k["gs_error"],
            distance_to_aim_m=k["dist_to_aim"],
            heading_error_rad=k["heading_error"],
            phi_rad=k["phi"],
            theta_rad=k["theta"],
            p_rps=float(self.state[10]),
            q_rps=float(self.state[11]),
            r_rps=float(self.state[12]),
            airspeed_mps=float(va),
            airspeed_ref_mps=self.cfg.approach_speed_mps,
            alpha_rad=float(alpha),
            beta_rad=float(beta),
            action=np.asarray(action, dtype=np.float64),
            prev_action=np.asarray(self.prev_action, dtype=np.float64),
            weight_on_wheels=bool(main or nose),
            touched_down=self.touched_down,
            runway_half_width_m=0.5 * self.cfg.runway.width_m,
            aim_point_x_m=self.cfg.runway.aim_point_m,
            touchdown=touchdown,
            crashed=self.crashed,
            crash_reason=self.crash_reason,
            success=self.success,
        )

    def _info(self, k: dict[str, float], breakdown) -> dict[str, Any]:
        info: dict[str, Any] = {
            "time_s": self.time_s,
            "kinematics": k,
            "crash_reason": self.crash_reason,
            "touched_down": self.touched_down,
            "success": self.success,
        }
        if breakdown is not None:
            info["reward_terms"] = breakdown.to_dict()
        return info

    # ------------------------------------------------------------- telemetry
    def telemetry_frame(self) -> dict[str, Any]:
        """Complete physical snapshot of the current simulator state (SI units)."""
        x = self.state
        phi, theta, psi = quat_to_euler(x[IDX_QUAT])
        va, alpha, beta, qbar = self._aux["airdata"]
        h = -float(x[2])
        return {
            "t": self.time_s,
            "position_ned": x[IDX_POS].tolist(),
            "velocity_body": x[IDX_VEL].tolist(),
            "velocity_ned": (quat_to_dcm(x[IDX_QUAT]) @ x[IDX_VEL]).tolist(),
            "quaternion_ned_frd": x[IDX_QUAT].tolist(),
            "euler_rad": [phi, theta, psi],
            "body_rates_rps": x[IDX_OMEGA].tolist(),
            "airspeed_mps": float(va),
            "alpha_rad": float(alpha),
            "beta_rad": float(beta),
            "dynamic_pressure_pa": float(qbar),
            "controls": {
                "elevator_rad": float(self.actuators[0]),
                "aileron_rad": float(self.actuators[1]),
                "rudder_rad": float(self.actuators[2]),
                "throttle": float(self.actuators[3]),
                "command": self.command.tolist(),
                "normalized_action": np.asarray(self.prev_action, dtype=np.float64).tolist(),
            },
            "wind_ned": {
                "mean": self.mean_wind.velocity_ned(h).tolist(),
                "turbulence": self._turb_ned.tolist(),
                "gust": self._gust_ned.tolist(),
                "total": self.wind_at(h).tolist(),
            },
            "gear_loads_n": [float(v) for v in self._aux["gear_loads"]],
            "touched_down": self.touched_down,
        }

    def render(self):
        if self.render_mode != "ansi":
            return None
        k = self._kinematics()
        va = self._aux["airdata"][0]
        return (
            f"t={self.time_s:6.2f}s x={k['x']:8.1f} y={k['y']:6.1f} h={k['h']:6.1f} "
            f"Va={va:5.1f} phi={math.degrees(k['phi']):5.1f} theta={math.degrees(k['theta']):5.1f} "
            f"psi={math.degrees(k['psi']):6.1f} hdot={k['h_dot']:5.2f}"
        )


def make_env_config(**overrides: Any) -> EnvConfig:
    """Convenience constructor: ``make_env_config(max_wind_speed_mps=12.0)`` etc.

    Keys matching ``WindConfig`` fields are routed to the wind config, the rest to
    :class:`EnvConfig`.
    """
    wind_keys = set(WindConfig.__dataclass_fields__)
    wind_over = {k: v for k, v in overrides.items() if k in wind_keys}
    env_over = {k: v for k, v in overrides.items() if k not in wind_keys}
    cfg = EnvConfig(**env_over)
    if wind_over:
        cfg.wind = replace(cfg.wind, **wind_over)
    return cfg
