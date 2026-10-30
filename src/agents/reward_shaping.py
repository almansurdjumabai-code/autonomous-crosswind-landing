"""Reward shaping for the autonomous crosswind landing task.

The reward is the sum of three families of terms:

1. **Potential-based shaping** (Ng, Harada & Russell, 1999)::

       F(s, s') = gamma * Phi(s') - Phi(s)

   where ``Phi`` encodes glideslope tracking, localizer (centerline) tracking and
   progress toward the aim point. Because ``F`` is a difference of potentials it
   leaves the optimal policy of the underlying sparse task unchanged while
   providing a dense learning signal. On terminal transitions ``Phi(s') := 0``.

2. **Dense regularisation penalties** (not potential-based, deliberately shaping
   the preferred *style* of flight): attitude envelope excursions, body-rate
   magnitude, airspeed / angle-of-attack protection, control effort and control
   rate, runway alignment (de-crab) and sink rate in the flare.

3. **Sparse terminal rewards**: touchdown quality (vertical speed, touchdown
   distance from the threshold relative to the aim point, lateral offset and
   crab angle), a rollout completion bonus, and crash penalties.

The module is deliberately independent of the simulator: the environment fills
in a :class:`FlightSnapshot` each control step and the shaper consumes it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

# Touchdown sink rate (h_dot, positive up) below which the landing is a crash.
CRASH_SINK_RATE_MPS: float = -1.5


@dataclass(frozen=True)
class TouchdownEvent:
    """Kinematic state of the aircraft at first main-gear contact."""

    time_s: float
    x_m: float
    y_m: float
    sink_rate_mps: float
    heading_error_rad: float
    roll_rad: float
    pitch_rad: float
    airspeed_mps: float
    lateral_speed_mps: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FlightSnapshot:
    """Everything the reward function needs at one control step.

    Distances are in metres in the runway frame (x along the runway from the
    threshold, y to the right of the centreline, h above ground level).
    """

    x_m: float
    y_m: float
    h_m: float
    h_dot_mps: float
    lateral_speed_mps: float
    glideslope_error_m: float
    distance_to_aim_m: float
    heading_error_rad: float
    phi_rad: float
    theta_rad: float
    p_rps: float
    q_rps: float
    r_rps: float
    airspeed_mps: float
    airspeed_ref_mps: float
    alpha_rad: float
    beta_rad: float
    action: np.ndarray
    prev_action: np.ndarray
    weight_on_wheels: bool
    touched_down: bool
    runway_half_width_m: float
    aim_point_x_m: float
    touchdown: Optional[TouchdownEvent] = None
    crashed: bool = False
    crash_reason: Optional[str] = None
    success: bool = False


@dataclass(frozen=True)
class RewardConfig:
    """Weights and tolerances of every reward term."""

    gamma: float = 0.99

    # Potential-based shaping weights (units: reward per normalised error).
    w_glideslope: float = 2.0
    w_localizer: float = 2.0
    w_progress: float = 1.0
    glideslope_tol_base_m: float = 4.0
    glideslope_tol_per_m: float = 0.03
    localizer_tol_base_m: float = 6.0
    localizer_tol_per_m: float = 0.02
    progress_scale_m: float = 1000.0

    # Dense penalties.
    roll_limit_rad: float = math.radians(25.0)
    pitch_min_rad: float = math.radians(-10.0)
    pitch_max_rad: float = math.radians(15.0)
    w_attitude_envelope: float = 2.0
    w_attitude_low_alt: float = 0.05
    low_alt_threshold_m: float = 15.0
    w_rates: float = 0.01
    w_airspeed: float = 0.01
    airspeed_tol_mps: float = 4.0
    alpha_warning_rad: float = math.radians(12.0)
    w_alpha: float = 5.0
    w_effort: float = 0.002
    w_effort_rate: float = 0.02
    w_decrab: float = 0.05
    w_rollout_centerline: float = 0.05
    # Flare: penalise sink rates steeper than ``flare_sink_target_mps`` close to the ground,
    # a dense precursor of the sparse hard-landing penalty.
    w_flare_sink: float = 0.4
    flare_height_m: float = 12.0
    flare_sink_target_mps: float = -0.8
    # Lower bound on the sum of the dense penalties of one step. Keeps the
    # per-step reward scale bounded in pathological states (deep stall, tumbling)
    # so the value function regression stays well conditioned.
    max_dense_penalty_per_step: float = 5.0

    # Sparse terminal terms.
    crash_sink_rate_mps: float = CRASH_SINK_RATE_MPS
    ideal_sink_rate_mps: float = -0.3
    sink_rate_sigma_mps: float = 0.5
    touchdown_x_sigma_m: float = 150.0
    touchdown_reward: float = 100.0
    touchdown_weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)
    crab_sigma_rad: float = math.radians(5.0)
    rollout_success_bonus: float = 50.0
    crash_penalty: float = -100.0
    crash_excess_sink_scale: float = 20.0
    crash_excess_sink_cap_mps: float = 10.0


@dataclass
class RewardBreakdown:
    """Per-term decomposition of the reward of a single step (for logging)."""

    shaping: float = 0.0
    attitude: float = 0.0
    rates: float = 0.0
    airspeed: float = 0.0
    alpha: float = 0.0
    effort: float = 0.0
    decrab: float = 0.0
    flare: float = 0.0
    rollout: float = 0.0
    touchdown: float = 0.0
    terminal: float = 0.0
    components: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (
            self.shaping
            + self.attitude
            + self.rates
            + self.airspeed
            + self.alpha
            + self.effort
            + self.decrab
            + self.flare
            + self.rollout
            + self.touchdown
            + self.terminal
        )

    def to_dict(self) -> dict:
        out = {
            "shaping": self.shaping,
            "attitude": self.attitude,
            "rates": self.rates,
            "airspeed": self.airspeed,
            "alpha": self.alpha,
            "effort": self.effort,
            "decrab": self.decrab,
            "flare": self.flare,
            "rollout": self.rollout,
            "touchdown": self.touchdown,
            "terminal": self.terminal,
            "total": self.total,
        }
        out.update(self.components)
        return out


def _pseudo_huber(e: float) -> float:
    """Smooth |e|-like cost, quadratic near zero: sqrt(1 + e^2) - 1."""
    return math.sqrt(1.0 + e * e) - 1.0


def _excess(value: float, lower: float, upper: float) -> float:
    """Distance of ``value`` outside the closed interval [lower, upper]."""
    if value > upper:
        return value - upper
    if value < lower:
        return lower - value
    return 0.0


class LandingRewardShaper:
    """Stateful reward function; call :meth:`reset` at every episode start."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._prev_potential: float = 0.0
        self._touchdown_rewarded: bool = False

    # ------------------------------------------------------------------ potential
    def potential(self, s: FlightSnapshot) -> float:
        """Shaping potential Phi(s) <= 0; zero means perfectly on the approach path.

        Once the aircraft has touched down the approach-path potential is frozen at
        zero so that the shaping term does not interfere with the rollout phase.
        """
        c = self.config
        if s.touched_down:
            return 0.0
        dist = max(s.distance_to_aim_m, 0.0)
        gs_tol = c.glideslope_tol_base_m + c.glideslope_tol_per_m * dist
        loc_tol = c.localizer_tol_base_m + c.localizer_tol_per_m * dist
        phi_gs = -c.w_glideslope * _pseudo_huber(s.glideslope_error_m / gs_tol)
        phi_loc = -c.w_localizer * _pseudo_huber(s.y_m / loc_tol)
        phi_prog = -c.w_progress * abs(s.distance_to_aim_m) / c.progress_scale_m
        return phi_gs + phi_loc + phi_prog

    def reset(self, s: FlightSnapshot) -> None:
        self._prev_potential = self.potential(s)
        self._touchdown_rewarded = False

    # --------------------------------------------------------------- touchdown
    def touchdown_reward(self, td: TouchdownEvent, runway_half_width_m: float, aim_x_m: float) -> tuple[float, dict]:
        """Sparse reward for the touchdown event. Returns (reward, factor dict).

        The factors f_i in [0, 1] are combined as a convex combination scaled by
        ``touchdown_reward``. A sink rate below the crash limit, an undershoot of the
        threshold or a runway-edge touchdown yields the crash penalty instead.
        """
        c = self.config
        if td.sink_rate_mps < c.crash_sink_rate_mps:
            # Capped so a non-physical state can never dominate the return scale.
            excess = min(c.crash_sink_rate_mps - td.sink_rate_mps, c.crash_excess_sink_cap_mps)
            return c.crash_penalty - c.crash_excess_sink_scale * excess, {"td_crash": 1.0}
        if td.x_m < 0.0 or abs(td.y_m) > runway_half_width_m:
            return c.crash_penalty, {"td_crash": 1.0}

        f_sink = math.exp(-(((td.sink_rate_mps - c.ideal_sink_rate_mps) / c.sink_rate_sigma_mps) ** 2))
        f_x = math.exp(-(((td.x_m - aim_x_m) / c.touchdown_x_sigma_m) ** 2))
        f_y = math.exp(-((td.y_m / (0.25 * runway_half_width_m)) ** 2))
        f_crab = math.exp(-((td.heading_error_rad / c.crab_sigma_rad) ** 2))
        w = c.touchdown_weights
        quality = w[0] * f_sink + w[1] * f_x + w[2] * f_y + w[3] * f_crab
        factors = {"td_crash": 0.0, "f_sink": f_sink, "f_x": f_x, "f_y": f_y, "f_crab": f_crab, "td_quality": quality}
        return c.touchdown_reward * quality, factors

    # -------------------------------------------------------------------- step
    def compute(self, s: FlightSnapshot, terminated: bool) -> RewardBreakdown:
        """Reward for the transition that produced snapshot ``s``."""
        c = self.config
        out = RewardBreakdown()

        # 1) Potential-based shaping; Phi(terminal) := 0.
        phi_next = 0.0 if terminated else self.potential(s)
        out.shaping = c.gamma * phi_next - self._prev_potential
        self._prev_potential = phi_next

        # 2) Dense penalties.
        roll_excess = _excess(s.phi_rad, -c.roll_limit_rad, c.roll_limit_rad)
        pitch_excess = _excess(s.theta_rad, c.pitch_min_rad, c.pitch_max_rad)
        out.attitude = -c.w_attitude_envelope * (roll_excess**2 + pitch_excess**2) / (math.radians(10.0) ** 2)
        if s.h_m < c.low_alt_threshold_m:
            # Close to the ground any bank is a wingtip-strike hazard.
            proximity = 1.0 - max(s.h_m, 0.0) / c.low_alt_threshold_m
            out.attitude -= c.w_attitude_low_alt * proximity * (s.phi_rad / math.radians(5.0)) ** 2

        out.rates = -c.w_rates * (s.p_rps**2 + s.q_rps**2 + s.r_rps**2)
        if not s.weight_on_wheels:
            out.airspeed = -c.w_airspeed * ((s.airspeed_mps - s.airspeed_ref_mps) / c.airspeed_tol_mps) ** 2
            alpha_excess = max(0.0, s.alpha_rad - c.alpha_warning_rad)
            out.alpha = -c.w_alpha * (alpha_excess / math.radians(2.0)) ** 2

        a = np.asarray(s.action, dtype=np.float64)
        da = a - np.asarray(s.prev_action, dtype=np.float64)
        out.effort = -c.w_effort * float(a @ a) - c.w_effort_rate * float(da @ da)

        if s.h_m < c.low_alt_threshold_m and not s.touched_down:
            proximity = 1.0 - max(s.h_m, 0.0) / c.low_alt_threshold_m
            out.decrab = -c.w_decrab * proximity * (
                (s.heading_error_rad / math.radians(5.0)) ** 2 + (s.lateral_speed_mps / 2.0) ** 2
            )

        if s.h_m < c.flare_height_m and not s.touched_down:
            proximity = 1.0 - max(s.h_m, 0.0) / c.flare_height_m
            excess = max(0.0, c.flare_sink_target_mps - s.h_dot_mps)
            out.flare = -c.w_flare_sink * proximity * excess * excess

        if s.touched_down and not s.crashed:
            out.rollout = -c.w_rollout_centerline * (s.y_m / s.runway_half_width_m) ** 2

        dense = out.attitude + out.rates + out.airspeed + out.alpha + out.effort + out.decrab + out.flare + out.rollout
        if dense < -c.max_dense_penalty_per_step:
            scale = c.max_dense_penalty_per_step / -dense
            out.attitude *= scale
            out.rates *= scale
            out.airspeed *= scale
            out.alpha *= scale
            out.effort *= scale
            out.decrab *= scale
            out.flare *= scale
            out.rollout *= scale

        # 3) Sparse terms.
        if s.touchdown is not None and not self._touchdown_rewarded:
            self._touchdown_rewarded = True
            r_td, factors = self.touchdown_reward(s.touchdown, s.runway_half_width_m, s.aim_point_x_m)
            out.touchdown = r_td
            out.components.update(factors)
            if factors.get("td_crash", 0.0) > 0.0:
                # The touchdown itself was the crash; do not double count below.
                return out

        if terminated:
            if s.crashed:
                out.terminal = c.crash_penalty
            elif s.success:
                out.terminal = c.rollout_success_bonus
        return out
