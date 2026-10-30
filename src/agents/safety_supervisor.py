"""Simplex runtime assurance for the learned landing policy.

A learned controller is allowed to fly only while the aircraft stays inside a
verified safe envelope. The architecture follows the Simplex / runtime-assurance
pattern used for certifying learning-enabled flight controls:

* the **primary** controller (the PPO policy) proposes an action every step;
* the **fallback** controller (the classical crosswind autoland) also runs every
  step, so its integrators and wind estimate are warm when it takes over;
* an **envelope monitor** checks the current state, extrapolated over a short
  look-ahead horizon, against altitude-scheduled limits (bank, pitch, angle of
  attack, airspeed, sink rate, glideslope and localizer deviation);
* a **switching logic** with hysteresis moves control authority from the primary
  to the fallback when a violation is predicted and latches the fallback until the
  state has stayed inside the envelope for a dwell time. Authority is rate-limited, so the commanded action is a smooth blend
  ``a = lambda * a_primary + (1 - lambda) * a_fallback``. Below the flare gate the
  fallback keeps authority once it has taken it (no hand-backs in the flare).

The per-step authority ``lambda`` and the active monitor reason are exposed for
telemetry so that every rollout states exactly who was flying.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from src.agents.baseline_autopilot import ClassicalAutopilot
from src.env.aircraft_env import CrosswindLandingEnv


@dataclass
class EnvelopeConfig:
    lookahead_s: float = 1.0
    bank_limit_high_deg: float = 30.0  # above 60 m AGL
    bank_limit_low_deg: float = 10.0  # at the runway
    pitch_min_deg: float = -10.0
    pitch_max_deg: float = 14.0
    alpha_max_deg: float = 11.0
    airspeed_margin_mps: float = 5.0  # below V_ref
    sink_limit_high_mps: float = 4.0  # above 60 m AGL
    sink_limit_low_mps: float = 1.2  # at the runway
    gs_dev_base_m: float = 10.0
    gs_dev_per_m: float = 0.05
    loc_dev_base_m: float = 12.0
    loc_dev_per_m: float = 0.03
    flare_gate_m: float = 15.0
    dwell_s: float = 3.0
    authority_rate_per_s: float = 2.5


class SafetySupervisor:
    """Blends a primary policy with the classical autoland under an envelope monitor."""

    def __init__(
        self,
        env: CrosswindLandingEnv,
        primary: Callable[[np.ndarray], np.ndarray],
        config: Optional[EnvelopeConfig] = None,
        fallback: Optional[ClassicalAutopilot] = None,
    ) -> None:
        self.env = env
        self.primary = primary
        self.cfg = config or EnvelopeConfig()
        # In residual action mode the environment already runs the classical autoland under the
        # policy, so falling back means commanding a zero residual.
        self.residual = env.cfg.action_mode == "residual"
        self.fallback = fallback or ClassicalAutopilot(env)
        self.reset()

    def reset(self) -> None:
        self.fallback.reset()
        self.authority = 1.0
        self.latched_until = -1.0
        self.reason: Optional[str] = None
        self.flare_lock = False

    # ----------------------------------------------------------------- monitor
    @staticmethod
    def _schedule(h: float, high: float, low: float) -> float:
        u = min(max(h / 60.0, 0.0), 1.0)
        return low + (high - low) * u

    def violation(self) -> Optional[str]:
        """Return the name of the first predicted envelope violation, or None."""
        env, c = self.env, self.cfg
        k = env._kinematics()
        va, alpha, _, _ = env._aux["airdata"]
        p, q, _ = (float(v) for v in env.state[10:13])
        if env.touched_down:
            return None
        h = k["h"]
        la = c.lookahead_s
        phi = math.degrees(k["phi"] + p * la)
        theta = math.degrees(k["theta"] + q * la)
        vz = k["h_dot"]
        h_pred = h + vz * la
        if abs(phi) > self._schedule(h_pred, c.bank_limit_high_deg, c.bank_limit_low_deg):
            return "bank"
        if not c.pitch_min_deg <= theta <= c.pitch_max_deg:
            return "pitch"
        if math.degrees(alpha) > c.alpha_max_deg:
            return "alpha"
        if va < env.cfg.approach_speed_mps - c.airspeed_margin_mps:
            return "airspeed"
        if vz < -self._schedule(h_pred, c.sink_limit_high_mps, c.sink_limit_low_mps) and h > 0.5:
            return "sink_rate"
        dist = max(k["dist_to_aim"], 0.0)
        if h > c.flare_gate_m and abs(k["gs_error"]) > c.gs_dev_base_m + c.gs_dev_per_m * dist:
            return "glideslope"
        if abs(k["y"] + k["y_dot"] * la) > c.loc_dev_base_m + c.loc_dev_per_m * dist:
            return "localizer"
        return None

    # --------------------------------------------------------------------- act
    def act(self, obs: np.ndarray) -> np.ndarray:
        env, c = self.env, self.cfg
        dt = env.cfg.control_dt
        t = env.time_s
        a_fallback = np.zeros(4) if self.residual else np.asarray(self.fallback.act(dt), dtype=np.float64)
        a_primary = np.asarray(self.primary(obs), dtype=np.float64)

        reason = self.violation()
        h_wheel = env._kinematics()["h"] - 1.0
        if reason is not None:
            # Any predicted violation (re-)arms the dwell latch: authority stays with the
            # fallback until the state has been inside the envelope for ``dwell_s``.
            self.reason = reason
            self.latched_until = t + c.dwell_s
        target = 0.0 if t < self.latched_until else 1.0
        if h_wheel < c.flare_gate_m and (self.authority < 0.999 or reason is not None):
            self.flare_lock = True  # no hand-back to the primary inside the flare gate
        if self.flare_lock:
            target = 0.0
            self.reason = self.reason or "flare_gate"
        elif target == 1.0:
            self.reason = None
        step = c.authority_rate_per_s * dt
        self.authority += max(-step, min(step, target - self.authority))
        a = self.authority * a_primary + (1.0 - self.authority) * a_fallback
        return np.clip(a, -1.0, 1.0).astype(np.float32)
