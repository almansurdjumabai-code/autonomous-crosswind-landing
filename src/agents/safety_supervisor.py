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
  fallback keeps authority once it has taken it (no hand-backs in the flare), and a
  primary that has been overridden ``max_reversions`` times in one approach is locked out
  for the rest of it (a heavy aircraft cannot afford to lose height on every hand-back).

The per-step authority ``lambda`` and the active monitor reason are exposed for
telemetry so that every rollout states exactly who was flying.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from src.agents.baseline_autopilot import ClassicalAutopilot
from src.env.aircraft_env import CrosswindLandingEnv, aircraft_key


@dataclass
class EnvelopeConfig:
    """Envelope limits. Defaults: Cessna-172-class aircraft (original tuning)."""

    lookahead_s: float = 1.0
    schedule_height_m: float = 60.0  # bank and sink limits relax linearly up to this height
    bank_limit_high_deg: float = 30.0  # at and above schedule_height_m
    bank_limit_low_deg: float = 10.0  # at the runway
    pitch_min_deg: float = -10.0
    pitch_max_deg: float = 14.0
    alpha_max_deg: float = 11.0
    airspeed_margin_mps: float = 5.0  # below V_ref
    sink_limit_high_mps: float = 4.0  # at and above schedule_height_m
    sink_limit_low_mps: float = 1.2  # at the runway
    gs_dev_base_m: float = 10.0
    gs_dev_per_m: float = 0.05
    loc_dev_base_m: float = 12.0
    loc_dev_per_m: float = 0.03
    flare_gate_m: float = 15.0  # wheel height
    dwell_s: float = 3.0
    authority_rate_per_s: float = 2.5
    max_reversions: int = 10**9  # fallbacks per approach before the primary is locked out


CESSNA_172_ENVELOPE = EnvelopeConfig()

# ATR 72-600: pitch bounded by the tail-strike attitude, bank by the propeller clearance,
# sink limits sized for a 2.9-3.6 m/s approach sink rate, a higher flare gate.
ATR_72_ENVELOPE = EnvelopeConfig(
    schedule_height_m=90.0,
    bank_limit_high_deg=25.0,
    bank_limit_low_deg=7.0,
    pitch_min_deg=-8.0,
    pitch_max_deg=8.0,
    alpha_max_deg=11.5,
    airspeed_margin_mps=5.0,
    sink_limit_high_mps=6.5,
    sink_limit_low_mps=2.0,
    gs_dev_base_m=10.0,
    gs_dev_per_m=0.012,  # about 0.7 deg, two dots of glideslope
    loc_dev_base_m=15.0,
    loc_dev_per_m=0.03,
    flare_gate_m=25.0,
    max_reversions=3,
)

ENVELOPE_BY_AIRCRAFT: dict[str, EnvelopeConfig] = {"atr72": ATR_72_ENVELOPE, "cessna172": CESSNA_172_ENVELOPE}


def envelope_for_aircraft(env: CrosswindLandingEnv) -> EnvelopeConfig:
    """Envelope tuned for the environment's airframe (Cessna limits for an unknown airframe)."""
    return ENVELOPE_BY_AIRCRAFT.get(aircraft_key(env.cfg.aircraft), CESSNA_172_ENVELOPE)


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
        self.cfg = config or envelope_for_aircraft(env)
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
        self.reversions = 0
        self.locked_out = False

    # ----------------------------------------------------------------- monitor
    def _schedule(self, h: float, high: float, low: float) -> float:
        u = min(max(h / self.cfg.schedule_height_m, 0.0), 1.0)
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
        h_wheel = env._kinematics()["h"] - env.wheel_height_m
        if reason is not None:
            # Any predicted violation (re-)arms the dwell latch: authority stays with the
            # fallback until the state has been inside the envelope for ``dwell_s``.
            if t >= self.latched_until:
                self.reversions += 1
                if self.reversions >= c.max_reversions:
                    self.locked_out = True
            self.reason = reason
            self.latched_until = t + c.dwell_s
        target = 0.0 if t < self.latched_until else 1.0
        if h_wheel < c.flare_gate_m and (self.authority < 0.999 or reason is not None):
            self.flare_lock = True  # no hand-back to the primary inside the flare gate
        if self.flare_lock or self.locked_out:
            target = 0.0
            # Once the dwell of the last violation has run out, name the lock that keeps the fallback flying.
            if reason is None and t >= self.latched_until:
                self.reason = "flare_gate" if self.flare_lock else "locked_out"
            self.reason = self.reason or ("flare_gate" if self.flare_lock else "locked_out")
        elif target == 1.0:
            self.reason = None
        step = c.authority_rate_per_s * dt
        self.authority += max(-step, min(step, target - self.authority))
        a = self.authority * a_primary + (1.0 - self.authority) * a_fallback
        return np.clip(a, -1.0, 1.0).astype(np.float32)
