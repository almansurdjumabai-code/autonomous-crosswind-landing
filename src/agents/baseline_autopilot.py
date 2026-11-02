"""Classical ILS-coupled crosswind autoland, used as the baseline and as the safety fallback.

The controller is a cascaded design acting on quantities available to a real
flight-control computer (inertial navigation + air data):

* **Wind estimation.** The wind vector is estimated as inertial ground velocity
  minus the air-relative velocity reconstructed from (V_a, alpha, beta). A slow
  low-pass gives the mean wind; the low-passed magnitude of the residual gives a
  gust estimate.
* **Speed.** The approach speed is gust-adjusted the way pilots add a gust
  factor: V_app = V_ref + min(0.5 * gust + 0.12 * crosswind, 6 m/s). This adds
  control authority and stall margin in turbulence. Thrust is held through the
  flare and retarded only in the last metre.
* **Longitudinal.** Glideslope -> vertical-speed command -> pitch command ->
  elevator (attitude + damper + integral). Below the flare height an exponential
  flare law h_dot_cmd = -h_wheel / tau - h_dot_td takes over with a faster loop.
  Pitch is limited below the tail-strike attitude, and an angle-of-attack
  protection caps the pitch command so that alpha stays below 12 deg (pushing
  the nose down above it), preventing gust-induced deep stalls.
* **Lateral, approach.** Localizer -> lateral-velocity command -> course error ->
  bank command -> aileron. The rudder coordinates (beta -> 0) and damps yaw, so the
  aircraft flies a crabbed approach whose crab angle the wind dictates.
* **Lateral, de-crab.** Below the de-crab height the rudder yaws the nose towards
  the runway heading. The commanded alignment is limited so that the required
  sideslip stays within ``max_decrab_slip``; beyond that, rudder authority and lift
  are lost (the failure mode of naive de-crabbing in strong crosswinds), so the
  aircraft touches down with a small residual crab instead. Ailerons hold the
  centreline wing-low.

The gains depend on the airframe: :func:`gains_for_aircraft` returns the set tuned for
the ATR 72-600 (tail-strike-limited pitch, slow turboprop power response, 55 m/s
approach) or for the original Cessna-172-class aircraft, by the preset the environment
was built from.

It returns normalised actions in [-1, 1]^4 so it can be evaluated by exactly the
same pipeline as a learned policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.env.aircraft_env import CrosswindLandingEnv, aircraft_key, quat_to_dcm


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class AutopilotGains:
    """Gains of the cascaded autoland. Defaults: Cessna-172-class aircraft (original tuning)."""

    # Longitudinal
    k_gs: float = 0.25
    k_hdot: float = 0.06
    ki_hdot: float = 0.03
    k_theta: float = 2.0
    k_q: float = 0.6
    k_va: float = 0.10
    ki_va: float = 0.02
    max_sink_cmd_mps: float = 5.0
    flare_height_m: float = 10.0  # wheel height at which the flare law takes over
    flare_tau_s: float = 2.8
    # True: the flare time constant is recomputed at flare entry so that the commanded sink
    # rate is continuous with the approach (tau = flare_height / (approach sink - touchdown sink));
    # False: fixed flare_tau_s, the command is then capped at the approach sink rate.
    flare_continuous_entry: bool = False
    touchdown_sink_mps: float = 0.25
    k_hdot_flare: float = 0.16
    ki_hdot_flare: float = 0.12
    flare_min_pitch_rad: float = math.radians(-8.0)  # pitch floor once the flare has started
    max_pitch_rate_rps: float = math.radians(1000.0)  # rate limit on the pitch command
    flare_speed_bleed_mps: float = 4.0  # speed given up during a smooth-air flare
    thrust_cut_height_m: float = 1.0  # wheel height below which the throttle is closed
    gusty_threshold_mps: float = 1.0
    alpha_protect_rad: float = math.radians(12.0)
    k_alpha_protect: float = 2.0
    min_pitch_rad: float = math.radians(-8.0)
    max_pitch_rad: float = math.radians(11.0)  # pitch command cap in the flare (tail strike)
    approach_max_pitch_rad: float = math.radians(11.0)  # pitch command cap on the approach (glideslope recapture)
    gust_speed_gain: float = 0.5
    crosswind_speed_gain: float = 0.12
    max_speed_additive_mps: float = 6.0
    # Lateral
    k_y: float = 0.08
    ki_y: float = 0.002
    max_lat_speed_mps: float = 5.0
    k_course: float = 0.5
    max_bank_rad: float = math.radians(20.0)
    k_phi: float = 1.2
    k_p: float = 0.15
    k_beta: float = 1.5
    k_r: float = 0.4
    # De-crab
    decrab_height_m: float = 4.0  # wheel height
    max_decrab_slip_deg: float = 12.0
    k_psi: float = 2.0
    k_wing_low: float = 0.06
    max_wing_low_rad: float = math.radians(8.0)
    # Estimator
    wind_tau_s: float = 3.0
    gust_tau_s: float = 4.0


CESSNA_172_GAINS = AutopilotGains()

# ATR 72-600, flaps 30. The pitch command is capped well below the tail-strike attitude
# (about 7 deg with the struts compressed), the flare starts higher and lasts longer because
# the approach sink rate is 2.9 m/s at 55 m/s, and the turboprop power loop is slower.
ATR_72_GAINS = AutopilotGains(
    k_gs=0.15,
    k_hdot=0.045,
    ki_hdot=0.02,
    k_theta=2.5,
    k_q=1.5,
    k_va=0.06,
    ki_va=0.012,
    max_sink_cmd_mps=7.0,
    flare_height_m=9.0,
    flare_tau_s=3.5,
    flare_continuous_entry=True,
    touchdown_sink_mps=0.6,
    k_hdot_flare=0.07,
    ki_hdot_flare=0.04,
    flare_min_pitch_rad=math.radians(1.0),
    max_pitch_rate_rps=math.radians(6.0),
    flare_speed_bleed_mps=2.0,
    thrust_cut_height_m=1.5,
    gusty_threshold_mps=1.5,
    alpha_protect_rad=math.radians(12.0),
    k_alpha_protect=2.0,
    min_pitch_rad=math.radians(-6.0),
    max_pitch_rad=math.radians(6.5),
    approach_max_pitch_rad=math.radians(10.0),
    gust_speed_gain=0.5,
    crosswind_speed_gain=0.10,
    max_speed_additive_mps=7.5,
    k_y=0.06,
    ki_y=0.0015,
    max_lat_speed_mps=6.0,
    k_course=0.5,
    max_bank_rad=math.radians(15.0),
    k_phi=1.8,
    k_p=0.25,
    k_beta=1.2,
    k_r=0.6,
    decrab_height_m=10.0,
    max_decrab_slip_deg=10.0,
    k_psi=2.0,
    k_wing_low=0.05,
    max_wing_low_rad=math.radians(5.0),
)

GAINS_BY_AIRCRAFT: dict[str, AutopilotGains] = {"atr72": ATR_72_GAINS, "cessna172": CESSNA_172_GAINS}


def gains_for_aircraft(env: CrosswindLandingEnv) -> AutopilotGains:
    """Gain set tuned for the environment's airframe (Cessna gains for an unknown airframe)."""
    return GAINS_BY_AIRCRAFT.get(aircraft_key(env.cfg.aircraft), CESSNA_172_GAINS)


class ClassicalAutopilot:
    """Stateful cascaded-PID crosswind autoland bound to an environment instance."""

    def __init__(self, env: CrosswindLandingEnv, gains: AutopilotGains | None = None) -> None:
        self.env = env
        self.g = gains or gains_for_aircraft(env)
        self.reset()

    def reset(self) -> None:
        self.int_hdot = 0.0
        self.int_va = 0.0
        self.int_y = 0.0
        self.wind = None  # low-passed wind estimate, NED [m/s]
        self.gust = 0.0  # low-passed residual magnitude [m/s]
        self.decrab_target_deg = None
        self.flaring = False
        self.flare_tau = None
        self.theta_cmd_prev = None

    # -------------------------------------------------------------- estimation
    def _estimate_wind(self, dt: float) -> np.ndarray:
        env = self.env
        va, alpha, beta, _ = env._aux["airdata"]
        r_nb = quat_to_dcm(env.state[6:10])
        v_ground = r_nb @ env.state[3:6]
        ca, sa, cb, sb = math.cos(alpha), math.sin(alpha), math.cos(beta), math.sin(beta)
        v_air = r_nb @ np.array([va * ca * cb, va * sb, va * sa * cb])
        w = v_ground - v_air
        if self.wind is None:
            self.wind = w.copy()
        a = 1.0 - math.exp(-dt / self.g.wind_tau_s)
        self.wind += (w - self.wind) * a
        b = 1.0 - math.exp(-dt / self.g.gust_tau_s)
        self.gust += (float(np.linalg.norm(w - self.wind)) - self.gust) * b
        return self.wind

    # --------------------------------------------------------------------- act
    def act(self, dt: float | None = None) -> np.ndarray:
        env, g = self.env, self.g
        dt = dt if dt is not None else env.cfg.control_dt
        k = env._kinematics()
        va, alpha, beta, _ = env._aux["airdata"]
        p, q, r = (float(v) for v in env.state[10:13])
        trim = env.trim
        rw = env.cfg.runway
        h_wheel = k["h"] - env.wheel_height_m
        wind = self._estimate_wind(dt)
        crosswind = abs(float(wind[1]))

        # --------------------------------------------------- longitudinal
        ground_speed = max(math.hypot(k["x_dot"], k["y_dot"]), 5.0)
        h_ref = max(k["dist_to_aim"], 0.0) * math.tan(rw.glideslope_rad)
        flaring = h_wheel < g.flare_height_m or env.touched_down
        approach_sink = -ground_speed * math.tan(rw.glideslope_rad)
        if flaring:
            if self.flare_tau is None:
                self.flare_tau = g.flare_tau_s
                if g.flare_continuous_entry:
                    self.flare_tau = max(g.flare_height_m / max(-approach_sink - g.touchdown_sink_mps, 0.3), g.flare_tau_s)
            # Exponential flare, never steeper than the approach sink rate: the law alone would
            # ask for a deeper descent at flare entry whenever flare_height / tau exceeds the
            # approach sink rate, and the aircraft would pitch down into the ground.
            hdot_cmd = max(-max(h_wheel, 0.0) / self.flare_tau - g.touchdown_sink_mps, approach_sink)
        else:
            hdot_cmd = approach_sink + g.k_gs * (h_ref - k["h"])
            hdot_cmd = _clip(hdot_cmd, -g.max_sink_cmd_mps, 3.0)
        e_hdot = hdot_cmd - k["h_dot"]
        if flaring and not self.flaring:
            # Keep the integral term continuous when the flare gains take over.
            self.int_hdot *= g.ki_hdot / max(g.ki_hdot_flare, 1e-9)
        self.flaring = flaring
        self.int_hdot = _clip(self.int_hdot + e_hdot * dt, -10.0, 10.0)
        # The flare needs a faster sink-rate loop than glideslope tracking.
        kp, ki = (g.k_hdot_flare, g.ki_hdot_flare) if flaring else (g.k_hdot, g.ki_hdot)
        theta_cmd = trim["theta"] + kp * e_hdot + ki * self.int_hdot
        theta_cmd = _clip(theta_cmd, g.flare_min_pitch_rad if flaring else g.min_pitch_rad,
                          g.max_pitch_rad if flaring else g.approach_max_pitch_rad)
        if self.theta_cmd_prev is not None:
            step = g.max_pitch_rate_rps * dt
            theta_cmd = _clip(theta_cmd, self.theta_cmd_prev - step, self.theta_cmd_prev + step)
        self.theta_cmd_prev = theta_cmd
        # Angle-of-attack protection (alpha floor): never let a sink-rate demand pitch the wing
        # into the stall; above alpha_protect the elevator pushes and the integrator unwinds.
        alpha_margin = g.alpha_protect_rad - alpha
        theta_cmd = min(theta_cmd, k["theta"] + alpha_margin)
        elevator = trim["elevator"] - g.k_theta * (theta_cmd - k["theta"]) + g.k_q * q
        if alpha_margin < 0.0:
            elevator += g.k_alpha_protect * (-alpha_margin)
            self.int_hdot = min(self.int_hdot, 0.0)

        v_app = env.cfg.approach_speed_mps + min(
            g.gust_speed_gain * self.gust + g.crosswind_speed_gain * crosswind, g.max_speed_additive_mps
        )
        if env.touched_down or h_wheel < g.thrust_cut_height_m:
            throttle = 0.0
        else:
            # Hold the gust-adjusted speed on the approach; in the flare bleed towards V_ref
            # but keep power so the flare is not flown on a decaying energy state.
            # In gusty air keep the gust additive through the flare: the energy is needed to
            # arrest gust-induced sink. In smooth air bleed towards V_ref for a gentle flare.
            if not flaring or self.gust > g.gusty_threshold_mps:
                target = v_app
            else:
                bleed = g.flare_speed_bleed_mps
                target = max(env.cfg.approach_speed_mps - 0.5 * bleed, v_app - bleed)
            e_va = target - va
            self.int_va = _clip(self.int_va + e_va * dt, -20.0, 20.0)
            throttle = trim["throttle"] + g.k_va * e_va + g.ki_va * self.int_va

        # -------------------------------------------------------- lateral
        self.int_y = _clip(self.int_y + k["y"] * dt, -200.0, 200.0)
        ydot_cmd = _clip(-g.k_y * k["y"] - g.ki_y * self.int_y, -g.max_lat_speed_mps, g.max_lat_speed_mps)
        course = math.atan2(k["y_dot"], max(k["x_dot"], 1.0))
        course_cmd = math.atan2(ydot_cmd, max(k["x_dot"], 1.0))
        phi_cmd = _clip(g.k_course * (course_cmd - course) * va / 9.81 * 2.0, -g.max_bank_rad, g.max_bank_rad)
        aileron = g.k_phi * (phi_cmd - k["phi"]) - g.k_p * p

        if h_wheel < g.decrab_height_m or env.touched_down:
            if self.decrab_target_deg is None:
                # Residual crab that keeps the de-crab sideslip within rudder authority.
                crab = math.degrees(k["heading_error"])
                excess = max(abs(crab) - g.max_decrab_slip_deg, 0.0)
                self.decrab_target_deg = math.copysign(excess, crab)
            target = 0.0 if env.touched_down else math.radians(self.decrab_target_deg)
            rudder = g.k_psi * (k["heading_error"] - target) + g.k_r * r
            if env.touched_down:
                phi_cmd = 0.0  # wings level on the ground; the rudder and nose wheel hold the centreline
            else:
                phi_cmd = _clip(-g.k_wing_low * 9.0 * k["y_dot"] - 0.01 * k["y"], -g.max_wing_low_rad, g.max_wing_low_rad)
            aileron = g.k_phi * (phi_cmd - k["phi"]) - g.k_p * p
        else:
            rudder = -g.k_beta * beta + g.k_r * r

        cmd = np.array([elevator, aileron, rudder, _clip(throttle, 0.0, 1.0)])
        lim = env.limits
        cmd[:3] = np.clip(cmd[:3], -lim[:3], lim[:3])
        return env.command_to_action(cmd)
