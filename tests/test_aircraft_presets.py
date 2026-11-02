"""Aircraft presets: the ATR 72-600 default, the Cessna-172-class legacy set, and the
geometry-dependent protections (tail strike, propeller clearance, wheel height)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.agents.baseline_autopilot import ATR_72_GAINS, CESSNA_172_GAINS, ClassicalAutopilot, gains_for_aircraft
from src.agents.safety_supervisor import ATR_72_ENVELOPE, CESSNA_172_ENVELOPE, SafetySupervisor
from src.env.aircraft_env import (
    AIRCRAFT_PRESETS,
    ATR_72_600,
    CESSNA_172,
    DEFAULT_AIRCRAFT,
    IDX_QUAT,
    IDX_VEL,
    KT_TO_MPS,
    CrosswindLandingEnv,
    EnvConfig,
    aircraft_key,
    euler_to_quat,
    make_env_config,
    quat_to_dcm,
)


def test_default_configuration_is_the_atr72_preset():
    assert DEFAULT_AIRCRAFT == "atr72"
    default, preset = EnvConfig(), make_env_config("atr72")
    assert default.aircraft == preset.aircraft == ATR_72_600
    assert default.approach_speed_mps == preset.approach_speed_mps == pytest.approx(107 * KT_TO_MPS, abs=0.5)
    assert default.reward == preset.reward
    assert default.runway == preset.runway
    assert aircraft_key(ATR_72_600) == "atr72" and aircraft_key(CESSNA_172) == "cessna172"


def test_cessna_preset_restores_the_original_operating_point():
    cfg = make_env_config("cessna172")
    assert cfg.aircraft.mass_kg == 1043.0
    assert cfg.approach_speed_mps == 33.0
    assert cfg.runway.length_m == 1600.0 and cfg.runway.width_m == 30.0
    assert cfg.reward.crash_sink_rate_mps == -1.5
    assert cfg.min_airspeed_mps == 18.0
    assert cfg.aircraft.wheel_height_m == 1.0


def test_make_env_config_routes_overrides_to_sub_configurations():
    cfg = make_env_config("atr72", max_wind_speed_mps=12.0, glideslope_rad=0.06, crash_sink_rate_mps=-3.0, action_mode="residual")
    assert cfg.wind.max_wind_speed_mps == 12.0
    assert cfg.runway.glideslope_rad == 0.06
    assert cfg.reward.crash_sink_rate_mps == -3.0
    assert cfg.action_mode == "residual"
    assert cfg.aircraft is ATR_72_600


def test_atr72_geometry_is_consistent():
    ac = ATR_72_600
    assert ac.wheel_height_m == pytest.approx(2.55)
    nose = next(g for g in ac.gear if not g.is_main)
    mains = [g for g in ac.gear if g.is_main]
    assert nose.position_body[0] - mains[0].position_body[0] == pytest.approx(10.75, abs=0.01)  # wheelbase
    assert mains[1].position_body[1] - mains[0].position_body[1] == pytest.approx(4.10, abs=0.01)  # track
    assert ac.span_m**2 / ac.wing_area_m2 == pytest.approx(12.0, abs=0.1)  # aspect ratio
    # Landing-configuration lift: C_Lmax about 2.7 at the stall angle, V_REF / V_S about 1.22.
    cl_max = ac.CL0 + ac.CL_alpha * ac.alpha_stall_rad
    assert 2.5 < cl_max < 2.9
    v_stall = math.sqrt(2 * ac.mass_kg * 9.80665 / (1.225 * ac.wing_area_m2 * cl_max))
    assert 1.18 < 55.0 / v_stall < 1.28


def test_atr72_trim_matches_a_flaps_30_approach():
    env = CrosswindLandingEnv(make_env_config("atr72"))
    assert env.trim["residual_norm"] < 1e-8
    assert 1.0 < math.degrees(env.trim["theta"]) < 5.0  # slightly nose-up on a 3 deg path
    assert 4.0 < math.degrees(env.trim["alpha"]) < 9.0
    assert 0.2 < env.trim["throttle"] < 0.5  # approach power well below the go-around reserve


def _grounded(env: CrosswindLandingEnv, roll_deg: float, pitch_deg: float, compression_m: float = 0.22) -> str | None:
    env.reset(seed=0, options={"wind_speed_mps": 0.0, "turbulence": False, "gusts": False})
    env.state[0:3] = [400.0, 0.0, -(env.cfg.aircraft.wheel_height_m - compression_m)]
    env.state[IDX_QUAT] = euler_to_quat(math.radians(roll_deg), math.radians(pitch_deg), 0.0)
    return env._structural_contact()


def test_atr72_tail_strike_attitude_is_about_seven_degrees():
    env = CrosswindLandingEnv(make_env_config("atr72"))
    assert _grounded(env, 0.0, 5.0) is None
    assert _grounded(env, 0.0, 8.5) == "tail_strike"
    # The autoland and the envelope monitor stay below that attitude.
    assert math.degrees(ATR_72_GAINS.max_pitch_rad) < 7.0
    assert ATR_72_ENVELOPE.pitch_max_deg <= 8.0


def test_atr72_propeller_is_the_limiting_point_in_a_bank():
    env = CrosswindLandingEnv(make_env_config("atr72"))
    assert _grounded(env, 12.0, 2.0) is None
    assert _grounded(env, 20.0, 2.0) == "prop_right_strike"
    assert _grounded(env, -20.0, 2.0) == "prop_left_strike"
    # Bank limits of the autoland and of the envelope monitor near the ground stay inside that.
    assert math.degrees(ATR_72_GAINS.max_wing_low_rad) <= 6.0
    assert ATR_72_ENVELOPE.bank_limit_low_deg <= 8.0


def test_gains_and_envelope_follow_the_aircraft():
    atr = CrosswindLandingEnv(make_env_config("atr72"))
    cessna = CrosswindLandingEnv(make_env_config("cessna172"))
    assert gains_for_aircraft(atr) is ATR_72_GAINS and gains_for_aircraft(cessna) is CESSNA_172_GAINS
    assert ClassicalAutopilot(atr).g is ATR_72_GAINS
    assert SafetySupervisor(atr, lambda obs: np.zeros(4)).cfg is ATR_72_ENVELOPE
    assert SafetySupervisor(cessna, lambda obs: np.zeros(4)).cfg is CESSNA_172_ENVELOPE


def test_wheel_height_drives_the_flare_and_the_flare_gate():
    """The autoland and the monitor measure height from the wheels, not from the CG."""
    env = CrosswindLandingEnv(make_env_config("atr72"))
    env.reset(seed=1, options={"wind_speed_mps": 0.0, "turbulence": False, "gusts": False})
    assert env.wheel_height_m == ATR_72_600.wheel_height_m
    ap = ClassicalAutopilot(env)
    # Place the aircraft just below the flare height measured at the wheels: the flare law engages.
    env.state[2] = -(env.wheel_height_m + ap.g.flare_height_m - 0.5)
    ap.act()
    assert ap.flaring
    env.state[2] = -(env.wheel_height_m + ap.g.flare_height_m + 5.0)
    ap.reset()
    ap.act()
    assert not ap.flaring


def test_atr72_autoland_lands_in_a_30_kt_steady_crosswind():
    env = CrosswindLandingEnv(make_env_config("atr72", turbulence_enabled=False, gusts_enabled=False))
    obs, _ = env.reset(seed=7, options={"wind_speed_mps": 30 * KT_TO_MPS, "wind_from_deg": 90.0, "start_y_m": 0.0})
    ap = ClassicalAutopilot(env)
    max_pitch = 0.0
    while True:
        obs, _, term, trunc, info = env.step(ap.act())
        max_pitch = max(max_pitch, math.degrees(env._kinematics()["theta"]))
        if term or trunc:
            break
    m = info["episode_metrics"]
    assert m["success"] == 1.0, m["crash_reason"]
    assert env.cfg.reward.crash_sink_rate_mps < m["touchdown_sink_rate_mps"] < 0.0
    assert abs(m["touchdown_y_m"]) < 0.5 * env.cfg.runway.width_m - 5.0
    assert abs(m["touchdown_crab_deg"]) < ATR_72_GAINS.max_decrab_slip_deg + 3.0  # partial de-crab
    assert max_pitch < 8.0  # never near the tail-strike attitude
    assert m["touchdown_x_m"] > 0.0


def test_telemetry_names_the_aircraft(tmp_path):
    from src.utils.telemetry import TelemetryRecorder

    env = TelemetryRecorder(CrosswindLandingEnv(make_env_config("atr72")), output_dir=tmp_path)
    env.reset(seed=3, options={"wind_speed_mps": 0.0, "turbulence": False, "gusts": False})
    ap = ClassicalAutopilot(env.unwrapped)
    for _ in range(4000):
        _, _, term, trunc, _ = env.step(ap.act())
        if term or trunc:
            break
    doc = env.last_episode
    assert doc["aircraft"]["name"] == "ATR 72-600"
    assert doc["aircraft"]["preset"] == "atr72"
    assert doc["aircraft"]["wheel_height_m"] == pytest.approx(2.55)
    assert doc["aircraft"]["approach_speed_mps"] == 55.0
