"""Tests for telemetry export (frame conversions, strict JSON) and reward shaping."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.agents.reward_shaping import (
    FlightSnapshot,
    LandingRewardShaper,
    RewardConfig,
    TouchdownEvent,
)
from src.env.aircraft_env import CrosswindLandingEnv, euler_to_quat
from src.utils.telemetry import (
    B_BLENDER,
    B_THREEJS,
    W_BLENDER,
    W_THREEJS,
    TelemetryRecorder,
    convert_pose,
    load_telemetry,
    quat_wxyz_to_matrix,
)


# ----------------------------------------------------------------------------- telemetry
@pytest.mark.parametrize("m", [W_BLENDER, W_THREEJS, B_BLENDER, B_THREEJS])
def test_frame_matrices_are_proper_rotations(m):
    assert np.allclose(m @ m.T, np.eye(3))
    assert np.isclose(np.linalg.det(m), 1.0)


def test_level_northbound_pose_in_target_conventions():
    q = euler_to_quat(0.0, 0.0, 0.0)  # nose north, wings level
    pos = np.array([100.0, 5.0, -20.0])  # 100 m north, 5 m east, 20 m up

    p_bl, q_bl = convert_pose(pos, q, W_BLENDER, B_BLENDER)
    r_bl = quat_wxyz_to_matrix([q_bl[3], *q_bl[:3]])
    assert np.allclose(p_bl, [5.0, 100.0, 20.0])
    assert np.allclose(r_bl @ [1, 0, 0], [0, 1, 0])  # model nose (+X) -> north (+Y)
    assert np.allclose(r_bl @ [0, 0, 1], [0, 0, 1])  # model up -> world up

    p_th, q_th = convert_pose(pos, q, W_THREEJS, B_THREEJS)
    r_th = quat_wxyz_to_matrix([q_th[3], *q_th[:3]])
    assert np.allclose(p_th, [5.0, 20.0, -100.0])
    assert np.allclose(r_th @ [0, 0, 1], [0, 0, -1])  # glTF nose (+Z) -> north (-Z)
    assert np.allclose(r_th @ [0, 1, 0], [0, 1, 0])  # glTF up -> world up


def test_right_bank_rolls_right_wing_down_in_blender():
    q = euler_to_quat(math.radians(30.0), 0.0, 0.0)
    _, q_bl = convert_pose(np.zeros(3), q, W_BLENDER, B_BLENDER)
    r = quat_wxyz_to_matrix([q_bl[3], *q_bl[:3]])
    right_wing_world = r @ np.array([0.0, -1.0, 0.0])  # model -Y is the right wing
    assert right_wing_world[2] < 0.0


def test_recorder_exports_strict_json(tmp_path):
    env = TelemetryRecorder(CrosswindLandingEnv(), output_dir=tmp_path)
    env.reset(seed=1, options={"wind_speed_mps": 6.0, "wind_from_deg": 90.0})
    for _ in range(4000):
        _, _, term, trunc, _ = env.step(np.zeros(4, dtype=np.float32))
        if term or trunc:
            break
    assert env.last_path is not None and env.last_path.exists()
    text = env.last_path.read_text(encoding="utf-8")
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    doc = load_telemetry(env.last_path)
    assert doc["episode"]["n_frames"] == len(doc["frames"]) > 10
    f = doc["frames"][-1]
    for key in ("ned", "blender", "threejs", "controls", "wind_ned", "airdata", "gear_loads_n"):
        assert key in f
    assert len(f["blender"]["rotation_quaternion"]) == 4
    q = np.array([fr["threejs"]["quaternion"] for fr in doc["frames"]])
    assert np.all(np.sum(q[1:] * q[:-1], axis=1) >= 0.0)  # hemisphere-continuous keys


# ----------------------------------------------------------------------------- reward
def _snapshot(**kw) -> FlightSnapshot:
    base = dict(
        x_m=-1000.0, y_m=0.0, h_m=68.0, h_dot_mps=-1.7, lateral_speed_mps=0.0,
        glideslope_error_m=0.0, distance_to_aim_m=1300.0, heading_error_rad=0.0,
        phi_rad=0.0, theta_rad=0.05, p_rps=0.0, q_rps=0.0, r_rps=0.0,
        airspeed_mps=33.0, airspeed_ref_mps=33.0, alpha_rad=0.1, beta_rad=0.0,
        action=np.zeros(4), prev_action=np.zeros(4), weight_on_wheels=False,
        touched_down=False, runway_half_width_m=15.0, aim_point_x_m=300.0,
    )
    base.update(kw)
    return FlightSnapshot(**base)


def test_potential_prefers_on_path_states():
    shaper = LandingRewardShaper()
    assert shaper.potential(_snapshot()) > shaper.potential(_snapshot(glideslope_error_m=30.0))
    assert shaper.potential(_snapshot()) > shaper.potential(_snapshot(y_m=40.0))


def test_shaping_telescopes_to_potential_difference():
    """With gamma = 1 the summed shaping reward equals Phi(s_T) - Phi(s_0)."""
    cfg = RewardConfig(gamma=1.0)
    shaper = LandingRewardShaper(cfg)
    traj = [_snapshot(glideslope_error_m=e, y_m=y) for e, y in [(20, 30), (10, 12), (3, 5), (0.5, 1)]]
    shaper.reset(traj[0])
    total = sum(shaper.compute(s, terminated=False).shaping for s in traj[1:])
    assert total == pytest.approx(shaper.potential(traj[-1]) - shaper.potential(traj[0]))


def _td(sink: float, x: float = 300.0, y: float = 0.0) -> TouchdownEvent:
    return TouchdownEvent(0.0, x, y, sink, 0.0, 0.0, 0.05, 30.0, 0.0)


def test_touchdown_reward_ordering_and_crash_threshold():
    shaper = LandingRewardShaper()
    gentle, _ = shaper.touchdown_reward(_td(-0.3), 15.0, 300.0)
    firm, _ = shaper.touchdown_reward(_td(-1.2), 15.0, 300.0)
    long_, _ = shaper.touchdown_reward(_td(-0.3, x=700.0), 15.0, 300.0)
    crash, f = shaper.touchdown_reward(_td(-1.6), 15.0, 300.0)
    undershoot, _ = shaper.touchdown_reward(_td(-0.3, x=-5.0), 15.0, 300.0)
    assert gentle > firm > 0.0
    assert gentle > long_ > 0.0
    assert crash < 0.0 and f["td_crash"] == 1.0
    assert undershoot == shaper.config.crash_penalty


def test_dense_penalty_is_bounded():
    shaper = LandingRewardShaper()
    s = _snapshot(phi_rad=1.2, theta_rad=0.7, alpha_rad=0.6, p_rps=3.0, q_rps=3.0, r_rps=3.0,
                  airspeed_mps=15.0, h_m=3.0, heading_error_rad=0.5, action=np.ones(4), prev_action=-np.ones(4))
    shaper.reset(s)
    b = shaper.compute(s, terminated=False)
    dense = b.attitude + b.rates + b.airspeed + b.alpha + b.effort + b.decrab + b.flare + b.rollout
    assert dense == pytest.approx(-shaper.config.max_dense_penalty_per_step)
