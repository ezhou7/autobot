"""PolicyController: observation building, scaling, safety limits."""
import numpy as np
import pytest

from autobot.control import PolicyController, SafetyLimits
from autobot.mavlink.vehicle import VehicleState


class FixedPolicy:
    """Returns a fixed action and remembers what it was fed."""

    def __init__(self, action):
        self.action = np.asarray(action, np.float32)
        self.calls = []

    def reset(self):
        self.calls.clear()

    def step(self, obs, is_first=False):
        self.calls.append((np.array(obs), is_first))
        return self.action.copy()


def state(pos=(0, 0, -2), vel=(0, 0, 0), q=(1, 0, 0, 0), rates=(0, 0, 0), t=100.0):
    return VehicleState(np.array(q, float), np.array(rates, float), t,
                        np.array(pos, float), np.array(vel, float), t,
                        "GUIDED", True, t)


def test_observation_at_handover():
    pol = FixedPolicy([0, 0, 0, 0])
    c = PolicyController(pol)
    s = state(vel=(1.0, 0, 0.5), rates=(0.1, 0.2, 0.3))
    c.handover(s)
    cmd = c.step(s, now=100.01)
    obs, first = pol.calls[0]
    assert first and cmd.ok
    np.testing.assert_allclose(obs[0:3], 0, atol=1e-6)                 # at the hover point
    np.testing.assert_allclose(obs[3:6], [1.0, 0, -0.5], atol=1e-6)    # north = +x, down = -z
    np.testing.assert_allclose(obs[6:15], np.eye(3).ravel(), atol=1e-6)
    np.testing.assert_allclose(obs[15:18], [0.1, -0.2, -0.3], atol=1e-6)
    np.testing.assert_allclose(obs[18:22], 0)


def test_position_offset_and_last_action():
    pol = FixedPolicy([0.1, 0, 0, 0])
    c = PolicyController(pol)
    c.handover(state(pos=(0, 0, -2)))
    c.step(state(pos=(0, 0, -2)), 100.01)
    c.step(state(pos=(0, 0, -1)), 100.03)    # sank to 1 m altitude
    obs, first = pol.calls[1]
    assert not first
    np.testing.assert_allclose(obs[0:3], [0, 0, -1.0], atol=1e-6)      # 1 m below the hover point
    np.testing.assert_allclose(obs[18:22], [0.1, 0, 0, 0], atol=1e-6)


def test_command_signs_and_scaling():
    # Policy: roll right, pitch nose down, yaw left, thrust 0.5.
    pol = FixedPolicy([0.2, 0.1, 0.05, 0.0])
    c = PolicyController(pol, SafetyLimits(max_rate=3.0), thrust_scale=0.8)
    s = state()
    c.handover(s)
    cmd = c.step(s, 100.01)
    np.testing.assert_allclose(cmd.rates_frd, [1.2, -0.6, -0.3], atol=1e-6)
    assert cmd.thrust == pytest.approx(0.4)
    assert not cmd.clipped


def test_rate_and_thrust_clipping():
    pol = FixedPolicy([1.0, -1.0, 0.0, 1.0])
    c = PolicyController(pol, SafetyLimits(max_rate=3.0, thrust_max=0.8))
    s = state()
    c.handover(s)
    cmd = c.step(s, 100.01)
    np.testing.assert_allclose(cmd.rates_frd, [3.0, 3.0, 0.0])
    assert cmd.thrust == pytest.approx(0.8) and cmd.clipped


def test_stale_telemetry_stops():
    c = PolicyController(FixedPolicy([0, 0, 0, 0]))
    s = state(t=100.0)
    c.handover(s)
    cmd = c.step(s, now=100.5)
    assert not cmd.ok and "stale" in cmd.reason


def test_large_tilt_stops():
    c = PolicyController(FixedPolicy([0, 0, 0, 0]), SafetyLimits(max_tilt_deg=45))
    c.handover(state())
    a = np.radians(60) / 2
    cmd = c.step(state(q=(np.cos(a), np.sin(a), 0, 0)), 100.01)    # rolled 60 deg
    assert not cmd.ok and "tilt" in cmd.reason
