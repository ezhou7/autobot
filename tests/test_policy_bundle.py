"""End-to-end checks with a real policy bundle (skipped without one).

    AUTOBOT_TEST_BUNDLE=~/models/skydreamer-smooth-9.4m-1c-2026-10-18 pytest tests

Feeds ArduPilot-style states through PolicyController and checks the
trained policy reacts in the physically right direction. This catches
frame/sign mistakes that unit tests on the conversion alone could miss.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from autobot.control import PolicyController
from autobot.mavlink.vehicle import VehicleState
from autobot.policy import load_policy

BUNDLE = os.environ.get("AUTOBOT_TEST_BUNDLE")
pytestmark = pytest.mark.skipif(
    not BUNDLE or not Path(BUNDLE).expanduser().is_dir(),
    reason="set AUTOBOT_TEST_BUNDLE to a policy bundle directory")


@pytest.fixture(scope="module")
def policy():
    pol, _ = load_policy(BUNDLE, sampling="deterministic")
    return pol


def mean_command(policy, pos_ned, yaw_deg=0.0, steps=15):
    """Hand over facing yaw_deg, then hold the drone at
    pos_ned (NED, m from the hand-over point) and average the commands."""
    a = np.radians(yaw_deg) / 2
    q = np.array([np.cos(a), 0, 0, np.sin(a)])
    c = PolicyController(policy)
    t = 100.0
    c.handover(VehicleState(q, np.zeros(3), t, np.zeros(3), np.zeros(3), t, "GUIDED", True, t))
    rates, thrust = [], []
    for i in range(steps):
        t += 0.02
        s = VehicleState(q, np.zeros(3), t, np.asarray(pos_ned, float), np.zeros(3), t,
                         "GUIDED", True, t)
        cmd = c.step(s, t)
        assert cmd.ok, cmd.reason
        rates.append(cmd.rates_frd)
        thrust.append(cmd.thrust)
    return np.mean(rates, 0), float(np.mean(thrust))


def test_at_hover_point_hovers(policy):
    rates, thrust = mean_command(policy, [0, 0, 0])
    assert 0.35 < thrust < 0.65
    assert np.all(np.abs(rates) < 1.0)


def test_below_hover_point_climbs(policy):
    _, hover = mean_command(policy, [0, 0, 0])
    _, below = mean_command(policy, [0, 0, 1.0])        # 1 m below (NED z down)
    assert below > hover + 0.03


@pytest.mark.parametrize("yaw_deg", [0, 90, -135])
def test_ahead_of_hover_point_pitches_nose_up(policy, yaw_deg):
    # 1 m ahead of the hover point along the hand-over heading: fly back,
    # i.e. pitch nose up (ArduPilot FRD pitch rate > 0).
    yaw = np.radians(yaw_deg)
    rates, _ = mean_command(policy, [np.cos(yaw), np.sin(yaw), 0], yaw_deg)
    assert rates[1] > 0.05


@pytest.mark.parametrize("yaw_deg", [0, 90])
def test_right_of_hover_point_banks_left(policy, yaw_deg):
    # 1 m to the right of the hover point: bank left (FRD roll rate < 0).
    yaw = np.radians(yaw_deg)
    right = [-np.sin(yaw), np.cos(yaw), 0]   # body +y (right) in NED
    rates, _ = mean_command(policy, right, yaw_deg)
    assert rates[0] < -0.05
