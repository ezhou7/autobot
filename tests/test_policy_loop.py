"""PolicyLoop with a fake vehicle: stop conditions and the fallback mode."""
import time

import numpy as np

from autobot.control import PolicyController, PolicyLoop
from autobot.mavlink.vehicle import VehicleState


class ZeroPolicy:
    def reset(self):
        pass

    def step(self, obs, is_first=False):
        return np.zeros(4, np.float32)


class FakeVehicle:
    """Level at 2 m in GUIDED. Switches to `switch_to` after `switch_after`
    commands; `telemetry_age` makes every state that many seconds old."""

    def __init__(self, switch_after=None, switch_to="LOITER", armed=True, telemetry_age=0.0):
        self.mode = "GUIDED"
        self.armed = armed
        self.switch_after = switch_after
        self.switch_to = switch_to
        self.telemetry_age = telemetry_age
        self.sent = []
        self.mode_requests = []

    def state(self):
        t = time.monotonic() - self.telemetry_age
        return VehicleState(np.array([1.0, 0, 0, 0]), np.zeros(3), t,
                            np.array([0, 0, -2.0]), np.zeros(3), t, self.mode, self.armed, t)

    def send_rates_thrust(self, rates_frd, thrust):
        self.sent.append((rates_frd, thrust))
        if self.switch_after is not None and len(self.sent) >= self.switch_after:
            self.mode = self.switch_to

    def set_mode(self, name):
        self.mode_requests.append(name)
        self.mode = name


def run(vehicle, duration_s):
    return PolicyLoop(vehicle, PolicyController(ZeroPolicy()), fallback_mode="BRAKE").run(duration_s)


def test_completes_and_logs_every_step():
    vehicle = FakeVehicle()
    result = run(vehicle, 0.1)
    assert result.completed and not result.taken_over
    assert result.steps == len(vehicle.sent) > 0
    assert result.log["obs"].shape == (result.steps, 22)
    assert result.summary()["steps"] == result.steps
    assert vehicle.mode_requests == []


def test_mode_change_is_a_takeover():
    vehicle = FakeVehicle(switch_after=3)
    result = run(vehicle, 5.0)
    assert result.taken_over and not result.completed
    assert result.steps == 3
    assert vehicle.mode_requests == []       # the pilot is in charge: no fallback


def test_disarm_is_a_takeover():
    result = run(FakeVehicle(armed=False), 5.0)
    assert result.taken_over and result.reason == "disarmed"


def test_safety_stop_requests_fallback():
    vehicle = FakeVehicle(telemetry_age=0.5)
    result = run(vehicle, 5.0)
    assert not result.completed and not result.taken_over
    assert "stale" in result.reason
    assert vehicle.mode_requests == ["BRAKE"]
    assert result.summary() == {"steps": 0}
