"""Turn vehicle state into the policy's observation, and the policy's
action into a clipped body-rate and thrust command.

No input/output, so it is unit-tested without a drone.
"""
from dataclasses import dataclass, field

import numpy as np

from autobot.mavlink.frames import (
    ModelFrame, gyro_frd_to_flu, quat_to_rot, rates_flu_to_frd)
from autobot.policy.spec import ACTION_DIM, get_obs_vector, scale_action

# Hover thrust of the F450 model the policy was trained on (0.501 in
# ezhou7/dreamerv3 aerial/configs/airframe_12in.py). Commands are scaled by
# MOT_THST_HOVER / TRAINED_HOVER_THRUST so the policy's hover thrust becomes
# this vehicle's hover throttle.
TRAINED_HOVER_THRUST = 0.50


@dataclass
class SafetyLimits:
    max_rate: float = 3.0          # rad/s, each body axis
    thrust_min: float = 0.10       # throttle, after scaling
    thrust_max: float = 0.80
    max_tilt_deg: float = 45.0     # stop if tilted more than this
    max_state_age: float = 0.10    # s; stop if telemetry is older


@dataclass
class Command:
    ok: bool
    reason: str = ""
    rates_frd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    thrust: float = 0.0
    action: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_DIM))
    obs: np.ndarray = None
    clipped: bool = False


class PolicyController:
    """Hovers at the drone's position at hand-over.
    thrust_scale: MOT_THST_HOVER / TRAINED_HOVER_THRUST.
    """

    def __init__(self, policy, limits=None, thrust_scale=1.0):
        self.policy = policy
        self.limits = limits or SafetyLimits()
        self.thrust_scale = float(thrust_scale)
        self.frame = None
        self._last_action = np.zeros(ACTION_DIM, np.float32)
        self._first = True

    def handover(self, state):
        """Start a new policy episode at the drone's current pose."""
        self.frame = ModelFrame.at(state.pos_ned, quat_to_rot(state.attitude_q))
        self.policy.reset()
        self._last_action = np.zeros(ACTION_DIM, np.float32)
        self._first = True

    def observe(self, state):
        # The hover point is the hand-over position: the origin of self.frame.
        R = self.frame.rotation(quat_to_rot(state.attitude_q))
        obs = get_obs_vector(
            pos=self.frame.position(state.pos_ned),
            vel=self.frame.vector(state.vel_ned),
            R=R,
            gyro=gyro_frd_to_flu(state.rates_frd),
            last_action=self._last_action,
            target=np.zeros(3))
        return obs, R

    def step(self, state, now):
        """Safety checks, then the model, then a command within the limits."""
        if self.frame is None:
            raise RuntimeError("call handover() first")

        att_age, pos_age = state.ages(now)
        if max(att_age, pos_age) > self.limits.max_state_age:
            return Command(False, f"stale telemetry (attitude {att_age * 1e3:.0f} ms, "
                                  f"position {pos_age * 1e3:.0f} ms)")
        obs, R = self.observe(state)
        if not np.all(np.isfinite(obs)):
            return Command(False, "non-finite observation", obs=obs)
        tilt_deg = np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
        if tilt_deg > self.limits.max_tilt_deg:
            return Command(False, f"tilt {tilt_deg:.0f} deg, limit "
                                  f"{self.limits.max_tilt_deg:.0f} deg", obs=obs)

        action = np.asarray(self.policy.step(obs, is_first=self._first), np.float32)
        self._first = False
        if not np.all(np.isfinite(action)):
            return Command(False, "non-finite action", obs=obs, action=action)
        self._last_action = action

        rates, thrust, clipped = self._apply_limits(action)
        return Command(True, "", rates_flu_to_frd(rates), thrust, action, obs, clipped)

    def _apply_limits(self, action):
        """Scale the model's action and clip it to the safety limits.
        Returns body rates (FLU, rad/s), throttle, and whether anything was clipped."""
        rates, thrust = scale_action(action)
        thrust *= self.thrust_scale
        safe_rates = np.clip(rates, -self.limits.max_rate, self.limits.max_rate)
        safe_thrust = float(np.clip(thrust, self.limits.thrust_min, self.limits.thrust_max))
        clipped = not np.allclose(safe_rates, rates) or safe_thrust != thrust
        return safe_rates, safe_thrust, clipped
