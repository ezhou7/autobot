"""Turn vehicle state and the camera into the tracking policy's inputs,
and its action into a clipped body-rate and thrust command.

Inputs each step (autobot/policy/vision_spec.py):
  image, depth  the newest frame from autobot/vision/perception.py,
                resampled into the training camera
  vector        attitude, gyro and velocity in the policy's frame (as the
                hover controller), height above ground, the last action,
                and the drone's position relative to the target: the
                target filter's estimate (YOLO + stereo depth) extrapolated
                to now

The command path and its limits are PolicyController's.
"""
from dataclasses import dataclass

import numpy as np

from autobot.control.policy_controller import Command, PolicyController, SafetyLimits
from autobot.mavlink.frames import gyro_frd_to_flu, quat_to_rot, rates_flu_to_frd
from autobot.policy import vision_spec as V
from autobot.policy.spec import ACTION_DIM

# Where things sit in the logged vector (see vision_spec.get_vision_vector).
LAYOUT = {"R": slice(0, 9), "rel": slice(20, 23), "standoff": V.VISION_STANDOFF_M}


@dataclass
class TrackingLimits(SafetyLimits):
    max_image_age: float = 0.15       # s, newest camera frame
    max_fix_age: float = 2.0          # s since YOLO last placed the target
    max_range_age: float = 3.0        # s since the last stereo range of it
    max_rangefinder_age: float = 0.3  # s, else height falls back to altitude


class VisionPolicyController(PolicyController):
    """Tracks the target perception has locked onto.

    agl: 'rangefinder' uses ArduPilot's downward rangefinder
    (DISTANCE_SENSOR, tilt-corrected) and falls back to altitude above
    the hand-over ground level; 'altitude' always uses the latter (flat
    ground only).
    """
    layout = LAYOUT

    def __init__(self, policy, perception, limits=None, thrust_scale=1.0,
                 agl="rangefinder"):
        super().__init__(policy, limits or TrackingLimits(), thrust_scale)
        self.perception = perception
        self.agl_mode = agl
        self.ground_z_ned = None
        self.extra = {}

    def handover(self, state):
        super().handover(state)
        # Ground level (NED z) below the hand-over point, for the altitude
        # fallback: from the rangefinder if it reads, else the EKF origin.
        R = quat_to_rot(state.attitude_q)
        if np.isfinite(state.range_down_m):
            self.ground_z_ned = state.pos_ned[2] + state.range_down_m * R[2, 2]
        else:
            self.ground_z_ned = 0.0

    def height_above_ground(self, state, now):
        R = quat_to_rot(state.attitude_q)
        fresh = now - state.range_time <= self.limits.max_rangefinder_age
        if self.agl_mode == "rangefinder" and fresh and np.isfinite(state.range_down_m):
            return state.range_down_m * R[2, 2], "rangefinder"
        return self.ground_z_ned - state.pos_ned[2], "altitude"

    def observe(self, state, now):
        """Returns (obs dict, R in the policy's frame) or raises
        _NotReady with the reason."""
        snap = self.perception.snapshot()
        if snap is None:
            raise _NotReady("no camera frames yet")
        if now - snap.image_t > self.limits.max_image_age:
            raise _NotReady(f"camera frame {1e3 * (now - snap.image_t):.0f} ms old")
        est, fix_age, range_age = self.perception.target_ned(now)
        if est is None:
            raise _NotReady("no target fix yet")
        if fix_age > self.limits.max_fix_age:
            raise _NotReady(f"target not seen for {fix_age:.1f} s")
        if range_age > self.limits.max_range_age:
            raise _NotReady(f"no stereo range of the target for {range_age:.1f} s")
        target_ned = est[0]
        R = self.frame.rotation(quat_to_rot(state.attitude_q))
        agl, agl_src = self.height_above_ground(state, now)
        vector = V.get_vision_vector(
            R=R, gyro=gyro_frd_to_flu(state.rates_frd),
            vel=self.frame.vector(state.vel_ned), agl=agl,
            last_action=self._last_action,
            rel_to_target=self.frame.vector(state.pos_ned - target_ned))
        self.extra = dict(
            target_ned=target_ned, fix_age=fix_age, range_age=range_age,
            image_age=now - snap.image_t, depth_age=now - snap.depth_t,
            target_locked=snap.target_locked, agl=agl,
            agl_rangefinder=agl_src == "rangefinder")
        return {"vector": vector, "image": snap.image, "depth": snap.depth}, R

    def step(self, state, now):
        if self.frame is None:
            raise RuntimeError("call handover() first")
        att_age, pos_age = state.ages(now)
        if max(att_age, pos_age) > self.limits.max_state_age:
            return Command(False, f"stale telemetry (attitude {att_age * 1e3:.0f} ms, "
                                  f"position {pos_age * 1e3:.0f} ms)")
        try:
            obs, R = self.observe(state, now)
        except _NotReady as e:
            return Command(False, str(e))
        vec = obs["vector"]
        if not np.all(np.isfinite(vec)):
            return Command(False, "non-finite observation", obs=vec)
        tilt_deg = np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
        if tilt_deg > self.limits.max_tilt_deg:
            return Command(False, f"tilt {tilt_deg:.0f} deg, limit "
                                  f"{self.limits.max_tilt_deg:.0f} deg", obs=vec)

        action = np.asarray(self.policy.step(obs, is_first=self._first), np.float32)
        self._first = False
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            return Command(False, "non-finite action", obs=vec, action=action)
        self._last_action = action
        rates, thrust, clipped = self._apply_limits(action)
        return Command(True, "", rates_flu_to_frd(rates), thrust, action, vec, clipped)


class _NotReady(Exception):
    pass
