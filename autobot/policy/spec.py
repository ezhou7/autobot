# Vendored from ezhou7/dreamerv3 @ e252b39 (aerial/spec/interface.py),
# the parts the deployed policy needs. Keep in sync with upstream.
"""Observation and action layout of the hover policy.

Conventions (the training simulator's):
  * World frame: z up.
  * R: rotation matrix body -> world. Body axes x forward, y left, z up
    (FLU). +roll banks right, +pitch is nose DOWN, +yaw turns left (CCW).
  * Gyro: body angular velocity, rad/s.
ArduPilot uses north-east-down / forward-right-down instead; see
autobot/mavlink/frames.py for the conversion.
"""
import numpy as np

# Control rate the policy was trained at.
STEP_RATE_HZ = 50

# Body-rate command scale: action[0:3] in [-1, 1] -> +-6 rad/s.
MAX_BODY_RATE_RPY = 6.0  # rad/s

# Action (4,), each in [-1, 1]:
#   0..2: body rate roll, pitch, yaw  (x MAX_BODY_RATE_RPY)
#   3:    collective thrust, (a + 1) / 2 in [0, 1] of max thrust
ACTION_DIM = 4

# Observation (22,): see get_obs_vector().
OBS_DIM = 22


def get_obs_vector(pos, vel, R, gyro, last_action, target):
    """Pack the 22-dim observation.

       [0:3]   position relative to target (world frame)
       [3:6]   linear velocity (world frame)
       [6:15]  rotation matrix body->world, row-major
       [15:18] gyro (body frame)
       [18:22] last action
    """
    pos = np.asarray(pos, dtype=np.float32).reshape(3)
    vel = np.asarray(vel, dtype=np.float32).reshape(3)
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    gyro = np.asarray(gyro, dtype=np.float32).reshape(3)
    last_action = np.asarray(last_action, dtype=np.float32).reshape(4)
    target = np.asarray(target, dtype=np.float32).reshape(3)

    obs = np.empty(OBS_DIM, dtype=np.float32)
    obs[0:3] = pos - target
    obs[3:6] = vel
    obs[6:15] = R.reshape(-1)
    obs[15:18] = gyro
    obs[18:22] = last_action
    return obs


def scale_action(raw_action):
    """Policy output [-1, 1]^4 -> (body rates rad/s (3,), thrust 0..1)."""
    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    body_rates = raw[:3] * MAX_BODY_RATE_RPY
    thrust_norm = float(np.clip((raw[3] + 1.0) * 0.5, 0.0, 1.0))
    return body_rates, thrust_norm
