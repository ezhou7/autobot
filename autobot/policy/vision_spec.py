# Constants vendored from ezhou7/dreamerv3 aerial/envs/track_12in_env.py (the
# Aerial-trackvisionpos/-trackvisionlook envs, on the training branch
# chore/aerial-remove-curriculum, not yet merged). Keep in sync with upstream.
"""Observation layout of the camera tracking policy.

The policy sees three inputs each 50 Hz step (training conventions; see
autobot/policy/spec.py for the frames):

  image   (128, 128, 3) uint8 RGB from a front camera: 90 deg field of
          view, square, tilted 10 deg up from the body x axis, mounted
          0.20 m forward and 0.03 m up from the drone's center
  depth   (128, 128, 1) uint8 nearness of the same view: 255 at the lens,
          0 at DEPTH_MAX_M or beyond (distance along the view axis)
  vector  (23,) float32, see get_vision_vector()

It was trained to hold VISION_STANDOFF_M from the target. The target's
position arrives in `vector` as the drone's position relative to it, as a
tracker would report it: the training env added 5 cm of noise and up to
0.1 s of delay. Here it comes from YOLO and stereo depth
(autobot/vision/).
"""
import numpy as np

CAM_RES = 128
CAM_FOV_DEG = 90.0
CAM_TILT_DEG = 10.0                      # up from the body x axis
CAM_OFFSET_FLU_M = np.array([0.20, 0.0, 0.03])
DEPTH_MAX_M = 20.0
RANGEFINDER_MAX_M = 40.0
VISION_STANDOFF_M = 3.0

VECTOR_DIM = 23
IMAGE_CHANNELS = 4                       # depth (1) + image (3), as the encoder stacks them


def get_vision_vector(R, gyro, vel, agl, last_action, rel_to_target):
    """Pack the 23-dim vector input.

       [0:9]    rotation matrix body -> world, row-major
       [9:12]   gyro (body frame, rad/s)
       [12:15]  linear velocity (world frame, m/s)
       [15]     height above the ground (m), clipped to 0..RANGEFINDER_MAX_M
       [16:20]  last action
       [20:23]  the drone's position minus the target's (world frame, m)
    """
    out = np.empty(VECTOR_DIM, np.float32)
    out[0:9] = np.asarray(R, np.float32).reshape(9)
    out[9:12] = np.asarray(gyro, np.float32).reshape(3)
    out[12:15] = np.asarray(vel, np.float32).reshape(3)
    out[15] = np.clip(float(agl), 0.0, RANGEFINDER_MAX_M)
    out[16:20] = np.asarray(last_action, np.float32).reshape(4)
    out[20:23] = np.asarray(rel_to_target, np.float32).reshape(3)
    return out


def nearness(depth_m):
    """Distance along the view axis (m; inf where unknown) -> the policy's
    uint8 depth input: 255 at the lens, 0 at DEPTH_MAX_M or beyond."""
    d = np.nan_to_num(np.asarray(depth_m, np.float32), nan=np.inf, posinf=np.inf)
    near = np.clip(1.0 - d / DEPTH_MAX_M, 0.0, 1.0) * 255.0
    return near.astype(np.uint8)
