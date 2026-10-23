"""Convert between ArduPilot's frames and the policy's frames.

ArduPilot (MAVLink):
  * world: north-east-down (NED)
  * body:  forward-right-down (FRD); +pitch rate = nose up,
    +yaw rate = turn right (clockwise seen from above)
Policy (training simulator, autobot/policy/spec.py):
  * world: z up
  * body:  forward-left-up (FLU); +pitch rate = nose DOWN,
    +yaw rate = turn left

Flipping the 2nd and 3rd axes, T = diag(1, -1, -1), maps NED to
north-west-up and FRD to FLU (T is its own inverse).

The policy was trained with the drone facing roughly along +x of its
world frame (heading within +-30 deg). ModelFrame therefore puts its
origin and +x axis at the drone's position and heading at hand-over,
so the policy starts at heading 0 wherever the drone points.
"""
from dataclasses import dataclass

import numpy as np
from pymavlink.quaternion import QuaternionBase

T = np.diag([1.0, -1.0, -1.0])


def quat_to_rot(q):
    """MAVLink quaternion (w, x, y, z), body -> world, to a 3x3 matrix."""
    q = np.asarray(q, dtype=float)
    return QuaternionBase(q / np.linalg.norm(q)).dcm


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_flu_to_world_up(R_ned):
    """FRD->NED rotation to FLU->north-west-up."""
    return T @ np.asarray(R_ned, dtype=float) @ T


def heading_up(R_up):
    """Heading in a z-up frame: angle of the body x axis, CCW from +x."""
    return float(np.arctan2(R_up[1, 0], R_up[0, 0]))


def rates_flu_to_frd(rates_flu):
    """Policy body rates (FLU) -> ArduPilot body rates (FRD)."""
    return T @ np.asarray(rates_flu, dtype=float)


def gyro_frd_to_flu(gyro_frd):
    return T @ np.asarray(gyro_frd, dtype=float)


@dataclass(frozen=True)
class ModelFrame:
    """The policy's world frame: z up, origin and +x axis at the drone's
    position and heading at hand-over (x forward, y left, z up)."""

    origin_ned: np.ndarray   # (3,) m
    heading: float           # rad, CCW from north in north-west-up

    @classmethod
    def at(cls, pos_ned, R_ned):
        return cls(np.asarray(pos_ned, dtype=float).copy(),
                   heading_up(body_flu_to_world_up(R_ned)))

    @property
    def H(self):
        return rot_z(self.heading)

    def position(self, pos_ned):
        """NED position -> model frame (m)."""
        return self.H.T @ (T @ (np.asarray(pos_ned, dtype=float) - self.origin_ned))

    def vector(self, vec_ned):
        """NED vector (e.g. velocity) -> model frame."""
        return self.H.T @ (T @ np.asarray(vec_ned, dtype=float))

    def rotation(self, R_ned):
        """FRD->NED attitude -> FLU->model frame rotation matrix."""
        return self.H.T @ body_flu_to_world_up(R_ned)
