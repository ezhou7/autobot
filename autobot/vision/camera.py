"""Camera geometry: pinhole intrinsics, where the camera sits on the drone,
and stereo calibration / rectification.

Frames:
  camera  OpenCV convention: x right, y down, z forward (the view axis)
  body    the policy's forward-left-up (FLU), origin at the drone's center
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    """Undistorted (rectified) pinhole camera; pixel centers at integer
    coordinates."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def ray(self, u, v):
        """Direction (camera frame) through pixel (u, v), with z = 1."""
        return np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])

    def scaled(self, width, height):
        """The same camera resampled to width x height."""
        sx, sy = width / self.width, height / self.height
        return PinholeCamera(self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5,
                             (self.cy + 0.5) * sy - 0.5, int(width), int(height))

    @classmethod
    def from_fov(cls, width, height, hfov_deg):
        f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        return cls(f, f, (width - 1) / 2, (height - 1) / 2, int(width), int(height))


def body_from_camera(tilt_up_deg, yaw_left_deg=0.0):
    """Rotation camera -> body (FLU) for a camera looking along body +x,
    pitched up by tilt_up_deg and turned left by yaw_left_deg."""
    t, y = np.radians(tilt_up_deg), np.radians(yaw_left_deg)
    z = np.array([np.cos(t) * np.cos(y), np.cos(t) * np.sin(y), np.sin(t)])  # view axis
    x = np.array([np.sin(y), -np.cos(y), 0.0])                                # image right
    return np.column_stack([x, np.cross(z, x), z])                           # y = z x x (down)


@dataclass(frozen=True)
class CameraMount:
    """Where the (rectified left) camera sits on the drone."""
    offset_flu: np.ndarray       # (3,) m, camera center in the body frame
    R_body_cam: np.ndarray       # (3, 3) camera -> body

    @classmethod
    def make(cls, offset_flu=(0.0, 0.0, 0.0), tilt_up_deg=0.0, yaw_left_deg=0.0,
             R_rect=None):
        """R_rect: stereo rectification rotation of the left camera
        (x_rect = R_rect @ x_cam), so the mount describes the rectified
        camera the images come from."""
        R = body_from_camera(tilt_up_deg, yaw_left_deg)
        if R_rect is not None:
            R = R @ np.asarray(R_rect, float).T
        return cls(np.asarray(offset_flu, float), R)

    def to_body(self, p_cam):
        return self.offset_flu + self.R_body_cam @ np.asarray(p_cam, float)


@dataclass
class StereoCalibration:
    """OpenCV stereo calibration of a camera pair (cv2.stereoCalibrate):
    intrinsics K/D of each camera and the pose (R, T) of the right camera
    relative to the left; T in meters."""
    image_size: tuple            # (width, height)
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray

    @classmethod
    def from_dict(cls, d):
        a = lambda k, shape=None: (np.asarray(d[k], float).reshape(shape)
                                   if shape else np.asarray(d[k], float))
        return cls(tuple(int(v) for v in d["image_size"]), a("K1", (3, 3)),
                   a("D1"), a("K2", (3, 3)), a("D2"), a("R", (3, 3)), a("T", (3,)))

    @property
    def baseline_m(self):
        return float(np.linalg.norm(self.T))


class Rectifier:
    """Undistorts and rectifies stereo pairs so matching rows line up.
    After rectification both cameras share `camera` (intrinsics) and are
    `baseline_m` apart along the image x axis."""

    def __init__(self, calib):
        import cv2
        self._cv2 = cv2
        size = calib.image_size
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            calib.K1, calib.D1, calib.K2, calib.D2, size, calib.R, calib.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.R_rect = R1
        self.maps = [cv2.initUndistortRectifyMap(K, D, Rr, P, size, cv2.CV_16SC2)
                     for K, D, Rr, P in ((calib.K1, calib.D1, R1, P1),
                                         (calib.K2, calib.D2, R2, P2))]
        self.camera = PinholeCamera(P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2], *size)
        self.baseline_m = float(-P2[0, 3] / P2[0, 0])

    def __call__(self, left, right):
        cv2 = self._cv2
        return tuple(cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
                     for img, (m1, m2) in zip((left, right), self.maps))
