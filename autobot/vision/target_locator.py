"""Where the target is, from its YOLO box and the stereo depth map.

Bearing: the ray through the box center. Range: the median depth over
the middle of the box (`core` of its width and height), which skips the
background at the box edges. Together they place the target in the
camera frame, then in the body frame through the camera mount.

Stereo range error grows with the square of the distance:
    sigma_z = z^2 / (f * baseline) * disparity_sigma
so the fix carries a covariance (camera frame) for the target filter.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class TargetFix:
    p_body: np.ndarray        # (3,) m, target relative to the drone's center, FLU
    cov_body: np.ndarray      # (3, 3) m^2
    range_m: float            # along the camera's view axis
    valid_share: float        # share of the box core with a depth reading


@dataclass
class LocatorConfig:
    core: float = 0.5                 # middle share of the box used for range
    min_valid_share: float = 0.2      # of the core, else no fix
    min_range_m: float = 0.3
    max_range_m: float = 15.0
    disparity_sigma_px: float = 0.5   # stereo matching noise
    bearing_sigma_px: float = 3.0     # box-center noise
    center_offset_m: float = 0.0      # add to the surface range to reach the target's center


class TargetLocator:

    def __init__(self, camera, mount, baseline_m, config=None):
        self.camera, self.mount = camera, mount
        self.baseline_m = float(baseline_m)
        self.cfg = config or LocatorConfig()

    def locate(self, box, depth_m):
        """box: (x0, y0, x1, y1) pixels in the rectified left image;
        depth_m: (H, W) distance along the view axis, inf = unknown.
        Returns a TargetFix, or None without a usable range."""
        c = self.cfg
        x0, y0, x1, y1 = (float(v) for v in box)
        u, v = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = c.core * (x1 - x0) / 2, c.core * (y1 - y0) / 2
        H, W = depth_m.shape
        r0, r1 = int(np.clip(np.floor(v - hh), 0, H - 1)), int(np.clip(np.ceil(v + hh), 0, H - 1))
        c0, c1 = int(np.clip(np.floor(u - hw), 0, W - 1)), int(np.clip(np.ceil(u + hw), 0, W - 1))
        patch = depth_m[r0:r1 + 1, c0:c1 + 1]
        ok = np.isfinite(patch) & (patch >= c.min_range_m) & (patch <= c.max_range_m)
        share = float(ok.mean()) if patch.size else 0.0
        if share < c.min_valid_share:
            return None
        ray = self.camera.ray(u, v)
        z = float(np.median(patch[ok])) + c.center_offset_m * ray[2] / np.linalg.norm(ray)
        p_cam = z * ray
        R = self.mount.R_body_cam
        return TargetFix(self.mount.to_body(p_cam), R @ self.cov_camera(z, ray) @ R.T,
                         z, share)

    def cov_camera(self, z, ray, range_sigma=None):
        """Covariance (camera frame) of a target at depth z along ray:
        stereo range error along the ray, pixel noise across it."""
        f = self.camera.fx
        s_z = (z * z / (f * self.baseline_m) * self.cfg.disparity_sigma_px
               if range_sigma is None else range_sigma)
        s_lat = z * self.cfg.bearing_sigma_px / f
        along = ray / np.linalg.norm(ray)
        return (s_z ** 2 - s_lat ** 2) * np.outer(along, along) + s_lat ** 2 * np.eye(3)
