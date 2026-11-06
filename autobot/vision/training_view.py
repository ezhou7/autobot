"""Resample the real camera into the camera the policy was trained with.

The policy saw a square 128x128 pinhole view, 90 deg field of view,
tilted 10 deg up from the body x axis (autobot/policy/vision_spec.py).
The real (rectified left) camera has its own intrinsics and mount. Each
training-camera pixel's ray is rotated into the real camera and sampled
there. The few centimeters between the two camera positions are ignored.

Depth: the policy's depth is the distance along ITS view axis, so the
real z-depth at the sampled pixel is rescaled: for a training ray d_v
(z = 1) seen in the real camera as d_r = R d_v, a point at real depth z_r
lies at z_r / d_r.z along the training view axis.

Where the real camera doesn't cover the training view (a narrower lens),
color repeats the nearest edge pixel and depth reads as far (0).
`coverage` is the covered share; aim for 1.0 with a >= 90x90 deg lens
(wider is fine).
"""
import numpy as np

from autobot.policy import vision_spec as V
from autobot.vision.camera import PinholeCamera, body_from_camera


class TrainingView:

    def __init__(self, real_camera, mount, res=V.CAM_RES, fov_deg=V.CAM_FOV_DEG,
                 tilt_deg=V.CAM_TILT_DEG):
        import cv2
        self._cv2 = cv2
        self.res = res
        virt = PinholeCamera.from_fov(res, res, fov_deg)
        R_real_virt = mount.R_body_cam.T @ body_from_camera(tilt_deg)
        u, v = np.meshgrid(np.arange(res), np.arange(res))
        d_v = np.stack([(u - virt.cx) / virt.fx, (v - virt.cy) / virt.fy,
                        np.ones_like(u, float)], -1)
        self._d_r = d_v @ R_real_virt.T          # training rays in the real camera
        # Color: shrink the frame first (area averaging) so the remap
        # doesn't alias; then sample it at about the training resolution.
        self.shrink = max(1, int(real_camera.fx / (1.5 * virt.fx)))
        small = real_camera.scaled(real_camera.width // self.shrink,
                                   real_camera.height // self.shrink)
        self._color_maps, _ = self._maps(small)
        self._depth_maps, self._inside = self._maps(real_camera)
        z = self._d_r[..., 2]
        self._depth_scale = np.where(self._inside, 1.0 / np.where(z > 1e-6, z, 1.0),
                                     0.0).astype(np.float32)
        self.coverage = float(self._inside.mean())
        self.real_camera = real_camera

    def _maps(self, cam):
        d = self._d_r
        front = d[..., 2] > 1e-6
        z = np.where(front, d[..., 2], 1.0)
        mx = np.where(front, cam.fx * d[..., 0] / z + cam.cx, -1.0)
        my = np.where(front, cam.fy * d[..., 1] / z + cam.cy, -1.0)
        # A pixel covers +-0.5 around its center.
        inside = (front & (mx >= -0.5) & (mx <= cam.width - 0.5)
                  & (my >= -0.5) & (my <= cam.height - 0.5))
        mx = np.where(inside, np.clip(mx, 0, cam.width - 1), mx)
        my = np.where(inside, np.clip(my, 0, cam.height - 1), my)
        return (mx.astype(np.float32), my.astype(np.float32)), inside

    def image(self, left_bgr):
        """Rectified left BGR frame -> the policy's (res, res, 3) RGB uint8."""
        cv2 = self._cv2
        if self.shrink > 1:
            h, w = left_bgr.shape[:2]
            left_bgr = cv2.resize(left_bgr, (w // self.shrink, h // self.shrink),
                                  interpolation=cv2.INTER_AREA)
        out = cv2.remap(left_bgr, *self._color_maps, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    def depth(self, depth_m):
        """Depth (m, inf = unknown) aligned with the rectified left frame ->
        the policy's (res, res, 1) uint8 nearness."""
        cv2 = self._cv2
        d = np.where(np.isfinite(depth_m), depth_m, 1e6).astype(np.float32)
        d = cv2.remap(d, *self._depth_maps, cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=1e6)
        d = np.where(self._inside, d * self._depth_scale, np.inf)
        return V.nearness(d)[..., None]
