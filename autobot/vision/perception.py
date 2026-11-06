"""The camera side of the tracking policy, in background threads.

For every stereo pair:
  1. rectify it (autobot/vision/camera.py)
  2. find the target in the left image with YOLO (detector.py)
  3. resample the left image into the policy's training camera
     (training_view.py)
A depth worker takes the newest pair whenever it is free (HITNet is far
slower than the camera) and:
  4. computes stereo depth (stereo_depth.py) -> the policy's depth input
  5. ranges the target in that same frame (target_locator.py) and updates
     the target filter with a full 3D fix (target_filter.py)
Between depth results, each frame's YOLO box still updates the filter's
direction to the target: a fix at the filter's predicted range, with a
large range uncertainty.

Positions use the drone's pose when the frame was grabbed (`pose_fn`,
e.g. the latest vehicle state then), so the target's motion is separated
from the drone's.
"""
import threading
import time
from dataclasses import dataclass

import numpy as np

from autobot.mavlink.frames import T, quat_to_rot
from autobot.vision.target_filter import TargetFilter


@dataclass
class Snapshot:
    """What the policy reads each step."""
    image: np.ndarray            # (128, 128, 3) uint8 RGB
    depth: np.ndarray            # (128, 128, 1) uint8 nearness
    image_t: float               # time.monotonic() of the frame
    depth_t: float               # of the frame the depth came from
    target_locked: bool          # YOLO sees the locked target in the newest frame


@dataclass
class Stats:
    frames: int = 0
    depth_maps: int = 0
    fixes: int = 0
    bearing_fixes: int = 0
    detect_ms: float = 0.0
    depth_ms: float = 0.0


class Perception:

    def __init__(self, capture, rectifier, detector, selector, depth, locator,
                 view, pose_fn, target_filter=None, max_depth_age=1.0):
        self.capture, self.rectifier = capture, rectifier
        self.detector, self.selector = detector, selector
        self.depth_fn, self.locator, self.view = depth, locator, view
        self.pose_fn = pose_fn                # () -> (pos_ned (3,), attitude quaternion (4,))
        self.filter = target_filter or TargetFilter()
        self.max_depth_age = max_depth_age
        self.stats = Stats()
        self._lock = threading.Lock()
        self._snapshot = None
        self._depth128 = None
        self._depth_t = -np.inf
        self._range_t = -np.inf               # frame time of the newest stereo range fix
        self._job = None                      # newest pair waiting for the depth worker
        self._job_ready = threading.Condition()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=f, daemon=True)
                         for f in (self._frame_loop, self._depth_loop)]

    def start(self):
        for th in self._threads:
            th.start()
        return self

    def stop(self):
        self._stop.set()
        with self._job_ready:
            self._job_ready.notify_all()
        for th in self._threads:
            th.join(timeout=2.0)

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def target_ned(self, t):
        """(estimate, fix age, range age) at time t: the filter's (position,
        velocity, cov) NED or None; seconds since the newest fix of any
        kind; seconds since the newest stereo range fix."""
        with self._lock:
            return self.filter.estimate(t), self.filter.age(t), t - self._range_t

    # --- threads --------------------------------------------------------

    def _frame_loop(self):
        index = -1
        while not self._stop.is_set():
            frame = self.capture.read(after=index, timeout=0.5)
            if frame is None:
                continue
            index = frame.index
            pose = self.pose_fn()
            left, right = self.rectifier(frame.left, frame.right)
            t0 = time.monotonic()
            det = self.selector.select(self.detector(left))
            self.stats.detect_ms = (time.monotonic() - t0) * 1e3
            image = self.view.image(left)
            with self._job_ready:
                self._job = (frame.t, left, right, det, pose)
                self._job_ready.notify()
            if det is not None:
                self._bearing_fix(frame.t, det.box, pose)
            with self._lock:
                depth = self._depth128
                if depth is None or frame.t - self._depth_t > self.max_depth_age:
                    depth = np.zeros((self.view.res, self.view.res, 1), np.uint8)
                self._snapshot = Snapshot(image, depth, frame.t, self._depth_t,
                                          det is not None)
            self.stats.frames += 1

    def _depth_loop(self):
        while not self._stop.is_set():
            with self._job_ready:
                self._job_ready.wait_for(lambda: self._job is not None or self._stop.is_set())
                if self._stop.is_set():
                    return
                t, left, right, det, pose = self._job
                self._job = None
            t0 = time.monotonic()
            depth = self.depth_fn(left, right)
            self.stats.depth_ms = (time.monotonic() - t0) * 1e3
            depth128 = self.view.depth(depth)
            fix = self.locator.locate(det.box, depth) if det is not None else None
            with self._lock:
                if t >= self._depth_t:
                    self._depth128, self._depth_t = depth128, t
                if fix is not None:
                    p, R = self._to_ned(fix.p_body, pose)
                    if self.filter.update(t, p, R @ fix.cov_body @ R.T):
                        self.stats.fixes += 1
                        self._range_t = max(self._range_t, t)
            self.stats.depth_maps += 1

    def _bearing_fix(self, t, box, pose):
        """Direction from this frame's box; range from the filter."""
        with self._lock:
            est = self.filter.estimate(t)
        if est is None:
            return
        pos_ned, q = pose
        R_ned_body = quat_to_rot(q) @ T                 # FLU body -> NED
        cam_ned = pos_ned + R_ned_body @ self.locator.mount.offset_flu
        R_ned_cam = R_ned_body @ self.locator.mount.R_body_cam
        x0, y0, x1, y1 = box
        ray = R_ned_cam @ self.locator.camera.ray((x0 + x1) / 2, (y0 + y1) / 2)
        unit = ray / np.linalg.norm(ray)
        r = float(unit @ (est[0] - cam_ned))
        if r <= self.locator.cfg.min_range_m:
            return
        ray_cam = self.locator.camera.ray((x0 + x1) / 2, (y0 + y1) / 2)
        z = r / np.linalg.norm(ray_cam)
        cov = R_ned_cam @ self.locator.cov_camera(z, ray_cam, range_sigma=1e3) @ R_ned_cam.T
        with self._lock:
            if self.filter.update(t, cam_ned + r * unit, cov):
                self.stats.bearing_fixes += 1

    @staticmethod
    def _to_ned(p_body_flu, pose):
        """Target position NED from a body-frame (FLU) fix and the drone's
        pose; also returns the FLU body -> NED rotation."""
        pos_ned, q = pose
        R = quat_to_rot(q) @ T
        return pos_ned + R @ np.asarray(p_body_flu, float), R
