"""The camera side of the tracking policy, without hardware.

Run:  .venv/bin/python -m pytest tests/test_vision.py -q
"""
import time

import numpy as np
import pytest

from autobot.policy import vision_spec as V
from autobot.vision.camera import CameraMount, PinholeCamera, body_from_camera
from autobot.vision.target_filter import TargetFilter
from autobot.vision.target_locator import TargetLocator
from autobot.vision.training_view import TrainingView

cv2 = pytest.importorskip("cv2")


def training_like_camera():
    return (PinholeCamera.from_fov(V.CAM_RES, V.CAM_RES, V.CAM_FOV_DEG),
            CameraMount.make(V.CAM_OFFSET_FLU_M, V.CAM_TILT_DEG))


def test_body_from_camera_axes():
    R = body_from_camera(0.0)
    np.testing.assert_allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-12)   # view axis -> forward
    np.testing.assert_allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-12)  # image right -> body right
    np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, -1], atol=1e-12)  # image down -> body down
    up = body_from_camera(10.0) @ [0, 0, 1]
    assert up[2] == pytest.approx(np.sin(np.radians(10)))


def test_training_view_is_identity_for_the_training_camera():
    cam, mount = training_like_camera()
    view = TrainingView(cam, mount)
    assert view.coverage == 1.0
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (V.CAM_RES, V.CAM_RES, 3), dtype=np.uint8)
    np.testing.assert_array_equal(view.image(bgr), bgr[..., ::-1])
    depth = rng.uniform(0.5, 30, (V.CAM_RES, V.CAM_RES)).astype(np.float32)
    np.testing.assert_array_equal(view.depth(depth)[..., 0], V.nearness(depth))


def test_training_view_tilt_and_depth_along_its_axis():
    """A level 640x480, 130 deg camera; a wall 5 m ahead. (Tilted 10 deg
    up, the training view reaches 55 deg above level; 130 x 116 deg
    covers that.)"""
    cam = PinholeCamera.from_fov(640, 480, 130.0)
    view = TrainingView(cam, CameraMount.make((0.2, 0, 0.03), tilt_up_deg=0.0))
    assert view.coverage == 1.0
    # The training camera looks 10 deg up, so the real camera's center
    # (level) shows up 64 * tan(10 deg) pixels below its center.
    mx, my = view._depth_maps
    v = 63.5 + 64 * np.tan(np.radians(10))
    row = int(round(v))
    assert mx[row, 64] == pytest.approx(cam.cx + cam.fx / 64 * 0.5, abs=1.0)
    assert my[row, 64] == pytest.approx(cam.cy, abs=cam.fy / 64 * 0.6)
    # A wall at z = 5 m is 5 / cos(10 deg) m along the training axis at its center.
    near = view.depth(np.full((480, 640), 5.0, np.float32))
    expect = V.nearness(np.array([5.0 / np.cos(np.radians(10))]))[0]
    assert abs(int(near[64, 64, 0]) - int(expect)) <= 1


def test_narrow_camera_reports_coverage():
    view = TrainingView(PinholeCamera.from_fov(640, 480, 60.0), CameraMount.make())
    assert view.coverage < 0.6
    near = view.depth(np.full((480, 640), 3.0, np.float32))
    assert near[0, 0, 0] == 0                     # uncovered corner reads far


def test_locator_places_the_target():
    cam = PinholeCamera.from_fov(640, 480, 90.0)
    mount = CameraMount.make((0.2, 0.0, 0.03), tilt_up_deg=10.0)
    loc = TargetLocator(cam, mount, baseline_m=0.06)
    depth = np.full((480, 640), np.inf, np.float32)
    depth[200:280, 330:390] = 4.0                 # the target
    fix = loc.locate((330, 200, 389, 279), depth)
    u, v = (330 + 389) / 2, (200 + 279) / 2
    p_cam = 4.0 * cam.ray(u, v)
    np.testing.assert_allclose(fix.p_body, mount.to_body(p_cam), atol=1e-9)
    assert fix.range_m == pytest.approx(4.0)
    # Range error ~ z^2 / (f B) * 0.5 px along the view axis.
    along = mount.R_body_cam @ (p_cam / np.linalg.norm(p_cam))
    s = np.sqrt(along @ fix.cov_body @ along)
    assert s == pytest.approx(16 / (320 * 0.06) * 0.5, rel=0.05)


def test_locator_needs_depth_in_the_box():
    cam = PinholeCamera.from_fov(640, 480, 90.0)
    loc = TargetLocator(cam, CameraMount.make(), 0.06)
    assert loc.locate((300, 200, 340, 260), np.full((480, 640), np.inf)) is None


def test_filter_follows_a_moving_target():
    rng = np.random.default_rng(0)
    f = TargetFilter(accel_sigma=1.0)
    vel = np.array([2.0, -1.0, 0.0])
    for k in range(40):                          # 10 Hz for 4 s
        t = 0.1 * k
        f.update(t, np.array([5, 0, -2]) + vel * t + rng.normal(0, 0.1, 3),
                 0.01 * np.eye(3))
    pos, v, _ = f.estimate(4.1)
    np.testing.assert_allclose(v, vel, atol=0.3)
    np.testing.assert_allclose(pos, np.array([5, 0, -2]) + vel * 4.1, atol=0.3)
    assert f.age(4.1) == pytest.approx(0.2)


def test_filter_takes_late_fixes_in_time_order():
    rng = np.random.default_rng(1)
    fixes = [(0.1 * k, rng.normal(0, 1, 3), (0.05 + 0.1 * (k % 3)) * np.eye(3))
             for k in range(20)]
    in_order, shuffled = TargetFilter(), TargetFilter()
    for fx in fixes:
        in_order.update(*fx)
    late = fixes[12]                              # arrives after the newer ones
    for fx in fixes[:12] + fixes[13:]:
        shuffled.update(*fx)
    assert shuffled.update(*late)
    for a, b in zip(in_order.estimate(2.5), shuffled.estimate(2.5)):
        np.testing.assert_allclose(a, b, atol=1e-9)
    assert not shuffled.update(-5.0, np.zeros(3), np.eye(3))   # older than the history


def test_sgbm_ranges_a_textured_plane():
    from autobot.vision.stereo_depth import SgbmDepth
    rng = np.random.default_rng(0)
    cam = PinholeCamera.from_fov(320, 240, 90.0)
    left = cv2.resize(rng.integers(0, 256, (60, 90), dtype=np.uint8), (360, 240),
                      interpolation=cv2.INTER_NEAREST)
    d = 12                                        # pixels of disparity
    right = left[:, d:d + 320]
    left = left[:, :320]
    depth = SgbmDepth(cam, 0.06, min_range_m=0.5)(
        cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right, cv2.COLOR_GRAY2BGR))
    core = depth[40:200, 120:280]
    assert np.median(core[np.isfinite(core)]) == pytest.approx(160 * 0.06 / d, rel=0.05)


def test_vision_vector_layout():
    R = np.arange(9).reshape(3, 3)
    vec = V.get_vision_vector(R, [1, 2, 3], [4, 5, 6], 50.0, [7, 8, 9, 10], [11, 12, 13])
    np.testing.assert_array_equal(vec[:9], np.arange(9))
    np.testing.assert_array_equal(vec[9:15], [1, 2, 3, 4, 5, 6])
    assert vec[15] == V.RANGEFINDER_MAX_M                    # clipped
    np.testing.assert_array_equal(vec[16:], [7, 8, 9, 10, 11, 12, 13])
    assert V.nearness(np.array([0.0, 10.0, 20.0, np.inf])).tolist() == [255, 127, 0, 0]


# --- the controller ---------------------------------------------------------

class FakePerception:
    def __init__(self, target_ned, image_t, fix_age=0.1, range_age=0.3):
        from autobot.vision.perception import Snapshot
        self._snap = Snapshot(np.full((128, 128, 3), 7, np.uint8),
                              np.full((128, 128, 1), 9, np.uint8), image_t, image_t, True)
        self.target, self.fix_age, self.range_age = np.asarray(target_ned, float), fix_age, range_age

    def snapshot(self):
        return self._snap

    def target_ned(self, t):
        return (self.target, np.zeros(3), np.eye(3)), self.fix_age, self.range_age


class RecordingPolicy:
    def __init__(self):
        self.obs = []

    def reset(self):
        pass

    def step(self, obs, is_first=False):
        self.obs.append(obs)
        return np.array([0.1, 0.0, 0.0, 0.2], np.float32)


def state_at(t, pos_ned, yaw_deg=0.0, range_down=float("nan")):
    from autobot.mavlink.vehicle import VehicleState
    a = np.radians(yaw_deg) / 2
    q = np.array([np.cos(a), 0, 0, np.sin(a)])
    return VehicleState(q, np.zeros(3), t, np.asarray(pos_ned, float), np.array([1.0, 0, 0]),
                        t, "GUIDED", True, t, range_down, t)


def test_controller_builds_the_policy_inputs():
    from autobot.control.vision_policy_controller import VisionPolicyController
    t = 100.0
    pol = RecordingPolicy()
    # Drone 2 m up facing east; target 5 m north of it, 1 m lower (so the
    # drone is 1 m above it: +1 up).
    c = VisionPolicyController(pol, FakePerception([5.0, 0.0, -1.0], t), agl="altitude")
    c.handover(state_at(t, [0, 0, -2.0], yaw_deg=90))
    cmd = c.step(state_at(t + 0.02, [0, 0, -2.0], yaw_deg=90), t + 0.02)
    assert cmd.ok, cmd.reason
    obs = pol.obs[-1]
    assert obs["image"].shape == (128, 128, 3) and obs["depth"].shape == (128, 128, 1)
    vec = obs["vector"]
    # The policy's frame: x along the hand-over heading (east), y left (north), z up.
    np.testing.assert_allclose(vec[20:23], [0.0, -5.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(vec[12:15], [0.0, 1.0, 0.0], atol=1e-6)   # 1 m/s north
    assert vec[15] == pytest.approx(2.0)                                 # height
    np.testing.assert_allclose(vec[0:9], np.eye(3).reshape(9), atol=1e-6)
    np.testing.assert_allclose(vec[16:20], 0.0)                          # first step
    c.step(state_at(t + 0.04, [0, 0, -2.0], yaw_deg=90), t + 0.04)
    np.testing.assert_allclose(pol.obs[-1]["vector"][16:20], [0.1, 0, 0, 0.2])


def test_controller_uses_the_rangefinder_tilt_corrected():
    from autobot.control.vision_policy_controller import VisionPolicyController
    t = 50.0
    c = VisionPolicyController(RecordingPolicy(), FakePerception([5, 0, -1], t))
    c.handover(state_at(t, [0, 0, -2.0], range_down=3.0))
    agl, src = c.height_above_ground(state_at(t, [0, 0, -2.0], range_down=3.0), t)
    assert (agl, src) == (pytest.approx(3.0), "rangefinder")
    agl, src = c.height_above_ground(state_at(t - 1.0, [0, 0, -2.5], range_down=3.0), t)
    assert src == "altitude" and agl == pytest.approx(3.5)   # stale reading: ground from hand-over


@pytest.mark.parametrize("kw, reason", [
    (dict(fix_age=3.0), "target not seen"),
    (dict(range_age=5.0), "no stereo range"),
    (dict(image_t=99.0), "camera frame"),
])
def test_controller_stops_without_fresh_perception(kw, reason):
    from autobot.control.vision_policy_controller import VisionPolicyController
    t = 100.0
    args = dict(target_ned=[5, 0, -1], image_t=t)
    args.update(kw)
    c = VisionPolicyController(RecordingPolicy(), FakePerception(**args))
    c.handover(state_at(t, [0, 0, -2.0]))
    cmd = c.step(state_at(t, [0, 0, -2.0]), t)
    assert not cmd.ok and reason in cmd.reason


# --- the threaded pipeline, end to end ----------------------------------------

class FakeCapture:
    """A rectified 320x240 pair of a static scene, at 30 Hz."""

    def __init__(self, left, right):
        from autobot.vision.stereo_capture import StereoFrame
        self._frame = lambda i: StereoFrame(time.monotonic(), left, right, i)
        self.i = 0

    def read(self, after=-1, timeout=1.0):
        time.sleep(1 / 30)
        self.i += 1
        return self._frame(self.i)

    def close(self):
        pass


def test_perception_estimates_the_target_position():
    """Drone 3 m up facing east; a target 4 m straight ahead of the camera."""
    from autobot.vision.build import build_perception
    from autobot.vision.detector import Detection
    w, h, B = 320, 240, 0.10       # 160 px * 0.10 m / 4 m = exactly 4 px of disparity
    f = w / 2
    cfg = {"calibration": {"image_size": [w, h],
                           "K1": [f, 0, (w - 1) / 2, 0, f, (h - 1) / 2, 0, 0, 1], "D1": [0] * 5,
                           "K2": [f, 0, (w - 1) / 2, 0, f, (h - 1) / 2, 0, 0, 1], "D2": [0] * 5,
                           "R": np.eye(3).ravel().tolist(), "T": [-B, 0, 0]},
           "mount": {"offset_m": [0.2, 0.0, 0.0], "tilt_up_deg": 0.0},
           "depth": {"backend": "sgbm"}}
    rng = np.random.default_rng(0)
    tex = cv2.resize(rng.integers(0, 256, (60, 100), dtype=np.uint8), (w + 40, h),
                     interpolation=cv2.INTER_NEAREST)
    d = int(round(f * B / 4.0))                    # a textured wall 4 m away
    assert d == f * B / 4.0
    left = cv2.cvtColor(tex[:, :w], cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(tex[:, d:d + w], cv2.COLOR_GRAY2BGR)
    box = np.array([w / 2 - 20, h / 2 - 20, w / 2 + 20, h / 2 + 20])
    yaw = np.radians(90) / 2
    pose = (np.array([0.0, 0.0, -3.0]), np.array([np.cos(yaw), 0, 0, np.sin(yaw)]))
    per = build_perception(cfg, lambda: pose, capture=FakeCapture(left, right),
                           detector=lambda img: [Detection(box, 0.9, 0, 1)]).start()
    try:
        deadline = time.monotonic() + 10
        while per.stats.fixes < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        est, fix_age, range_age = per.target_ned(time.monotonic())
        snap = per.snapshot()
    finally:
        per.stop()
    assert per.stats.fixes >= 3 and per.stats.bearing_fixes > 0
    # East of the drone: camera 0.2 m ahead + 4 m.
    np.testing.assert_allclose(est[0], [0.0, 4.2, -3.0], atol=0.25)
    assert fix_age < 0.2 and range_age < 1.0
    assert snap.target_locked and snap.depth.max() > 0


# --- a real camera bundle (skipped without one) ---------------------------------

import os  # noqa: E402

BUNDLE = os.environ.get("AUTOBOT_TEST_CAMERA_BUNDLE")


@pytest.mark.skipif(not BUNDLE, reason="set AUTOBOT_TEST_CAMERA_BUNDLE to a camera policy bundle")
def test_camera_bundle_runs():
    from autobot.policy import load_policy, zero_observation
    pol, info = load_policy(BUNDLE, sampling="deterministic")
    assert info.kind == "camera"
    obs = zero_observation("camera")
    for i in range(3):
        a = pol.step(obs, is_first=(i == 0))
        assert a.shape == (4,) and np.isfinite(a).all() and np.abs(a).max() <= 1
