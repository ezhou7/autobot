"""Build the tracking policy's perception from a YAML file
(config/stereo_camera.example.yaml shows every setting)."""
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml

from autobot.vision.camera import CameraMount, Rectifier, StereoCalibration
from autobot.vision.detector import RknnDetector, TargetSelector, UltralyticsDetector
from autobot.vision.perception import Perception
from autobot.vision.stereo_capture import StereoCapture
from autobot.vision.stereo_depth import make_depth
from autobot.vision.target_filter import TargetFilter
from autobot.vision.target_locator import LocatorConfig, TargetLocator
from autobot.vision.training_view import TrainingView


def load_config(path):
    return yaml.YAML(typ="safe").load(Path(path).expanduser().read_text())


def build_geometry(cfg):
    """(rectifier, mount, training view) from the calibration and mount."""
    rect = Rectifier(StereoCalibration.from_dict(cfg["calibration"]))
    m = cfg.get("mount", {})
    mount = CameraMount.make(m.get("offset_m", (0, 0, 0)), m.get("tilt_up_deg", 0.0),
                             m.get("yaw_left_deg", 0.0), R_rect=rect.R_rect)
    return rect, mount, TrainingView(rect.camera, mount)


def build_depth(cfg, rect):
    d = dict(cfg.get("depth", {"backend": "sgbm"}))
    kind = d.pop("backend")
    if "model" in d:
        d["model_path"] = str(Path(d.pop("model")).expanduser())
    return make_depth(kind, rect.camera, rect.baseline_m, **d)


def build_detector(cfg):
    d = cfg.get("detector", {})
    kind, model = d.get("backend", "ultralytics"), d.get("model", "yolov8n.pt")
    if kind == "rknn":
        return RknnDetector(str(Path(model).expanduser()), fps=cfg["capture"].get("fps", 30))
    if kind == "ultralytics":
        return UltralyticsDetector(model, classes=d.get("classes", [0]),
                                   conf=d.get("conf", 0.25))
    raise ValueError(f"unknown detector backend {kind!r}")


def build_perception(cfg, pose_fn, capture=None, detector=None):
    """Everything from the camera to the target filter. capture/detector
    can be given (tests, recorded data); otherwise they come from cfg."""
    rect, mount, view = build_geometry(cfg)
    if capture is None:
        c = cfg["capture"]
        w, h = cfg["calibration"]["image_size"]
        capture = StereoCapture(c.get("left", 0), c.get("right"), w, h, c.get("fps", 30))
    locator = TargetLocator(rect.camera, mount, rect.baseline_m,
                            LocatorConfig(**cfg.get("locator", {})))
    return Perception(capture, rect, detector or build_detector(cfg),
                      TargetSelector(**cfg.get("selector", {})), build_depth(cfg, rect),
                      locator, view, pose_fn, TargetFilter(**cfg.get("filter", {})))


def static_pose(height_m=2.0):
    """A level, still drone `height_m` above the origin: pose_fn for bench
    tests without a flight controller."""
    pos, q = np.array([0.0, 0.0, -height_m]), np.array([1.0, 0, 0, 0])
    return lambda: (pos, q)
