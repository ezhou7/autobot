"""YOLO detections of the target, and which one the drone follows.

Two backends, both with BoT-SORT track IDs:
  UltralyticsDetector  YOLOv8 through ultralytics (a PC, or the Pi's CPU)
  RknnDetector         the YOLOv5 RKNN model on the Orange Pi's NPU, with
                       autobot's existing post-processing (person class)

TargetSelector locks onto one track and keeps it; it takes the most
confident detection when it has none, or after the locked track has been
gone for `relock_after` frames.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    box: np.ndarray          # (4,) x0, y0, x1, y1 pixels
    score: float
    cls: int
    track_id: int = -1


class UltralyticsDetector:

    def __init__(self, weights="yolov8n.pt", classes=(0,), conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.classes, self.conf = list(classes), conf

    def __call__(self, frame_bgr):
        r = self.model.track(frame_bgr, persist=True, classes=self.classes,
                             conf=self.conf, verbose=False, tracker="botsort.yaml")[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        ids = b.id.cpu().numpy().astype(int) if b.id is not None else -np.ones(len(b), int)
        return [Detection(xyxy, float(s), int(c), int(i)) for xyxy, s, c, i in zip(
            b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy(), ids)]


class RknnDetector:
    """autobot's YOLOv5 RKNN path (autobot/tracker.py), returning boxes in
    the input frame's pixels."""

    def __init__(self, model_path, fps=30.0):
        import yaml
        from ultralytics.trackers import BOTSORT

        from autobot import get_resource
        from autobot.model import YoloModel
        from autobot.utils import Properties
        from autobot.utils.yolov5 import IMG_SIZE
        self.yolo = YoloModel()
        self.yolo.load(model_path)
        args = yaml.safe_load(get_resource("botsort.yaml"))
        self.tracker = BOTSORT(args=Properties(args), frame_rate=fps)
        self.size = IMG_SIZE

    def __call__(self, frame_bgr):
        import cv2
        from ultralytics.engine.results import Results

        from autobot.utils.postprocess import post_process_rknn
        from autobot.utils.yolov5 import CLASSES
        h, w = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, self.size)
        boxes, classes, scores = post_process_rknn(
            self.yolo.infer([np.expand_dims(img, 0)]), img)
        if not len(boxes):
            return []
        rows = np.hstack([boxes, np.arange(len(boxes))[:, None], scores[:, None],
                          classes[:, None]])
        res = Results(img, "", dict(enumerate(CLASSES)), rows, scores)
        tracked = self.tracker.update(res.boxes, img=img)
        scale = np.array([w / self.size[0], h / self.size[1]] * 2)
        return [Detection(np.asarray(t[:4]) * scale, float(t[5]), int(t[6]), int(t[4]))
                for t in tracked]


class TargetSelector:

    def __init__(self, relock_after=15):
        self.track_id = None
        self.relock_after = relock_after
        self._missing = 0

    def select(self, detections):
        """The locked target's detection this frame, or None."""
        if self.track_id is not None:
            for d in detections:
                if d.track_id == self.track_id:
                    self._missing = 0
                    return d
            self._missing += 1
            if self._missing < self.relock_after:
                return None
        if not detections:
            return None
        best = max(detections, key=lambda d: d.score)
        self.track_id, self._missing = best.track_id, 0
        return best
