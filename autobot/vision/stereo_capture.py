"""Grab synchronized frames from a stereo camera.

Two layouts:
  two devices    left and right are separate V4L2 cameras
  side by side   one device whose frames hold left | right halves (most
                 USB stereo modules)
A background thread keeps only the newest pair, stamped with
time.monotonic() at capture, so readers never see a stale queue.
"""
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class StereoFrame:
    t: float                 # time.monotonic() when grabbed
    left: np.ndarray         # BGR
    right: np.ndarray
    index: int


class StereoCapture:

    def __init__(self, left=0, right=None, width=640, height=480, fps=30):
        """right=None: `left` is one side-by-side device of 2*width x height."""
        import cv2
        self._cv2 = cv2
        self.side_by_side = right is None
        devices = [left] if self.side_by_side else [left, right]
        self.caps = []
        for dev in devices:
            cap = cv2.VideoCapture(dev)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * width if self.side_by_side else width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError(f"can't open camera {dev!r}")
            self.caps.append(cap)
        self._latest = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _grab(self):
        # grab() both first, then retrieve(): the two exposures are closest in time.
        if not all(cap.grab() for cap in self.caps):
            return None
        t = time.monotonic()
        frames = [cap.retrieve()[1] for cap in self.caps]
        if self.side_by_side:
            f = frames[0]
            half = f.shape[1] // 2
            return t, f[:, :half], f[:, half:]
        return t, frames[0], frames[1]

    def _run(self):
        index = 0
        while not self._stop.is_set():
            got = self._grab()
            if got is None:
                time.sleep(0.01)
                continue
            with self._cond:
                self._latest = StereoFrame(got[0], got[1], got[2], index)
                self._cond.notify_all()
            index += 1

    def read(self, after=-1, timeout=1.0):
        """The newest pair with index > after, waiting up to timeout."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._latest is not None and self._latest.index > after, timeout)
            return self._latest if ok else None

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        for cap in self.caps:
            cap.release()
