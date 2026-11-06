"""Check the stereo camera pipeline on the ground: YOLO lock, stereo
range, the target's position, timings, and what the policy would see.

    python device_scripts/stereo_check.py --camera config/stereo_camera.yaml \\
        --seconds 30 --save logs/stereo_check

Prints one line per second. With --save, writes the rectified left frame
with the target's box, a depth image, and the policy's 128x128 image and
depth every 2 s.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from autobot.vision.build import build_depth, build_detector, build_geometry, load_config
from autobot.vision.detector import TargetSelector
from autobot.vision.stereo_capture import StereoCapture
from autobot.vision.target_locator import LocatorConfig, TargetLocator


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--camera", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--save", default=None, help="folder for sample images")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.camera)
    rect, mount, view = build_geometry(cfg)
    print(f"rectified: f {rect.camera.fx:.1f} px, baseline {rect.baseline_m * 100:.1f} cm, "
          f"policy view coverage {view.coverage:.0%}")
    depth_fn, detector = build_depth(cfg, rect), build_detector(cfg)
    selector = TargetSelector(**cfg.get("selector", {}))
    locator = TargetLocator(rect.camera, mount, rect.baseline_m,
                            LocatorConfig(**cfg.get("locator", {})))
    c = cfg["capture"]
    w, h = cfg["calibration"]["image_size"]
    cap = StereoCapture(c.get("left", 0), c.get("right"), w, h, c.get("fps", 30))
    out = Path(args.save) if args.save else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    start = last_print = last_save = time.monotonic()
    index, times = -1, {"rectify": [], "detect": [], "depth": [], "view": []}
    try:
        while time.monotonic() - start < args.seconds:
            frame = cap.read(after=index)
            if frame is None:
                continue
            index = frame.index
            t0 = time.perf_counter()
            left, right = rect(frame.left, frame.right)
            t1 = time.perf_counter()
            det = selector.select(detector(left))
            t2 = time.perf_counter()
            depth = depth_fn(left, right)
            t3 = time.perf_counter()
            image, near = view.image(left), view.depth(depth)
            t4 = time.perf_counter()
            for k, a, b in (("rectify", t0, t1), ("detect", t1, t2), ("depth", t2, t3),
                            ("view", t3, t4)):
                times[k].append((b - a) * 1e3)
            fix = locator.locate(det.box, depth) if det is not None else None
            now = time.monotonic()
            if now - last_print > 1.0:
                last_print = now
                ms = ", ".join(f"{k} {np.median(v[-30:]):.0f}" for k, v in times.items())
                if fix is not None:
                    print(f"target: {fix.range_m:5.2f} m along the view axis, body "
                          f"(fwd, left, up) {np.round(fix.p_body, 2)}, depth valid "
                          f"{fix.valid_share:.0%} | ms: {ms}")
                else:
                    print(f"target: {'seen, no range' if det is not None else 'not seen'} | ms: {ms}")
            if out and now - last_save > 2.0:
                last_save = now
                stamp = f"{now - start:06.1f}"
                vis = left.copy()
                if det is not None:
                    x0, y0, x1, y1 = (int(v) for v in det.box)
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
                d = np.nan_to_num(np.clip(1 - depth / 20, 0, 1), posinf=0) * 255
                cv2.imwrite(str(out / f"{stamp}_left.jpg"), vis)
                cv2.imwrite(str(out / f"{stamp}_depth.png"),
                            cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_MAGMA))
                cv2.imwrite(str(out / f"{stamp}_policy.png"), np.concatenate(
                    [image[..., ::-1], cv2.cvtColor(near[..., 0], cv2.COLOR_GRAY2BGR)], 1))
    finally:
        cap.close()


if __name__ == "__main__":
    main()
