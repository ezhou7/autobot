"""Calibrate a stereo camera with a printed checkerboard; writes the
`calibration:` block of the stereo camera YAML.

Hold the board in front of both cameras at different distances, angles
and image positions. Pairs are taken automatically whenever both cameras
see the whole board (at most one per --interval seconds), so it runs
headless on the Orange Pi.

    python device_scripts/stereo_calibrate.py --left 0 --pairs 25 \\
        --board 9x6 --square 0.025 --out config/stereo_camera.yaml

--board counts INNER corners (columns x rows); --square is one square's
side in meters (the scale of the baseline, and so of every range).
"""
import argparse
import time

import cv2
import numpy as np
import ruamel.yaml as yaml

from autobot.vision.stereo_capture import StereoCapture


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--left", default="0")
    ap.add_argument("--right", default=None, help="omit for one side-by-side device")
    ap.add_argument("--width", type=int, default=640, help="one camera's frame width")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--board", default="9x6", help="inner corners, columns x rows")
    ap.add_argument("--square", type=float, required=True, help="square side, m")
    ap.add_argument("--pairs", type=int, default=25)
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--out", required=True, help="YAML to write (other blocks are kept)")
    return ap.parse_args()


def device(x):
    return int(x) if x is not None and str(x).isdigit() else x


def main():
    args = parse_args()
    cols, rows = (int(v) for v in args.board.split("x"))
    board = np.zeros((cols * rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square
    cap = StereoCapture(device(args.left), device(args.right), args.width, args.height)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    obj, pts_l, pts_r, last, index = [], [], [], 0.0, -1
    print(f"show the {cols}x{rows} board to both cameras")
    try:
        while len(obj) < args.pairs:
            frame = cap.read(after=index)
            if frame is None:
                continue
            index = frame.index
            if time.monotonic() - last < args.interval:
                continue
            grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (frame.left, frame.right)]
            found = [cv2.findChessboardCorners(g, (cols, rows), flags) for g in grays]
            if not all(ok for ok, _ in found):
                continue
            corners = [cv2.cornerSubPix(g, c, (5, 5), (-1, -1), crit)
                       for g, (_, c) in zip(grays, found)]
            obj.append(board)
            pts_l.append(corners[0])
            pts_r.append(corners[1])
            last = time.monotonic()
            print(f"pair {len(obj)}/{args.pairs}")
    finally:
        cap.close()
    size = (args.width, args.height)
    err_l, K1, D1, _, _ = cv2.calibrateCamera(obj, pts_l, size, None, None)
    err_r, K2, D2, _, _ = cv2.calibrateCamera(obj, pts_r, size, None, None)
    err, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        obj, pts_l, pts_r, K1, D1, K2, D2, size, criteria=crit,
        flags=cv2.CALIB_FIX_INTRINSIC)
    print(f"reprojection error: left {err_l:.2f} px, right {err_r:.2f} px, "
          f"stereo {err:.2f} px (aim for < 0.5)")
    print(f"baseline {np.linalg.norm(T) * 100:.1f} cm")
    y = yaml.YAML()
    try:
        cfg = y.load(open(args.out))
    except FileNotFoundError:
        cfg = {}
    cfg["calibration"] = {
        "image_size": [args.width, args.height],
        "K1": K1.ravel().tolist(), "D1": D1.ravel().tolist(),
        "K2": K2.ravel().tolist(), "D2": D2.ravel().tolist(),
        "R": R.ravel().tolist(), "T": T.ravel().tolist()}
    with open(args.out, "w") as fh:
        y.dump(cfg, fh)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
