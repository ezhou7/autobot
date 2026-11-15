"""Depth from a rectified stereo pair.

Both backends return the distance along the view axis (m) for every pixel
of the rectified left image, np.inf where unknown:

  HitnetDepth  HITNet (Tankovich et al., CVPR 2021), a real-time stereo
               network, run with ONNX Runtime. Use PINTO0309's ONNX
               conversion of the ETH3D model at 240x320 (grayscale input):
               on rendered test scenes it measured a target's range to
               0.12 m at 2-4 m, 0.27 m at 4-6 m and 1.1 m at 6-10 m with a
               6-12 cm baseline (docs/tracking_integration.md). ~0.2 s per
               pair on a desktop CPU; much slower on the Orange Pi's CPU.
  SgbmDepth    OpenCV's semi-global block matching. No model file, ~4x
               faster, noisier past ~4 m and leaves textureless areas
               empty.

depth = f * baseline / disparity, with f the rectified focal length in
pixels of the image the disparity is measured in.
"""
import numpy as np

MIN_DISPARITY_PX = 0.3     # below this the range is meaningless


def disparity_to_depth(disp, f_px, baseline_m):
    disp = np.asarray(disp, np.float32)
    with np.errstate(divide="ignore"):
        depth = np.float32(f_px * baseline_m) / disp
    depth[~(disp > MIN_DISPARITY_PX)] = np.inf
    return depth


class HitnetDepth:
    """HITNet stereo via ONNX Runtime. The input size comes from the model
    (e.g. 240x320); images are resized to it and the disparity is scaled
    back to the rectified image's pixels."""

    def __init__(self, model_path, camera, baseline_m, threads=2):
        import cv2
        import onnxruntime as ort
        self._cv2 = cv2
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self._input = inp.name
        self._output = self.session.get_outputs()[0].name
        _, channels, self.in_h, self.in_w = inp.shape
        self.gray = channels == 2               # ETH3D model: two gray images
        self.camera, self.baseline_m = camera, float(baseline_m)

    def __call__(self, left, right):
        """left, right: rectified BGR (H, W, 3) uint8 at camera's size."""
        cv2 = self._cv2
        imgs = [cv2.resize(im, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)
                for im in (left, right)]
        if self.gray:
            imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None] for im in imgs]
        else:
            imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
        x = (np.concatenate(imgs, -1).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        disp = np.squeeze(self.session.run([self._output], {self._input: x})[0])
        W, H = self.camera.width, self.camera.height
        disp = cv2.resize(disp, (W, H), interpolation=cv2.INTER_NEAREST) * (W / self.in_w)
        return disparity_to_depth(disp, self.camera.fx, self.baseline_m)


class SgbmDepth:
    """OpenCV semi-global block matching on the rectified pair."""

    def __init__(self, camera, baseline_m, min_range_m=0.5, block=5):
        import cv2
        self._cv2 = cv2
        max_disp = camera.fx * baseline_m / min_range_m
        n = int(np.ceil(max_disp / 16) * 16)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=max(16, n), blockSize=block,
            P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2, uniquenessRatio=10,
            speckleWindowSize=50, speckleRange=2, disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.camera, self.baseline_m = camera, float(baseline_m)

    def __call__(self, left, right):
        cv2 = self._cv2
        gl, gr = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (left, right))
        disp = self.matcher.compute(gl, gr).astype(np.float32) / 16.0
        return disparity_to_depth(disp, self.camera.fx, self.baseline_m)


def make_depth(kind, camera, baseline_m, model_path=None, **kw):
    if kind == "hitnet":
        if not model_path:
            raise ValueError("hitnet needs model_path (an ONNX file)")
        return HitnetDepth(model_path, camera, baseline_m, **kw)
    if kind == "sgbm":
        return SgbmDepth(camera, baseline_m, **kw)
    raise ValueError(f"unknown depth backend {kind!r}")
