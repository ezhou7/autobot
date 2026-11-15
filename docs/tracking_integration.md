# Flying the camera tracking policy

autobot can fly the camera tracking policy trained in
[ezhou7/dreamerv3](https://github.com/ezhou7/dreamerv3) (DreamerV3, F450
model, `visionlook-2m-1c-2026-11-04`). The policy follows a target and
keeps 3 m from it. Each 50 Hz step it reads:

- a 128×128 color image and a matching depth image from a front camera;
- its attitude, rates, velocity and height above ground;
- its position relative to the target.

In simulation the target's position came from the simulator. Here it
comes from **YOLO** (where the target is in the image) and **stereo depth**
(how far away it is).

```
stereo camera ──▶ rectify ──┬─▶ YOLO + BoT-SORT ─▶ target box ─┐
  (30 fps)                  │                                   ├─▶ target position ─▶ Kalman filter (NED)
                            ├─▶ HITNet stereo depth ─▶ range ───┘        (per frame)          │
                            │        (~4/s)        └──────────▶ depth image ┐               │ extrapolated to now
                            └─▶ resample into the training camera ─▶ image ─┤               │
ArduPilot ──attitude, position, rangefinder (50 Hz)──────────────▶ vector ◀──────────────────┘
   ▲                                                                 │
   └──────── SET_ATTITUDE_TARGET: body rates + thrust ◀── policy (NumPy) + safety limits
```

## Pieces

| Module | What it does |
|---|---|
| `autobot/policy/reference.py`, `weights.py` | NumPy forward pass, now with the camera policy's image encoder (4 convolution layers). Vendored from dreamerv3 ([PR #10](https://github.com/ezhou7/dreamerv3/pull/10)), where it matches JAX within 2e-4 (`aerial/tests/test_npu_reference_camera.py`, and the trained model on its training branch) |
| `autobot/policy/vision_spec.py` | The camera policy's inputs: image, depth (nearness, 0–20 m), 23-number vector |
| `autobot/vision/camera.py` | Pinhole cameras, where the camera sits on the drone, stereo calibration and rectification |
| `autobot/vision/stereo_capture.py` | Two cameras or one side-by-side device; keeps the newest pair |
| `autobot/vision/stereo_depth.py` | HITNet (ONNX Runtime) or OpenCV SGBM → depth in meters |
| `autobot/vision/detector.py` | YOLOv5 on the NPU (autobot's RKNN path) or YOLOv8 (ultralytics), BoT-SORT IDs, and a selector that stays locked on one target |
| `autobot/vision/target_locator.py` | Box center → direction; median depth over the middle of the box → range; together → the target's position, with an uncertainty that grows with range² |
| `autobot/vision/target_filter.py` | Constant-velocity Kalman filter in NED; accepts late range fixes and replays newer ones |
| `autobot/vision/training_view.py` | Resamples the real camera into the training camera: 90° view, tilted 10° up |
| `autobot/vision/perception.py` | Two threads: per frame (rectify, YOLO, direction update, image) and depth (stereo, range fix, depth image) |
| `autobot/control/vision_policy_controller.py` | Builds the inputs each step and stops on stale data. Height above ground comes from the downward rangefinder, tilt-corrected, falling back to altitude |
| `flight_scripts/policy_track.py` | `--bench`: perception and the policy with no commands sent. Otherwise: wait for a ranged target, take off, track, hand back, land |
| `device_scripts/stereo_calibrate.py` | Checkerboard calibration → the YAML `calibration:` block (runs headless) |
| `device_scripts/stereo_check.py` | Live check of lock, range, position and timings; saves what the policy sees |
| `device_scripts/get_hitnet.sh` | Downloads the HITNet model (checksum-verified) |
| `config/stereo_camera.example.yaml` | Every camera and perception setting |

## Setup

```bash
pip install -r requirements-tracking.txt        # + rknn-toolkit-lite2 on the Orange Pi
bash device_scripts/get_hitnet.sh               # -> ~/models/hitnet/eth3d_240x320.onnx
# The policy bundle: exported in ezhou7/dreamerv3 with
#   python aerial/deploy/export_model.py --logdir <run> --name visionlook-2m-1c-2026-11-04
# then copied to the drone, e.g. ~/models/visionlook-2m-1c-2026-11-04/
```

The first load writes `policy_weights.npz` (28 MB) into the bundle.

### The camera

What matters most, roughly in order:

1. **Field of view.** The policy saw a 90°×90° view tilted 10° up. After
   rectification the camera should cover that, so it needs about 100°+
   horizontally *and* vertically. `TrainingView.coverage` (printed at
   start) says how much is covered. Uncovered parts read as far away,
   which the policy may take as open space.
2. **Synchronized exposures and a global shutter.** A rolling shutter on a
   moving drone breaks stereo matching.
3. **Baseline 6–12 cm.** Wider measures range better but sees less up close.
4. **Mount:** forward, about 0.2 m ahead of center, tilted 10° up, as in
   training. Enter the real offset and tilt under `mount:`.

### Calibration

```bash
python device_scripts/stereo_calibrate.py --left 0 --board 9x6 --square 0.025 \
    --pairs 25 --out config/stereo_camera.yaml
```

Aim for a stereo reprojection error under 0.5 px. Then copy the other
blocks from `config/stereo_camera.example.yaml` and fill in `mount:`,
`depth:` and `detector:`.

## Checking on the ground

```bash
# Camera, YOLO, depth and range, with timings; saves sample images
python device_scripts/stereo_check.py --camera config/stereo_camera.yaml --save logs/check
# The whole chain with the policy at 50 Hz, nothing sent (props off).
# With --connect it uses the flight controller's attitude and position.
python -m flight_scripts.policy_track --bench --model ~/models/visionlook-2m-1c-2026-11-04 \
    --camera config/stereo_camera.yaml [--connect /dev/ttyS6] --seconds 60
```

Walk in front of it: the printed distance to the target should match a
tape measure to about 0.2 m at 2–4 m.

## Flying

Use the same ArduPilot parameters as the hover policy
(`config/policy.parm`, docs/policy_integration.md), plus a downward
rangefinder if you have one (`--agl rangefinder`, the default; it falls
back to altitude above the take-off point).

```bash
python -m flight_scripts.policy_track --model ~/models/visionlook-2m-1c-2026-11-04 \
    --camera config/stereo_camera.yaml --connect /dev/ttyS6 --seconds 30
```

It won't take off until YOLO has locked a target and stereo has ranged
it. In flight the policy stops, and ArduPilot holds position (BRAKE),
when:
- the target hasn't been seen for 2 s;
- it hasn't been ranged for 3 s;
- the camera frame is older than 0.15 s;
- telemetry is stale, or the drone tilts past 45°;
- the pilot leaves GUIDED.

The limits are in `TrackingLimits`.

ArduPilot SITL has no camera, so this policy can't fly in SITL as is.
`--bench --connect tcp:127.0.0.1:5760` runs it against SITL telemetry
with the real camera, without sending commands.

## What was measured (2026-11-05, desktop i7-9700K)

**Policy port.** The NumPy encoder, recurrent step and action head match
the trained model in JAX within 2e-4. That covers 12 steps of real
camera observations in a forest and a town.

**Resampling into the training camera.** A wide (130°) level camera
resampled by `TrainingView` against the training camera rendered at the
same spot, in 8 simulator scenes: color within about 1/255 per pixel on
average, depth exact to the input's 8 cm steps.

**Target position, stereo pairs rendered in our simulator worlds** (90°
camera, 6–12 cm baseline, 84 scenes, target boxes from ground truth):

| Depth | 2–4 m | 4–6 m | 6–11 m | Time per pair |
|---|---|---|---|---|
| HITNet ETH3D 240×320, 320×240 images | 0.17 m | 0.34 m | 0.88 m | 0.16 s |
| HITNet, 640×480 images | 0.22 m | 0.39 m | 1.00 m | 0.16 s |
| SGBM, 640×480 images | 0.29 m | 0.52 m | 1.12 m | 0.01 s |

These are median 3D errors. HITNet also fills the whole depth image;
SGBM leaves about a quarter of it empty on blank surfaces, and empty
reads as far away.

**The whole stack**, replaying simulator stereo frames with HITNet: 197 of
200 policy steps in 4 s, 17–18 ms each. That needs the thread budget
`policy_track.py` sets: 4 math-library threads, HITNet on 2. With the
defaults (8 + 4 threads on 8 cores) a step took 88 ms.

## Limits and next steps

- **Speed on the Orange Pi.** The policy step is 12–18 ms on the desktop,
  almost all of it the image encoder. On the Pi's A76 cores expect
  several times that, below the trained 50 Hz. HITNet on the Pi's CPU is
  around a second per pair.
  - Next: run the encoder (and HITNet, or SGBM) on the NPU, as YOLO
    already does. dreamerv3 has the RKNN tooling for the vector policy
    (`aerial/deploy/npu/`).
  - Until then, `policy_track.py` warns when a step exceeds 20 ms.
- **This model's behavior.** In simulation it halved crashes from flying
  blind, but:
  - **Hovering:** at rest it circles about 1.2 m too close.
  - **Losing sight of the target:** while moving it looks where it flies,
    with the target in view only 30–50% of the time. Here, out of view
    means the target's position is extrapolated, and the controller stops
    after 2 s.
  - `visionlook2` (training as of 2026-11-05) brings back watching the
    target while hovering. It uses the same inputs, so it's a drop-in
    replacement.
- **Range past 6 m** is good to about 1 m, against 5 cm of noise in
  training. Up close it's about 0.2 m.
- **Training target vs real target.** Training used a magenta ball; YOLO
  here finds people. The policy reads the target's position from the
  vector, but the image looks different from anything it trained on.
- **Not flown yet.** Next: calibrate the real camera, run `stereo_check.py`
  and `--bench`, then a tethered test.
