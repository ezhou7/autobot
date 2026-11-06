"""Fly the camera tracking policy: follow a target found by YOLO and
ranged with stereo depth.

connect -> start the camera pipeline -> wait for a target fix -> GUIDED ->
arm -> take off -> hand over to the policy -> track -> ArduPilot holds
position -> LAND. Every policy step is saved to a .npz log.

--bench runs the camera pipeline and the policy at 50 Hz WITHOUT sending
any command (props off): with --connect it uses the vehicle's attitude
and position, without it a level drone 2 m up. It prints the target
estimate and the policy's commands so the whole chain can be checked on
the ground.

Setup and limits: docs/tracking_integration.md. Fly with a pilot on the RC
mode switch: leaving GUIDED stops the policy at once.
"""
import argparse
import json
import os
import time

# Thread budget, set before NumPy loads: the policy's matrix math, HITNet
# (depth.threads in the camera YAML) and YOLO share the CPU. Oversubscribed,
# a 12 ms policy step took 88 ms on an 8-core desktop.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np  # noqa: E402

from autobot.control import PolicyLoop  # noqa: E402
from autobot.control.vision_policy_controller import TrackingLimits, VisionPolicyController  # noqa: E402
from autobot.mavlink import StateRecorder, Vehicle  # noqa: E402
from autobot.policy import STEP_RATE_HZ, load_policy, warm_up, zero_observation  # noqa: E402
from autobot.utils.logging import stdout_logger  # noqa: E402
from autobot.vision.build import build_perception, load_config, static_pose  # noqa: E402
from flight_scripts.policy_hover import (  # noqa: E402
    check_params, get_thrust_scale, hand_back, land, read_param_file, save_log, take_off)

log = stdout_logger("policy_track")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="camera policy bundle directory")
    ap.add_argument("--camera", required=True, help="stereo camera YAML (config/stereo_camera.example.yaml)")
    ap.add_argument("--connect", default=None, help="MAVLink URL; required unless --bench")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--bench", action="store_true", help="never command the vehicle")
    ap.add_argument("--params", default=None, help="parameter file to load first")
    ap.add_argument("--takeoff-alt", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=30.0, help="policy flight time")
    ap.add_argument("--thrust-scale", default="auto")
    ap.add_argument("--max-rate", type=float, default=3.0)
    ap.add_argument("--agl", default="rangefinder", choices=["rangefinder", "altitude"],
                    help="height above ground source")
    ap.add_argument("--wait-target", type=float, default=60.0,
                    help="seconds to wait for a ranged target before giving up")
    ap.add_argument("--fallback-mode", default="BRAKE")
    ap.add_argument("--no-land", action="store_true")
    ap.add_argument("--land-on-abort", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    if not args.bench and not args.connect:
        ap.error("--connect is required unless --bench")
    return args


def time_per_step(policy, steps=20):
    obs = zero_observation("camera")
    start = time.perf_counter()
    for i in range(steps):
        policy.step(obs, is_first=(i == 0))
    return (time.perf_counter() - start) / steps


def vehicle_pose_fn(vehicle):
    def pose():
        s = vehicle.state()
        return s.pos_ned.copy(), s.attitude_q.copy()
    return pose


def wait_for_target(perception, timeout):
    """Until the filter has a fresh, ranged fix of the locked target."""
    log.info("waiting for a target: YOLO lock and a stereo range")
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        now = time.monotonic()
        est, fix_age, range_age = perception.target_ned(now)
        if est is not None and fix_age < 0.5 and range_age < 1.5:
            log.info(f"target at NED {np.round(est[0], 2)} "
                     f"(range fixes {perception.stats.fixes})")
            return True
        time.sleep(0.2)
    return False


def bench(policy, perception, vehicle, seconds, limits):
    """The 50 Hz loop without commands: print what the policy would do."""
    from autobot.mavlink.vehicle import VehicleState
    controller = VisionPolicyController(policy, perception, limits)
    if vehicle is not None:
        state_fn = vehicle.state
    else:
        pos, q = static_pose()()

        def state_fn():
            now = time.monotonic()
            return VehicleState(q, np.zeros(3), now, pos, np.zeros(3), now, "GUIDED", True,
                                now, float(-pos[2]), now)
    controller.handover(state_fn())
    start = last_print = time.monotonic()
    period = 1.0 / STEP_RATE_HZ
    while time.monotonic() - start < seconds:
        now = time.monotonic()
        cmd = controller.step(state_fn(), now)
        if now - last_print > 0.5:
            last_print = now
            st = perception.stats
            if cmd.ok:
                rel = cmd.obs[20:23]
                log.info(f"target {np.linalg.norm(rel):5.2f} m away (rel {np.round(rel, 2)}), "
                         f"rates {np.round(cmd.rates_frd, 2)} rad/s, thrust {cmd.thrust:.2f} | "
                         f"fix age {controller.extra['fix_age']:.2f} s, range age "
                         f"{controller.extra['range_age']:.2f} s | detect {st.detect_ms:.0f} ms, "
                         f"depth {st.depth_ms:.0f} ms")
            else:
                log.info(f"not ready: {cmd.reason} | frames {st.frames}, depth maps "
                         f"{st.depth_maps}, range fixes {st.fixes}")
        time.sleep(max(0.0, period - (time.monotonic() - now)))


def main():
    args = parse_args()
    policy, info = load_policy(args.model)
    if info.kind != "camera":
        raise SystemExit(f"{info.name} is a {info.kind} policy; use policy_hover.py")
    warm_up(policy, kind="camera")
    step_ms = time_per_step(policy) * 1e3
    log.info(f"policy {info.name}: {step_ms:.1f} ms per step")
    if step_ms > 1e3 / STEP_RATE_HZ:
        log.warning(f"the policy step ({step_ms:.0f} ms) is longer than the "
                    f"{1e3 / STEP_RATE_HZ:.0f} ms period it was trained at")

    vehicle = None
    if args.connect:
        vehicle = Vehicle(args.connect, baud=args.baud)
        vehicle.connect()
        for msg in ("ATTITUDE_QUATERNION", "LOCAL_POSITION_NED"):
            vehicle.request_rate(msg, STEP_RATE_HZ)
        vehicle.request_rate("DISTANCE_SENSOR", 20)

    cfg = load_config(args.camera)
    perception = build_perception(
        cfg, vehicle_pose_fn(vehicle) if vehicle else static_pose())
    log.info(f"training view coverage {perception.view.coverage:.0%} of the policy's camera")
    if perception.view.coverage < 0.95:
        log.warning("the camera doesn't fill the policy's 90x90 deg view; "
                    "uncovered parts read as far away")
    perception.start()
    limits = TrackingLimits(max_rate=args.max_rate)
    try:
        if args.bench:
            bench(policy, perception, vehicle, args.seconds, limits)
            return
        if not wait_for_target(perception, args.wait_target):
            raise SystemExit("no ranged target; not taking off")
        if args.params:
            for name, value in read_param_file(args.params).items():
                vehicle.set_param(name, value)
        check_params(vehicle)
        thrust_scale = get_thrust_scale(vehicle, args.thrust_scale)
        take_off(vehicle, args.takeoff_alt)
        if not wait_for_target(perception, 10.0):
            log.error("lost the target during take-off; landing")
            land(vehicle)
            return

        controller = VisionPolicyController(policy, perception, limits,
                                            thrust_scale=thrust_scale, agl=args.agl)
        loop = PolicyLoop(vehicle, controller, fallback_mode=args.fallback_mode)
        result = loop.run(args.seconds)
        summary = result.summary()
        summary.update(completed=result.completed, reason=result.reason, model=info.name,
                       connect=args.connect, thrust_scale=thrust_scale,
                       perception=perception.stats.__dict__)
        log.info("summary: " + json.dumps(summary, indent=1, default=float))
        with StateRecorder(vehicle) as recorder:
            if hand_back(vehicle, result, args):
                land(vehicle)
            else:
                time.sleep(3.0)
        save_log(args.log, summary, result.log, recorder.arrays())
    finally:
        perception.stop()
        perception.capture.close()
        if vehicle is not None:
            vehicle.close()


if __name__ == "__main__":
    main()
