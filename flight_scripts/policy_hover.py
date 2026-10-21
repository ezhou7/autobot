"""Fly the hover policy: ArduPilot takes off, the policy flies, ArduPilot lands.

connect -> load parameters -> GUIDED -> arm -> take off -> hand over to the
policy -> hover -> ArduPilot holds position -> LAND.
Every policy step is saved to a .npz log. Setup, SITL (software in the
loop) and the real drone: docs/policy_integration.md.

Fly with a pilot on the RC (radio control) mode switch: leaving GUIDED
stops the policy at once.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from autobot.control import (
    TRAINED_HOVER_THRUST, PolicyController, PolicyLoop, SafetyLimits)
from autobot.mavlink import StateRecorder, Vehicle
from autobot.policy import OBS_DIM, STEP_RATE_HZ, load_policy, warm_up
from autobot.utils.logging import stdout_logger

log = stdout_logger("policy_hover")

# GUID_OPTIONS bit 3: ArduPilot reads SET_ATTITUDE_TARGET thrust as
# throttle. Without it, thrust is a climb rate.
GUID_OPTIONS_THRUST_IS_THROTTLE = 8


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="policy bundle directory")
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--params", default=None, help="parameter file to load first")
    ap.add_argument("--takeoff-alt", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=30.0, help="policy flight time")
    ap.add_argument("--thrust-scale", default="auto",
                    help="'auto' = MOT_THST_HOVER / trained hover, or a number")
    ap.add_argument("--max-rate", type=float, default=3.0)
    ap.add_argument("--fallback-mode", default="BRAKE",
                    help="mode if the policy stops early; BRAKE ignores the sticks")
    ap.add_argument("--no-land", action="store_true",
                    help="after a normal finish, hold position instead of landing")
    ap.add_argument("--land-on-abort", action="store_true",
                    help="land after a safety stop (default: hold the fallback mode)")
    ap.add_argument("--log", default=None, help="output .npz (default: logs/policy_<time>.npz)")
    return ap.parse_args()


def time_per_step(policy, steps=50):
    start = time.perf_counter()
    for i in range(steps):
        policy.step(np.zeros(OBS_DIM, np.float32), is_first=(i == 0))
    return (time.perf_counter() - start) / steps


def read_param_file(path):
    """ArduPilot parameter file -> {name: value}. Accepts 'NAME value'
    and Mission Planner's 'NAME,value'; '#' starts a comment."""
    params = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, value = line.replace(",", " ").split()[:2]
            params[name] = float(value)
    return params


def check_params(vehicle):
    guid_options = int(vehicle.get_param("GUID_OPTIONS"))
    if not guid_options & GUID_OPTIONS_THRUST_IS_THROTTLE:
        raise SystemExit(f"GUID_OPTIONS={guid_options}: bit 3 (8) must be set so ArduPilot "
                         "reads the policy's thrust as throttle, not climb rate")
    angle_boost = vehicle.get_param("ATC_ANGLE_BOOST")
    if angle_boost != 0:
        raise SystemExit(f"ATC_ANGLE_BOOST={angle_boost:g}, policy needs 0 "
                         "(it already adds thrust when tilted)")


def get_thrust_scale(vehicle, setting):
    if setting != "auto":
        return float(setting)
    hover = vehicle.get_param("MOT_THST_HOVER")
    scale = hover / TRAINED_HOVER_THRUST
    log.info(f"MOT_THST_HOVER {hover:.3f} -> thrust scale {scale:.3f}")
    return scale


def take_off(vehicle, altitude):
    log.info("waiting for position and attitude telemetry")
    vehicle.wait_for(lambda s: max(s.ages()) < 0.2, 60.0, "position and attitude telemetry")
    vehicle.set_mode("GUIDED")
    vehicle.arm()
    vehicle.takeoff(altitude)
    vehicle.wait_for(lambda s: -s.pos_ned[2] > altitude - 0.15, 60.0, f"altitude {altitude} m")
    time.sleep(3.0)   # let ArduPilot's take-off settle before handing over
    log.info(f"at {-vehicle.state().pos_ned[2]:.2f} m, handing over to the policy")


def hand_back(vehicle, result, args):
    """Leave the vehicle with ArduPilot. Returns whether to land now."""
    if result.completed:
        # A GUIDED position target: ArduPilot holds it without stick input.
        # Repeated for 2 s so one lost message doesn't matter.
        position = vehicle.state().pos_ned
        for _ in range(20):
            vehicle.hold_position(position)
            time.sleep(0.1)
        return not args.no_land
    if result.taken_over:
        log.warning(f"{result.reason}; leaving the vehicle to the pilot / ArduPilot")
        return False
    # Safety stop: the loop has already asked for the fallback mode.
    log.warning(f"safety stop ({result.reason}); vehicle in {vehicle.state().mode}")
    return args.land_on_abort


def land(vehicle):
    try:
        vehicle.set_mode("LAND")
        log.info("landing")
        vehicle.wait_for(lambda s: not s.armed, 90.0, "landed and disarmed")
    except TimeoutError as e:
        log.error(str(e))


def save_log(path, summary, policy_log, after_policy):
    out = Path(path or f"logs/policy_{time.strftime('%Y%m%dT%H%M%S')}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    after_policy = {f"after_{name}": values for name, values in after_policy.items()}
    np.savez(out, summary=json.dumps(summary), **policy_log, **after_policy)
    log.info(f"log saved to {out}")


def main():
    args = parse_args()

    policy, info = load_policy(args.model)
    warm_up(policy)
    log.info(f"policy {info.name}: {time_per_step(policy) * 1e3:.2f} ms per step")

    vehicle = Vehicle(args.connect, baud=args.baud)
    vehicle.connect()
    vehicle.request_rate("ATTITUDE_QUATERNION", STEP_RATE_HZ)
    vehicle.request_rate("LOCAL_POSITION_NED", STEP_RATE_HZ)
    if args.params:
        for name, value in read_param_file(args.params).items():
            vehicle.set_param(name, value)
            log.info(f"set {name} = {value:g}")
    check_params(vehicle)
    thrust_scale = get_thrust_scale(vehicle, args.thrust_scale)

    take_off(vehicle, args.takeoff_alt)

    controller = PolicyController(
        policy, SafetyLimits(max_rate=args.max_rate), thrust_scale=thrust_scale)
    loop = PolicyLoop(vehicle, controller, fallback_mode=args.fallback_mode)
    result = loop.run(args.seconds)

    summary = result.summary()
    summary.update(completed=result.completed, reason=result.reason,
                   model=info.name, connect=args.connect, thrust_scale=thrust_scale)
    log.info("summary: " + json.dumps(summary, indent=1))

    with StateRecorder(vehicle) as recorder:
        should_land = hand_back(vehicle, result, args)
        if should_land:
            land(vehicle)
        else:
            time.sleep(3.0)   # record a few seconds of what ArduPilot does next
    after_policy = recorder.arrays()
    if len(after_policy["t"]):
        log.info(f"after the policy: min altitude {-after_policy['pos_ned'][:, 2].max():.2f} m, "
                 f"max sink rate {after_policy['vel_ned'][:, 2].max():.2f} m/s")

    save_log(args.log, summary, result.log, after_policy)
    vehicle.close()


if __name__ == "__main__":
    main()
