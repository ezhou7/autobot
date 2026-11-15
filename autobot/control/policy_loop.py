"""Run the policy against a live vehicle at 50 Hz.

The loop stops sending commands as soon as anything looks wrong. With no
commands, ArduPilot's GUID_TIMEOUT levels the drone and holds altitude.
After a safety stop the loop also asks for `fallback_mode`. Use BRAKE,
which holds position without pilot input. LOITER and ALT_HOLD follow the
throttle stick: a stick at the bottom (or SITL with no transmitter) means
"descend at PILOT_SPD_DN", which in SITL hit the ground at 2.5 m/s.
"""
import time
from dataclasses import dataclass

import numpy as np

from autobot.mavlink.vehicle import GUIDED_MODES
from autobot.policy.spec import STEP_RATE_HZ
from autobot.utils.logging import stdout_logger

# Logged once per policy step.
LOG_FIELDS = ("t", "obs", "action", "rates_frd", "thrust", "pos_ned", "vel_ned",
              "q", "att_age", "pos_age", "loop_dt", "clipped")


HOVER_LAYOUT = {"R": slice(6, 15), "rel": slice(0, 3), "standoff": 0.0}


@dataclass
class LoopResult:
    completed: bool     # flew the whole duration
    taken_over: bool    # the pilot or ArduPilot left GUIDED or disarmed
    reason: str
    log: dict           # LOG_FIELDS (+ the controller's extras) -> one row per step
    layout: dict = None  # where the controller's observation keeps R and the target offset

    @property
    def steps(self):
        return len(self.log["t"])

    def summary(self):
        if self.steps == 0:
            return {"steps": 0}
        lay = self.layout or HOVER_LAYOUT
        # Distance from the hold point (the hover point, or standoff meters
        # from the target), skipping the hand-over transient.
        error = np.abs(np.linalg.norm(self.log["obs"][:, lay["rel"]], axis=1)
                       - lay["standoff"])
        settled = error[len(error) // 2:]
        cos_tilt = self.log["obs"][:, lay["R"]].reshape(-1, 3, 3)[:, 2, 2]   # body z . world z
        loop_dt = self.log["loop_dt"][1:]          # the first entry is not a full period
        telemetry_age = np.maximum(self.log["att_age"], self.log["pos_age"])
        rate_change = np.diff(self.log["rates_frd"], axis=0)
        return {
            "steps": self.steps,
            "seconds": float(self.log["t"][-1]),
            "loop_hz_mean": float(1.0 / np.mean(loop_dt)) if len(loop_dt) else 0.0,
            "loop_dt_max_ms": float(np.max(loop_dt) * 1e3) if len(loop_dt) else 0.0,
            "telemetry_age_p90_ms": float(np.percentile(telemetry_age, 90) * 1e3),
            "clipped_fraction": float(np.mean(self.log["clipped"])),
            "dist_to_hover_point_median_m": float(np.median(settled)),
            "dist_to_hover_point_p90_m": float(np.percentile(settled, 90)),
            "max_tilt_deg": float(np.degrees(np.arccos(np.clip(cos_tilt.min(), -1, 1)))),
            "thrust_mean": float(np.mean(self.log["thrust"])),
            "rate_cmd_change_rms": float(np.sqrt(np.mean(np.sum(rate_change ** 2, axis=1)))),
        }


class PolicyLoop:
    def __init__(self, vehicle, controller, fallback_mode="BRAKE",
                 rate_hz=STEP_RATE_HZ):
        self.vehicle = vehicle
        self.controller = controller
        self.fallback_mode = fallback_mode
        self.period = 1.0 / rate_hz
        self.log = stdout_logger(__name__)

    def run(self, duration_s):
        """Hand over and hover for duration_s."""
        state = self.vehicle.state()
        self.controller.handover(state)
        self.log.info(f"hand-over at NED {np.round(state.pos_ned, 2)}, heading "
                      f"{np.degrees(self.controller.frame.heading):.0f} deg (north-west-up)")
        result = self._fly(duration_s)
        if not result.completed:
            self.log.warning(f"policy stopped: {result.reason}")
            if not result.taken_over:
                self._fall_back()
        return result

    def _fly(self, duration_s):
        rows = []
        start = time.monotonic()
        next_tick = last = start
        while True:
            now = time.monotonic()
            t = now - start
            if t >= duration_s:
                return self._result(rows, completed=True, reason="duration reached")
            state = self.vehicle.state()
            if state.mode not in GUIDED_MODES:
                return self._result(rows, taken_over=True, reason=f"mode changed to {state.mode}")
            if not state.armed:
                return self._result(rows, taken_over=True, reason="disarmed")
            cmd = self.controller.step(state, now)
            if not cmd.ok:
                return self._result(rows, reason=cmd.reason)
            extra = dict(self.controller.extra)
            self.vehicle.send_rates_thrust(cmd.rates_frd, cmd.thrust)

            att_age, pos_age = state.ages(now)
            rows.append(dict(
                t=t, obs=cmd.obs, action=cmd.action, rates_frd=cmd.rates_frd,
                thrust=cmd.thrust, pos_ned=state.pos_ned, vel_ned=state.vel_ned,
                q=state.attitude_q, att_age=att_age, pos_age=pos_age,
                loop_dt=now - last, clipped=cmd.clipped, **extra))
            last = now

            next_tick += self.period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()   # overran; don't try to catch up

    def _result(self, rows, completed=False, taken_over=False, reason=""):
        names = list(LOG_FIELDS) + [k for k in (rows[0] if rows else {})
                                    if k not in LOG_FIELDS]
        log = {name: np.asarray([row[name] for row in rows]) for name in names}
        return LoopResult(completed, taken_over, reason, log,
                          getattr(self.controller, "layout", None))

    def _fall_back(self):
        # Don't override a pilot who switched modes in the meantime.
        if not self.fallback_mode or self.vehicle.state().mode not in GUIDED_MODES:
            return
        try:
            self.vehicle.set_mode(self.fallback_mode)
            self.log.warning(f"switched to {self.fallback_mode}")
        except Exception as e:
            self.log.error(f"fallback to {self.fallback_mode} failed: {e}")
