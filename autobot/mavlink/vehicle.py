"""MAVLink link to an ArduPilot copter (pymavlink).

A background thread keeps the latest state; `state()` returns a copy.
Connection examples: SITL `tcp:127.0.0.1:5760`, or the Orange Pi serial
port wired to TELEM2 (`/dev/ttyS6`, 921600 baud).
"""
import copy
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from pymavlink import mavutil

from autobot.utils.logging import stdout_logger

mavlink = mavutil.mavlink

GUIDED_MODES = ("GUIDED", "GUIDED_NOGPS")

# SET_ATTITUDE_TARGET: use the body rates and thrust, ignore the attitude.
TYPE_MASK_BODY_RATES = mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE

# SET_POSITION_TARGET_LOCAL_NED: use x, y, z only.
TYPE_MASK_POSITION_ONLY = (
    mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE | mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
    | mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE | mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE | mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE | mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)


@dataclass
class VehicleState:
    """Latest vehicle state; the *_time fields are time.monotonic() at reception."""
    attitude_q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    rates_frd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude_time: float = 0.0
    pos_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position_time: float = 0.0
    mode: str = ""
    armed: bool = False
    heartbeat_time: float = 0.0

    def ages(self, now=None):
        """Seconds since the last attitude and the last position message."""
        now = time.monotonic() if now is None else now
        return now - self.attitude_time, now - self.position_time


class Vehicle:
    def __init__(self, url, baud=921600, source_system=255,
                 source_component=mavlink.MAV_COMP_ID_ONBOARD_COMPUTER):
        self.log = stdout_logger(__name__)
        self.url = url
        self.conn = mavutil.mavlink_connection(
            url, baud=baud, source_system=source_system,
            source_component=source_component, autoreconnect=True)
        self.texts = []          # (time, text) of every STATUSTEXT
        self._state = VehicleState()
        self._state_lock = threading.Lock()
        self._send_lock = threading.Lock()
        # Replies (PARAM_VALUE, COMMAND_ACK) that the reader thread hands
        # to a waiting request.
        self._replies = threading.Condition()
        self._params = {}
        self._acks = {}
        self._stop = threading.Event()
        self._threads = []

    def connect(self, timeout=30.0):
        self.log.info(f"waiting for heartbeat on {self.url}")
        heartbeat = self._wait_for_autopilot_heartbeat(timeout)
        self.conn.target_system = heartbeat.get_srcSystem()
        self.conn.target_component = heartbeat.get_srcComponent()
        self.log.info(f"connected: system {self.conn.target_system} "
                      f"component {self.conn.target_component}")
        self._handle(heartbeat)
        for loop in (self._read_loop, self._heartbeat_loop):
            thread = threading.Thread(target=loop, daemon=True)
            thread.start()
            self._threads.append(thread)

    def close(self):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self.conn.close()

    def state(self):
        with self._state_lock:
            return copy.deepcopy(self._state)

    def wait_for(self, predicate, timeout, what):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate(self.state()):
                return
            time.sleep(0.05)
        raise TimeoutError(f"timed out waiting for {what}")

    def request_rate(self, message_name, hz):
        msg_id = getattr(mavlink, f"MAVLINK_MSG_ID_{message_name}")
        self.command(mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz, wait=False)

    def command(self, cmd, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0,
                wait=True, timeout=3.0):
        """Send COMMAND_LONG. Returns the MAV_RESULT, or None if wait is False."""
        args = (self.conn.target_system, self.conn.target_component,
                cmd, 0, p1, p2, p3, p4, p5, p6, p7)
        if not wait:
            self._send(self.conn.mav.command_long_send, *args)
            return None
        result = self._request(self._acks, cmd, timeout, self.conn.mav.command_long_send, *args)
        if result is None:
            raise TimeoutError(f"no COMMAND_ACK for command {cmd}")
        return result

    def set_mode(self, name, timeout=5.0):
        mapping = self.conn.mode_mapping()
        if name not in mapping:
            raise ValueError(f"unknown mode {name}; known: {sorted(mapping)}")
        # Confirmed by the mode in ArduPilot's heartbeat, not by COMMAND_ACK.
        self.command(mavlink.MAV_CMD_DO_SET_MODE,
                     mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mapping[name], wait=False)
        self.wait_for(lambda s: s.mode == name, timeout, f"mode {name}")

    def arm(self, timeout=10.0):
        result = self.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        if result != mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError(f"arming refused (result {result}); "
                               f"last ArduPilot messages: {self.texts[-3:]}")
        self.wait_for(lambda s: s.armed, timeout, "armed")

    def takeoff(self, altitude_m):
        result = self.command(mavlink.MAV_CMD_NAV_TAKEOFF, p7=altitude_m)
        if result != mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError(f"takeoff refused (result {result})")

    def get_param(self, name, timeout=3.0):
        value = self._request(self._params, name, timeout,
                              self.conn.mav.param_request_read_send,
                              self.conn.target_system, self.conn.target_component,
                              name.encode(), -1)
        if value is None:
            raise TimeoutError(f"no value for parameter {name}")
        return value

    def set_param(self, name, value, timeout=3.0):
        value = float(value)
        reported = self._request(self._params, name, timeout,
                                 self.conn.mav.param_set_send,
                                 self.conn.target_system, self.conn.target_component,
                                 name.encode(), value, mavlink.MAV_PARAM_TYPE_REAL32)
        if reported is None:
            raise TimeoutError(f"parameter {name} not acknowledged")
        if abs(reported - value) > 1e-4 * max(1.0, abs(value)):
            raise RuntimeError(f"parameter {name}: set {value}, ArduPilot reports {reported}")

    def hold_position(self, pos_ned):
        """GUIDED position target (NED, m). Unlike LOITER, ArduPilot holds
        it without reading the pilot's sticks."""
        x, y, z = (float(v) for v in pos_ned)
        self._send(self.conn.mav.set_position_target_local_ned_send,
                   self._time_boot_ms(), self.conn.target_system, self.conn.target_component,
                   mavlink.MAV_FRAME_LOCAL_NED, TYPE_MASK_POSITION_ONLY,
                   x, y, z, 0, 0, 0, 0, 0, 0, 0, 0)

    def send_rates_thrust(self, rates_frd, thrust):
        """Body rates (rad/s, forward-right-down) and thrust (0..1).

        ArduPilot uses this only in GUIDED / GUIDED_NOGPS, and reads thrust
        as throttle only if GUID_OPTIONS bit 3 (8) is set; otherwise thrust
        is a climb rate.
        """
        roll, pitch, yaw = (float(v) for v in rates_frd)
        self._send(self.conn.mav.set_attitude_target_send,
                   self._time_boot_ms(), self.conn.target_system, self.conn.target_component,
                   TYPE_MASK_BODY_RATES, [1.0, 0.0, 0.0, 0.0],
                   roll, pitch, yaw, float(thrust))

    def _wait_for_autopilot_heartbeat(self, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            heartbeat = self.conn.recv_match(type="HEARTBEAT", blocking=True,
                                             timeout=max(0.1, end - time.monotonic()))
            # Ground stations and other components send heartbeats too.
            if (heartbeat is not None
                    and heartbeat.autopilot != mavlink.MAV_AUTOPILOT_INVALID
                    and heartbeat.type != mavlink.MAV_TYPE_GCS):
                return heartbeat
        raise TimeoutError(f"no autopilot heartbeat on {self.url} in {timeout} s")

    def _heartbeat_loop(self):
        while not self._stop.is_set():
            self._send(self.conn.mav.heartbeat_send,
                       mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                       mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self._stop.wait(1.0)

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                msg = self.conn.recv_match(blocking=True, timeout=0.2)
            except Exception as e:  # serial hiccup; autoreconnect reopens the port
                self.log.warning(f"read error: {e}")
                continue
            if msg is not None:
                self._handle(msg)

    def _handle(self, msg):
        if msg.get_srcSystem() != self.conn.target_system:
            return
        now = time.monotonic()
        kind = msg.get_type()
        if kind == "ATTITUDE_QUATERNION":
            with self._state_lock:
                self._state.attitude_q = np.array([msg.q1, msg.q2, msg.q3, msg.q4])
                self._state.rates_frd = np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])
                self._state.attitude_time = now
        elif kind == "LOCAL_POSITION_NED":
            with self._state_lock:
                self._state.pos_ned = np.array([msg.x, msg.y, msg.z])
                self._state.vel_ned = np.array([msg.vx, msg.vy, msg.vz])
                self._state.position_time = now
        elif kind == "HEARTBEAT":
            if msg.get_srcComponent() != self.conn.target_component:
                return
            with self._state_lock:
                self._state.mode = mavutil.mode_string_v10(msg)
                self._state.armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self._state.heartbeat_time = now
        elif kind == "PARAM_VALUE":
            self._store_reply(self._params, msg.param_id, msg.param_value)
        elif kind == "COMMAND_ACK":
            self._store_reply(self._acks, msg.command, msg.result)
        elif kind == "STATUSTEXT":
            self.texts.append((now, msg.text))
            self.log.info(f"ArduPilot: {msg.text}")

    def _send(self, send_fn, *args):
        with self._send_lock:
            send_fn(*args)

    def _request(self, replies, key, timeout, send_fn, *args):
        """Send a message, then wait for the reader thread to put the reply
        in replies[key]. Returns the reply, or None on timeout."""
        with self._replies:
            replies.pop(key, None)
        self._send(send_fn, *args)
        with self._replies:
            if self._replies.wait_for(lambda: key in replies, timeout):
                return replies.pop(key)
        return None

    def _store_reply(self, replies, key, value):
        with self._replies:
            replies[key] = value
            self._replies.notify_all()

    @staticmethod
    def _time_boot_ms():
        return int(time.monotonic() * 1000) & 0xFFFFFFFF
