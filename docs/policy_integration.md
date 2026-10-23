# Flying the hover policy with autobot

autobot can load a hover policy trained in
[ezhou7/dreamerv3](https://github.com/ezhou7/dreamerv3) (DreamerV3, F450
model), run it on the CPU with NumPy, and fly it through ArduPilot over
MAVLink. ArduPilot still takes off and lands; the policy flies in
between, at 50 Hz.

```
ArduPilot ──ATTITUDE_QUATERNION, LOCAL_POSITION_NED (50 Hz)──▶ autobot
   ▲                                                          │ frames (NED/FRD -> z-up/FLU)
   │                                                          │ policy (NumPy, ~2.5 ms on an A76 core)
   └──────── SET_ATTITUDE_TARGET: body rates + thrust ◀────────┘ safety limits
```

## Pieces

| Module | What it does |
|---|---|
| `autobot/policy/` | Loads a release bundle; NumPy forward pass (`reference.py`, `weights.py` vendored from dreamerv3 @ e252b39); observation/action layout (`spec.py`) |
| `autobot/mavlink/frames.py` | ArduPilot north-east-down (NED) / forward-right-down (FRD) ↔ the policy's z-up / forward-left-up. The policy's frame has its origin and +x axis at the drone's position and heading at hand-over |
| `autobot/mavlink/vehicle.py` | pymavlink link: telemetry thread, modes, arm, takeoff, parameters, `send_rates_thrust()`, `hold_position()` |
| `autobot/mavlink/recorder.py` | records position, velocity and mode in the background (used after the hand-back) |
| `autobot/control/policy_controller.py` | state → 22-dim observation → policy → clipped body rates + thrust. The policy hovers at the hand-over point |
| `autobot/control/policy_loop.py` | the 50 Hz loop; stops on mode change, disarm, stale telemetry, too much tilt; flight summary |
| `flight_scripts/policy_hover.py` | take off → hand over → hover → hand back → land; saves a log |
| `config/policy.parm` | ArduPilot parameters the policy needs |
| `flight_scripts/sitl/start_sitl.sh` | starts ArduCopter SITL (software in the loop) |

## Setup

```bash
pip install -r requirements-policy.txt     # numpy<2, ruamel.yaml, pymavlink
# Get a policy bundle (ezhou7/dreamerv3 release), e.g.
gh release download model-skydreamer-smooth-9.4m-1c-2026-10-18 -R ezhou7/dreamerv3
tar -xzf skydreamer-smooth-9.4m-1c-2026-10-18.tar.gz -C ~/models/
```

The first load writes `policy_weights.npz` (~23 MB) into the bundle.

## ArduPilot parameters (`config/policy.parm`)

| Parameter | Value | Why |
|---|---|---|
| `GUID_OPTIONS` | 8 | bit 3: SET_ATTITUDE_TARGET thrust is throttle, not a climb rate |
| `GUID_TIMEOUT` | 0.3 | if autobot stops sending, ArduPilot levels and holds altitude after 0.3 s |
| `ATC_ANGLE_BOOST` | 0 | the policy already adds thrust when tilted |
| `ATC_ACC_R_MAX`, `ATC_ACC_P_MAX`, `ATC_ACC_Y_MAX` | defaults | the policy was trained with ArduPilot's default rate slew |

Thrust is scaled by `MOT_THST_HOVER / 0.50` (0.50 is the trained
vehicle's hover thrust), so the policy's hover maps to this vehicle's
hover throttle. `MOT_THST_EXPO` should match the props so throttle is
linear in thrust.

## SITL

Build ArduCopter SITL once (tested with Copter-4.7.2):

```bash
git clone --depth 1 --branch Copter-4.7.2 --recurse-submodules --shallow-submodules \
    https://github.com/ArduPilot/ardupilot.git
cd ardupilot && ./waf configure --board sitl && ./waf copter
```

Then:

```bash
flight_scripts/sitl/start_sitl.sh --wipe        # terminal 1, MAVLink on tcp:127.0.0.1:5760
python -m flight_scripts.policy_hover \
    --connect tcp:127.0.0.1:5760 --params config/policy.parm \
    --model ~/models/skydreamer-smooth-9.4m-1c-2026-10-18 --seconds 30
```

SITL wind: `SIM_WIND_SPD`, `SIM_WIND_DIR`, `SIM_WIND_TURB`.

### Watching in Mission Planner

SITL accepts a second MAVLink client on TCP port 5762. In Mission
Planner, pick **TCP** in the connection drop-down, press **Connect**, and
enter host `127.0.0.1` (or this computer's IP address when Mission
Planner runs on another computer) and port `5762`. Connect before
starting `policy_hover.py`: the map and the heads-up display show the
take-off, the policy's hover in GUIDED and the landing.

### Results (2026-10-20 and 2026-10-22, Copter-4.7.2 SITL built-in quad, 9.4M policy)

| Test | Result |
|---|---|
| 30 s hover, no wind (4 runs) | 50.0 Hz loop, telemetry age p90 12-21 ms, 8.7-9.4 cm median from the hover point, max tilt 5-11°, no clipping |
| Pilot switches to BRAKE mid-flight | policy stops, autobot leaves the vehicle alone |

## On the drone

- Wire an Orange Pi UART (serial port) to TELEM2; `SERIAL2_PROTOCOL = 2`,
  `SERIAL2_BAUD = 921`. Run with `--connect /dev/ttyXXX --baud 921600`
  (the UART device you enabled).
- Fly with a pilot on the RC (radio control) mode switch. Leaving GUIDED stops the
  policy at once; ArduPilot ignores SET_ATTITUDE_TARGET outside GUIDED.
- Steps: props off (check signs and rates in the log), tethered hover,
  then free hover.
- The policy needs ArduPilot's position and velocity estimate
  (`LOCAL_POSITION_NED`): it hovers relative to a point.

## Logs

Each run saves `logs/policy_<time>.npz`: per step `t, obs, action,
rates_frd, thrust, pos_ned, vel_ned, q, att_age, pos_age,
loop_dt, clipped`, plus `after_*` (altitude, speed and mode for a few
seconds after the policy stops) and a `summary` JSON.

## Safety behaviour

- Body rates clipped to ±3 rad/s, thrust to [0.10, 0.80].
- The loop stops sending if telemetry is older than 100 ms, the tilt
  exceeds 45°, the observation or action is not finite, the mode leaves
  GUIDED, or the drone disarms.
- Safety stop: the loop requests BRAKE (holds position without stick
  input) and the script does not land unless `--land-on-abort`.
- Pilot or ArduPilot takeover (mode change or disarm): autobot sends
  nothing more and requests no fallback mode.
- Normal finish: ArduPilot holds the position in GUIDED, then LAND.
- Do not use LOITER / ALT_HOLD as an automatic fallback: they follow the
  throttle stick, and a stick at the bottom (or SITL without a
  transmitter) means "descend at `PILOT_SPD_DN`" (in SITL: hit the
  ground at 2.5 m/s).

## Known issues / next steps

- **Hand-back sag:** when ArduPilot takes thrust back, altitude drops
  0.25-0.32 m before its altitude controller catches up.
- **Mode changes are seen at 1 Hz** (heartbeat). Harmless, since ArduPilot
  ignores the policy's commands outside GUIDED.
- The ArduPilot quad in SITL is a generic model, not the F450.
  Next SITL step: drive SITL with the dreamerv3 F450 physics through
  ArduPilot's JSON interface.
