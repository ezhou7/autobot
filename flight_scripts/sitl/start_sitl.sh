#!/usr/bin/env bash
# Start ArduCopter SITL (built-in quad physics) for testing autobot.
# MAVLink on tcp:127.0.0.1:5760. Ctrl-C to stop.
#
#   ARDUPILOT=~/workspace/ardupilot flight_scripts/sitl/start_sitl.sh [--wipe]
#
# Build once (Copter-4.7.2 tested):
#   git clone --depth 1 --branch Copter-4.7.2 --recurse-submodules \
#       --shallow-submodules https://github.com/ArduPilot/ardupilot.git
#   cd ardupilot && ./waf configure --board sitl && ./waf copter
set -euo pipefail
ARDUPILOT="${ARDUPILOT:-$HOME/workspace/ardupilot}"
BIN="$ARDUPILOT/build/sitl/bin/arducopter"
[[ -x "$BIN" ]] || { echo "no SITL binary at $BIN; build it first (see header)" >&2; exit 1; }
RUN_DIR="${SITL_DIR:-/tmp/autobot-sitl}"
mkdir -p "$RUN_DIR" && cd "$RUN_DIR"
WIPE=()
[[ "${1:-}" == "--wipe" ]] && WIPE=(-w)
exec "$BIN" --model quad --speedup 1 -I0 "${WIPE[@]}" \
    --defaults "$ARDUPILOT/Tools/autotest/default_params/copter.parm"
