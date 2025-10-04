import asyncio
import queue
from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError, VelocityNedYaw
from simple_pid import PID
from common.lidar import get_distance

# -- Constants
# --- Flight Behavior Constants ---
TAKEOFF_ALTITUDE_M = 10.0 # Meters
FORWARD_SPEED_M_S = 2.0  # Constant forward speed during tracking
MIN_DISTANCE_M = 2.0  # Safety stop distance

# --- P-Controller Gains ---
# These gains will need tuning. Start with small values.
# Y-axis control (left/right movement)
PID_KP_Y = 0.01
# Z-axis control (up/down movement)
PID_KP_Z = 0.01

# --- Camera/Frame Constants ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


async def retry_command(command, retries=3, delay=1, backoff=2):
    """Attempts to execute an async command with retries and exponential backoff."""
    last_exception = None
    for i in range(retries):
        try:
            return await command()
        except (ActionError, OffboardError) as e:
            last_exception = e
            print(f"Command failed with error: {e}. Attempt {i + 1} of {retries}.")
            if i < retries - 1:
                await asyncio.sleep(delay)
                delay *= backoff
    raise last_exception


async def monitor_in_air(drone, stop_flag):
    """ Monitors the drone's in_air state and sets the stop_flag if it lands unexpectedly. """
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("-- Drone has landed or is not in the air. Stopping mission.")
            stop_flag.set()
            break

async def run(stop_flag, target_queue):
    """ Connects to the drone, takes off, flies a square pattern using velocity commands, and lands. """
    drone = System()
    # Connect to the drone. 'udpin://0.0.0.0:14550' is the default for SITL.
    # For a real drone, you might use 'serial:///dev/ttyUSB0:57600' or similar.
    await drone.connect(system_address="udpin://0.0.0.0:14550")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    # --- Check if drone is already in the air ---
    print("-- Checking drone state...")
    in_air = False # Default to assuming the drone is on the ground
    try:
        in_air = await asyncio.wait_for(anext(aiter(drone.telemetry.in_air())),
                                      timeout=5.0)
        if in_air:
            print("-- Drone is already in the air. Skipping takeoff and taking control.")
        else:
            print("-- Drone is on the ground. Proceeding with takeoff.")
    except asyncio.TimeoutError:
        in_air = False # Explicitly set to false on timeout

    # --- Initialize PID controllers ---
    print("-- Initializing PID controllers")
    center_x, center_y = FRAME_WIDTH / 2, FRAME_HEIGHT / 2
    pid_y = PID(Kp=PID_KP_Y, Ki=0.001, Kd=0.001, setpoint=0, output_limits=(-5, 5))
    pid_z = PID(Kp=PID_KP_Z, Ki=0.001, Kd=0.001, setpoint=0, output_limits=(-5, 5))

    if not in_air:
        # --- Takeoff --- 
        print("-- Arming")
        await retry_command(drone.action.arm)
        print("-- Setting takeoff altitude")
        await drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE_M)
        
        print("-- Taking off")
        await retry_command(drone.action.takeoff)
        # Wait for the drone to reach a stable altitude
        await asyncio.sleep(10)

    # --- Start Offboard Mode & State Monitoring ---
    print("-- Starting offboard mode")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    await retry_command(drone.offboard.start)

    print("-- Starting in-air state monitor")
    monitor_task = asyncio.create_task(monitor_in_air(drone, stop_flag))

    y_velocity = 0.0
    z_velocity = 0.0
    current_forward_speed = FORWARD_SPEED_M_S

    # --- Tracking Loop ---
    print("-- Starting tracking loop")
    while not stop_flag.is_set():
        try:
            # Block and wait for a target for up to 1 second.
            target_x, target_y = target_queue.get(timeout=0.01)
            # print(f"[Tracker] Target coordinates: ({target_x}, {target_y})")

            if target_x == -1 and target_y == -1:
                # If the queue is empty after the timeout, the target is lost. Hover in place.
                print("-- Target lost. Hovering.")
                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                )
                # Reset PID controllers to prevent integral windup
                pid_y.reset()
                pid_z.reset()
                continue

            error_x = target_x - center_x
            error_y = target_y - center_y

            # --- PID Control Logic ---
            y_velocity = pid_y(error_x)
            z_velocity = pid_z(error_y)

            print(f"-- PID Control: y_velocity={y_velocity:.2f}, z_velocity={z_velocity:.2f}")

            # --- Safety Stop Logic ---
            distance = get_distance()
            current_forward_speed = FORWARD_SPEED_M_S if distance > MIN_DISTANCE_M else 0.0
            if current_forward_speed == 0.0:
                print(f"-- Proximity alert! Distance: {distance:.2f}m. Halting forward movement.")

            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(current_forward_speed, y_velocity, z_velocity, 0.0)
            )

        except queue.Empty:
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(current_forward_speed, y_velocity, z_velocity, 0.0)
            )

    print("-- Stopping Copter")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)

    # --- Stop Offboard Mode ---
    print("-- Stopping offboard mode")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed with error code: {error._result.result}")

    print("-- Landing")
    await drone.action.land()

    # Clean up the monitor task
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        print("-- In-air state monitor stopped.")


