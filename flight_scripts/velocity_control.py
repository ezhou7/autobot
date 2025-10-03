import asyncio
from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError, VelocityNedYaw
from simple_pid import PID
from common.lidar import get_distance

# -- Constants
# --- Flight Behavior Constants ---
FORWARD_SPEED_M_S = 2.0  # Constant forward speed during tracking
MIN_DISTANCE_M = 2.0  # Safety stop distance

# --- P-Controller Gains ---
# These gains will need tuning. Start with small values.
# Y-axis control (left/right movement)
P_GAIN_Y = 0.01
# Z-axis control (up/down movement)
P_GAIN_Z = -0.01  # Negative because a higher pixel y-coord means moving down

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

    print("Checking if drone is armable...")
    async for is_armable in drone.telemetry.health():
        if is_armable:
            print("-- Drone is armable")
            break
        await asyncio.sleep(1)

    print("-- Arming")
    await retry_command(drone.action.arm)
    print("-- Waiting for EKF to settle after arming...")
    await asyncio.sleep(5) # Increase delay to 5 seconds

    print("-- Setting takeoff altitude")
    await drone.action.set_takeoff_altitude(5.0)

    print("-- Taking off")
    await retry_command(drone.action.takeoff)
    await asyncio.sleep(10) # Wait for takeoff to complete

    # --- Start Offboard Mode ---
    # Set an initial setpoint before starting offboard mode
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    await retry_command(drone.offboard.start)

    # --- Tracking Loop ---
    print("-- Initializing PID controllers")
    center_x, center_y = FRAME_WIDTH / 2, FRAME_HEIGHT / 2
    # The setpoint is the center of the frame, so we want the error to be 0.
    pid_y = PID(Kp=P_GAIN_Y, Ki=0.001, Kd=0.001, setpoint=0, output_limits=(-5, 5))
    pid_z = PID(Kp=P_GAIN_Z, Ki=-0.001, Kd=-0.001, setpoint=0, output_limits=(-5, 5))

    print("-- Starting tracking loop")
    while not stop_flag.is_set():
        try:
            target_x, target_y = target_queue.get_nowait()
            error_x = target_x - center_x
            error_y = target_y - center_y

            # --- PID Control Logic ---
            y_velocity = pid_y(error_x)
            z_velocity = pid_z(error_y)

            # --- Safety Stop Logic ---
            distance = get_distance()
            current_forward_speed = FORWARD_SPEED_M_S if distance > MIN_DISTANCE_M else 0.0
            if current_forward_speed == 0.0:
                print(f"-- Proximity alert! Distance: {distance:.2f}m. Halting forward movement.")

            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(current_forward_speed, y_velocity, z_velocity, 0.0)
            )

        except asyncio.QueueEmpty:
            # If the target is lost, hover in place.
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            )
            # Reset PID controllers when the target is lost to prevent integral windup
            pid_y.reset()
            pid_z.reset()

        await asyncio.sleep(0.1) # Loop at ~10 Hz

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


if __name__ == "__main__":
    # Run the asyncio event loop
    asyncio.run(run())
