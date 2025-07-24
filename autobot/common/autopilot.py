import asyncio

import mavsdk
import math

from typing import NamedTuple, Callable, List
from mavsdk.action import ActionError
from mavsdk.telemetry import LandedState, FlightMode
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed, PositionNedYaw
from autobot.common import utils

class Action(NamedTuple):
    func: Callable
    kwargs: dict


class System:
    STOP_VELOCITY = VelocityBodyYawspeed(0.0,0.0,0.0,0.0)
    SLEEP_TIME = 0.05
    DEFAULT_SERIAL_ADDRESS = "/dev/ttyUSB0"
    DEFAULT_UDP_PORT = 14540
    TIMEOUT = 15

    def __init__(self, ip=None, port=None, use_serial=False, serial_address=None):
        self.ip = ip
        self.is_ready = False
        self.actions:List[Action] = []
        self.current_action = ""
        self.use_serial = use_serial
        self.port = port or self.DEFAULT_UDP_PORT
        self.serial = (serial_address if serial_address else self.DEFAULT_SERIAL_ADDRESS)
        self.mav = mavsdk.System()
        self.log = utils.stdout_logger(__name__)

    def close(self):
        del self.mav

    async def start_queue(self):
        try:
            while True:
                if self.actions:
                    action = self.actions.pop(0)
                    self.current_action = action.func.__name__
                    self.log.info(f"running action {self.current_action}")
                    try:
                        await asyncio.wait_for(action.func(self,**action.kwargs), timeout=10)
                    except asyncio.exceptions.TimeoutError:
                        self.log.warning(f"timeout waiting for {self.current_action}")
                    self.current_action = ""
                else:
                    await asyncio.sleep(self.SLEEP_TIME)
        except asyncio.exceptions.CancelledError:
           self.log.warning("system stop")

    def queue_action(self, func: Callable, interrupt: bool = False, **kwargs: dict):
        if interrupt:
            self.clear_queue()
        if func is None:
            return
        
        action = Action(func, kwargs)
        self.actions.append(action)
        self.log.info(f"queue action: {func.__name__}")

    def clear_queue(self):
        self.actions.clear()
        self.log.warning("queue cleared")

    async def connect(self):
        if self.use_serial and self.serial:
            address = f"serial://{self.serial}"
        else:
            address = f"tcp://{self.ip or ''}:{self.port}"
        
        self.log.info(f"waiting for drone connection on address {address}")
        
        await asyncio.wait_for(self.mav.connect(system_address=address), timeout=self.TIMEOUT)

        await System.wait_for_async_value(self.mav.core.connection_state(), is_connected=True)
        self.log.info("connection established.")

        # currently controlling is only available via GPS lock
        await System.wait_for_async_value(self.mav.telemetry.health(), is_global_position_ok=False)
        self.log.info("system ready.")

        self.is_ready = True
    
    async def is_connected(self):
        """Check if the system is connected via MAVLink."""
        return (await System.get_async_generated(self.mav.core.connection_state())).is_connected
    
    async def kill_motors(self):
        await self.mav.action.kill()

    async def hold(self):
        try:
            await self.mav.action.hold()
        except Exception as e:
            self.log.error(e)
            await self.return_home()

    async def return_home(self):
        """Return to home position and land."""
        try:
            await self.mav.action.return_to_launch()
        except ActionError as error:
            self.log.error("RETURN: " + str(error))

        await self.__landing_finished()
    
    async def takeoff(self, check_state=True):
        """Takeoff.
        
        Finishes when the system arrives at the minimum takeoff altitude."""
        if check_state and await self.get_landed_state() != LandedState.ON_GROUND:
            self.log.warning("Cannot take-off, not on ground")
            return
            
        try:
            print("UNARMED")
            await self.mav.action.arm()
            print("ARMED")
        except ActionError as error:
            # if already armed, ignore
            self.log.error("ARM: " + str(error))

        try:
            print("NOT TAKEN OFF")
            await self.mav.action.takeoff()
            await self.__wait_for_landed_state(LandedState.IN_AIR)
            print("TAKEOFF")
        except ActionError as error:
            self.log.error("TAKEOFF: " + str(error))


    async def land(self):
        """Land.
        
        Finishes when the system is in the ground."""
        try:
            await self.mav.action.land()
        except ActionError as error:
            self.log.error("LAND: " + str(error))

        await self.__landing_finished()
    
    async def toggle_takeoff_land(self):
        if await self.get_landed_state() == LandedState.ON_GROUND:
            await self.takeoff(False)
        else:
            await self.land()

    async def start_offboard(self):
        """Start offboard mode where the system's movement can be directly controlled."""
        if await self.get_landed_state() == LandedState.ON_GROUND:
            self.log.warning("Cannot start offboard mode while drone is on the ground")
            return

        await self.mav.offboard.set_velocity_body(self.STOP_VELOCITY)
        try:
            await self.mav.offboard.start()
        except OffboardError as error:
            self.log.error(f"Starting offboard mode failed with error code: {error._result.result}")
            await self.return_home()
            return

        self.log.info("System in offboard mode")
    
    async def stop_offboard(self):
        """Exit offboard mode and return to hold."""
        await self.mav.offboard.set_velocity_body(self.STOP_VELOCITY)

        try:
            await self.mav.offboard.stop()
        except OffboardError as error:
            self.log.error(f"Stopping offboard mode failed with error code: {error._result.result}")
            return

        self.log.info("System exited offboard mode")

    async def toggle_offboard(self):
        """Toggle offboard according to the current state."""
        if await self.is_offboard():
            await self.stop_offboard()
        else:
            await self.start_offboard()

    async def set_velocity(self, forward=0.0, right=0.0, up=0.0, yaw=0.0):
        """Set the system's velocity in body coordinates."""
        if not await self.is_offboard():
            self.log.warning("System is not in offboard move, it cannot move")
        else:
            await self.mav.offboard.set_velocity_body(
                VelocityBodyYawspeed(forward, right, -up, yaw))

    async def move_body_velocity(self, forward=0.0, right=0.0, up=0.0, yaw=0.0, time=1):
        """Move in a particular direction for a set time."""
        await self.set_velocity(forward, right, up, yaw)
        await asyncio.sleep(time)
        await self.set_velocity()

    
    async def set_position_ned_yaw(self, position: PositionNedYaw):
        """Move the system to a target position."""
        if not await self.is_offboard():
            self.log.warning("System is not in offboard move, it cannot move")
        else:
            await self.mav.offboard.set_position_ned(position)


    # Utility functions to move in every direction
    async def move_yaw_right(self): await self.move_body_velocity(yaw=2)
    async def move_yaw_left(self): await self.move_body_velocity(yaw=-2)
    async def move_fwd_positive(self): await self.move_body_velocity(forward=1)
    async def move_fwd_negative(self): await self.move_body_velocity(forward=-1)
    async def move_right(self): await self.move_body_velocity(right=1)
    async def move_left(self): await self.move_body_velocity(right=-1)
    async def move_up(self): await self.move_body_velocity(up=0.5)
    async def move_down(self): await self.move_body_velocity(up=-0.5)

    async def get_position(self):
        return await System.get_async_generated(self.mav.telemetry.position())

    async def get_position_ned_yaw(self):
        pos_ned = (await System.get_async_generated(self.mav.telemetry.position_velocity_ned())).position
        yaw = (await System.get_async_generated(self.mav.telemetry.heading())).heading_deg
        return PositionNedYaw(pos_ned.north_m, pos_ned.east_m, pos_ned.down_m, yaw)

    async def get_attitude(self):
        return await System.get_async_generated(self.mav.telemetry.attitude_euler())

    async def get_yaw_velocity(self):
        yaw_vel = (await System.get_async_generated(self.mav.telemetry.attitude_angular_velocity_body()))
        return yaw_vel.yaw_rad_s * 180 / math.pi

    async def get_ground_velocity(self):
        return await System.get_async_generated(self.mav.telemetry.velocity_ned())

    async def get_ground_velocity_mag(self):
        vel_ned =  await self.get_ground_velocity()
        return (vel_ned.north_m_s ** 2 + vel_ned.east_m_s ** 2) ** 0.5

    async def get_landed_state(self):
        """Return current system landed state"""
        return await System.get_async_generated(self.mav.telemetry.landed_state())

    async def get_flight_mode(self):
        """Return current system flight mode"""
        return await System.get_async_generated(self.mav.telemetry.flight_mode())

    async def is_offboard(self):
        return await self.mav.offboard.is_active()

    async def __landing_finished(self):
        """Runs until the drone has finished landing."""
        await self.__wait_for_landed_state(LandedState.ON_GROUND)
        await self.wait_for_async_value(self.mav.telemetry.armed(), False)
        self.log.info("Landing complete")

    async def __wait_for_landed_state(self, landed_state: LandedState):
        """Wait until the system's landed state match the expected one."""
        await System.wait_for_async_value(self.mav.telemetry.landed_state(), landed_state)
    
    @staticmethod
    async def get_async_generated(generator):
        async for item in generator:
            return item

    @staticmethod
    async def wait_for_async_value(generator, value = None, **kwargs):
        async for item in generator:
            if ((value is None or item == value) and
                (kwargs is None or len(kwargs) == 0 or all([getattr(item, key) == kwargs[key] for key in kwargs.keys()]))):
                break


##############################
############ TEST ############
##############################
async def test():
    drone = System(ip="54.144.75.153", port=5760, use_serial=False)
    await drone.connect()
    # await drone.connect(system_address="serial:///dev/serial0:921600")  ### Serial - UART OrangePi
    # await drone.connect(system_address="serial:///dev/ttyUSB0:57600")  ### Telemetry OrangePi
    async for state in drone.mav.core.connection_state():
        if state.is_connected:
            print("Drone discovered!")
            break

    print(await System.get_async_generated(drone.mav.telemetry.position()))

    await asyncio.sleep(1)
    await drone.takeoff()
    await asyncio.sleep(3)
    await drone.land()

    await drone.close()
