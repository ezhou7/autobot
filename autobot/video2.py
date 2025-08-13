import asyncio
import numpy as np
from threading import Event

from autobot.common.autopilot import System
from autobot.common.queue import MessageBroker
from autobot.orchestrator import Orchestrator
from autobot.follow.controller import Controller
from autobot.common.lidar import LIDARLite
from autobot.common.tf_luna_lidar import TFLunaLidar


def execute_input_thread(stop_flag: Event, msg_broker: MessageBroker):
    while not stop_flag.is_set():
        user_input = input("Enter 'q' to quit: ")
        if user_input == 'q':
            stop_flag.set()
        msg_broker.put("user_input", user_input)


def lidar_thread(stop_flag: Event, msg_broker: MessageBroker):
    distance_error_threshold = 10
    lidar = LIDARLite()
    while not stop_flag.is_set():
        if not msg_broker.empty("tracker_to_lidar"):
            fxc, fyc, xc, yc = msg_broker.get("tracker_to_lidar")
            if np.linalg.norm(np.array([fxc, fyc]) - np.array([xc, yc])) < distance_error_threshold:
                dist_to_target = lidar.read_distance_v3hp()
                print(f"Distance to target={dist_to_target}")


def downward_lidar_thread(stop_flag: Event, msg_broker: MessageBroker):
    lidar = TFLunaLidar()
    print(lidar.get_version())
    lidar.set_sample_rate(100)

    while not stop_flag.is_set():
        dist, strength, temp = lidar.read_data()
        print(f"height={dist}, signal strength={strength}, temperature={temp}")

    lidar.ser.close()


async def sitl_function(stop_flag: Event, msg_broker: MessageBroker):
    drone = System(ip="54.144.75.153", port=5760, use_serial=False)
    await drone.connect()
    async for state in drone.mav.core.connection_state():
        if state.is_connected:
            print("Drone discovered!")
            break

    # print(await System.get_async_generated(drone.mav.telemetry.position()))

    controller = Controller(320, 320)
    await drone.takeoff(check_state=False)
    await drone.start_offboard()
    while not stop_flag.is_set():
        if not msg_broker.empty("tracker_to_follower"):
            _, _, xc, yc = msg_broker.get("tracker_to_follower")
            # print(f"Received centroids: obj={(xc, yc)}")
            yaw, fwd = controller.control(np.array((0, 0)), np.array((xc, yc)))
            print(f"yaw={yaw}, fwd={fwd}")
            await drone.set_velocity(forward=fwd, yaw=yaw)
        await asyncio.sleep(2)

    await drone.land()


def async_thread(stop_flag: Event, msg_broker: MessageBroker):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(sitl_function(stop_flag, msg_broker))
    loop.close()


if __name__ == "__main__":
    orchestrator = Orchestrator([
        execute_input_thread,
        lidar_thread,
        async_thread
    ])
    orchestrator.execute()
