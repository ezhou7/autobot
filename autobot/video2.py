import asyncio
from threading import Event

from autobot.common.autopilot import System
from autobot.common.queue import MessageBroker
from autobot.orchestrator import Orchestrator


def execute_input_thread(stop_flag: Event, msg_broker: MessageBroker):
    while not stop_flag.is_set():
        user_input = input("Enter 'q' to quit: ")
        if user_input == 'q':
            stop_flag.set()
        msg_broker.put("user_input", user_input)


async def sitl_thread(stop_flag: Event, msg_broker: MessageBroker):
    drone = System(ip="54.144.75.153", port=5760, use_serial=False)
    await drone.connect()
    async for state in drone.mav.core.connection_state():
        print(state)
        if state.is_connected:
            print("Drone discovered!")
            break

    # print(await System.get_async_generated(drone.mav.telemetry.position()))

    await drone.takeoff(check_state=False)
    while not stop_flag.is_set():
        if not msg_broker.empty("tracker_to_follower"):
            fxc, fyc, xc, yc = msg_broker.get("tracker_to_follower")
            print(f"Received centroids: frame={(fxc, fyc)}, obj={(xc, yc)}")
        await asyncio.sleep(2)

    await drone.land()
    print("Disconnected")



def async_thread(stop_flag: Event, msg_broker: MessageBroker):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(sitl_thread(stop_flag, msg_broker))
    loop.close()


if __name__ == "__main__":
    orchestrator = Orchestrator([execute_input_thread, async_thread])
    orchestrator.execute()
