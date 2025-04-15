import asyncio
from queue import Queue
from threading import Event

from autobot.common.autopilot import test
from autobot.orchestrator import Orchestrator


def execute_input_thread(stop_flag: Event, input_queue: Queue):
    while not stop_flag.is_set():
        user_input = input("Enter 'q' to quit: ")
        input_queue.put(user_input)
        if user_input == 'q':
            stop_flag.set()
            break


def async_thread(stop_flag: Event, input_queue: Queue):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test())
    loop.close()


if __name__ == "__main__":
    orchestrator = Orchestrator([execute_input_thread, async_thread])
    orchestrator.execute()
