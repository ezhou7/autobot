import asyncio
import multiprocessing as mp
from .staging_tracker import tracker_process
from .velocity_control import run as flight_controller_run


async def async_main():
    """
    Main function to orchestrate the tracker and flight controller.
    """
    # Create a multiprocessing context
    ctx = mp.get_context('spawn')
    stop_flag = ctx.Event()
    target_queue = ctx.Queue()

    # --- Start the Tracker Process ---
    tracker = ctx.Process(
        target=tracker_process,
        args=(stop_flag, target_queue)
    )
    tracker.start()
    print("[Main] Started tracker process.")

    # --- Start the Flight Controller ---
    # The flight controller will run in the main process's asyncio loop
    try:
        await flight_controller_run(stop_flag, target_queue)
    except Exception as e:
        print(f"[Main] An error occurred in the flight controller: {e}")
        raise Exception(e)
    finally:
        # --- Cleanup ---
        print("[Main] Mission finished. Cleaning up...")
        stop_flag.set() # Signal all processes to stop
        tracker.join(timeout=5) # Wait for the tracker process to finish
        if tracker.is_alive():
            print("[Main] Tracker process did not terminate gracefully. Terminating.")
            tracker.terminate()
        print("[Main] Cleanup complete.")


def main():
    """Synchronous entry point for the console script."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt received. Shutting down.")
