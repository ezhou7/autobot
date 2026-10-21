"""Record a vehicle's position, velocity and mode in the background."""
import threading
import time

import numpy as np


class StateRecorder:
    """Samples vehicle.state() at `hz` while the `with` block runs.

        with StateRecorder(vehicle) as recorder:
            ...
        arrays = recorder.arrays()
    """

    def __init__(self, vehicle, hz=20.0):
        self.vehicle = vehicle
        self.period = 1.0 / hz
        self.rows = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._start_time = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def arrays(self):
        if not self.rows:
            return {"t": np.zeros(0)}
        t, pos_ned, vel_ned, mode = zip(*self.rows)
        return {"t": np.array(t), "pos_ned": np.array(pos_ned),
                "vel_ned": np.array(vel_ned), "mode": np.array(mode)}

    def _run(self):
        while not self._stop.is_set():
            s = self.vehicle.state()
            self.rows.append((time.monotonic() - self._start_time, s.pos_ned, s.vel_ned, s.mode))
            self._stop.wait(self.period)
