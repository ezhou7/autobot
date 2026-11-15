"""Constant-velocity Kalman filter for the target's position.

Works in ArduPilot's local north-east-down (NED) frame, so the target's
own motion is separated from the drone's. Fixes arrive stamped with the
time their camera frame was taken; the policy asks for the estimate at
the current time, which the filter extrapolates.

Fixes can arrive out of order: a stereo range (slow) for a frame that
is older than the direction-only fixes of later frames. The filter keeps
the last `history_s` of fixes; a late fix rewinds to the state before it,
goes in at its own time, and the later fixes are applied again.

The policy only flies while the estimate is fresh: `age()` is the time
since the newest fix, and controllers stop past their limit (the target
left the camera's view for too long).
"""
import bisect

import numpy as np

H = np.hstack([np.eye(3), np.zeros((3, 3))])


class TargetFilter:

    def __init__(self, accel_sigma=3.0, init_vel_sigma=2.0, history_s=2.0):
        self.accel_sigma = float(accel_sigma)     # m/s^2, the target's unmodeled acceleration
        self.init_vel_sigma = float(init_vel_sigma)
        self.history_s = float(history_s)
        self.reset()

    def reset(self):
        self._fixes = []       # (t, p, cov), time order
        self._states = []      # (x, P) after each fix
        self.fixes = 0

    @property
    def t(self):
        return self._fixes[-1][0] if self._fixes else None

    def update(self, t, p, cov):
        """A fix: position p (3,) NED with covariance cov (3, 3), at time
        t. Returns False if it is older than the history kept."""
        p, cov = np.asarray(p, float), np.asarray(cov, float)
        if self._fixes and t < self._fixes[-1][0] - self.history_s:
            return False
        i = bisect.bisect_right([f[0] for f in self._fixes], t)
        later = self._fixes[i:]
        del self._fixes[i:], self._states[i:]
        for fix in [(t, p, cov)] + later:
            self._apply(*fix)
        self.fixes += 1
        cutoff = self._fixes[-1][0] - self.history_s
        while len(self._fixes) > 1 and self._fixes[0][0] < cutoff:
            self._fixes.pop(0)
            self._states.pop(0)
        return True

    def _apply(self, t, p, cov):
        if not self._states:
            x = np.concatenate([p, np.zeros(3)])
            P = np.block([[cov, np.zeros((3, 3))],
                          [np.zeros((3, 3)), self.init_vel_sigma ** 2 * np.eye(3)]])
        else:
            x, P = self._predict(*self._states[-1], t - self._fixes[-1][0])
            S = H @ P @ H.T + cov
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (p - H @ x)
            P = (np.eye(6) - K @ H) @ P
            P = (P + P.T) / 2
        self._fixes.append((t, p, cov))
        self._states.append((x, P))

    def estimate(self, t):
        """(position (3,), velocity (3,), position covariance (3, 3)) at
        time t, or None before the first fix. Doesn't change the state."""
        if not self._states:
            return None
        x, P = self._predict(*self._states[-1], max(0.0, t - self.t))
        return x[:3], x[3:], P[:3, :3]

    def age(self, t):
        return np.inf if not self._fixes else t - self.t

    def _predict(self, x, P, dt):
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        q = self.accel_sigma ** 2
        Q = np.block([[dt ** 4 / 4 * q * np.eye(3), dt ** 3 / 2 * q * np.eye(3)],
                      [dt ** 3 / 2 * q * np.eye(3), dt ** 2 * q * np.eye(3)]])
        return F @ x, F @ P @ F.T + Q
