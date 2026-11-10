"""A small, reusable PID controller for a scalar error signal."""

import numpy as np


class PID:
    """kp/ki/kd gains on a scalar error. step(error, dt) returns the control output. The integral term is
    clamped to i_limit to avoid windup, and the output is clamped to out_limit; either may be None to disable
    its clamp. dt defaults to 1 (one step() call = one time step) for callers that don't track real time."""

    def __init__(self, kp, ki=0.0, kd=0.0, i_limit=None, out_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit, self.out_limit = i_limit, out_limit
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def step(self, error, dt=1.0):
        self.integral += error * dt
        if self.i_limit is not None:
            self.integral = float(np.clip(self.integral, -self.i_limit, self.i_limit))
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(out, -self.out_limit, self.out_limit)) if self.out_limit is not None else out
