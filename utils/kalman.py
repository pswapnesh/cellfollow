"""A small, reusable constant-velocity Kalman filter for one point in d dimensions."""

import numpy as np


class ConstantVelocityKalman:
    """State (2d,): position (d,) then velocity (d,), assuming constant velocity between steps. q, r: process
    and measurement noise (scalar, applied to the identity); p0: initial state uncertainty."""

    def __init__(self, dim=2, q=1.0, r=1.0, p0=1e3):
        self.dim = dim
        self.x = np.zeros(2 * dim)
        self.F = np.eye(2 * dim)
        self.F[:dim, dim:] = np.eye(dim)
        self.H = np.zeros((dim, 2 * dim))
        self.H[:, :dim] = np.eye(dim)
        self.Q = q * np.eye(2 * dim)
        self.R = r * np.eye(dim)
        self.P = p0 * np.eye(2 * dim)

    @property
    def position(self):
        return self.x[:self.dim]

    @property
    def velocity(self):
        return self.x[self.dim:]

    def reset(self, position, velocity=None):
        """Re-seed at a known position (velocity defaults to zero), with the initial uncertainty restored."""
        self.x[:self.dim] = position
        self.x[self.dim:] = 0.0 if velocity is None else velocity
        self.P = self.P[0, 0] * np.eye(2 * self.dim)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.position

    def update(self, z):
        """z: a (dim,) position measurement."""
        err = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ err
        self.P = (np.eye(2 * self.dim) - K @ self.H) @ self.P
        return self.position

    def step(self, z):
        """predict() then update(z): the common case of one measurement per time step."""
        self.predict()
        return self.update(z)
