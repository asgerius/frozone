from __future__ import annotations

import enum

import numpy as np


from frozone.simulations import Simulation

class Ball(Simulation):

    class ProcessVariables(enum.IntEnum):
        X = 0
        Y = 1
        DX = 2
        DY = 3

    class ControlVariables(enum.IntEnum):
        F0 = 0
        F1 = 1
        F2 = 2

    @classmethod
    def sample_init_process_vars(cls, n: int) -> np.ndarray:
        process_states = super().sample_init_process_vars(n)
        process_states[:, cls.ProcessVariables.X] = np.random.uniform(-0.5, 0.5, n)
        process_states[:, cls.ProcessVariables.Y] = np.random.uniform(-0.5, 0.5, n)
        process_states[:, cls.ProcessVariables.DX] = 0
        process_states[:, cls.ProcessVariables.DY] = 0
        return process_states

    @classmethod
    def sample_init_control_vars(cls, n: int) -> np.ndarray:
        return np.zeros((n, len(cls.ControlVariables)), dtype=np.float32)

    @classmethod
    def sample_control_vars(cls, process_states: np.ndarray, prev_control_states: np.ndarray, dt: float) -> np.ndarray:
        f0 = np.random.uniform(np.maximum(prev_control_states[:, 0] - dt, 0), np.minimum(prev_control_states[:, 0] + dt, 1))
        f1 = np.random.uniform(np.maximum(prev_control_states[:, 1] - dt, 0), np.minimum(prev_control_states[:, 1] + dt, 1))
        f2 = np.random.uniform(np.maximum(prev_control_states[:, 2] - dt, 0), np.minimum(prev_control_states[:, 2] + dt, 1))
        return np.vstack((f0, f1, f2)).T

    @classmethod
    def forward(self, process_states: np.ndarray, control_states: np.ndarray, dt: float) -> np.ndarray:
        mass = 1
        k_drag = 1

        x, y, dx, dy = process_states.T
        f0, f1, f2 = control_states.T

        d = np.sqrt(x ** 2 + y ** 2)
        x_neg = x < 0
        theta = np.arctan(y / x)
        theta[x_neg] += np.pi
        v = np.sqrt(dx ** 2 + dy ** 2)

        Frx = -d ** 2 * np.cos(theta)
        Fry = -d ** 2 * np.sin(theta)

        Ftx = -(
            f0 * np.cos(0) + \
            f1 * np.cos(2 / 3 * np.pi) + \
            f2 * np.cos(4 / 3 * np.pi)
        )
        Fty = -(
            f0 * np.sin(0) + \
            f1 * np.sin(2 / 3 * np.pi) + \
            f2 * np.sin(4 / 3 * np.pi)
        )

        Fdx = -k_drag * v * dx
        Fdy = -k_drag * v * dy

        Fx = 0.5 * Frx + Ftx + Fdx
        Fy = 0.5 * Fry + Fty + Fdy

        new_x = x + dt * dx
        new_y = y + dt * dy
        new_dx = dx + dt * Fx / mass
        new_dy = dy + dt * Fy / mass

        return np.vstack((new_x, new_y, new_dx, new_dy)).T
