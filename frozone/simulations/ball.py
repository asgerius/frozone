from __future__ import annotations

import enum

import numpy as np

from frozone.simulations import Simulation


class Ball(Simulation):

    class ProcessVariables(enum.IntEnum):
        X = 0
        Y = 1

    class ControlVariables(enum.IntEnum):
        F0 = 0
        F1 = 1
        F2 = 2

    class HiddenVariables(enum.IntEnum):
        VX = 0
        VY = 1

    class StaticVariables(enum.IntEnum):
        THRUST_TYPE = 0

    static_value_count = [2 ** 3]

    @classmethod
    def sample_init_process_vars(cls, n: int) -> tuple[np.ndarray, np.ndarray]:
        process_states = super().sample_init_process_vars(n)
        process_states[:, cls.ProcessVariables.X] = np.random.uniform(-0.2, 0.2, n)
        process_states[:, cls.ProcessVariables.Y] = np.random.uniform(-0.2, 0.2, n)
        return process_states

    @classmethod
    def sample_init_control_vars(cls, n: int) -> np.ndarray:
        return np.zeros((n, len(cls.ControlVariables)), dtype=np.float32)

    @classmethod
    def sample_new_control_vars(cls, process_states: np.ndarray, prev_control_states: np.ndarray, dt: float) -> np.ndarray:
        reg = 10
        f0 = np.random.normal(prev_control_states[:, 0], dt / reg)
        f1 = np.random.normal(prev_control_states[:, 1], dt / reg)
        f2 = np.random.normal(prev_control_states[:, 2], dt / reg)
        return 1 / (1 + dt / reg) * np.maximum(np.vstack((f0, f1, f2)).T, 0)

    @classmethod
    def sample_init_hidden_vars(cls, n: int) -> np.ndarray:
        return np.zeros((n, len(cls.HiddenVariables)), dtype=np.float32)

    @classmethod
    def sample_init_static_vars(cls, n: int) -> np.ndarray:
        static_states = super().sample_init_static_vars(n)
        static_states[...] = np.random.randint(0, 8, static_states.shape)
        return static_states

    @classmethod
    def forward(self, X: np.ndarray, U: np.ndarray, Z: np.ndarray, S: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mass = 2
        k_drag = 5

        x, y = X.T
        vx, vy = Z.T
        f0, f1, f2 = U.T.copy()
        f0[np.bitwise_and(S.flat, 0b1)   == 0b1]   *= 0.97
        f1[np.bitwise_and(S.flat, 0b10)  == 0b10]  *= 0.98
        f2[np.bitwise_and(S.flat, 0b100) == 0b100] *= 0.99

        d = np.sqrt(x ** 2 + y ** 2)
        x_neg = x < 0
        theta = np.arctan(y / x)
        theta[x_neg] += np.pi
        v = np.sqrt(vx ** 2 + vy ** 2)

        Frx = -d * np.cos(theta)
        Fry = -d * np.sin(theta)

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

        Fdx = -k_drag * v * vx
        Fdy = -k_drag * v * vy

        Fx = 0.5 * Frx + Ftx + Fdx
        Fy = 0.5 * Fry + Fty + Fdy

        new_x = x + dt * vx
        new_y = y + dt * vy
        new_vx = vx + dt * Fx / mass
        new_vy = vy + dt * Fy / mass

        return np.vstack((new_x, new_y)).T, np.vstack((new_vx, new_vy)).T, S
