from __future__ import annotations

import enum

import numpy as np

from frozone.environments import Environment


class Ball(Environment):

    is_simulation = True

    class XLabels(enum.IntEnum):
        X = 0
        Y = 1

    class ULabels(enum.IntEnum):
        F0 = 0
        F1 = 1
        F2 = 2

    class SLabels(enum.IntEnum):
        F0_DECREASE = 0
        F1_DECREASE = 1
        F2_DECREASE = 2

    class ZLabels(enum.IntEnum):
        VX = 0
        VY = 1

    S_bin_count = (1, 1, 1)

    @classmethod
    def sample_init_process_vars(cls, n: int) -> tuple[np.ndarray, np.ndarray]:
        process_states = super().sample_init_process_vars(n)
        process_states[:, cls.XLabels.X] = np.random.uniform(-0.2, 0.2, n)
        process_states[:, cls.XLabels.Y] = np.random.uniform(-0.2, 0.2, n)
        return process_states

    @classmethod
    def sample_init_control_vars(cls, n: int) -> np.ndarray:
        return np.zeros((n, len(cls.ULabels)), dtype=cls.U_dtype)

    @classmethod
    def sample_new_control_vars(cls, process_states: np.ndarray, prev_control_states: np.ndarray, dt: float) -> np.ndarray:
        reg = 10
        f0 = np.random.normal(prev_control_states[:, 0], dt / reg)
        f1 = np.random.normal(prev_control_states[:, 1], dt / reg)
        f2 = np.random.normal(prev_control_states[:, 2], dt / reg)
        return 1 / (1 + dt / reg) * np.maximum(np.vstack((f0, f1, f2)).T, 0)

    @classmethod
    def sample_init_hidden_vars(cls, n: int) -> np.ndarray:
        return np.zeros((n, len(cls.ZLabels)), dtype=cls.X_dtype)

    @classmethod
    def sample_init_static_vars(cls, n: int) -> np.ndarray:
        static_states = super().sample_init_static_vars(n)
        static_states[...] = np.random.randint(0, 1, static_states.shape)
        return static_states

    @classmethod
    def forward(self, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mass = 2
        k_drag = 5

        x, y = X.T
        vx, vy = Z.T
        f0, f1, f2 = U.T.copy()

        f0[S[:, 0].astype(bool)] *= 0.97
        f1[S[:, 1].astype(bool)] *= 0.98
        f2[S[:, 2].astype(bool)] *= 0.99

        d = np.sqrt(x ** 2 + y ** 2)
        x_neg = x < 0
        theta = np.arctan(y / (x + 1e-9))
        theta[x_neg] += np.pi
        v = np.sqrt(vx ** 2 + vy ** 2)

        Fx_grav = -d * np.cos(theta)
        Fy_grav = -d * np.sin(theta)

        Fx_thrust = -(
            f0 * np.cos(0) + \
            f1 * np.cos(2 / 3 * np.pi) + \
            f2 * np.cos(4 / 3 * np.pi)
        )
        Fy_thrust = -(
            f0 * np.sin(0) + \
            f1 * np.sin(2 / 3 * np.pi) + \
            f2 * np.sin(4 / 3 * np.pi)
        )

        Fx_drag = -k_drag * v * vx
        Fy_drag = -k_drag * v * vy

        Fx = 0.5 * Fx_grav + Fx_thrust + Fx_drag
        Fy = 0.5 * Fy_grav + Fy_thrust + Fy_drag

        new_x = x + dt * vx
        new_y = y + dt * vy
        new_vx = vx + dt * Fx / mass
        new_vy = vy + dt * Fy / mass

        return np.vstack((new_x, new_y)).T, S, np.vstack((new_vx, new_vy)).T
