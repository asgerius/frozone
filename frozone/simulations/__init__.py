from __future__ import annotations

import abc
import enum

import numpy as np


class Simulation(abc.ABC):

    class ProcessVariables(enum.IntEnum):
        pass

    class ControlVariables(enum.IntEnum):
        pass

    @abc.abstractclassmethod
    def sample_init_process_vars(cls, n: int) -> np.ndarray:
        process_states = np.empty((n, len(cls.ProcessVariables)), dtype=np.float32)
        return process_states

    @abc.abstractclassmethod
    def sample_init_control_vars(cls, n: int) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def sample_control_vars(cls, process_states: np.ndarray, prev_control_states: np.ndarray, dt: float) -> np.ndarray:
        pass

    @classmethod
    def simulate(cls, n: int, iters: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
        process_states = np.empty((n, iters + 1, len(cls.ProcessVariables)), dtype=np.float32)
        process_states[:, 0] = cls.sample_init_process_vars(n)

        control_states = np.empty((n, iters, len(cls.ControlVariables)), dtype=np.float32)

        for i in range(iters):
            if i == 0:
                control_states[:, i] = cls.sample_control_vars(
                    process_states[:, i],
                    cls.sample_init_control_vars(n),
                    dt = dt,
                )
            else:
                control_states[:, i] = cls.sample_control_vars(
                    process_states[:, i],
                    control_states[:, i - 1],
                    dt = dt
                )

            process_states[:, i + 1] = cls.forward(process_states[:, i], control_states[:, i], dt)

        return process_states, control_states

    @abc.abstractclassmethod
    def forward(cls, process_states: np.ndarray, control_states: np.ndarray, dt: float) -> np.ndarray:
        pass

    @classmethod
    def forward_multiple(cls, process_states: np.ndarray, control_states: np.ndarray, dt: float) -> np.ndarray:
        n, iters = control_states.shape[:2]

        tmp = process_states
        process_states = np.empty((n, iters + 1, len(cls.ProcessVariables)), dtype=np.float32)
        process_states[:, 0] = tmp

        for i in range(iters):
            process_states[:, i + 1] = cls.forward(process_states[:, i], control_states[:, i], dt)

        return process_states
