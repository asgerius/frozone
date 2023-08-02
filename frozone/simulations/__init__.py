from __future__ import annotations

import abc
import enum

import numpy as np


class Simulation(abc.ABC):

    class ProcessVariables(enum.IntEnum):
        """ States observable to the controller. """

    class ControlVariables(enum.IntEnum):
        """ Variables that are set by the controller, generally referred to as u in literature. """

    class HiddenVariables(enum.IntEnum):
        """ Variables hidden from the controller. """

    @abc.abstractclassmethod
    def sample_init_process_vars(cls, n: int) -> np.ndarray:
        process_states = np.empty((n, len(cls.ProcessVariables)), dtype=np.float32)
        return process_states

    @abc.abstractclassmethod
    def sample_init_control_vars(cls, n: int) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def sample_control_vars(cls, process_states: np.ndarray, prev_control_states: np.ndarray, dt: float) -> np.ndarray:
        """ Sample new control variables for next time step. """

    @abc.abstractclassmethod
    def sample_init_hidden_vars(cls, n: int) -> np.ndarray:
        pass

    @classmethod
    def simulate(cls, n: int, iters: int, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        process_states = np.empty((n, iters + 1, len(cls.ProcessVariables)), dtype=np.float32)
        process_states[:, 0] = cls.sample_init_process_vars(n)

        hidden_states = np.empty((n, iters + 1, len(cls.HiddenVariables)), dtype=np.float32)
        hidden_states[:, 0] = cls.sample_init_hidden_vars(n)

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

            process_states[:, i + 1], hidden_states[:, i + 1] = cls.forward(process_states[:, i], hidden_states[:, i], control_states[:, i], dt)

        return process_states, hidden_states, control_states

    @abc.abstractclassmethod
    def forward(cls, X: np.ndarray, Z: np.ndarray, U: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """ Does a single forward pass. All input matrices should have shape n x d, where n is the number of concurrent simulations
        and d is the number of dimensions of that particular variable. """

    @classmethod
    def forward_multiple(cls, X: np.ndarray, Z: np.ndarray, U: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """ Same as forward, but does multiple forward iterations. X and Z have the same shape as in forward, but U has shape
        n x iterations x d. """
        n, iters = U.shape[:2]

        tmp = X
        X = np.empty((n, iters + 1, len(cls.ProcessVariables)), dtype=np.float32)
        X[:, 0] = tmp

        tmp = Z
        Z = np.empty((n, iters + 1, len(cls.HiddenVariables)), dtype=np.float32)
        Z[:, 0] = tmp

        for i in range(iters):
            X[:, i + 1], Z[:, i + 1] = cls.forward(X[:, i], Z[:, i], U[:, i], dt)

        return X, Z

from .ball import Ball
