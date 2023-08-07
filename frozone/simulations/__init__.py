from __future__ import annotations

import abc
import enum

import numpy as np


class Simulation(abc.ABC):

    class ProcessVariables(enum.IntEnum):
        """ Mutable states observable to the controller. """

    class ControlVariables(enum.IntEnum):
        """ Variables that are set by the controller, generally referred to as u in literature. """

    class HiddenVariables(enum.IntEnum):
        """ Variables hidden from the controller. """

    class StaticVariables(enum.IntEnum):
        """ Categorical and/or discrete properties. These can change over time, but do so independently of control. """

    # The number of values each static variable can take in the order of StaticVariables
    static_value_count = list()

    @abc.abstractclassmethod
    def sample_init_process_vars(cls, n: int) -> np.ndarray:
        process_states = np.empty((n, len(cls.ProcessVariables)), dtype=np.float32)
        return process_states

    @abc.abstractclassmethod
    def sample_init_control_vars(cls, n: int) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def sample_new_control_vars(cls, process_states: np.ndarray, prev_control_states: np.ndarray, dt: float) -> np.ndarray:
        """ Sample new control variables for next time step. """

    @abc.abstractclassmethod
    def sample_init_hidden_vars(cls, n: int) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def sample_init_static_vars(cls, n: int) -> np.ndarray:
        static_states = np.empty((n, len(cls.StaticVariables)), dtype=int)
        return static_states

    @classmethod
    def simulate(cls, n: int, iters: int, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = np.empty((n, iters + 1, len(cls.ProcessVariables)), dtype=np.float32)
        X[:, 0] = cls.sample_init_process_vars(n)

        Z = np.empty((n, iters + 1, len(cls.HiddenVariables)), dtype=np.float32)
        Z[:, 0] = cls.sample_init_hidden_vars(n)

        U = np.empty((n, iters, len(cls.ControlVariables)), dtype=np.float32)

        S = np.empty((n, iters + 1, len(cls.StaticVariables)), dtype=int)
        S[:, 0] = cls.sample_init_static_vars(n)

        for i in range(iters):
            if i == 0:
                U[:, i] = cls.sample_new_control_vars(
                    X[:, i],
                    cls.sample_init_control_vars(n),
                    dt = dt,
                )
            else:
                U[:, i] = cls.sample_new_control_vars(
                    X[:, i],
                    U[:, i - 1],
                    dt = dt
                )

            X[:, i + 1], Z[:, i + 1], S[:, i + 1] = cls.forward(
                X[:, i],
                U[:, i],
                Z[:, i],
                S[:, i],
                dt,
            )

        return X, Z, U, S

    @abc.abstractclassmethod
    def forward(cls, X: np.ndarray, U: np.ndarray, Z: np.ndarray, S: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Does a single forward pass. All input matrices should have shape n x d, where n is the number of concurrent simulations
        and d is the number of dimensions of that particular variable. """

    @classmethod
    def forward_multiple(cls, X: np.ndarray, U: np.ndarray, Z: np.ndarray, S: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Same as forward, but does multiple forward iterations. X, Z, and U have the same shape as in forward, but U has shape
        n x iterations x d. """
        n, iters = U.shape[:2]

        tmp = X
        X = np.empty((n, iters + 1, len(cls.ProcessVariables)), dtype=np.float32)
        X[:, 0] = tmp

        tmp = Z
        Z = np.empty((n, iters + 1, len(cls.HiddenVariables)), dtype=np.float32)
        Z[:, 0] = tmp

        tmp = S
        S = np.empty((n, iters + 1, len(cls.StaticVariables)), dtype=int)
        S[:, 0] = tmp

        for i in range(iters):
            X[:, i + 1], Z[:, i + 1], S[:, i + 1] = cls.forward(X[:, i], U[:, i], Z[:, i], S[:, i], dt)

        return X, Z, S

from .ball import Ball
