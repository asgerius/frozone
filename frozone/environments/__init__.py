from __future__ import annotations

import abc
import enum

import numpy as np
from tqdm import tqdm


class Environment(abc.ABC):

    X_dtype = np.float32
    U_dtype = np.float32
    S_dtype = np.uint8

    dt: float
    is_simulation: bool

    class XLabels(enum.IntEnum):
        """ Mutable states observable to the controller. """

    class ULabels(enum.IntEnum):
        """ Variables that are set by the controller, generally referred to as u in literature. """

    class SLabels(enum.IntEnum):
        """ Categorical and/or discrete properties. These can change over time, but do so independently of control. """

    class ZLabels(enum.IntEnum):
        """ Variables hidden from the controller. """

    # The number of 0 or 1 needed to represent each value in SLabels
    S_bin_count: tuple[int] = tuple()

    # Variables that go into the system dynamics but are not predicted, as they have no reference values
    no_reference_variables: tuple[XLabels] = tuple()

    @classmethod
    def reference_variables(cls) -> tuple[XLabels]:
        """ Reverse of no_reference_variables """
        return tuple(xlab for xlab in cls.XLabels if xlab not in cls.no_reference_variables)

    def __init_subclass__(cls):
        super().__init_subclass__()
        assert len(cls.SLabels) == len(cls.S_bin_count)
        # assert all(x >= 1 for x in cls.S_bin_count)

    @abc.abstractclassmethod
    def sample_init_process_vars(cls, n: int) -> np.ndarray:
        process_states = np.empty((n, len(cls.XLabels)), dtype=cls.X_dtype)
        return process_states

    @abc.abstractclassmethod
    def sample_init_control_vars(cls, n: int) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def sample_new_control_vars(cls, process_states: np.ndarray, prev_control_states: np.ndarray) -> np.ndarray:
        """ Sample new control variables for next time step. """

    @abc.abstractclassmethod
    def sample_init_hidden_vars(cls, n: int) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def sample_init_static_vars(cls, n: int) -> np.ndarray:
        static_states = np.empty((n, sum(cls.S_bin_count)), dtype=cls.S_dtype)
        return static_states

    @classmethod
    def simulate(cls, n: int, iters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not cls.is_simulation:
            raise NotImplementedError("Environment %s cannot be simulated" % cls.__name__)

        X = np.empty((n, iters + 1, len(cls.XLabels)), dtype=cls.X_dtype)
        X[:, 0] = cls.sample_init_process_vars(n)

        Z = np.empty((n, iters + 1, len(cls.ZLabels)), dtype=cls.X_dtype)
        Z[:, 0] = cls.sample_init_hidden_vars(n)

        U = np.empty((n, iters + 1, len(cls.ULabels)), dtype=cls.U_dtype)

        S = np.empty((n, iters + 1, sum(cls.S_bin_count)), dtype=cls.S_dtype)
        S[:, 0] = cls.sample_init_static_vars(n)

        for i in tqdm(range(iters)):
            if i == 0:
                U[:, i] = cls.sample_new_control_vars(
                    X[:, i],
                    cls.sample_init_control_vars(n),
                )
            else:
                U[:, i] = cls.sample_new_control_vars(X[:, i], U[:, i - 1])

            X[:, i + 1], S[:, i + 1], Z[:, i + 1] = cls.forward(
                X[:, i], U[:, i],
                S[:, i], Z[:, i],
            )

        U[:, -1] = cls.sample_new_control_vars(X[:, -1], U[:, -2])

        return X, U, S, Z

    @abc.abstractclassmethod
    def forward(cls, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Does a single forward pass. All input matrices should have shape n x D, where n is the number of concurrent simulations
        and D is the dimensionality of that particular variable. Return X, S, Z. """

    @classmethod
    def forward_multiple(cls, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Same as forward, but does multiple forward iterations. X, Z, and U have the same shape as in forward, but U has shape
        n x iterations x d. Returns X, S, Z. """
        if not cls.is_simulation:
            raise NotImplementedError("Environment %s cannot be simulated" % cls.__name__)

        n, iters = U.shape[:2]

        tmp = X
        X = np.empty((n, iters + 1, len(cls.XLabels)), dtype=cls.X_dtype)
        X[:, 0] = tmp

        tmp = Z
        Z = np.empty((n, iters + 1, len(cls.ZLabels)), dtype=cls.U_dtype)
        Z[:, 0] = tmp

        tmp = S
        S = np.empty((n, iters + 1, len(cls.SLabels)), dtype=cls.S_dtype)
        S[:, 0] = tmp

        for i in range(iters):
            X[:, i + 1], S[:, i + 1], Z[:, i + 1] = cls.forward(X[:, i], U[:, i], S[:, i], Z[:, i])

        return X, S, Z

from .ball import Ball
from .floatzone import FloatZone
from .steuermann import Steuermann
