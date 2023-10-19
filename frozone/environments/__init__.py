from __future__ import annotations

import abc
import enum
import sys
from typing import Optional

import numpy as np
from tqdm import tqdm

import frozone.train


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
    reference_variables: tuple[XLabels]

    # Control values that are predefined and so are not predicted by the networ
    predefined_control: tuple[ULabels] = tuple()
    predicted_control: tuple[ULabels]

    control_limits: dict[ULabels, tuple[float | None, float | None]] = dict()

    units: dict[tuple[str, enum.IntEnum], str] = dict()

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.reference_variables = tuple(xlab for xlab in cls.XLabels if xlab not in cls.no_reference_variables)
        cls.predicted_control = tuple(ulab for ulab in cls.ULabels if ulab not in cls.predefined_control)

        assert len(cls.SLabels) == len(cls.S_bin_count)

    @classmethod
    def format_label(cls, label: enum.IntEnum):
        label_class = label.__class__.__name__[0]
        label_name = label.name
        unit = cls.units.get((label_class, label))
        if unit is not None:
            return f"{label_class} {label_name} [{unit}]"
        else:
            return f"{label_class} {label_name}"

    @classmethod
    def limit_control(cls, U: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> np.ndarray:
        if mean is None:
            mean = np.zeros(len(cls.ULabels))
            std = np.ones(len(cls.ULabels))
        U = U.copy()
        for ulab, (lower, upper) in cls.control_limits.items():
            if lower is not None:
                U[..., ulab] = np.maximum((lower - mean[ulab]) / std[ulab], U[..., ulab])
            if upper is not None:
                U[..., ulab] = np.minimum((upper - mean[ulab]) / std[ulab], U[..., ulab])
        return U

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
    def simulate(cls, n: int, timesteps: int, with_tqdm=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not cls.is_simulation:
            raise NotImplementedError("Environment %s cannot be simulated" % cls.__name__)

        X = np.empty((n, timesteps, len(cls.XLabels)), dtype=cls.X_dtype)
        X[:, 0] = cls.sample_init_process_vars(n)

        Z = np.empty((n, timesteps, len(cls.ZLabels)), dtype=cls.X_dtype)
        Z[:, 0] = cls.sample_init_hidden_vars(n)

        U = np.empty((n, timesteps, len(cls.ULabels)), dtype=cls.U_dtype)

        S = np.empty((n, timesteps, sum(cls.S_bin_count)), dtype=cls.S_dtype)
        S[:, 0] = cls.sample_init_static_vars(n)

        for i in tqdm(range(timesteps-1), file=sys.stdout, disable=not with_tqdm):
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
    def forward_standardized(
        cls,
        X: np.ndarray,
        U: np.ndarray,
        S: np.ndarray,
        Z: np.ndarray,
        train_results: frozone.train.TrainResults,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Same as forward, but X and U are taken in the standardized domain. The returned X is also standardized. """
        eps = 1e-6
        X = X * (train_results.std_x + eps) + train_results.mean_x
        U = U * (train_results.std_u + eps) + train_results.mean_u

        X, S, Z = cls.forward(X, U, S, Z)

        X = (X - train_results.mean_x) / (train_results.std_x + eps)

        return X, S, Z

from .ball import Ball
from .floatzone import FloatZone
from .steuermann import Steuermann
