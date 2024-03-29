from __future__ import annotations

import math
from dataclasses import dataclass
from pprint import pformat
from typing import Optional, Type

import numpy as np
from pelutils import DataStorage

import frozone.environments as environments


# This global variable is used to indicate to threads whether or not a training is running.
# If it is false, threads should stop within a very short amount of time.
is_doing_training = False

def history_only_process(env: Type[environments.Environment]) -> np.ndarray:

    x_target_include = np.ones(len(env.XLabels), dtype=np.float32)
    for xlab in env.no_reference_variables:
        x_target_include[xlab] = 0

    return x_target_include

def history_only_control(env: Type[environments.Environment]) -> np.ndarray:

    u_target_include = np.ones(len(env.ULabels), dtype=np.float32)
    for ulab in env.predefined_control:
        u_target_include[ulab] = 0

    return u_target_include

@dataclass
class TrainConfig(DataStorage):

    # Environment should match name of a class in simulations
    env:                str
    data_path:          str
    phase:              Optional[str]

    # Windows are given in seconds
    history_window:     float
    prediction_window:  float
    history_interp:     float
    prediction_interp:  float

    # Training stuff
    batches:            int
    batch_size:         int
    lr:                 float
    num_models:         int
    # Dataset stuff
    max_num_data_files: int
    eval_size:          int

    # Loss stuff
    loss_fn:            str
    huber_delta:        float

    def __post_init__(self):
        env = self.get_env()
        assert env.dt <= self.history_window
        assert env.dt <= self.prediction_window
        assert 0 <= self.history_interp <= 1
        assert 0 <= self.prediction_interp <= 1

    @property
    def H(self) -> int:
        return int(self.history_window / self.get_env().dt)

    @property
    def F(self) -> int:
        return int(self.prediction_window / self.get_env().dt)

    @property
    def Hi(self) -> int:
        return int(self.history_window * self.history_interp / self.get_env().dt)

    @property
    def Fi(self) -> int:
        return int(self.prediction_window * self.prediction_interp / self.get_env().dt)

    @property
    def num_eval_batches(self) -> int:
        return math.ceil(self.eval_size / self.batch_size)

    def get_env(self) -> Type[environments.Environment]:
        return getattr(environments, self.env)

    def __str__(self) -> str:
        return pformat(vars(self))

@dataclass
class TrainResults(DataStorage):

    # Feature-wise means and standard deviations of training data
    mean_x: Optional[np.ndarray]
    std_x:  Optional[np.ndarray]
    mean_u: Optional[np.ndarray]
    std_u:  Optional[np.ndarray]

    checkpoints:        list[int]  # Batches before which there was a checkpoint
    train_loss_x:       list[list[float]]  # Models outermost, batches innermost
    train_loss_u:       list[list[float]]
    test_loss_x:        list[list[float]]
    test_loss_u:        list[list[float]]
    test_loss_x_std:    list[list[float]]
    test_loss_u_std:    list[list[float]]
    ensemble_loss_x:    list[float]
    ensemble_loss_u:    list[float]
    lr:                 list[float]

    @classmethod
    def empty(cls, num_models: int) -> TrainResults:
        return TrainResults(
            # Mean and standard deviation
            None, None, None, None,
            # Everything else
            list(), *([list() for _ in range(num_models)] for _ in range(6)),
            list(), list(), list(),
        )
