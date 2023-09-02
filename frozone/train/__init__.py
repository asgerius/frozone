from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pelutils import DataStorage


@dataclass
class TrainConfig(DataStorage):

    # Environment should match name of a class in simulations
    env:                str = "FloatZone"
    data_path:          str = "data-floatzone"

    # Windows are given in seconds
    dt:                 float = 6
    history_window:     float = 42
    prediction_window:  float = 30

    # Training stuff
    batches:            int = 15000
    batch_size:         int = 200
    lr:                 float = 2e-5
    # How many data points to include in each evaluation
    eval_size:          int = 4000

    # Loss weight - 0 for only process and 1 for only control
    alpha:              float = 0.5

    # Data augmentation
    # Standard deviation of the generated noise
    # This is given as a multiplier to the feature-wise standard deviation in the data
    epsilon:            float = 0.05

    @property
    def H(self) -> int:
        return int(self.history_window / self.dt)

    @property
    def F(self) -> int:
        return int(self.prediction_window / self.dt)

    @property
    def num_eval_batches(self) -> int:
        return math.ceil(self.eval_size / self.batch_size)

@dataclass
class TrainResults(DataStorage):

    # Feature-wise means and standard deviations of training data
    mean_x:             Optional[np.ndarray]
    std_x:              Optional[np.ndarray]
    mean_u:             Optional[np.ndarray]
    std_u:              Optional[np.ndarray]

    checkpoints:        list[int]
    train_loss_x:       list[float]
    train_loss_u:       list[float]
    train_loss:         list[float]
    test_loss_x:        list[float]
    test_loss_u:        list[float]
    test_loss:          list[float]
    test_loss_x_std:    list[float]
    test_loss_u_std:    list[float]
    test_loss_std:      list[float]
    lr:                 list[float]

    @classmethod
    def empty(cls) -> TrainResults:
        return TrainResults(
            # Mean and standard deviation
            None, None, None, None,
            # Everything else
            list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(),
        )
