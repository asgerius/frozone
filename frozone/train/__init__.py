from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pelutils import DataStorage


@dataclass
class TrainConfig(DataStorage):

    # Environment should match name of a class in simulations
    env:                str = "Ball"
    data_path:          str = "data-ball"

    # Windows are given in seconds
    dt:                 float = 0.1
    history_window:     float = 1
    prediction_window:  float = 3

    # Training stuff
    batches:            int = 2000
    batch_size:         int = 100
    lr:                 float = 1e-5
    # How many data points to include in each evaluation
    eval_size:          int = 5000

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
    mean_x:             np.ndarray
    std_x:              np.ndarray
    mean_u:             np.ndarray
    std_u:              np.ndarray

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
            list(), list(), list(), list(),
            # Everything else
            list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(),
        )
