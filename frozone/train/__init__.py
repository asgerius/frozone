from __future__ import annotations

import math
from dataclasses import dataclass

from pelutils import DataStorage


@dataclass
class TrainConfig(DataStorage):

    # Environment should match name of a class in simulations
    env:                str = "Ball"

    # Windows are given in seconds
    dt:                 float = 0.1
    history_window:     float = 1
    prediction_window:  float = 5

    # Training stuff
    batches:            int = 50000
    batch_size:         int = 500
    # How many data points to include in each evaluation
    eval_size:          int = 5000

    # Loss weight - 0 for only process and 1 for only control
    alpha:              float = 0.5

    # Data augmentation
    # Standard deviation of the generated noise
    # This is given as a multiplier to the feature-wise standard deviation in the data
    epsilon:            float = 0.05

    @property
    def history_steps(self) -> int:
        return int(self.history_window / self.dt)

    @property
    def predict_steps(self) -> int:
        return int(self.prediction_window / self.dt)

    @property
    def num_eval_batches(self) -> int:
        return math.ceil(self.eval_size / self.batch_size)

@dataclass
class TrainResults(DataStorage):

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

    @classmethod
    def empty(cls) -> TrainResults:
        return TrainResults(list(), list(), list(), list(), list(), list(), list(), list(), list(), list())
