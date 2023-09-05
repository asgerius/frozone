from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pelutils import DataStorage


@dataclass
class TrainConfig(DataStorage):

    # Environment should match name of a class in simulations
    env:                str
    data_path:          str

    # Windows are given in seconds
    dt:                 float
    history_window:     float
    prediction_window:  float

    # Training stuff
    batches:            int
    batch_size:         int
    lr:                 float
    # Dataset stuff
    max_num_data_files: int
    eval_size:          int

    # Loss weight - 0 for only process and 1 for only control
    alpha:              float

    # Data augmentation
    # Standard deviation of the generated noise
    # This is given as a multiplier to the feature-wise standard deviation in the data
    epsilon:            float

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
