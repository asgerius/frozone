from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pelutils import DataStorage


# This global variable is used to indicate to threads whether or not a training is running.
# If it is false, threads should stop within a very short amount of time.
is_doing_training = False

@dataclass
class TrainConfig(DataStorage):

    # Environment should match name of a class in simulations
    env:                str
    data_path:          str
    phase:              Optional[str]

    # Windows are given in seconds
    dt:                 float
    history_window:     float
    prediction_window:  float

    # Training stuff
    batches:            int
    batch_size:         int
    lr:                 float
    num_models:         int
    # Dataset stuff
    max_num_data_files: int
    eval_size:          int

    # Loss weight - 0 for only dynamics and 1 for only control
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
    mean_x: Optional[np.ndarray]
    std_x:  Optional[np.ndarray]
    mean_u: Optional[np.ndarray]
    std_u:  Optional[np.ndarray]

    checkpoints:          list[int]  # Batches before which there was a checkpoint
    train_loss_x:         list[list[float]]  # Models outermost, batches innermost
    train_loss_u:         list[list[float]]
    train_loss:           list[list[float]]
    test_loss_x:          list[list[float]]
    test_loss_u:          list[list[float]]
    test_loss:            list[list[float]]
    test_loss_x_std:      list[list[float]]
    test_loss_u_std:      list[list[float]]
    test_loss_std:        list[list[float]]
    ensemble_test_loss_x: list[float]
    ensemble_test_loss_u: list[float]
    ensemble_test_loss:   list[float]
    lr:                   list[float]

    @classmethod
    def empty(cls, num_models: int) -> TrainResults:
        return TrainResults(
            # Mean and standard deviation
            None, None, None, None,
            # Everything else
            list(), *([list() for _ in range(num_models)] for _ in range(9)),
            list(), list(), list(), list(),
        )
