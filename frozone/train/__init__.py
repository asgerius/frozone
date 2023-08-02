from __future__ import annotations

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

    # Hyperparameter stuff
    batches:            int = 50000
    batch_size:         int = 300

    # Loss weight - 0 for only process and 1 for only control
    alpha:              float = 0.5

    @property
    def history_steps(self) -> int:
        return int(self.history_window / self.dt)

    @property
    def predict_steps(self) -> int:
        return int(self.prediction_window / self.dt)

@dataclass
class TrainResults(DataStorage):

    checkpoints:    list[int]
    train_loss_x:   list[float]
    train_loss_u:   list[float]
    train_loss:     list[float]
    test_loss_x:    list[float]
    test_loss_u:    list[float]
    test_loss:      list[float]

    @classmethod
    def empty(cls) -> TrainResults:
        return TrainResults(list(), list(), list(), list(), list(), list(), list())
