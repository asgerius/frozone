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
    prediction_window:  float = 10

    # Hyperparameter stuff
    batches:            int = 100000
    batch_size:         int = 200

    @property
    def history_window_steps(self) -> int:
        return int(self.history_window / self.dt)

    @property
    def predict_window_steps(self) -> int:
        return int(self.prediction_window / self.dt)

@dataclass
class TrainResults(DataStorage):

    checkpoints:    list[int]
    train_loss:     list[float]
    eval_loss:      list[float]

    @classmethod
    def empty(cls) -> TrainResults:
        return TrainResults(list(), list(), list())
