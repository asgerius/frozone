from dataclasses import dataclass

from pelutils import DataStorage


@dataclass
class TrainConfig(DataStorage):

    # Environment should match name of a class in simulations
    env:                str = "Ball"

    # Windows are given in seconds
    dt:                 float = 0.1
    history_window:     float = 2
    prediction_window:  float = 5

    # Hyperparameter stuff
    batches:            int = 10000
    batch_size:         int = 100

    @property
    def history_window_steps(self) -> int:
        return int(self.history_window / self.dt)

    @property
    def predict_window_steps(self) -> int:
        return int(self.prediction_window / self.dt)
