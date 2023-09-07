from dataclasses import dataclass

from pelutils import DataStorage


@dataclass
class ForwardConfig(DataStorage):
    num_samples: int = 10
    prediction_window: float = 100
