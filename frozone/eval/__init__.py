from dataclasses import dataclass
from typing import Type

from pelutils import DataStorage

from frozone.environments import Environment
from frozone.train import TrainConfig


@dataclass
class ForwardConfig(DataStorage):
    num_samples: int
    num_sequences: int
    opt_steps: int
    step_size: float

@dataclass
class SimulationConfig(DataStorage):
    num_samples: int
    control_every: float  # In seconds
    opt_steps: int
    step_size: float

    def __post_init__(self):
        assert self.num_samples > 0
        assert self.step_size > 0
        assert self.opt_steps >= 0

    def control_every_steps(self, env: Type[Environment], train_cfg: TrainConfig) -> int:
        return min(max(int(self.control_every / env.dt), 1), train_cfg.F)
