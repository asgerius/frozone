from dataclasses import dataclass
from typing import Type

from pelutils import DataStorage

from frozone.environments import Environment


@dataclass
class ForwardConfig(DataStorage):
    num_samples: int
    num_sequences: int

@dataclass
class SimulationConfig(DataStorage):
    num_samples: int
    simulation_length: float  # In seconds
    control_every: float  # In seconds
    opt_steps: int
    step_size: float

    def __post_init__(self):
        assert self.num_samples > 0
        assert self.simulation_length > 0
        assert self.step_size > 0
        assert self.opt_steps >= 0

    def simulation_steps(self, env: Type[Environment]) -> int:
        return int(self.simulation_length / env.dt)

    def control_every_steps(self, env: Type[Environment]) -> int:
        return max(int(self.control_every / env.dt), 1)
