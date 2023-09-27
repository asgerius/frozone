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

    def simulation_steps(self, env: Type[Environment]) -> int:
        return self.simulation_length // env.dt
