from dataclasses import dataclass

from pelutils import DataStorage


@dataclass
class ForwardConfig(DataStorage):
    num_samples: int
    num_sequences: int
