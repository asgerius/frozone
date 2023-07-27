from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from pelutils import DataStorage

from frozone import device


class Frozone(nn.Module):

    @dataclass
    class Config(DataStorage):
        num_process_var: int
        num_control_var: int

        history_window_steps: int
        predict_window_steps: int

        num_hidden_layers: int
        layer_size: int

        def __post_init__(self):
            assert self.num_process_var > 0
            assert self.num_control_var > 0
            assert self.history_window_steps > 0
            assert self.predict_window_steps > 0
            assert self.num_hidden_layers >= 0
            assert self.layer_size > 0

    def __init__(self, config: Frozone.Config):
        super().__init__()

        self.config = config

        in_size = self.config.num_process_var * (self.config.history_window_steps + self.config.predict_window_steps) + self.config.num_control_var * (self.config.history_window_steps - 1)
        out_size = self.config.num_control_var * self.config.predict_window_steps
        layer_sizes = (in_size, *[self.config.layer_size] * self.config.num_hidden_layers, out_size)

        layers = list()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.BatchNorm1d(in_size))
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        layers.pop()  # Remove final relu

        self.layers = nn.Sequential(*layers)

    def forward(self, history_process: torch.FloatTensor, history_control: torch.FloatTensor, target_process: torch.FloatTensor) -> torch.FloatTensor:

        batch_size = history_process.shape[0]

        x = torch.cat((history_process.view(batch_size, -1), history_control.view(batch_size, -1), target_process.view(batch_size, -1)), dim=-1)
        return self.layers(x).reshape(len(x), self.config.predict_window_steps, self.config.num_control_var)

    def numel(self) -> int:
        """ Number of model parameters. Further docs here: https://pokemondb.net/pokedex/numel """
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def _state_dict_file_name(cls) -> str:
        return cls.__name__ + ".pt"

    def save(self, path: str):
        torch.save(self.state_dict(), os.path.join(path, self._state_dict_file_name()))
        self.config.save(path)

    @classmethod
    def load(cls, path: str) -> Frozone:
        config = cls.Config.load(path, cls.__name__)
        model = cls(config).to(device)
        model.load_state_dict(torch.load(os.path.join(path, cls._state_dict_file_name()), map_location=device))
        return model
