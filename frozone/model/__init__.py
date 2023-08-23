from __future__ import annotations

import abc
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from pelutils import DataStorage

from frozone import device


@dataclass
class FzConfig(DataStorage):

    Dx: int  # Number of process variables
    Du: int  # Number of control variables
    Ds: int  # Number of static variables after binary encoding
    Dz: int  # Number of latent variables

    H: int  # Number of history steps
    F: int  # Number of prediction steps

    history_encoder_name:  str
    target_encoder_name:   str
    dynamics_network_name: str
    control_network_name:  str

    fc_hidden_num:  int
    fc_hidden_size: int

    dropout_p: float = 0
    activation_fn: str = "ReLU"

    def __post_init__(self):
        assert self.Dx > 0
        assert self.Du > 0
        assert self.Ds >= 0
        assert self.Dz > 0
        assert self.H > 0
        assert self.F > 0

        assert self.fc_hidden_num > 0

        assert 0 <= self.dropout_p <= 1

    def get_activation_fn(self) -> nn.Module:
        return getattr(nn, self.activation_fn)()

class _FloatzoneModule(nn.Module, abc.ABC):

    def __init__(self, config: FzConfig):
        super().__init__()

        self.config = config

    @classmethod
    def concat_to_feature_vec(cls, *tensors: torch.FloatTensor):
        """ Concatenates input tensors to feature vectors. Batches in first dimension are respected. """
        batch_size = tensors[0].shape[0]
        assert all(tensor.shape[0] == batch_size for tensor in tensors)
        return torch.concat([tensor.view(batch_size, -1) for tensor in tensors], dim=-1)

    def numel(self) -> int:
        """ Number of model parameters. Further docs here: https://pokemondb.net/pokedex/numel """
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def _state_dict_file_name(cls) -> str:
        return cls.__name__ + ".pt"

    def save(self, path: str):
        torch.save(self.state_dict(), os.path.join(path, self._state_dict_file_name()))
        self.config.save(path, self.__class__.__name__)

    @classmethod
    def load(cls, path: str) -> _FloatzoneModule:
        config = FzConfig.load(path, cls.__name__)
        model = cls(config).to(device)
        model.load_state_dict(torch.load(os.path.join(path, cls._state_dict_file_name()), map_location=device))
        return model

    def build_linear_layers(self, in_size: int, out_size: int) -> list[nn.Module]:
        layer_sizes = [in_size, *[self.config.fc_hidden_size] * self.config.fc_hidden_num, out_size]

        modules = list()
        modules.append(nn.BatchNorm1d(layer_sizes[0]))

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            modules.append(nn.Linear(in_size, out_size))

            is_last = i == len(layer_sizes) - 2
            if not is_last:
                modules += (
                    self.config.get_activation_fn(),
                    nn.BatchNorm1d(out_size),
                    nn.Dropout1d(p=self.config.dropout_p),
                )

        return modules

