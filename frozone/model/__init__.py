from __future__ import annotations

import abc
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from pelutils import DataStorage

from frozone import device


@dataclass
class FzConfig(DataStorage):

    dx: int  # Number of process variables
    du: int  # Number of control variables
    ds: int  # Number of static variables after binary encoding
    dz: int  # Number of latent variables

    H: int  # Number of history steps
    F: int  # Number of target steps

    encoder_name: str
    decoder_name: str

    # 0 for both dynamics and control, 1 for dynamics only, and 2 for control only
    mode: int

    fc_layer_num:  int
    fc_layer_size: int

    dropout: float
    activation_fn: str = "ReLU"

    def __post_init__(self):
        assert self.dx > 0
        assert self.du > 0
        assert self.ds >= 0
        assert self.dz > 0
        assert self.H > 0
        assert self.F > 0

        assert self.fc_layer_num > 0
        assert self.fc_layer_size > 0

        assert 0 <= self.dropout < 1

    def get_activation_fn(self) -> nn.Module:
        return getattr(nn, self.activation_fn)()

    @property
    def has_dynamics(self) -> bool:
        return self.mode in { 0, 1 }

    @property
    def has_control(self) -> bool:
        return self.mode in { 0, 2 }

class _FloatzoneModule(nn.Module, abc.ABC):

    def __init__(self, config: FzConfig):
        super().__init__()

        self.config = config

    @classmethod
    def concat_to_feature_vec(cls, *tensors: torch.FloatTensor):
        """ Concatenates input tensors to feature vectors. Batches in first dimension are respected. """
        batch_size = tensors[0].shape[0]
        assert all(tensor.shape[0] == batch_size for tensor in tensors)
        return torch.concat([tensor.reshape(batch_size, -1) for tensor in tensors], dim=-1)

    def numel(self) -> int:
        """ Number of model parameters. Further docs here: https://pokemondb.net/pokedex/numel """
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def _state_dict_file_name(cls, num: int) -> str:
        return cls.__name__ + "_%i.pt" % num

    def save(self, path: str, num: int):
        torch.save(self.state_dict(), os.path.join(path, self._state_dict_file_name(num)))
        self.config.save(path, self.__class__.__name__ + "_%i" % num)

    @classmethod
    def load(cls, path: str, num: int) -> _FloatzoneModule:
        config = FzConfig.load(path, cls.__name__ + "_%i" % num)
        model = cls(config).to(device)
        model.load_state_dict(torch.load(os.path.join(path, cls._state_dict_file_name(num)), map_location=device))
        return model

    def build_fully_connected(self, in_size: int, out_size: int) -> list[nn.Module]:
        """ Builds consecutive modules for a fully connected network. """
        layer_sizes = [in_size, *[self.config.fc_layer_size] * self.config.fc_layer_num, out_size]

        modules = list()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            modules.append(nn.Linear(in_size, out_size))

            is_last = i == len(layer_sizes) - 2
            if not is_last:
                modules += (
                    self.config.get_activation_fn(),
                    nn.BatchNorm1d(out_size),
                    nn.Dropout(p=self.config.dropout),
                )

        return modules
