from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from pprint import pformat
from typing import Optional

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

    # Fully-connected parameters
    fc_layer_num:  int
    fc_layer_size: int

    # Transformer parameters
    t_layer_num: int
    t_nhead: int
    t_d_feedforward: int

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

    def __str__(self) -> str:
        return pformat(vars(self))

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

    def build_positional_encoding(self, seq_length: int) -> torch.FloatTensor:
        PE = torch.empty(seq_length, self.config.dz)
        pos = torch.arange(seq_length)
        index_sin = 2 * torch.arange(self.config.dz // 2 + self.config.dz % 2)
        index_cos = 1 + 2 * torch.arange(self.config.dz // 2)
        PE[:, index_sin] = torch.sin(torch.outer(pos, 1 / 10000 ** (index_sin / self.config.dz)))
        PE[:, index_cos] = torch.cos(torch.outer(pos, 1 / 10000 ** ((index_cos - 1) / self.config.dz)))

        return nn.Parameter(PE, requires_grad=False)

class BaseTransformer(_FloatzoneModule):

    def __init__(self, config: FzConfig, input_d: int, positional_encoding: Optional[nn.Parameter] = None):
        super().__init__(config)

        assert config.dz % config.t_nhead == 0, "dz must be divisble by the number of attention heads, t_nhead"

        self.embedding = nn.Linear(input_d, config.dz)

        self.positional_encoding = positional_encoding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = config.dz,
            nhead = config.t_nhead,
            dim_feedforward = config.t_d_feedforward,
            dropout = config.dropout,
            activation = config.activation_fn.lower(),
            batch_first = True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers = config.t_layer_num,
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        assert len(x.shape) == 3  # Batch, sequence, feature

        embedding = self.embedding(x)

        if self.positional_encoding is not None:
            embedding += self.positional_encoding

        return self.encoder(embedding)
