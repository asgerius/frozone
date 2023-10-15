from __future__ import annotations

import abc
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
    H_interp: int
    F_interp: int

    encoder_name: str
    decoder_name: str

    # Fully-connected parameters
    fc_layer_num:  int
    fc_layer_size: int

    # Transformer parameters
    t_layer_num: int
    t_nhead: int
    t_d_feedforward: int

    alpha: float
    dropout: float
    activation_fn: str

    def __post_init__(self):
        assert self.dx > 0
        assert self.du > 0
        assert self.ds >= 0
        assert self.dz > 0
        assert self.H > 0
        assert self.F > 0
        assert 0 < self.H_interp <= self.H
        assert 0 < self.F_interp <= self.F

        assert self.fc_layer_num > 0
        assert self.fc_layer_size > 0

        assert 0 <= self.dropout < 1
        assert 0 <= self.alpha <= 1

    def get_activation_fn(self) -> nn.Module:
        return getattr(nn, self.activation_fn)()

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

        return PE.to(device)

class BaseTransformer(_FloatzoneModule):

    def __init__(self, config: FzConfig, input_d: int, positional_encoding: Optional[torch.FloatTensor] = None, *, embedding=True):
        super().__init__(config)

        assert config.dz % config.t_nhead == 0, "dz must be divisble by the number of attention heads, t_nhead"

        self.has_embedding = embedding
        if embedding:
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

        if self.has_embedding:
            x = self.embedding(x)

        if self.positional_encoding is not None:
            x += self.positional_encoding

        return self.encoder(x)
