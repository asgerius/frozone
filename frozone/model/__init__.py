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
    F: int  # Number of target steps

    history_encoder_name:  str
    target_encoder_name:   str
    dynamics_network_name: str
    control_network_name:  str

    # 0 for both dynamics and control, 1 for dynamics only, and 2 for control only
    mode: int

    fc_layer_num:  int
    fc_layer_size: int

    resnext_cardinality: int

    dropout_p: float
    activation_fn: str = "ReLU"

    def __post_init__(self):
        assert self.Dx > 0
        assert self.Du > 0
        assert self.Ds >= 0
        assert self.Dz > 0
        assert self.H > 0
        assert self.F > 0

        assert self.fc_layer_num > 0
        assert self.fc_layer_size > 0

        assert self.resnext_cardinality > 0

        assert 0 <= self.dropout_p <= 1

    def get_activation_fn(self) -> nn.Module:
        return getattr(nn, self.activation_fn)()

    @property
    def has_dynamics_network(self) -> bool:
        return self.mode in { 0, 1 }

    @property
    def has_control_and_target(self) -> bool:
        return self.mode in { 0, 2 }

class ResnextBlock(nn.Module):
    """ ResNeXt block as proposed in https://arxiv.org/pdf/1611.05431.pdf.
    Each residual block uses the PreResNet architecture from https://arxiv.org/pdf/1603.05027.pdf.
    The ordering of batch normalization and activation function is changed though. See
    https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout. """

    def __init__(self, size: int, config: FzConfig):
        super().__init__()

        self.config = config

        for i in range(config.resnext_cardinality):
            setattr(self, f"resnext_{i}", nn.Sequential(
                # config.get_activation_fn(),
                # nn.BatchNorm1d(size),
                nn.Linear(size, size),
                config.get_activation_fn(),
                nn.BatchNorm1d(size),
                nn.Linear(size, size),
            ))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        Fx = torch.zeros_like(x)
        for i in range(self.config.resnext_cardinality):
            Fx += getattr(self, f"resnext_{i}")(x)

        return x + Fx

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
                    nn.Dropout(p=self.config.dropout_p),
                )

        return modules

    def build_resnext(self, in_size: int, out_size: int) -> list[nn.Module]:
        """ Same as a build_fully_connected, but with aggregated residual blocks as described in
        https://arxiv.org/pdf/1512.03385.pdf (ResNet) and https://arxiv.org/pdf/1611.05431.pdf (ResNeXt).

        For simplicity, this reuses the fc_layer_num and fc_layer_size config parameters for the blocks.
        A linear layer is added at each end to reshape the data into the appropriate shapes. """

        modules = list()

        def add_intermediate_ops():
            nonlocal modules
            modules += (
                self.config.get_activation_fn(),
                nn.BatchNorm1d(self.config.fc_layer_size),
                nn.Dropout(p=self.config.dropout_p),
            )

        modules.append(nn.Linear(in_size, self.config.fc_layer_size))
        add_intermediate_ops()

        for i in range(self.config.fc_layer_num):
            modules.append(ResnextBlock(self.config.fc_layer_size, self.config))
            add_intermediate_ops()

        modules.append(nn.Linear(self.config.fc_layer_size, out_size))

        return modules
