import abc

import torch
import torch.nn as nn

from . import FzConfig, _FloatzoneModule


class _DynamicsNetwork(_FloatzoneModule, abc.ABC):

    def __call__(self, Uf: torch.FloatTensor, Sf: torch.FloatTensor, z1: torch.FloatTensor) -> torch.FloatTensor:
        return super().__call__(Uf, Sf, z1)

class FullyConnected(_DynamicsNetwork):

    def __init__(self, config: FzConfig):
        super().__init__(config)

        in_size = self.config.Dz + config.F * (self.config.Du + self.config.Ds)

        self.layers = nn.Sequential(*self.build_fully_connected(in_size, config.F * self.config.Dx))

    def forward(self, Uf: torch.FloatTensor, Sf: torch.FloatTensor, z1: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(Uf, Sf, z1)
        return self.layers(x).view(-1, self.config.F, self.config.Dx)

class ResNext(_DynamicsNetwork):

    def __init__(self, config: FzConfig):
        super().__init__(config)

        in_size = self.config.Dz + config.F * (self.config.Du + self.config.Ds)

        self.layers = nn.Sequential(*self.build_resnext(in_size, config.F * self.config.Dx))

    def forward(self, Uf: torch.FloatTensor, Sf: torch.FloatTensor, z1: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(Uf, Sf, z1)
        return self.layers(x).view(-1, self.config.F, self.config.Dx)
