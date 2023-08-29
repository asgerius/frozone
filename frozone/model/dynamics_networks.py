import abc

import torch
import torch.nn as nn

from . import FzConfig, _FloatzoneModule


class _DynamicsNetwork(_FloatzoneModule, abc.ABC):

    def __call__(self, u: torch.FloatTensor, s: torch.FloatTensor, z1: torch.FloatTensor) -> torch.FloatTensor:
        return super().__call__(u, s, z1)

class FullyConnected(_DynamicsNetwork):

    def __init__(self, config: FzConfig):
        super().__init__(config)

        in_size = self.config.Dz + self.config.Du + 2 * self.config.Ds

        self.layers = nn.Sequential(*self.build_fully_connected(in_size, self.config.Dx))

    def forward(self, Xh: torch.FloatTensor, Uh: torch.FloatTensor, Sh: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(Xh, Uh, Sh)
        return self.layers(x)

class ResNext(_DynamicsNetwork):

    def __init__(self, config: FzConfig):
        super().__init__(config)

        in_size = self.config.Dz + self.config.Du + 2 * self.config.Ds

        self.layers = nn.Sequential(*self.build_resnext(in_size, self.config.Dx))

    def forward(self, Xh: torch.FloatTensor, Uh: torch.FloatTensor, Sh: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(Xh, Uh, Sh)
        return self.layers(x)
