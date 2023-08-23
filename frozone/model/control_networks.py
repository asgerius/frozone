import abc

import torch
import torch.nn as nn

from . import FzConfig, _FloatzoneModule


class _ControlNetwork(_FloatzoneModule, abc.ABC):

    def __call__(self, z1: torch.FloatTensor, z2: torch.FloatTensor) -> torch.FloatTensor:
        return super().__call__(z1, z2)

class FullyConnected(_ControlNetwork):

    def __init__(self, config: FzConfig):
        super().__init__(config)

        in_size = 2 * self.config.Dz

        self.layers = nn.Sequential(*self.build_linear_layers(in_size, self.config.Du))

    def forward(self, z1: torch.FloatTensor, z2: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(z1, z2)
        return self.layers(x)
