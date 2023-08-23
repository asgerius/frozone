import abc

import torch
import torch.nn as nn

from . import FzConfig, _FloatzoneModule


class _TargetEncoder(_FloatzoneModule, abc.ABC):

    def __call__(self, Xf: torch.FloatTensor, Sf: torch.FloatTensor) -> torch.FloatTensor:
        return super().__call__(Xf, Sf)

class FullyConnected(_TargetEncoder):

    def __init__(self, config: FzConfig):
        super().__init__(config)

        in_size = (self.config.Dx + self.config.Ds) * self.config.F

        self.layers = nn.Sequential(*self.build_linear_layers(in_size, self.config.Dz))

    def forward(self, Xf: torch.FloatTensor, Sf: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(Xf, Sf)
        return self.layers(x)
