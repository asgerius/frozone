import abc

import torch
import torch.nn as nn

from . import FzConfig, _FloatzoneModule


class EncoderH(_FloatzoneModule, abc.ABC):

    def __call__(self, Xh: torch.FloatTensor, Uh: torch.FloatTensor, Sh: torch.FloatTensor) -> torch.FloatTensor:
        return super().__call__(Xh, Uh, Sh)

class FullyConnected(EncoderH):

    def __init__(self, config: FzConfig):
        super().__init__(config)

        in_size = config.H * (config.dx + config.du + config.ds)

        self.layers = nn.Sequential(*self.build_fully_connected(in_size, self.config.dz))

    def forward(self, Xh: torch.FloatTensor, Uh: torch.FloatTensor, Sh: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(Xh, Uh, Sh)
        return self.layers(x)
