import abc

import torch
import torch.nn as nn

from . import FzConfig, _FloatzoneModule


class Decoder(_FloatzoneModule, abc.ABC):

    def __init__(self, config: FzConfig, *, is_x: bool):
        super().__init__(config)
        self._is_x = is_x

    @property
    def is_x(self) -> bool:
        return self._is_x

    @property
    def is_u(self) -> bool:
        return not self._is_x

    def __call__(self, zh: torch.FloatTensor, Z: torch.FloatTensor) -> torch.FloatTensor:
        return super().__call__(zh, Z)

class FullyConnected(Decoder):

    def __init__(self, config: FzConfig, *, is_x: bool):
        super().__init__(config, is_x=is_x)

        in_size = (1 + config.H) * self.config.dz
        self.out_d = config.dx if self.is_x else config.du
        out_size = config.F * self.out_d

        self.layers = nn.Sequential(*self.build_fully_connected(in_size, out_size))

    def forward(self, zh: torch.FloatTensor, Z: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(zh, Z)
        return self.layers(x).view(-1, self.config.F, self.out_d)
