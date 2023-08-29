import abc

import torch
import torch.nn as nn

from . import FzConfig, _FloatzoneModule


class _HistoryEncoder(_FloatzoneModule, abc.ABC):

    def __call__(self, Xh: torch.FloatTensor, Uh: torch.FloatTensor, Sh: torch.FloatTensor) -> torch.FloatTensor:
        return super().__call__(Xh, Uh, Sh)

class FullyConnected(_HistoryEncoder):

    def __init__(self, config: FzConfig):
        super().__init__(config)

        in_size = (self.config.Dx + self.config.Du + self.config.Ds) * self.config.H

        self.layers = nn.Sequential(*self.build_fully_connected(in_size, self.config.Dz))

    def forward(self, Xh: torch.FloatTensor, Uh: torch.FloatTensor, Sh: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(Xh, Uh, Sh)
        return self.layers(x)

class ResNext(_HistoryEncoder):

    def __init__(self, config: FzConfig):
        super().__init__(config)

        in_size = (self.config.Dx + self.config.Du + self.config.Ds) * self.config.H

        self.layers = nn.Sequential(*self.build_resnext(in_size, self.config.Dz))

    def forward(self, Xh: torch.FloatTensor, Uh: torch.FloatTensor, Sh: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(Xh, Uh, Sh)
        return self.layers(x)
