import abc

import torch
import torch.nn as nn

from . import BaseTransformer, FzConfig, _FloatzoneModule


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

class Transformer(EncoderH):

    def __init__(self, config: FzConfig):
        super().__init__(config)

        positional_encoding = self.build_positional_encoding(config.H)
        self.embedding = nn.Linear(config.dx + config.du + config.ds, config.dz)
        self.transformer = BaseTransformer(config, config.dz, positional_encoding)

    def forward(self, Xh: torch.FloatTensor, Uh: torch.FloatTensor, Sh: torch.FloatTensor) -> torch.FloatTensor:
        x = torch.concat((Xh, Uh, Sh), dim=2)
        x_e = self.embedding(x)
        x_t = self.transformer(x_e)
        # Question: Return last layer, or use linear decoder to go from H x dz to dz?
        return x_t[:, -1]
