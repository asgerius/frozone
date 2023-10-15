import abc

import torch
import torch.nn as nn

from . import BaseTransformer, FzConfig, _FloatzoneModule


class EncoderF(_FloatzoneModule, abc.ABC):

    def __init__(self, config: FzConfig, input_d: int):
        super().__init__(config)
        self.input_d = input_d

    def __call__(self, Sf: torch.FloatTensor, Xf_or_Uf: torch.FloatTensor) -> torch.FloatTensor:
        return super().__call__(Sf, Xf_or_Uf)

class FullyConnected(EncoderF):

    def __init__(self, config: FzConfig, input_d: int):
        super().__init__(config, input_d)

        in_size = config.F_interp * (config.ds + input_d)
        out_size = config.F_interp * config.dz

        self.layers = nn.Sequential(*self.build_fully_connected(in_size, out_size))

    def forward(self, Sf: torch.FloatTensor, Xf_or_Uf: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(Sf, Xf_or_Uf)
        return self.layers(x).view(-1, self.config.F_interp, self.config.dz)

class Transformer(EncoderF):

    def __init__(self, config: FzConfig, input_d: int):
        super().__init__(config, input_d)

        positional_encoding = self.build_positional_encoding(config.F_interp)
        self.transformer = BaseTransformer(config, config.ds + input_d, positional_encoding)

    def forward(self, Sf: torch.FloatTensor, Xf_or_Uf: torch.FloatTensor) -> torch.FloatTensor:
        x = torch.concat((Sf, Xf_or_Uf), dim=2)
        x_t = self.transformer(x)
        return x_t
