import abc

import torch
import torch.nn as nn

from . import FzConfig, _FloatzoneModule, BaseTransformer


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

    @property
    def out_d(self) -> int:
        return self.config.dx if self.is_x else self.config.du

    def __call__(self, zh: torch.FloatTensor, Z: torch.FloatTensor) -> torch.FloatTensor:
        return super().__call__(zh, Z)

class FullyConnected(Decoder):

    def __init__(self, config: FzConfig, *, is_x: bool):
        super().__init__(config, is_x=is_x)

        in_size = (1 + config.F_interp) * self.config.dz
        out_size = config.F_interp * self.out_d

        self.layers = nn.Sequential(*self.build_fully_connected(in_size, out_size))

    def forward(self, zh: torch.FloatTensor, Z: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(zh, Z)
        return self.layers(x).view(-1, self.config.F_interp, self.out_d)

class Transformer(Decoder):

    def __init__(self, config: FzConfig, *, is_x: bool):
        super().__init__(config, is_x=is_x)

        positional_encoding = self.build_positional_encoding(1 + config.F_interp)
        self.transformer = BaseTransformer(config, config.dz, positional_encoding, embedding=False)
        self.decoder_layer = nn.Linear(config.dz, self.out_d)

    def forward(self, zh: torch.FloatTensor, Z: torch.FloatTensor) -> torch.FloatTensor:
        x = torch.concat((zh.unsqueeze(dim=1), Z), dim=1)
        x_t = self.transformer(x)
        out = self.decoder_layer(x_t)
        return out[:, 1:]
