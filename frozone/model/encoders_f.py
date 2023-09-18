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

        in_size = config.F * (config.ds + input_d)
        out_size = config.F * config.dz

        self.layers = nn.Sequential(*self.build_fully_connected(in_size, out_size))

    def forward(self, Sf: torch.FloatTensor, Xf_or_Uf: torch.FloatTensor) -> torch.FloatTensor:
        x = self.concat_to_feature_vec(Sf, Xf_or_Uf)
        return self.layers(x).view(-1, self.config.F, self.config.dz)

class Transformer(EncoderF):

    def __init__(self, config: FzConfig, input_d: int):
        super().__init__(config, input_d)

        positional_encoding = self.build_positional_encoding(config.F)
        self.embedding = nn.Linear(config.ds + input_d, config.dz)
        self.transformer = BaseTransformer(config, config.dz, positional_encoding)

    def forward(self, Sf: torch.FloatTensor, Xf_or_Uf: torch.FloatTensor) -> torch.FloatTensor:
        x = torch.concat((Sf, Xf_or_Uf), dim=2)
        x_e = self.embedding(x)
        x_t = self.transformer(x_e)
        return x_t

class GatedTransformer(EncoderF):
    """ https://arxiv.org/pdf/2103.14438.pdf """

    def __init__(self, config: FzConfig, input_d: int):
        super().__init__(config, input_d)

        positional_encoding = self.build_positional_encoding(config.F)
        channel_d = config.ds + input_d
        self.time_embedding = nn.Linear(channel_d, config.dz)
        self.time_transformer = BaseTransformer(config, config.dz, positional_encoding)

        self.channel_embedding = nn.Linear(config.F, config.dz)
        self.channel_transformer = BaseTransformer(config, config.dz)

        self.gate = nn.Linear(config.dz * (config.F + channel_d), 2)

        self.out_map = nn.Linear(config.dz * (config.F + channel_d), config.dz)

    def forward(self, Sf: torch.FloatTensor, Xf_or_Uf: torch.FloatTensor) -> torch.FloatTensor:
        # Time step-wise tower
        x_s = torch.concat((Sf, Xf_or_Uf), dim=2)
        x_s_e = self.time_embedding(x_s)
        x_s_t = self.time_transformer(x_s_e)

        # Channel-wise tower
        x_c = x_s.permute(0, 2, 1)
        x_c_e = self.channel_embedding(x_c)
        x_c_t = self.channel_transformer(x_c_e)

        h = self.gate(torch.concat((x_s_t, x_c_t), dim=1).flatten(1, -1))
        g = torch.softmax(h, dim=-1)
        y = torch.concat((
            (g[:, 0] * x_s_t.permute(1, 2, 0)).permute(2, 0, 1),
            (g[:, 1] * x_c_t.permute(1, 2, 0)).permute(2, 0, 1),
        ), dim=1).flatten(1, -1)

        return self.out_map(y)
