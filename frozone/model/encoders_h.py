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
        return x_t[:, -1]

class GatedTransformer(EncoderH):
    """ https://arxiv.org/pdf/2103.14438.pdf """

    def __init__(self, config: FzConfig):
        super().__init__(config)

        positional_encoding = self.build_positional_encoding(config.H)
        channel_d = config.dx + config.du + config.ds
        self.time_embedding = nn.Linear(channel_d, config.dz)
        self.time_transformer = BaseTransformer(config, config.dz, positional_encoding)

        self.channel_embedding = nn.Linear(config.H, config.dz)
        self.channel_transformer = BaseTransformer(config, config.dz)

        self.gate = nn.Linear(config.dz * (config.H + channel_d), 2)

        self.out_map = nn.Linear(config.dz * (config.H + channel_d), config.dz)

    def forward(self, Xh: torch.FloatTensor, Uh: torch.FloatTensor, Sh: torch.FloatTensor) -> torch.FloatTensor:
        # Time step-wise tower
        x_s = torch.concat((Xh, Uh, Sh), dim=2)
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
