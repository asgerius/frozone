from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from pelutils import DataStorage

from frozone import device
from frozone.model import Frozone


class FFFrozone(Frozone):

    @dataclass
    class Config(Frozone.Config):

        num_hidden_layers: int
        layer_size: int

        def __post_init__(self):
            super().__post_init__()
            assert self.num_hidden_layers >= 0
            assert self.layer_size > 0

    def __init__(self, config: Config):
        super().__init__(config)

        self.latent_model = _FFLatentModel(config)
        self.control_model = _FFControlModel(config)
        self.process_model = _FFProcessModel(config)

    def forward(
        self,
        XH: torch.FloatTensor,
        UH: torch.FloatTensor,
        XF: Optional[torch.FloatTensor] = None,
        UF: Optional[torch.FloatTensor] = None,
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        ZH = self.latent_model(XH, UH)
        pred_XF = None
        pred_UF = None

        if XF is not None:
            pred_UF = self.control_model(ZH, XF)
        if UF is not None:
            pred_XF = self.process_model(ZH, UF)

        return ZH, pred_UF, pred_XF

def _build_ff_layers(in_size: int, out_size: int, config: FFFrozone.Config) -> list[nn.Module]:
        layer_sizes = (in_size, *[config.layer_size] * config.num_hidden_layers, out_size)

        layers = list()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.BatchNorm1d(in_size))
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        layers.pop()  # Remove final relu

        return layers

class _FFLatentModel(Frozone):

    def __init__(self, config: FFFrozone.Config):
        super().__init__(config)

        in_size = self.config.D * self.config.h + self.config.d * (self.config.h - 1)
        self.layers = nn.Sequential(*_build_ff_layers(in_size, self.config.K, self.config))

    def forward(
        self,
        XH: torch.FloatTensor,
        UH: torch.FloatTensor,
    ) -> torch.FloatTensor:

        batch_size = XH.shape[0]

        x = torch.cat((XH.view(batch_size, -1), UH.view(batch_size, -1)), dim=-1)
        return self.layers(x)

class _FFControlModel(Frozone):

    def __init__(self, config: FFFrozone.Config):
        super().__init__(config)

        in_size = self.config.K + self.config.D * self.config.f
        out_size = self.config.d * self.config.f
        self.layers = nn.Sequential(*_build_ff_layers(in_size, out_size, self.config))

    def forward(
        self,
        ZH: torch.FloatTensor,
        XF: torch.FloatTensor,
    ) -> torch.FloatTensor:

        batch_size = ZH.shape[0]

        x = torch.cat((ZH, XF.view(batch_size, -1)), dim=-1)
        return self.layers(x).view(batch_size, self.config.f, self.config.d)

class _FFProcessModel(Frozone):

    def __init__(self, config: FFFrozone.Config):
        super().__init__(config)

        in_size = self.config.K + self.config.d * self.config.f
        out_size = self.config.D * self.config.f
        self.bnorm, self.linear, *layers = self.layers = _build_ff_layers(in_size, out_size, self.config)
        self.layers = nn.Sequential(*layers)

        self.U_size = self.config.d * self.config.f
        self.U_layer = nn.Parameter(torch.zeros(1, self.config.K + self.U_size))

    def forward(
        self,
        ZH: torch.FloatTensor,
        UF: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:

        batch_size = ZH.shape[0]

        if UF is not None:
            UF = UF.view(batch_size, -1)
            x = torch.cat((ZH, UF), dim=-1)
            x = self.layers(self.linear(self.bnorm(x)))
        else:
            x_ZH = torch.cat((ZH, torch.zeros((batch_size, self.U_size)).to(device)), dim=-1)
            x_ZH = self.linear(self.bnorm(x_ZH))

            x_UH = self.linear(self.bnorm(self.U_layer))

            x = self.layers(x_ZH + x_UH)

        return x.view(batch_size, self.config.f, self.config.D)

    def set_UF(self, UF: torch.FloatTensor):
        with torch.no_grad():
            self.U_layer.zero_()
            self.U_layer[0, self.config.K:] = UF.flatten()
