from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import frozone.model.history_encoders as history_encoders
import frozone.model.target_encoders as target_encoders
import frozone.model.dynamics_networks as dynamics_networks
import frozone.model.control_networks as control_networks
from frozone.model import FzConfig, _FloatzoneModule

class FzNetwork(_FloatzoneModule):

    history_encoder:  history_encoders._HistoryEncoder
    target_encoder:   target_encoders._TargetEncoder
    dynamics_network: dynamics_networks._DynamicsNetwork
    control_network:  control_networks._ControlNetwork

    def __init__(self, config: _FloatzoneModule.Config):
        super().__init__(config)

        self.bnorm_Sh = nn.BatchNorm1d(config.Ds)
        self.bnorm_Sf = nn.BatchNorm1d(config.Ds)

        self.history_encoder  = getattr(history_encoders,  self.config.history_encoder_name)(config)
        self.target_encoder   = getattr(target_encoders,   self.config.target_encoder_name)(config)
        self.dynamics_network = getattr(dynamics_networks, self.config.dynamics_network_name)(config)
        self.control_network  = getattr(control_networks,  self.config.control_network_name)(config)

    def __call__(
        self,
        Xh: torch.FloatTensor,
        Uh: torch.FloatTensor,
        Sh: torch.FloatTensor,
        Xf: torch.FloatTensor,
        Sf: torch.FloatTensor,
        u: Optional[torch.FloatTensor] = None,
        s: Optional[torch.FloatTensor] = None,
    ) -> tuple[Optional[torch.FloatTensor], torch.FloatTensor]:
        """ This method implementation does not change anything, but it adds type support for forward calls. """
        return super().__call__(Xh, Uh, Sh, Xf, Sf, u, s)

    def forward(
        self,
        Xh: torch.FloatTensor,
        Uh: torch.FloatTensor,
        Sh: torch.FloatTensor,
        Xf: torch.FloatTensor,
        Sf: torch.FloatTensor,
        u: Optional[torch.FloatTensor] = None,
        s: Optional[torch.FloatTensor] = None,
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor], torch.FloatTensor]:
        z1 = self.history_encoder(Xh, Uh, self.bnorm_Sh(Sh))
        z2 = self.target_encoder(Xf, self.bnorm_Sf(Sf))

        if u is not None:
            assert s is not None
            x_pred = self.dynamics_network(u, s, z1)
        else:
            x_pred = None
        u_pred = self.control_network(z1, z2)

        return z1, x_pred, u_pred
