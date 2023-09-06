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

    def __init__(self, config: FzConfig):
        super().__init__(config)

        self.history_encoder = getattr(history_encoders,  self.config.history_encoder_name)(config)
        self.z1_post_encoder_layers = nn.Sequential(
            config.get_activation_fn(),
            nn.BatchNorm1d(config.Dz),
            nn.Dropout(config.dropout_p),
        )
        if config.has_control_and_target:
            self.target_encoder = getattr(target_encoders,   self.config.target_encoder_name)(config)
            self.z2_post_encoder_layers = nn.Sequential(
                config.get_activation_fn(),
                nn.BatchNorm1d(config.Dz),
                nn.Dropout(config.dropout_p),
            )

        if config.has_dynamics_network:
            self.dynamics_network = getattr(dynamics_networks, self.config.dynamics_network_name)(config)
        if config.has_control_and_target:
            self.control_network  = getattr(control_networks,  self.config.control_network_name)(config)


    def __call__(
        self,
        Xh: torch.FloatTensor,
        Uh: torch.FloatTensor,
        Sh: torch.FloatTensor,
        Xf: Optional[torch.FloatTensor] = None,
        Sf: Optional[torch.FloatTensor] = None,
        u: Optional[torch.FloatTensor] = None,
        s: Optional[torch.FloatTensor] = None,
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        """ This method implementation does not change anything, but it adds type support for forward calls. """
        return super().__call__(Xh, Uh, Sh, Xf, Sf, u, s)

    def forward(
        self,
        Xh: torch.FloatTensor,
        Uh: torch.FloatTensor,
        Sh: torch.FloatTensor,
        Xf: Optional[torch.FloatTensor],
        Sf: Optional[torch.FloatTensor],
        u: Optional[torch.FloatTensor],
        s: Optional[torch.FloatTensor],
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        z1 = self.history_encoder(Xh, Uh, Sh)
        z1 = self.z1_post_encoder_layers(z1)

        if self.config.has_control_and_target:
            z2 = self.target_encoder(Xf, Sf)
            z2 = self.z2_post_encoder_layers(z2)
            u_pred = self.control_network(z1, z2)
        else:
            u_pred = None

        if self.config.has_dynamics_network and u is not None:
            assert s is not None
            x_pred = self.dynamics_network(u, s, z1)
        else:
            x_pred = None

        return z1, x_pred, u_pred
