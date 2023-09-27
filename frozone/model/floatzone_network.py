from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import frozone.model.encoders_h as encoders_h
import frozone.model.encoders_f as encoders_f
import frozone.model.decoders as decoders
from frozone.model import FzConfig, _FloatzoneModule

class FzNetwork(_FloatzoneModule):

    def __init__(self, config: FzConfig):
        super().__init__(config)

        # Question: Return last layer, or use linear decoder to go from H x dz to dz?
        self.Eh: encoders_h.EncoderH = getattr(encoders_h, config.encoder_name)(config)

        encoder_name = "Transformer" if config.encoder_name == "GatedTransformer" else config.encoder_name
        if config.has_dynamics:
            self.Eu: encoders_f.EncoderF = getattr(encoders_f, encoder_name)(config, config.du)
            self.Dx: decoders.Decoder    = getattr(decoders, self.config.decoder_name)(config, is_x=True)
        if config.has_control:
            self.Ex: encoders_f.EncoderF = getattr(encoders_f, encoder_name)(config, config.dx)
            self.Du: decoders.Decoder    = getattr(decoders, self.config.decoder_name)(config, is_x=False)

        predict_ref_weights_data = 0.5 ** torch.arange(config.H).flip(0)
        self.predict_ref_weights = nn.Parameter(predict_ref_weights_data / predict_ref_weights_data.sum(), requires_grad=False)

    def __call__(
        self,
        Xh: torch.FloatTensor,
        Uh: torch.FloatTensor,
        Sh: torch.FloatTensor,
        Sf: torch.FloatTensor, *,
        Xf: Optional[torch.FloatTensor] = None,
        Uf: Optional[torch.FloatTensor] = None,
    ) -> tuple[tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]], Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        """ This method implementation does not change anything, but it adds type support for forward calls. """
        return super().__call__(Xh, Uh, Sh, Sf, Xf=Xf, Uf=Uf)

    def forward(
        self,
        Xh: torch.FloatTensor,
        Uh: torch.FloatTensor,
        Sh: torch.FloatTensor,
        Sf: torch.FloatTensor, *,
        Xf: Optional[torch.FloatTensor],
        Uf: Optional[torch.FloatTensor],
    ) -> tuple[tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]], Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        zh = self.Eh(Xh, Uh, Sh)
        Zu = Zx = Xf_pred = Uf_pred = None

        if self.config.has_dynamics and Uf is not None:
            Zu = self.Eu(Sf, Xf_or_Uf=Uf)
            Xh_smooth = Xh.permute(0, 2, 1) @ self.predict_ref_weights
            Xf_pred = self.Dx(zh, Zu) + Xh_smooth.unsqueeze(dim=1)
            Xf_pred = self.Dx(zh, Zu) + Xh[:, -1].unsqueeze(dim=1)

        if self.config.has_control and Xf is not None:
            Zx = self.Ex(Sf, Xf_or_Uf=Xf)
            Uh_smooth = Uh.permute(0, 2, 1) @ self.predict_ref_weights
            Uf_pred = self.Du(zh, Zx) + Uh_smooth.unsqueeze(dim=1)
            Uf_pred = self.Du(zh, Zx) + Uh[:, -1].unsqueeze(dim=1)

        return (zh, Zu, Zx), Xf_pred, Uf_pred
