from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

import frozone.model.encoders_h as encoders_h
import frozone.model.encoders_f as encoders_f
import frozone.model.decoders as decoders
from frozone import device
from frozone.model import FzConfig, _FloatzoneModule

class FzNetwork(_FloatzoneModule):

    def __init__(self, config: FzConfig, *, for_control: bool):
        super().__init__(config)

        self.for_control = for_control
        self.for_dynamics = not for_control

        # Question: Return last layer, or use linear decoder to go from H x dz to dz?
        self.Eh: encoders_h.EncoderH = getattr(encoders_h, config.encoder_name)(config)

        encoder_name = "Transformer" if config.encoder_name == "GatedTransformer" else config.encoder_name
        self.Ef: encoders_f.EncoderF = getattr(encoders_f, encoder_name)(config, config.du if self.for_dynamics else config.dx)
        self.D:  decoders.Decoder    = getattr(decoders, self.config.decoder_name)(config, is_x=self.for_dynamics)

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
    ) -> torch.FloatTensor:
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
    ) -> torch.FloatTensor:
        h = Xh if self.for_dynamics else Uh
        f = Uf if self.for_dynamics else Xf

        h_smooth = h.permute(0, 2, 1) @ self.predict_ref_weights

        zh = self.Eh(Xh, Uh, Sh)
        Zf = self.Ef(Sf, Xf_or_Uf=f)

        pred = self.D(zh, Zf) + h_smooth.unsqueeze(dim=1)

        return pred

    @classmethod
    def _state_dict_file_name(cls, num: int, for_control: bool) -> str:
        return cls.__name__ + ".%s.%i.pt" % ("X" if for_control else "U", num)

    def save(self, path: str, num: int):
        if num == 0 and self.for_dynamics:
            self.config.save(path)
        path = os.path.join(path, "models")
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, self._state_dict_file_name(num, self.for_control)))

    @classmethod
    def load(cls, path: str, num: int) -> tuple[_FloatzoneModule, _FloatzoneModule]:
        config = FzConfig.load(path)

        dynamics_model = cls(config, for_control=False).to(device)
        control_model = cls(config, for_control=True).to(device)

        path = os.path.join(path, "models")

        dynamics_model.load_state_dict(torch.load(os.path.join(path, cls._state_dict_file_name(num, False)), map_location=device))
        control_model.load_state_dict(torch.load(os.path.join(path, cls._state_dict_file_name(num, True)), map_location=device))

        return dynamics_model, control_model
