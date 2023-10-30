from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

import frozone.model.encoders_h as encoders_h
import frozone.model.encoders_f as encoders_f
import frozone.model.decoders as decoders
from frozone import device
from frozone.model import FzConfig, _FloatzoneModule
from frozone.train import TrainConfig, history_only_control, history_only_process


def interpolate(n: int, fp: np.ndarray | torch.Tensor, train_cfg: Optional[TrainConfig]=None, *, h=False) -> torch.Tensor:

    if h:
        scale = (train_cfg.Hi - (train_cfg.Fi - 1) / (train_cfg.Fi - 1)) / (train_cfg.Hi - 1)
    else:
        scale = 1

    is_numpy = isinstance(fp, np.ndarray)
    if is_numpy:
        fp = torch.from_numpy(fp)

    fp = fp.transpose(-2, -1)
    x = torch.arange(fp.shape[-1], device=fp.device)
    a = fp[..., 1:] - fp[..., :-1]
    b = fp[..., :-1] - a * x[..., :-1]

    xi = scale * torch.linspace(0, fp.shape[-1] - 1, n, device=fp.device)
    if scale == 1:
        xi[-1] -= 1
    index = xi.long()
    yi = a[..., index] * xi + b[..., index]
    if scale == 1:
        yi[..., -1] = fp[..., -1]
    yi = yi.transpose(-2, -1)

    return yi.numpy() if is_numpy else yi

class FzNetwork(_FloatzoneModule):

    def __init__(self, config: FzConfig, train_cfg: TrainConfig, *, for_control: bool):
        super().__init__(config)

        self.for_control = for_control
        self.for_dynamics = not for_control

        self.Eh: encoders_h.EncoderH = getattr(encoders_h, config.encoder_name)(config)

        encoder_name = "Transformer" if config.encoder_name == "GatedTransformer" else config.encoder_name
        self.Ef: encoders_f.EncoderF = getattr(encoders_f, encoder_name)(config, config.du if self.for_dynamics else config.dr)
        self.D:  decoders.Decoder    = getattr(decoders, self.config.decoder_name)(config, is_x=self.for_dynamics)

        self.smoothing_kernel_h = self._build_low_pass_matrix(config.Hi)
        self.smoothing_kernel_f = self._build_low_pass_matrix(config.Fi)

        # Build loss functions
        if train_cfg.loss_fn == "l1":
            loss_fn = torch.nn.L1Loss(reduction="none")
        elif train_cfg.loss_fn == "l2":
            loss_fn = torch.nn.MSELoss(reduction="none")
        elif train_cfg.loss_fn == "huber":
            _loss_fn = torch.nn.HuberLoss(reduction="none", delta=train_cfg.huber_delta)
            loss_fn = lambda target, input: 1 / train_cfg.huber_delta * _loss_fn(target, input)
        self.loss_weight = torch.ones(train_cfg.Fi)
        self.loss_weight = self.loss_weight / self.loss_weight.sum()

        self.target_weights_process = history_only_process(train_cfg.get_env())
        self.target_weights_process = torch.from_numpy(self.target_weights_process) * \
            len(self.target_weights_process) / \
            self.target_weights_process.sum()

        self.target_weights_control = history_only_control(train_cfg.get_env())
        self.target_weights_control = torch.from_numpy(self.target_weights_control) * \
            len(self.target_weights_control) / \
            self.target_weights_control.sum()

        def loss_fn_x(x_target: torch.FloatTensor, x_pred: torch.FloatTensor) -> torch.FloatTensor:
            loss: torch.FloatTensor = loss_fn(x_target, x_pred).mean(dim=0)
            return (loss.T @ self.loss_weight * self.target_weights_process).mean()
        def loss_fn_u(u_target: torch.FloatTensor, u_pred: torch.FloatTensor) -> torch.FloatTensor:
            loss: torch.FloatTensor = loss_fn(u_target, u_pred).mean(dim=0)
            return (loss.T @ self.loss_weight * self.target_weights_control).mean()

        self._loss_fn_x = loss_fn_x
        self._loss_fn_u = loss_fn_u

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

        Xh_smooth = self.smoothen_h(Xh)
        Uh_smooth = self.smoothen_h(Uh)
        h_smooth = Xh_smooth if self.for_dynamics else Uh_smooth

        f = Uf if self.for_dynamics else Xf

        zh = self.Eh(Xh, Uh, Sh)
        Zf = self.Ef(Sf, Xf_or_Uf=f)

        pred = self.D(zh, Zf)
        pred = pred + h_smooth[:, -1].unsqueeze(dim=1)
        pred = self.smoothen_f(pred)

        return pred

    def smoothen_h(self, h):
        return torch.matmul(self.smoothing_kernel_h, h)

    def smoothen_f(self, f):
        return torch.matmul(self.smoothing_kernel_f, f)

    def _build_low_pass_matrix(self, size: int) -> torch.FloatTensor:
        kernel = torch.empty(size, size)
        for i in range(size):
            kernel[i, :i+1] = (1 - self.config.alpha) ** torch.arange(i+1).flip(0)
            kernel[i, i+1:] = (1 - self.config.alpha) ** (torch.arange(size-i-1) + 1)

        kernel = (kernel.T / kernel.sum(dim=1)).T

        return kernel

    def loss(self, target_F: torch.FloatTensor, pred_F: torch.FloatTensor) -> torch.FloatTensor:
        target_F = self.smoothen_f(target_F)

        if self.for_control:
            return self._loss_fn_u(target_F, pred_F)
        else:
            return self._loss_fn_x(target_F, pred_F)

    def to(self, device):
        self.smoothing_kernel_h = self.smoothing_kernel_h.to(device)
        self.smoothing_kernel_f = self.smoothing_kernel_f.to(device)
        self.loss_weight = self.loss_weight.to(device)
        self.target_weights_process = self.target_weights_process.to(device)
        self.target_weights_control = self.target_weights_control.to(device)
        self.Eh.to(device)
        self.Ef.to(device)
        self.D.to(device)
        return super().to(device)

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
        train_cfg = TrainConfig.load(path)

        dynamics_model = cls(config, train_cfg, for_control=False).to(device)
        control_model = cls(config, train_cfg, for_control=True).to(device)

        path = os.path.join(path, "models")

        dynamics_model.load_state_dict(torch.load(os.path.join(path, cls._state_dict_file_name(num, False)), map_location=device))
        control_model.load_state_dict(torch.load(os.path.join(path, cls._state_dict_file_name(num, True)), map_location=device))

        return dynamics_model, control_model
