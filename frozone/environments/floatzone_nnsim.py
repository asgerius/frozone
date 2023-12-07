import math
from typing import Optional

import numpy as np
import torch

from frozone import device
from frozone.data.utils import numpy_to_torch_device
from frozone.environments.floatzone import FloatZone
from frozone.model.floatzone_network import FzNetwork, FzConfig, interpolate


class FloatZoneNNSim(FloatZone):

    is_simulation = True

    model_path = "/work3/s183912/floatzone/out/FLOATZONE_DYNAMICUS"
    # train_cfg: frozone.train.TrainConfig
    model_config: FzConfig
    models: list[tuple[FzNetwork, FzNetwork]]

    @classmethod
    def load(cls, train_cfg, train_res):
        cls.train_cfg = train_cfg
        cls.train_res = train_res
        cls.model_config = FzConfig.load(cls.model_path)
        cls.models = [FzNetwork.load(cls.model_path, i) for i in range(train_cfg.num_models)]
        for dm, cm in cls.models:
            dm.eval()
            cm.eval()

    @classmethod
    def init_hidden_vars(cls, U: np.ndarray) -> np.ndarray:
        """ Some wanker decided this absolutely has to be implemented. """
        return np.empty_like(U, dtype=cls.X_dtype)[..., []]

    @classmethod
    @torch.inference_mode()
    def forward_standardized_multiple(
        cls,
        Xh: np.ndarray,
        Uh: np.ndarray,
        Sh: np.ndarray,
        Sf: np.ndarray,
        Uf: np.ndarray,
        timesteps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_forwards = math.ceil(timesteps / cls.train_cfg.F)

        H = Xh.shape[1]
        Xh, Uh, Sh, Sf, Uf = numpy_to_torch_device(Xh, Uh, Sh, Sf, Uf)

        Xf = torch.zeros((Xh.shape[0], timesteps, len(cls.XLabels)), device=device)
        X = torch.concat((Xh, Xf), dim=1)
        U = torch.concat((Uh, Uf), dim=1)
        S = torch.concat((Sh, Sf), dim=1)

        full_poly_dia = X[:, 0, cls.XLabels.FullPolyDia] * cls.train_res.std_x[cls.XLabels.FullPolyDia] + cls.train_res.mean_x[cls.XLabels.FullPolyDia]
        std_poly_dia = lambda x: (x - cls.train_res.mean_x[cls.XLabels.PolyDia]) / (cls.train_res.std_x[cls.XLabels.PolyDia] + 1e-6)
        poly_angle_0_std = -cls.train_res.mean_x[cls.XLabels.PolyAngle] / (cls.train_res.std_x[cls.XLabels.PolyAngle] + 1e-6)
        for i in range(num_forwards):
            seq_start = i * cls.train_cfg.F
            seq_mid = seq_start + cls.train_cfg.H
            seq_end = min(seq_mid + cls.train_cfg.F, timesteps + cls.train_cfg.H)
            timesteps_this_forward = seq_end - seq_mid
            for j, (dynamics_model, control_model) in enumerate(cls.models):
                Xf_pred = dynamics_model(
                    interpolate(cls.train_cfg.Hi, X[:, seq_start:seq_mid], cls.train_cfg, h=True),
                    interpolate(cls.train_cfg.Hi, U[:, seq_start:seq_mid], cls.train_cfg, h=True),
                    interpolate(cls.train_cfg.Hi, S[:, seq_start:seq_mid], cls.train_cfg, h=True),
                    Sf = interpolate(cls.train_cfg.Fi, S[:, seq_mid:seq_end]),
                    Uf = interpolate(cls.train_cfg.Fi, U[:, seq_mid:seq_end]),
                ) / cls.train_cfg.num_models

                X[:, seq_mid:seq_end] += interpolate(cls.train_cfg.F, Xf_pred)[:, :timesteps_this_forward]

            X[..., cls.XLabels.FullPolyDia] = Xh[:, -1, cls.XLabels.FullPolyDia]
            for j in range(Xh.shape[0]):
                for k in range(timesteps_this_forward):
                    X[j, seq_mid+k, cls.XLabels.PolyAngle] = max(poly_angle_0_std, X[j, seq_mid+k, cls.XLabels.PolyAngle])
                    X[j, seq_mid+k, cls.XLabels.PolyDia] = min(std_poly_dia(full_poly_dia), X[j, seq_mid+k, cls.XLabels.PolyDia])

        return X[:, H:].cpu().numpy()
