from __future__ import annotations

import math
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from pelutils import DataStorage

import frozone.simulations as simulations
from frozone import device
from frozone.model import Frozone, FFFrozone
from frozone.train import TrainConfig


@dataclass
class ForwardEvalConfig(DataStorage):

    train_cfg: TrainConfig

    n: int

    control_window: float
    simulation_window: float

    gradient_steps: int
    gamma: float

    def __post_init__(self):
        assert isinstance(self.n, int) and self.n > 0
        assert 0 < self.control_window <= self.train_cfg.prediction_window

    @property
    def control_steps(self) -> int:
        return int(self.control_window / self.train_cfg.dt)

    @property
    def simulation_steps(self) -> int:
        return int(self.simulation_window / self.train_cfg.dt)

    @property
    def total_steps(self) -> int:
        return self.train_cfg.history_steps + self.train_cfg.predict_steps \
            + self.control_steps + self.simulation_steps

@dataclass
class ForwardEvalResults(DataStorage):
    loss: list[float]

def forward_eval(path: str, eval_cfg: ForwardEvalConfig, model: Frozone) -> ForwardEvalResults:
    train_cfg = eval_cfg.train_cfg

    model.eval()

    loss_fn = nn.MSELoss()

    env: simulations.Simulation = getattr(simulations, train_cfg.env)

    import matplotlib.pyplot as plt
    import pelutils.ds.plots as plots

    with plots.Figure(f"{path}/X.png"):
        for i in range(eval_cfg.n):
            # Simulate new
            target_X, target_Z, target_U = env.simulate(1, eval_cfg.total_steps, train_cfg.dt)

            # Only dealing with one simulation at a time, so discard first dimension
            target_X: np.ndarray = target_X[0]
            target_U: np.ndarray = target_U[0]
            target_Z: np.ndarray = target_Z[0]

            for j in range(2):

                actual_X = target_X[:len(target_X) - train_cfg.predict_steps + eval_cfg.control_steps - 1].copy()
                actual_U = target_U[:len(target_U) - train_cfg.predict_steps + eval_cfg.control_steps - 1].copy()
                actual_Z = target_Z[:len(target_Z) - train_cfg.predict_steps + eval_cfg.control_steps - 1].copy()

                index = 0
                while index + train_cfg.history_steps + train_cfg.predict_steps <= eval_cfg.total_steps:

                    XH = actual_X[index : index + train_cfg.history_steps]
                    UH = actual_U[index : index + train_cfg.history_steps - 1]

                    XF = target_X[index + train_cfg.history_steps : index + train_cfg.history_steps + train_cfg.predict_steps]

                    XH_d = torch.from_numpy(XH).to(device).unsqueeze(0)
                    UH_d = torch.from_numpy(UH).to(device).unsqueeze(0)
                    XF_d = torch.from_numpy(XF).to(device).unsqueeze(0)

                    with torch.inference_mode():
                        pred_ZH_d, pred_UF_d, _ = model(XH_d, UH_d, XF_d)

                    print(index)
                    model.process_model.requires_grad_(False)
                    model.process_model.U_layer.requires_grad_(True)
                    model.process_model.set_UF(pred_UF_d)
                    for k in range((j > 0) * eval_cfg.gradient_steps):
                        optim = torch.optim.AdamW(model.process_model.parameters(), lr=eval_cfg.gamma)
                        pred_XF_d = model.process_model(pred_ZH_d)
                        loss = loss_fn(XF_d, pred_XF_d)
                        print(loss.item())
                        loss.backward()
                        model.process_model.U_layer.grad[:, :model.config.K] = 0
                        optim.step()
                        optim.zero_grad()

                    pred_UF_d = model.process_model.U_layer.data[0, model.config.K:].detach().view(train_cfg.predict_steps, len(env.ControlVariables)).cpu().numpy()

                    control_slice = slice(index + train_cfg.history_steps - 1, index + train_cfg.history_steps + eval_cfg.control_steps - 1)
                    actual_U[control_slice] = pred_UF_d[:eval_cfg.control_steps]
                    XC, ZC = env.forward_multiple(
                        np.expand_dims(actual_X[index + train_cfg.history_steps - 1], 0),
                        np.expand_dims(actual_Z[index + train_cfg.history_steps - 1], 0),
                        np.expand_dims(actual_U[control_slice], 0),
                        train_cfg.dt,
                    )
                    process_slice = slice(index + train_cfg.history_steps, index + train_cfg.history_steps + eval_cfg.control_steps)

                    actual_X[process_slice] = XC.squeeze(axis=0)[1:]
                    actual_Z[process_slice] = ZC.squeeze(axis=0)[1:]

                    index += eval_cfg.control_steps
                plt.subplot(eval_cfg.n, 2, 2 * i + j + 1)
                plt.plot(*target_X.T, label="Target")
                plt.plot(*actual_X.T, label="Actual")

                plt.grid()
                plt.legend()

            # with plots.Figure(f"{path}/Xvars-%i.png" % i):
            #     plt.plot(target_X[:, 0], label="Target $x$")
            #     plt.plot(target_X[:, 1], label="Target $y$")
            #     plt.plot(actual_X[:, 0], label="Actual $x$")
            #     plt.plot(actual_X[:, 1], label="Actual $y$")

            #     plt.grid()
            #     plt.legend()

            # with plots.Figure(f"{path}/Uvars-%i.png" % i):
            #     plt.plot(target_U[:, 0], label="Target $f_0$")
            #     plt.plot(target_U[:, 1], label="Target $f_1$")
            #     plt.plot(target_U[:, 2], label="Target $f_2$")
            #     plt.plot(actual_U[:, 0], label="Actual $f_0$")
            #     plt.plot(actual_U[:, 1], label="Actual $f_1$")
            #     plt.plot(actual_U[:, 2], label="Actual $f_2$")

            #     plt.grid()
            #     plt.legend()

if __name__ == "__main__":
    path = "out/2023-08-03_01-36-53"
    train_cfg = TrainConfig.load(path)
    model = FFFrozone.load(path)
    eval_cfg = ForwardEvalConfig(train_cfg, 2, 0.2, 100, 20, 1e-3)

    forward_eval(path, eval_cfg, model)
