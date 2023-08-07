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

    control_window: float
    simulation_window: float

    gradient_steps: int
    gamma: float

    def __post_init__(self):
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
    error: list[float]
    forward_loss: list[list[float]]

    @classmethod
    def empty(cls) -> ForwardEvalResults:
        return ForwardEvalResults(list(), list())

def forward_eval(eval_cfg: ForwardEvalConfig, model: Frozone, eval_results: ForwardEvalResults):
    """ Runs a forward simulation using a given controller defined by eval_cfg and eval_cfg.train_cfg.
    Nothing is returned, but information is appended in-place to the relevant fields in eval_results. """
    train_cfg = eval_cfg.train_cfg

    model.eval()

    loss_fn = nn.MSELoss()

    env: simulations.Simulation = getattr(simulations, train_cfg.env)

    # Simulate new
    target_X, target_Z, target_U, target_S = env.simulate(1, eval_cfg.total_steps, train_cfg.dt)

    # Only dealing with one simulation at a time, so discard first dimension
    target_X: np.ndarray = target_X[0]
    target_U: np.ndarray = target_U[0]
    target_Z: np.ndarray = target_Z[0]
    target_S: np.ndarray = target_S[0]

    actual_X = target_X[:len(target_X) - train_cfg.predict_steps + eval_cfg.control_steps - 1].copy()
    actual_U = target_U[:len(target_U) - train_cfg.predict_steps + eval_cfg.control_steps - 1].copy()
    actual_Z = target_Z[:len(target_Z) - train_cfg.predict_steps + eval_cfg.control_steps - 1].copy()
    actual_S = target_S[:len(target_S) - train_cfg.predict_steps + eval_cfg.control_steps - 1].copy()

    eval_results.forward_loss.append(list())

    index = 0
    while index + train_cfg.history_steps + train_cfg.predict_steps <= eval_cfg.total_steps:

        XH = actual_X[index : index + train_cfg.history_steps]
        UH = actual_U[index : index + train_cfg.history_steps - 1]
        SH = actual_S[index : index + train_cfg.history_steps]

        XF = target_X[index + train_cfg.history_steps : index + train_cfg.history_steps + train_cfg.predict_steps]
        SF = target_S[index + train_cfg.history_steps : index + train_cfg.history_steps + train_cfg.predict_steps]

        XH_d = torch.from_numpy(XH).to(device).unsqueeze(0)
        UH_d = torch.from_numpy(UH).to(device).unsqueeze(0)
        SH_d = torch.from_numpy(SH).to(device).unsqueeze(0)
        XF_d = torch.from_numpy(XF).to(device).unsqueeze(0)
        SF_d = torch.from_numpy(SF).to(device).unsqueeze(0)

        with torch.inference_mode():
            pred_ZH_d, pred_UF_d, _ = model(XH_d, UH_d, SH_d, SF_d, XF_d)

        model.process_model.requires_grad_(False)
        model.process_model.U_layer.requires_grad_(True)
        model.process_model.set_UF(pred_UF_d)
        # print()
        # print(index)
        for k in range(eval_cfg.gradient_steps):
            optim = torch.optim.AdamW(model.process_model.parameters(), lr=eval_cfg.gamma)
            pred_XF_d = model.process_model(pred_ZH_d, SF_d)
            loss = loss_fn(XF_d, pred_XF_d)
            loss.backward()
            model.process_model.U_layer.grad[:, :-model.process_model.U_size] = 0
            optim.step()
            optim.zero_grad()
            eval_results.forward_loss[-1].append(loss.item())
            # print(loss.item())

        pred_UF_d = model.process_model.U_layer.data[0, -model.process_model.U_size:] \
            .detach() \
            .view(train_cfg.predict_steps, len(env.ControlVariables)) \
            .cpu() \
            .numpy()

        control_slice = slice(index + train_cfg.history_steps - 1, index + train_cfg.history_steps + eval_cfg.control_steps - 1)
        actual_U[control_slice] = pred_UF_d[:eval_cfg.control_steps]
        XC, ZC, _ = env.forward_multiple(
            np.expand_dims(actual_X[index + train_cfg.history_steps - 1], 0),
            np.expand_dims(actual_U[control_slice], 0),
            np.expand_dims(actual_Z[index + train_cfg.history_steps - 1], 0),
            np.expand_dims(actual_S[index + train_cfg.history_steps - 1], 0),
            train_cfg.dt,
        )
        process_slice = slice(index + train_cfg.history_steps, index + train_cfg.history_steps + eval_cfg.control_steps)

        actual_X[process_slice] = XC.squeeze(axis=0)[1:]
        actual_Z[process_slice] = ZC.squeeze(axis=0)[1:]

        index += eval_cfg.control_steps

    eval_results.error.append(loss_fn(
        torch.from_numpy(target_X[train_cfg.history_steps:-train_cfg.predict_steps+1]),
        torch.from_numpy(actual_X[train_cfg.history_steps:]),
    ))

if __name__ == "__main__":
    path = "out/2023-08-07_18-03-07"
    train_cfg = TrainConfig.load(path)
    model = FFFrozone.load(path)
    eval_cfg = ForwardEvalConfig(train_cfg, 0.2, 40, 20, 5e-3)

    results_opt = ForwardEvalResults.empty()
    results_no_opt = ForwardEvalResults.empty()

    from tqdm import tqdm

    n_opt = 30
    n_no_opt = 50

    for i in tqdm(range(n_opt)):
        forward_eval(eval_cfg, model, results_opt)
    for i in tqdm(range(n_no_opt)):
        eval_cfg.gradient_steps = 0
        forward_eval(eval_cfg, model, results_no_opt)

    results_opt.save(path, "ForwardEvalResults_opt")
    results_no_opt.save(path, "ForwardEvalResults_no_opt")

    print("OPT")
    print("Mean    = %f" % np.mean(results_opt.error))
    print("Std(mu) = %f" % (np.std(results_opt.error, ddof=1) / np.sqrt(n_opt)))

    print("NO OPT")
    print("Mean    = %f" % np.mean(results_no_opt.error))
    print("Std(mu) = %f" % (np.std(results_no_opt.error, ddof=1) / np.sqrt(n_no_opt)))
