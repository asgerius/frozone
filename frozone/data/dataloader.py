from __future__ import annotations

import random
from typing import Generator, Type

import numpy as np
import torch
from pelutils import TT

from frozone import device
from frozone.simulations import Simulation
from frozone.train import TrainConfig


def simulation_dataloader(
    simulation: Type[Simulation],
    train_cfg: TrainConfig,
    num_simulations = 1000,
    central_iters = 10000,
) -> Generator[tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], None, None]:

    iters = central_iters + train_cfg.history_steps + train_cfg.predict_steps
    with TT.profile("Generate new states"):
        X, _, U, S = simulation.simulate(num_simulations, iters, train_cfg.dt)

    while True:
        start_iter = np.random.uniform(train_cfg.history_steps, central_iters, train_cfg.batch_size)
        XH = np.empty((train_cfg.batch_size, train_cfg.history_steps, len(simulation.ProcessVariables)), dtype=np.float32)
        UH = np.empty((train_cfg.batch_size, train_cfg.history_steps - 1, len(simulation.ControlVariables)), dtype=np.float32)
        SH = np.empty((train_cfg.batch_size, train_cfg.history_steps, len(simulation.StaticVariables)), dtype=int)
        XF = np.empty((train_cfg.batch_size, train_cfg.predict_steps, len(simulation.ProcessVariables)), dtype=np.float32)
        UF = np.empty((train_cfg.batch_size, train_cfg.predict_steps, len(simulation.ControlVariables)), dtype=np.float32)
        SF = np.empty((train_cfg.batch_size, train_cfg.predict_steps, len(simulation.StaticVariables)), dtype=int)

        TT.profile("Sample")
        for i in range(train_cfg.batch_size):
            num_sim = np.random.randint(num_simulations)
            start_iter = np.random.randint(0, central_iters)
            XH[i] = X[num_sim, start_iter:start_iter + train_cfg.history_steps]
            UH[i] = U[num_sim, start_iter:start_iter + train_cfg.history_steps - 1]
            SH[i] = S[num_sim, start_iter:start_iter + train_cfg.history_steps]
            XF[i] = X[num_sim, start_iter + train_cfg.history_steps:start_iter + train_cfg.history_steps + train_cfg.predict_steps]
            UF[i] = U[num_sim, start_iter + train_cfg.history_steps - 1:start_iter + train_cfg.history_steps + train_cfg.predict_steps - 1]
            SF[i] = S[num_sim, start_iter + train_cfg.history_steps:start_iter + train_cfg.history_steps + train_cfg.predict_steps]
        TT.end_profile()

        with TT.profile("To device"):
            XH = torch.from_numpy(XH).to(device)
            UH = torch.from_numpy(UH).to(device)
            SH = torch.from_numpy(SH).to(device)
            XF = torch.from_numpy(XF).to(device)
            UF = torch.from_numpy(UF).to(device)
            SF = torch.from_numpy(SF).to(device)

        yield XH, UH, SH, XF, UF, SF
