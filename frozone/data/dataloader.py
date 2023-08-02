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
) -> Generator[tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], None, None]:

    iters = central_iters + train_cfg.history_steps + train_cfg.predict_steps
    with TT.profile("Generate new states"):
        process_states, _, control_states = simulation.simulate(num_simulations, iters, train_cfg.dt)

    while True:
        start_iter = np.random.uniform(train_cfg.history_steps, central_iters, train_cfg.batch_size)
        XH = np.empty((train_cfg.batch_size, train_cfg.history_steps, len(simulation.ProcessVariables)), dtype=np.float32)
        UH = np.empty((train_cfg.batch_size, train_cfg.history_steps - 1, len(simulation.ControlVariables)), dtype=np.float32)
        XF = np.empty((train_cfg.batch_size, train_cfg.predict_steps, len(simulation.ProcessVariables)), dtype=np.float32)
        UF = np.empty((train_cfg.batch_size, train_cfg.predict_steps, len(simulation.ControlVariables)), dtype=np.float32)

        TT.profile("Sample")
        for i in range(train_cfg.batch_size):
            num_sim = np.random.randint(num_simulations)
            start_iter = np.random.randint(0, central_iters)
            XH[i] = process_states[num_sim, start_iter:start_iter + train_cfg.history_steps]
            UH[i] = control_states[num_sim, start_iter:start_iter + train_cfg.history_steps - 1]
            XF[i] = process_states[num_sim, start_iter + train_cfg.history_steps:start_iter + train_cfg.history_steps + train_cfg.predict_steps]
            UF[i] = control_states[num_sim, start_iter + train_cfg.history_steps - 1:start_iter + train_cfg.history_steps + train_cfg.predict_steps - 1]
        TT.end_profile()

        with TT.profile("To device"):
            XH = torch.from_numpy(XH).to(device)
            UH = torch.from_numpy(UH).to(device)
            XF = torch.from_numpy(XF).to(device)
            UF = torch.from_numpy(UF).to(device)

        yield XH, UH, XF, UF
