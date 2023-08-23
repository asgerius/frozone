from __future__ import annotations

from typing import Generator, Type

import numpy as np
import torch
from pelutils import TT

from frozone import device
from frozone.environments import Environment
from frozone.train import TrainConfig


def simulation_dataloader(
    simulation: Type[Environment],
    train_cfg: TrainConfig,
    num_simulations = 1000,
    central_iters = 10000,
) -> Generator[tuple[torch.FloatTensor], None, None]:
    """ Yields Xh, Uh, Sh, Xf, Sf, u, s on device. """

    iters = central_iters + train_cfg.H + train_cfg.F
    with TT.profile("Generate new states"):
        X, _, U, S = simulation.simulate(num_simulations, iters, train_cfg.dt)

    while True:
        start_iter = np.random.uniform(train_cfg.H, central_iters, train_cfg.batch_size)
        Xh = np.empty((train_cfg.batch_size, train_cfg.H, len(simulation.XLabels)), dtype=Environment.X_dtype)
        Uh = np.empty((train_cfg.batch_size, train_cfg.H, len(simulation.ULabels)), dtype=Environment.U_dtype)
        Sh = np.empty((train_cfg.batch_size, train_cfg.H, sum(simulation.S_bin_count)), dtype=Environment.S_dtype)
        Xf = np.empty((train_cfg.batch_size, train_cfg.F, len(simulation.XLabels)), dtype=Environment.X_dtype)
        Sf = np.empty((train_cfg.batch_size, train_cfg.F, len(simulation.SLabels)), dtype=Environment.S_dtype)
        u = np.empty((train_cfg.batch_size, len(simulation.ULabels)), dtype=Environment.U_dtype)
        s = np.empty((train_cfg.batch_size, 2, sum(simulation.S_bin_count)), dtype=Environment.S_dtype)

        TT.profile("Sample")
        for i in range(train_cfg.batch_size):
            num_sim = np.random.randint(num_simulations)
            start_iter = np.random.randint(0, central_iters)
            Xh[i] = X[num_sim, start_iter:start_iter + train_cfg.H]
            Uh[i] = U[num_sim, start_iter:start_iter + train_cfg.H]
            Sh[i] = S[num_sim, start_iter:start_iter + train_cfg.H]
            Xf[i] = X[num_sim, start_iter + train_cfg.H:start_iter + train_cfg.H + train_cfg.F]
            Sf[i] = S[num_sim, start_iter + train_cfg.H:start_iter + train_cfg.H + train_cfg.F]
            u[i] = U[num_sim, start_iter + train_cfg.H]
            s[i] = S[num_sim, start_iter + train_cfg.H:start_iter + train_cfg.H + 2]
        TT.end_profile()

        with TT.profile("To device"):
            Xh = torch.from_numpy(Xh).to(device)
            Uh = torch.from_numpy(Uh).to(device)
            Sh = torch.from_numpy(Sh).to(device)
            Xf = torch.from_numpy(Xf).to(device)
            Sf = torch.from_numpy(Sf).to(device)
            u = torch.from_numpy(u).to(device)
            s = torch.from_numpy(s).to(device)

        yield Xh, Uh, Sh, Xf, Sf, u, s
