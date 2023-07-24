from __future__ import annotations

import random
from typing import Generator, Type

import numpy as np
import torch
from pelutils import TT

from frozone import device
from frozone.model import Frozone
from frozone.simulations import Simulation


def simulation_dataloader(
    simulation: Type[Simulation],
    batch_size: int,
    model: Frozone,
    num_simulations = 1000,
    central_iters = 10000,
) -> Generator[tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], None, None]:

    iters = central_iters + model.config.history_window_steps + model.config.predict_window_steps
    with TT.profile("Generate new states"):
        process_states, _, control_states = simulation.simulate(num_simulations, iters, model.config.dt)

    while True:
        start_iter = np.random.uniform(model.config.history_window_steps, central_iters, batch_size)
        history_process = np.empty((batch_size, model.config.history_window_steps, len(simulation.ProcessVariables)), dtype=np.float32)
        history_control = np.empty((batch_size, model.config.history_window_steps - 1, len(simulation.ControlVariables)), dtype=np.float32)
        target_process = np.empty((batch_size, model.config.predict_window_steps, len(simulation.ProcessVariables)), dtype=np.float32)
        target_control = np.empty((batch_size, model.config.predict_window_steps, len(simulation.ControlVariables)), dtype=np.float32)

        TT.profile("Sample")
        for i in range(batch_size):
            num_sim = np.random.randint(num_simulations)
            start_iter = np.random.randint(0, central_iters)
            history_process[i] = process_states[num_sim, start_iter:start_iter + model.config.history_window_steps]
            history_control[i] = control_states[num_sim, start_iter:start_iter + model.config.history_window_steps - 1]
            target_process[i] = process_states[num_sim, start_iter + model.config.history_window_steps:start_iter + model.config.history_window_steps + model.config.predict_window_steps]
            target_control[i] = control_states[num_sim, start_iter + model.config.history_window_steps - 1:start_iter + model.config.history_window_steps + model.config.predict_window_steps - 1]
        TT.end_profile()

        with TT.profile("To device"):
            history_process = torch.from_numpy(history_process).to(device)
            history_control = torch.from_numpy(history_control).to(device)
            target_process = torch.from_numpy(target_process).to(device)
            target_control = torch.from_numpy(target_control).to(device)

        yield history_process, history_control, target_process, target_control
