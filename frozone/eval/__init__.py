from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
import torch

from frozone.model import Frozone
from frozone.simulations import Ball, Simulation


@torch.no_grad()
def forward(simulation: type[Simulation], process_states: np.ndarray, hidden_states: np.ndarray, model: Frozone, control_window: int) -> np.ndarray:
    n, iters, *_ = process_states.shape
    num_windows = (iters - 1 - model.config.window) // control_window
    control_iters = control_window * num_windows

    print(model.config)
    print(process_states.shape, num_windows, iters)
    result_process_states, result_control_states = simulation.simulate(len(process_states), control_iters, model.config.dt)
    result_process_states[:, 0] = process_states[:, 0]

    for i in range(num_windows):
        index = i * control_window
        target_process = process_states[:, index:index + model.config.window + 1]
        predicted_control = model(torch.from_numpy(target_process)).cpu().numpy()

        result_control_states[:, index:index + control_window] = predicted_control[:, :control_window]
        result_process_states[:, index:index + control_window + 1] = simulation.forward_multiple(
            result_process_states[:, index],
            predicted_control[:, :control_window],
            model.config.dt,
        )

    return result_process_states, result_control_states

if __name__ == "__main__":
    model = Frozone.load("out", torch.device("cpu"))
    model.eval()
    m, n = 4, 4
    iters = model.config.window
    control_window = 5
    target_process, target_hidden, target_control = Ball.simulate(m * n, 2 * iters, model.config.dt)

    with torch.no_grad():
        result_process_states, result_control_states = forward(Ball, target_process, target_hidden, model, control_window)

        forward_control_states = model(torch.from_numpy(target_process[:, :model.config.window + 1])).cpu().numpy()
        forward_process_states = Ball.forward_multiple(target_process[:, 0], forward_control_states, model.config.dt)

    print(target_process.shape, target_control.shape)
    print(forward_process_states.shape, forward_control_states.shape)
    print(result_process_states.shape, result_control_states.shape)

    with plots.Figure('out/forward-predict.png', figsize=(40, 30)), torch.no_grad():
        for i in range(m):
            for j in range(n):
                index = n * i + j

                plt.subplot(m, n, index + 1)
                plt.plot(
                    target_process[index, :result_control_states.shape[1], Ball.ProcessVariables.X],
                    target_process[index, :result_control_states.shape[1], Ball.ProcessVariables.Y],
                    label="True behavior",
                )
                # plt.plot(
                #     result_process_states[index, :, Ball.ProcessVariables.X],
                #     result_process_states[index, :, Ball.ProcessVariables.Y],
                #     "--",
                #     label="Forward window %i" % control_window,
                # )
                plt.plot(
                    forward_process_states[index, :, Ball.ProcessVariables.X],
                    forward_process_states[index, :, Ball.ProcessVariables.Y],
                    "--",
                    label="Forward direct",
                )

                plt.legend()
                plt.grid()

