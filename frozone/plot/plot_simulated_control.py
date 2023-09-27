import math
import os
import shutil
from typing import Optional, Type

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots

from frozone.environments import Environment
from frozone.eval import SimulationConfig
from frozone.train import TrainConfig, TrainResults


# Use a non-gui backend. For God knows what reason, using the default TkAgg GUI based backend
# completely breaks when having an asynchronous data loader.
matplotlib.use('Agg')

_plot_folder = "control-plots"

def plot_simulated_control(
    path: str,
    env: Type[Environment],
    train_cfg: TrainConfig,
    train_results: TrainResults,
    simulation_cfg: SimulationConfig,
    X_true: np.ndarray,
    U_true: np.ndarray,
    X_pred: np.ndarray,
    U_pred: np.ndarray,
    X_pred_by_model: np.ndarray,
    U_pred_by_model: np.ndarray,
):
    shutil.rmtree(os.path.join(path, _plot_folder), ignore_errors=True)

    timesteps_true = np.arange(X_true.shape[1]) * env.dt
    timesteps_pred_index = train_cfg.H + np.arange(simulation_cfg.simulation_steps(env))
    timesteps_pred = timesteps_pred_index * env.dt

    width = math.ceil(math.sqrt(len(env.XLabels) + len(env.ULabels)))
    height = math.ceil((len(env.XLabels) + len(env.ULabels)) / width)

    def get_next_label():
        i = 0
        while i < len(env.XLabels) + len(env.ULabels):
            if i < len(env.XLabels):
                yield env.XLabels(i)
            else:
                yield env.ULabels(i - len(env.XLabels))
            i += 1

    for i in range(simulation_cfg.num_samples):
        label_maker = get_next_label()
        with plots.Figure(os.path.join(path, _plot_folder, "sample_%i.png" % i), figsize=(12.5 * width, height * 10)):
            for j in range(width * height):
                try:
                    label = next(label_maker)
                except StopIteration:
                    break
                plt.subplot(width, height, j + 1)
                is_x = isinstance(label, env.XLabels)
                if is_x:
                    true = X_true
                    pred = X_pred
                    pred_by_model = X_pred_by_model
                else:
                    true = U_true
                    pred = U_pred
                    pred_by_model = U_pred_by_model

                plt.plot(timesteps_true, true[i, :, label], label="True value")
                for k in range(train_cfg.num_models):
                    plt.plot(
                        timesteps_pred,
                        pred_by_model[i, k, timesteps_pred_index, label],
                        alpha=0.4,
                        color="grey",
                        label="Individual predictions" if k == 0 else None,
                    )
                plt.plot(
                    timesteps_pred,
                    pred[i, timesteps_pred_index, label],
                    color=plots.tab_colours[1],
                    label="Mean prediction",
                )

                plt.xlabel("Time [s]")
                plt.ylabel(label.name)
                plt.legend()

                margin = 0.4
                true_min = true[i, :, label].min()
                true_max = true[i, :, label].max()
                plt.ylim(
                    bottom = true_min - margin * (true_max - true_min),
                    top = true_max + margin * (true_max - true_min))

                plt.grid()
