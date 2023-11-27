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

_plot_folder = "simulation-plots"

def plot_simulated_control(
    path: str,
    env: Type[Environment],
    train_cfg: TrainConfig,
    train_results: TrainResults,
    simulation_cfg: SimulationConfig,
    X_true: np.ndarray,
    U_true: np.ndarray,
    R_true: np.ndarray,
    X_pred: np.ndarray,
    U_pred: np.ndarray,
    X_pred_opt: np.ndarray,
    U_pred_opt: np.ndarray,
    X_pred_by_model: np.ndarray,
    U_pred_by_model: np.ndarray,
    sample_no: int,
):
    timesteps_true = np.arange(X_true.shape[0]) * env.dt
    timesteps_pred_index = np.arange(X_pred.shape[0])[train_cfg.H:]
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

    label_maker = get_next_label()
    with plots.Figure(os.path.join(path, _plot_folder, "sample_%i.png" % sample_no), figsize=(12.5 * width, 10 * height)):
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
                pred_opt = X_pred_opt
                pred_by_model = X_pred_by_model
            else:
                true = U_true
                pred = U_pred
                pred_opt = U_pred_opt
                pred_by_model = U_pred_by_model

            plt.plot(timesteps_true, true[:, label], lw=1.2, label="True value")
            if is_x and label in env.reference_variables:
                plt.plot(timesteps_true, R_true[:, env.reference_variables.index(label)], lw=1.2, color="red", label="True value")
            if is_x or label in env.predicted_control:
                for k in range(train_cfg.num_models):
                    plt.plot(
                        timesteps_pred,
                        pred_by_model[k, timesteps_pred_index, label],
                        alpha=0.7,
                        lw=1.2,
                        color="grey",
                        label="Individual predictions" if k == 0 else None,
                    )
                plt.plot(
                    timesteps_pred,
                    pred[timesteps_pred_index, label],
                    lw=1.2,
                    color=plots.tab_colours[1],
                    label="Ensemble",
                )
                plt.plot(
                    timesteps_pred,
                    pred_opt[timesteps_pred_index, label],
                    lw=1.2,
                    color=plots.tab_colours[2],
                    label="Ensemble (opt)",
                )

            plt.xlabel("Time [s]")
            plt.ylabel(env.format_label(label))
            plt.legend()

            plt.grid()

        plt.suptitle(
            f"{env.__name__} controller - sample {sample_no}",
            fontsize="xx-large",
        )
