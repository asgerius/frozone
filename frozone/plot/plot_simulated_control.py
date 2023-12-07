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
    timesteps_true = np.arange(X_true.shape[0]) * env.dt / 3600
    timesteps_pred_index = np.arange(X_pred.shape[0])[train_cfg.H:]
    timesteps_pred = timesteps_pred_index * env.dt / 3600

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

            if not is_x and label in env.predefined_control:
                plt.plot(timesteps_true, true[:, label], lw=1.2, color="grey")
            if is_x and label in env.reference_variables:
                plt.plot(timesteps_true, R_true[:, env.reference_variables.index(label)], lw=1.2, color="red", label="Reference")
            if is_x or label in env.predicted_control:
                # To plot for all models, use
                for k in range(train_cfg.num_models):
                # k = 0
                    plt.plot(
                        timesteps_pred,
                        pred_by_model[k, timesteps_pred_index, label],
                        alpha=0.7,
                        lw=1.2,
                        color=plots.tab_colours[0],
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

            plt.xlabel("Time [h]")
            plt.ylabel(env.format_label(label))
            plt.legend()

            plt.grid()

        plt.suptitle(
            f"{env.__name__} controller - sample {sample_no}",
            fontsize="xx-large",
        )

def plot_error(
    path: str,
    env: Type[Environment],
    train_cfg: TrainConfig,
    simulation_cfg: SimulationConfig,
    error_calcs: dict[int, dict[str, dict[str, np.ndarray]]],
):

    with plots.Figure(os.path.join(path, _plot_folder, "error.png"), figsize=(12.5 * len(env.reference_variables), 10)):
        for i, rlab in enumerate(error_calcs):
            plt.subplot(1, len(env.reference_variables), i + 1)
            plt.plot([0], [0], c="grey", label="Mean")
            plt.plot([0], [0], "--", c="grey", label="80'th %-tile")
            plt.plot([0], [0], ":", c="grey", label="100'th %-tile")
            plt.axvline(env.dt * train_cfg.H / 3600, c="black", label="Controller start")
            for j, control_method in enumerate(error_calcs[rlab]):
                timesteps = np.arange(len(error_calcs[rlab][control_method]["error_mean"])) * env.dt / 3600
                plt.plot(timesteps, error_calcs[rlab][control_method]["error_mean"], c=plots.tab_colours[j], label=control_method)
                plt.plot(timesteps, error_calcs[rlab][control_method]["error_80"], "--", c=plots.tab_colours[j])
                plt.plot(timesteps, error_calcs[rlab][control_method]["error_100"],  ":", c=plots.tab_colours[j])

            plt.legend(loc=1)
            plt.grid()
            plt.xlabel("Time [h]")
            plt.ylabel(env.format_label(rlab))


        plt.suptitle(
            f"Error in control over {simulation_cfg.num_samples:,} simulations",
            fontsize="xx-large",
        )
