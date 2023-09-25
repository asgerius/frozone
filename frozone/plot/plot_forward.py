import math
import os
import shutil
from typing import Type

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots

from frozone.environments import Environment
from frozone.eval import ForwardConfig
from frozone.train import TrainConfig, TrainResults


# Use a non-gui backend. For God knows what reason, using the default TkAgg GUI based backend
# completely breaks when having an asynchronous data loader.
matplotlib.use('Agg')

_plot_folder = "forward-plots"

def plot_forward(
    path: str,
    env: Type[Environment],
    train_cfg: TrainConfig,
    train_results: TrainResults,
    forward_cfg: ForwardConfig,
    X_true: np.ndarray,
    X_pred_m: np.ndarray,
    X_pred_p: np.ndarray,
    X_pred_i: np.ndarray,
    U_true: np.ndarray,
):
    shutil.rmtree(os.path.join(path, _plot_folder), ignore_errors=True)

    timestamps_true = np.arange(X_true.shape[1]) * env.dt
    timestamps_pred = np.arange(X_pred_m.shape[1])[train_cfg.H - 1:] * env.dt
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

    for i in range(forward_cfg.num_samples):
        label_maker = get_next_label()
        with plots.Figure(os.path.join(path, _plot_folder, "sample_%i.png" % i), figsize=(50, 30)):
            for j in range(width * height):
                try:
                    label = next(label_maker)
                except StopIteration:
                    break
                plt.subplot(width, height, j + 1)
                is_x = isinstance(label, env.XLabels)
                if is_x:
                    for k in range(train_cfg.num_models):
                        plt.plot(
                            timestamps_pred, X_pred_i[i, train_cfg.H - 1 :, k, j], "-o", color="grey", alpha=0.5,
                            label="Individual estimates" if k == 0 else None,
                        )
                        plt.plot(
                            timestamps_true[:train_cfg.H + train_cfg.F], X_pred_p[i, :, k, j], "-P", color="grey", alpha=0.5,
                            label="Prediction horizon" if k == 0 else None,
                        )
                    plt.plot(timestamps_pred, X_pred_i[i, train_cfg.H - 1 :, :, j].mean(axis=1), "-o", label="Mean individual estimates")
                    plt.plot(timestamps_pred, X_pred_m[i, train_cfg.H - 1 :, :, j].mean(axis=1), "-o", label="Mean estimate")
                    plt.plot(timestamps_true[:train_cfg.H + train_cfg.F], X_pred_p[i, :, :, j].mean(axis=1), "-o", label="Mean prediction horizon")

                    plt.plot(timestamps_true, X_true[i, :, j], "-o", label="True trajectory")
                else:
                    plt.plot(timestamps_true, U_true[i, :, j - len(env.XLabels)], "-o", label="True trajectory")

                plt.xlabel("Time [s]")
                plt.ylabel(label.name)
                plt.legend()

                plt.grid()
