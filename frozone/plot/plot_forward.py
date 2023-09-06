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

_save_folder = "forward-plots"

def plot_forward(
    path: str,
    env: Type[Environment],
    train_cfg: TrainConfig,
    train_results: TrainResults,
    forward_cfg: ForwardConfig,
    X_true: np.ndarray,
    X_pred: np.ndarray,
    U_true: np.ndarray,
):
    shutil.rmtree(os.path.join(path, _save_folder), ignore_errors=True)

    timestamps = np.arange(X_true.shape[1]) * train_cfg.dt
    for i in range(forward_cfg.num_samples):
        for j in range(len(env.XLabels)):
            with plots.Figure(os.path.join(path, _save_folder, "sample_%i" % i, env.XLabels(j).name + ".png")):
                for k in range(train_cfg.num_models):
                    plt.plot(
                        timestamps, X_pred[i, :, k, j], "-o", color="grey", alpha=0.5,
                        label="Individual estimates" if k == 0 else None,
                    )
                plt.plot(timestamps, X_true[i, :, j], "-o", label="True trajectory")
                plt.plot(timestamps[train_cfg.H:], X_pred[i, train_cfg.H:, :, j].mean(axis=1), "-o", label="Mean estimate")

                plt.xlabel("Time [s]")
                plt.ylabel(env.XLabels(j).name)
                plt.legend()

                plt.grid()
