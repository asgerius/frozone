import math
import os
import shutil
import warnings
from typing import Optional, Type

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots

from frozone.environments import Environment
from frozone.eval import ForwardConfig
from frozone.train import TrainConfig, TrainResults


warnings.filterwarnings("error")

# Use a non-gui backend. For God knows what reason, using the default TkAgg GUI based backend
# completely breaks when having an asynchronous data loader.
matplotlib.use('Agg')

_plot_folder = "predict-plots"

def plot_forward(
    path: str,
    env: Type[Environment],
    train_cfg: TrainConfig,
    train_results: TrainResults,
    forward_cfg: ForwardConfig,
    X_true: np.ndarray,
    U_true: np.ndarray,
    X_pred: Optional[np.ndarray],
    U_pred: Optional[np.ndarray],
):
    shutil.rmtree(os.path.join(path, _plot_folder), ignore_errors=True)

    sequence_length = train_cfg.H + train_cfg.F
    timesteps = np.arange(forward_cfg.num_sequences * sequence_length) * env.dt

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
                    plot_preds = label not in env.no_reference_variables
                else:
                    true = U_true
                    pred = U_pred
                    plot_preds = True

                plt.plot(timesteps, true[i, :, label], "-o", label="True value")
                if pred is not None and plot_preds:
                    for k in range(forward_cfg.num_sequences):
                        seq_mid = k * sequence_length + train_cfg.H
                        seq_end = (k + 1) * sequence_length
                        for l in range(train_cfg.num_models):
                            plt.plot(
                                timesteps[seq_mid-1:seq_end],
                                pred[i, l, seq_mid-1:seq_end, label],
                                "-o",
                                alpha=0.4,
                                color="grey",
                                label="Individual predictions" if k == l == 0 else None,
                            )
                        plt.plot(
                            timesteps[seq_mid-1:seq_end],
                            pred[i, :, seq_mid-1:seq_end, label].mean(axis=0),
                            "-o",
                            color=plots.tab_colours[1],
                            label="Mean prediction" if k == 0 else None,
                        )

                plt.xlabel("Time [s]")
                plt.ylabel(label.name)
                plt.legend()

                margin = 0.4
                true_min = true[i, :, label].min()
                true_max = true[i, :, label].max()
                try:
                    plt.ylim(
                        bottom = true_min - margin * (true_max - true_min),
                        top = true_max + margin * (true_max - true_min),
                    )
                except UserWarning:
                    pass

                plt.grid()
