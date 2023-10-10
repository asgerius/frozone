import math
import os
import shutil
from typing import Optional, Type

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
from pelutils import log

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
    U_true: np.ndarray,
    X_true_smooth: np.ndarray,
    U_true_smooth: np.ndarray,
    X_pred: Optional[np.ndarray],
    U_pred: Optional[np.ndarray],
    U_pred_opt: Optional[np.ndarray],
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
        log.debug("Sample %i / %i" % (i + 1, forward_cfg.num_samples))
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
                    true_smooth = X_true_smooth
                    pred = X_pred
                    pred_opt = None
                    plot_preds = label not in env.no_reference_variables
                else:
                    true = U_true
                    true_smooth = U_true_smooth
                    pred = U_pred
                    pred_opt = U_pred_opt
                    plot_preds = True

                plt.plot(timesteps, true[i, :, label], color="grey", label="True value")

                for k in range(forward_cfg.num_sequences):
                    seq_start = k * sequence_length
                    seq_mid = k * sequence_length + train_cfg.H
                    seq_end = (k + 1) * sequence_length
                    plt.plot(timesteps[seq_start:seq_mid], true_smooth[i, seq_start:seq_mid, label], lw=1.2, c=plots.tab_colours[0], label="True value (smooth)" if k == 0 else None)
                    plt.plot(timesteps[seq_mid:seq_end], true_smooth[i, seq_mid:seq_end, label], lw=1.2, c=plots.tab_colours[0])

                    if pred is not None and plot_preds:
                        for l in range(train_cfg.num_models):
                            plt.plot(
                                timesteps[seq_mid-1:seq_end],
                                pred[i, l, seq_mid-1:seq_end, label],
                                alpha=0.7,
                                color="grey",
                                label="Individual predictions" if k == l == 0 else None,
                            )
                        plt.plot(
                            timesteps[seq_mid-1:seq_end],
                            pred[i, :, seq_mid-1:seq_end, label].mean(axis=0),
                            lw=1.2,
                            color=plots.tab_colours[1],
                            label="Ensemble" if k == 0 else None,
                        )

                if pred_opt is not None:
                    for k in range(forward_cfg.num_sequences):
                        seq_mid = k * sequence_length + train_cfg.H
                        seq_end = (k + 1) * sequence_length
                        plt.plot(
                            timesteps[seq_mid-1:seq_end],
                            pred_opt[i, seq_mid-1:seq_end, label],
                            lw=1.2,
                            color=plots.tab_colours[2],
                            label="Ensemble (opt)" if k == 0 else None,
                        )

                plt.xlabel("Time [s]")
                plt.ylabel(env.format_label(label))
                plt.legend()

                plt.grid()
