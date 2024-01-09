import math
import os
import shutil
from typing import Optional, Type

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
from pelutils import log

from frozone.data import Metadata, PHASES
from frozone.environments import Environment, FloatZone
from frozone.eval import ForwardConfig
from frozone.plot import get_figure_args
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
    metadatas: list[Metadata],
    X_true: np.ndarray,
    U_true: np.ndarray,
    X_true_smooth: np.ndarray,
    U_true_smooth: np.ndarray,
    X_pred: Optional[np.ndarray],
    U_pred: Optional[np.ndarray],
    U_pred_opt: Optional[np.ndarray],
    U_pred_ref: Optional[np.ndarray],
    R_true: np.ndarray,
):
    shutil.rmtree(os.path.join(path, _plot_folder), ignore_errors=True)

    sequence_length = train_cfg.H + train_cfg.F
    timesteps = env.dt * np.arange(forward_cfg.num_sequences * sequence_length) / 3600

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

    for i in range(X_pred.shape[0]):
        log.debug("Sample %i / %i" % (i + 1, X_pred.shape[0]))
        label_maker = get_next_label()
        with plots.Figure(os.path.join(path, _plot_folder, "sample_%i.pdf" % i), figsize=(12.5 * width, height * 10)):
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
                    pred_ref = None
                    plot_preds = label not in env.no_reference_variables
                else:
                    true = U_true
                    true_smooth = U_true_smooth
                    pred = U_pred
                    pred_opt = U_pred_opt
                    pred_ref = U_pred_ref
                    plot_preds = label not in env.predefined_control

                plt.plot(timesteps, true[i, :, label], color="grey", label="True value")
                if is_x and label in env.reference_variables:
                    plt.plot(timesteps, R_true[i, :, env.reference_variables.index(label)], color="red", label="Target")

                for k in range(forward_cfg.num_sequences):
                    seq_start = k * sequence_length
                    seq_mid = k * sequence_length + train_cfg.H
                    seq_end = (k + 1) * sequence_length
                    plt.plot(timesteps[seq_start:seq_mid], true_smooth[i, seq_start:seq_mid, label], lw=1.2, c=plots.tab_colours[0], label="True value (smooth)" if k == 0 else None)
                    plt.plot(timesteps[seq_mid:seq_end], true_smooth[i, seq_mid:seq_end, label], lw=1.2, c=plots.tab_colours[0])

                    if pred is not None and plot_preds:
                        for l in range(train_cfg.num_models):
                            plt.plot(
                                timesteps[seq_mid:seq_end],
                                pred[i, l, seq_mid:seq_end, label],
                                alpha=0.7,
                                color="grey",
                                label="Individual predictions" if k == l == 0 else None,
                            )
                        plt.plot(
                            timesteps[seq_mid:seq_end],
                            pred[i, :, seq_mid:seq_end, label].mean(axis=0),
                            lw=1.2,
                            color=plots.tab_colours[1],
                            label="Ensemble" if k == 0 else None,
                        )

                    if pred_opt is not None and plot_preds:
                        plt.plot(
                            timesteps[seq_mid:seq_end],
                            pred_opt[i, seq_mid:seq_end, label],
                            lw=1.2,
                            color=plots.tab_colours[2],
                            label="Ensemble (opt)" if k == 0 else None,
                        )

                    if pred_ref is not None and plot_preds:
                        plt.plot(
                            timesteps[seq_mid:seq_end],
                            pred_ref[i, seq_mid:seq_end, label],
                            lw=1.2,
                            color="red",
                            label="Ensemble (ref)" if k == 0 else None,
                        )

                plt.xlabel("Time [h]")
                plt.ylabel(env.format_label(label))
                plt.legend()

                plt.grid()

            plt.suptitle(
                f"Forward - {os.path.basename(metadatas[i].raw_file) if metadatas[i].raw_file is not None else 'Simulation'} ({metadatas[i].date.year})",
                fontsize="xx-large",
            )

def plot_forward_for_report(
    path: str,
    env: Type[Environment],
    train_cfg: TrainConfig,
    train_results: TrainResults,
    forward_cfg: ForwardConfig,
    metadatas: list[Metadata],
    X_true: np.ndarray,
    U_true: np.ndarray,
    S_true: np.ndarray,
    X_true_smooth: np.ndarray,
    U_true_smooth: np.ndarray,
    X_pred: Optional[np.ndarray],
    U_pred: Optional[np.ndarray],
    U_pred_opt: Optional[np.ndarray],
    U_pred_ref: Optional[np.ndarray],
    R_true: np.ndarray,
):

    sequence_length = train_cfg.H + train_cfg.F
    timesteps = env.dt * np.arange(forward_cfg.num_sequences * sequence_length) / 3600

    width = math.ceil(math.sqrt(len(env.reference_variables) + len(env.predicted_control)))
    # I don't think this works in general because is is not ==, but it is fine for now
    height = math.ceil((len(env.reference_variables) + len(env.predicted_control)) / width)

    def get_next_label():
        i = 0
        while i < len(env.reference_variables) + len(env.predicted_control):
            if i < len(env.reference_variables):
                yield env.reference_variables[i]
            else:
                yield env.predicted_control[i-len(env.reference_variables)]
            i += 1

    for i in range(X_pred.shape[0]):
        log.debug("Target %i / %i" % (i + 1, X_pred.shape[0]))
        label_maker = get_next_label()
        with plots.Figure(os.path.join(path, _plot_folder, "target_%i.pdf" % i),
                          **get_figure_args(height, width, w=12, h=10, fontsize=28, other_rc_params={"lines.linewidth": 2.5}, legend_fontsize=1)):
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
                else:
                    true = U_true
                    pred = U_pred

                if env is FloatZone:
                    for j, phase in enumerate((8, 16, 512)):
                        is_phase = FloatZone.is_phase(phase, S_true).cpu().numpy()
                        plt.plot(timesteps[is_phase[i]], true[i, is_phase[i], label], c=plots.tab_colours[j], label="Observed (%s)" % PHASES[phase])
                else:
                    plt.plot(timesteps, true[i, :, label], label="Observed")

                for k in range(forward_cfg.num_sequences):
                    seq_start = k * sequence_length
                    seq_mid = k * sequence_length + train_cfg.H
                    seq_end = (k + 1) * sequence_length

                    plt.plot(
                        timesteps[seq_mid:seq_end],
                        pred[i, :, seq_mid:seq_end, label].mean(axis=0),
                        color="black",
                        label="Predicted" if k == 0 else None,
                    )

                plt.xlabel("Time [h]")
                plt.title(env.format_label(label))
                plt.legend()

                plt.grid()
