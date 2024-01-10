import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from pelutils import TT

from frozone.plot import get_figure_args
from frozone.train import TrainConfig, TrainResults


# Use a non-gui backend. For God knows what reason, using the default TkAgg GUI based backend
# completely breaks when having an asynchronous data loader.
matplotlib.use('Agg')

_plot_folder = "train-plots"

def plot_loss(loc: str, train_cfg: TrainConfig, train_results: TrainResults):
    with TT.profile("Plot loss"), plots.Figure(
        os.path.join(loc, _plot_folder, "loss.pdf"),
        **get_figure_args(w=14, h=9, fontsize=24),
    ):

        train_loss_x = np.array(train_results.train_loss_x)
        train_loss_u = np.array(train_results.train_loss_u)

        test_loss_x = np.array(train_results.test_loss_x)
        test_loss_u = np.array(train_results.test_loss_u)

        for i, loss in enumerate(test_loss_x):
            plt.plot(train_results.checkpoints, loss, "-o", color="dodgerblue", lw=1.5, label="Test loss $X$" if i == 0 else None)

        for i, loss in enumerate(test_loss_u):
            plt.plot(train_results.checkpoints, loss, "-o", color="tab:orange", lw=1.5, label="Test loss $U$" if i == 0 else None)

        plt.plot(*plots.moving_avg(train_loss_x.mean(axis=0), neighbors=12), color="black", lw=3.5)
        plt.plot(*plots.moving_avg(train_loss_x.mean(axis=0), neighbors=12), label="Train loss $X$", color="deepskyblue", lw=2.2)
        plt.plot(*plots.moving_avg(train_loss_u.mean(axis=0), neighbors=12), color="black", lw=3.5)
        plt.plot(*plots.moving_avg(train_loss_u.mean(axis=0), neighbors=12), label="Train loss $U$", color="coral", lw=2.2)

        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.gca().yaxis.set_minor_locator(LogLocator(10, (2, 3, 5, 7)))
        plt.gca().yaxis.set_minor_formatter(LogFormatterSciNotation(10, minor_thresholds=(10, 10)))
        plt.legend(loc=1)
        plt.grid(visible=True, which="major")
        plt.grid(visible=True, which="minor", ls="--")

def plot_lr(loc: str, train_cfg: TrainConfig, train_results: TrainResults):
    with TT.profile("Plot learning rate"), plots.Figure(os.path.join(loc, _plot_folder, "lr.pdf")):
        plt.plot(train_results.lr)

        plt.xlabel("Batch")
        plt.ylabel("Learning rate")

        plt.grid()

if __name__ == "__main__":
    path = sys.argv[1]

    train_cfg = TrainConfig.load(path)
    train_res = TrainResults.load(path)

    plot_loss(path, train_cfg, train_res)
    plot_lr(path, train_cfg, train_res)
