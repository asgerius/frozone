import os

import matplotlib.pyplot as plt
import pelutils.ds.plots as plots
from pelutils import TT

from frozone.train import TrainConfig, TrainResults


_save_folder = "train-plots"

def plot_loss(loc: str, train_cfg: TrainConfig, train_results: TrainResults):
    with TT.profile("Plot loss"), plots.Figure(os.path.join(loc, _save_folder, "loss.png")):
        plot_both = 0 < train_cfg.alpha < 1
        plot_dynamics = plot_both or train_cfg.alpha == 0
        plot_control = plot_both or train_cfg.alpha == 1

        if plot_dynamics:
            plt.plot(train_results.train_loss_x, color="grey", alpha=0.2)
        if plot_control:
            plt.plot(train_results.train_loss_u, color="grey", alpha=0.2)
        if plot_both:
            plt.plot(train_results.train_loss, color="grey", alpha=0.2)
        if plot_dynamics:
            plt.plot(*plots.moving_avg(train_results.train_loss_x, neighbors=12), label="Train loss $X$", color=plots.tab_colours[0], alpha=0.8)
        if plot_control:
            plt.plot(*plots.moving_avg(train_results.train_loss_u, neighbors=12), label="Train loss $U$", color=plots.tab_colours[1], alpha=0.8)
        if plot_both:
            plt.plot(*plots.moving_avg(train_results.train_loss, neighbors=12), label="Train loss", color=plots.tab_colours[2], alpha=0.8)

        if plot_dynamics:
            plt.plot(train_results.checkpoints, train_results.test_loss_x, "-o", color="black", lw=2.7, ms=8)
        if plot_control:
            plt.plot(train_results.checkpoints, train_results.test_loss_u, "-o", color="black", lw=2.7, ms=8)
        if plot_both:
            plt.plot(train_results.checkpoints, train_results.test_loss, "-o", color="black", lw=2.7, ms=8)
        if plot_dynamics:
            plt.plot(train_results.checkpoints, train_results.test_loss_x, "-o", label="Test loss $X$", color=plots.tab_colours[0], lw=1.5, ms=6)
        if plot_control:
            plt.plot(train_results.checkpoints, train_results.test_loss_u, "-o", label="Test loss $U$", color=plots.tab_colours[1], lw=1.5, ms=6)
        if plot_both:
            plt.plot(train_results.checkpoints, train_results.test_loss, "-o", label="Test loss", color=plots.tab_colours[2], lw=1.5, ms=6)

        plt.title(train_cfg.env)
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.yscale("log")

        plt.legend(loc=1)
        plt.grid()

def plot_lr(loc: str, train_cfg: TrainConfig, train_results: TrainResults):
    with TT.profile("Plot learning rate"), plots.Figure(os.path.join(loc, _save_folder, "lr.png")):
        plt.plot(train_results.lr)

        plt.xlabel("Batch")
        plt.ylabel("Learning rate")

        plt.grid()
