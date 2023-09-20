import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
from pelutils import TT

from frozone.train import TrainConfig, TrainResults


# Use a non-gui backend. For God knows what reason, using the default TkAgg GUI based backend
# completely breaks when having an asynchronous data loader.
matplotlib.use('Agg')

_plot_folder = "train-plots"

def plot_loss(loc: str, train_cfg: TrainConfig, train_results: TrainResults):
    with TT.profile("Plot loss"), plots.Figure(os.path.join(loc, _plot_folder, "loss.png")):
        plot_both = 0 < train_cfg.alpha < 1
        plot_dynamics = plot_both or train_cfg.alpha == 0
        plot_control = plot_both or train_cfg.alpha == 1

        train_loss_x = np.array(train_results.train_loss_x)
        train_loss_u = np.array(train_results.train_loss_u)
        train_loss = np.array(train_results.train_loss)

        test_loss_x = np.array(train_results.test_loss_x)
        test_loss_u = np.array(train_results.test_loss_u)
        test_loss = np.array(train_results.test_loss)

        if plot_dynamics:
            plt.plot(train_loss_x.mean(axis=0), color="grey", alpha=0.2)
        if plot_control:
            plt.plot(train_loss_u.mean(axis=0), color="grey", alpha=0.2)
        if plot_both:
            plt.plot(train_loss.mean(axis=0), color="grey", alpha=0.2)
        if plot_dynamics:
            plt.plot(*plots.moving_avg(train_loss_x.mean(axis=0), neighbors=12), label="Train loss $X$", color=plots.tab_colours[0], alpha=0.8)
        if plot_control:
            plt.plot(*plots.moving_avg(train_loss_u.mean(axis=0), neighbors=12), label="Train loss $U$", color=plots.tab_colours[1], alpha=0.8)
        if plot_both:
            plt.plot(*plots.moving_avg(train_loss.mean(axis=0), neighbors=12), label="Train loss", color=plots.tab_colours[2], alpha=0.8)

        if plot_dynamics:
            plt.plot(train_results.checkpoints, test_loss_x.mean(axis=0), color="black", lw=2.7)
            plt.plot(train_results.checkpoints, test_loss_x.mean(axis=0), color=plots.tab_colours[0], lw=1.5)
            for i, loss in enumerate(test_loss_x):
                plt.scatter(train_results.checkpoints, loss, color="black", s=64)
                plt.scatter(train_results.checkpoints, loss, color=plots.tab_colours[0], s=36, label="Test loss $X$" if i == 0 else None)
            plt.plot(train_results.checkpoints, train_results.ensemble_loss_x, "-o", color="black", lw=2.7, ms=8)
            plt.plot(train_results.checkpoints, train_results.ensemble_loss_x, "--o", color=plots.tab_colours[0], lw=1.5, ms=6, label="Ensemble loss $X$")
        if plot_control:
            plt.plot(train_results.checkpoints, test_loss_u.mean(axis=0), color="black", lw=2.7)
            plt.plot(train_results.checkpoints, test_loss_u.mean(axis=0), color=plots.tab_colours[0], lw=1.5)
            for loss in test_loss_u:
                plt.scatter(train_results.checkpoints, loss, color="black", s=64)
                plt.scatter(train_results.checkpoints, loss, color=plots.tab_colours[1], s=36, label="Test loss $U$")
        if plot_both:
            plt.plot(train_results.checkpoints, test_loss.mean(axis=0), color="black", lw=2.7)
            plt.plot(train_results.checkpoints, test_loss.mean(axis=0), color=plots.tab_colours[2], lw=1.5)
            for loss in test_loss:
                plt.scatter(train_results.checkpoints, loss, color="black", s=64)
                plt.scatter(train_results.checkpoints, loss, color=plots.tab_colours[2], s=36, label="Test loss")

        plt.title(train_cfg.env)
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.yscale("log")

        plt.legend(loc=1)
        plt.grid()

def plot_lr(loc: str, train_cfg: TrainConfig, train_results: TrainResults):
    with TT.profile("Plot learning rate"), plots.Figure(os.path.join(loc, _plot_folder, "lr.png")):
        plt.plot(train_results.lr)

        plt.xlabel("Batch")
        plt.ylabel("Learning rate")

        plt.grid()
