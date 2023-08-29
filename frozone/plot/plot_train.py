import os

import matplotlib.pyplot as plt
import pelutils.ds.plots as plots
from pelutils import TT

from frozone.train import TrainConfig, TrainResults


_save_folder = "train-plots"

def plot_loss(loc: str, train_cfg: TrainConfig, train_results: TrainResults):
    with TT.profile("Plot loss"), plots.Figure(os.path.join(loc, _save_folder, "loss.png")):
        plt.plot(train_results.train_loss_x, label="Train loss $X$")
        plt.plot(train_results.train_loss_u, label="Train loss $U$")
        plt.plot(train_results.train_loss, label="Train loss")

        plt.plot(train_results.checkpoints, train_results.test_loss_x, label="Test loss $X$")
        plt.plot(train_results.checkpoints, train_results.test_loss_u, label="Test loss $U$")
        plt.plot(train_results.checkpoints, train_results.test_loss, label="Test loss")

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
