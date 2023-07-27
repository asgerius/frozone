import os

import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots

from frozone.train import TrainConfig, TrainResults


_save_folder = "train-plots"

def plot_loss(loc: str, train_cfg: TrainConfig, train_results: TrainResults):
    with plots.Figure(os.path.join(loc, _save_folder, "loss.png")):
        plt.plot(train_results.train_loss, label="Train loss")
        plt.plot(train_results.checkpoints, train_results.test_loss, label="Test loss")

        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.yscale("log")

        plt.legend(loc=1)
        plt.grid()
