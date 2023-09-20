import random

import numpy as np
from pelutils import TickTock

from frozone.train import TrainConfig


def linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    assert x.shape == y.shape, "Shape of x, %s, does not match shape of y, %s" % (x.shape, y.shape)
    Sx = x.sum()
    Sy = y.sum()
    Sxy = (x * y).sum()
    Sxx = (x * x).sum()
    n = len(x)
    b = (n * Sxy - Sx * Sy) / (n * Sxx - Sx ** 2)
    a = (Sy - b * Sx) / n
    return a, b

def running_avg(x: np.ndarray, neighbours: int) -> np.ndarray:
    padded = np.pad(x, neighbours, "edge")
    kernel = 1 + np.arange(2 * neighbours + 1)
    kernel[2 * neighbours - neighbours:] = kernel[:neighbours][::-1]
    kernel = kernel / kernel.sum()
    return np.convolve(padded, kernel, "valid")

def augment(X: np.ndarray, U: np.ndarray, train_cfg: TrainConfig, tt: TickTock):

    for feature in train_cfg.get_env().XLabels:
        if random.random() > train_cfg.augment_prob:
            continue

        n = train_cfg.H + train_cfg.F
        timesteps = np.arange(n)
        x = np.ascontiguousarray(X[:, feature])

        with tt.profile("Linear regression"):
            a, b = linear_regression(timesteps, x)
            x_linear = a * timesteps + b
        with tt.profile("Residuals and std."):
            residuals = x - x_linear
            std = residuals.std(ddof=1)

        with tt.profile("White noise"):
            white_noise = np.random.uniform(0, train_cfg.epsilon) * std * np.random.randn(n)
            x += white_noise

        X[:, feature] = x