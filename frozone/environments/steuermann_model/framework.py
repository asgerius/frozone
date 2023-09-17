"""
Moduls for framework functions
"""


import numpy as np
import matplotlib.pyplot as plt

from frozone.environments.steuermann_model.model.system import System


def rungekutta(tstart, tend, h, f, x, **kwargs):
    """
    Calculates rungekutter from t start to t end.

    :param tend:
    :param tstart:
    :param h:
    :param f:
    :param x:
    :param kwargs: dictionary containing inputs
    :return:
    """

    n = (tend - tstart) / h
    for i in range(1, int(n + 1)):
        k1 = h * f(x, **kwargs)
        k2 = h * f(x + 0.5 * k1, **kwargs)
        k3 = h * f(x + 0.5 * k2, **kwargs)
        k4 = h * f(x + k3, **kwargs)

        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x


def simulate(x, u, t0, t1, f, **kwargs) -> np.ndarray:
    """
    :param x: initialstate
    :param f: system of differential equations
    :param u: insput sequence
    :return: Future state prediction
    """

    sys = System()

    x_pred = rungekutta(t0, t1, (t1 - t0) / 20, f, x=x, u=u, **kwargs)

    sys.set_within_limits(x_pred)

    return x_pred


def plot(x, labels):
    """Generate plot for N-1 Columns in matrix

    :param labels:
    :param x: Matrix where column is the x axis
    """

    print("start plotting")

    time = list(np.arange(0, 60, 1/60)) + [60]

    for i in range(x.shape[1]):
        data = x[:, i]
        plt.plot(time, data)
        plt.grid(True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xlabel("Time [s]")
        plt.ylabel(labels[i])
        plt.xlim(-2, float(max(time)))
        plt.ylim(float(min(data))*0.99, float(max(data)*1.01))
        plt.show()
