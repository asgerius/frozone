import numpy as np

from frozone.environments.steuermann_model.model.system import System


np.set_printoptions(precision=4, linewidth=200)

def rungekutta(tstart, tend, h, f, x, **kwargs):
    """
    Calculates runge kutta from t start to t end.

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
    :param x: initial state
    :param f: system of differential equations
    :param u: insput sequence
    :return: Future state prediction
    """

    sys = System()

    x_pred = rungekutta(t0, t1, (t1 - t0) / 20, f, x=x, u=u, **kwargs)

    sys.set_within_limits(x_pred)

    return x_pred
