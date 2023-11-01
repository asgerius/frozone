import numpy as np


class ulb:
    """
    Method for declaring upper and lower bound
    """

    def __init__(self, lower, upper):
        """
        Define upper and lower limit parameters

        Parameter
        ---------
        lower : int
            integer representing lowerbound
        upper : int
            integer representing upperbound

        """
        self.lower = lower
        self.upper = upper

    def iswithin(self, x: float) -> float:
        """
        Validate if the param x is with upper and lower limit
        """
        if x is None:
            return 1
        elif x < self.lower:
            return self.lower
        elif x > self.upper:
            return self.upper
        else:
            return x

class System:

    def __init__(self):
        self.phase = 'CONE'
        self.limits = (
            ulb(0.001, 120),  # Rf[mm]
            ulb(0.001, 120),  # Rc[mm]
            ulb(0.001, 20),  # Hf[mm]
            ulb(0.001, 20),  # Hc[mm]
            ulb(0.0000001, 2000),  # V[cm3]
            ulb(-10, 10),  # vMe[mm/s]
            ulb(-10, 10),  # vGr[mm/s]
            ulb(-np.pi, np.pi),  # crystal angle[rad]
            ulb(0.0001, 15),  # Ud[kV]
            ulb(-60, 60),  # RN[mm]
            ulb(-10 ** 8, 10 ** 8),  # vfd[mm/s]
            ulb(-10 ** 8, 10 ** 8),  # vcd[mm/s]
        )

    def set_within_limits(self, x):
        """
        Checks if all variables is within upper and lower limits

        Parameters
        ----------
        x : list

        Return
        ______
        updated state vector
        """

        for i in range(len(x)):
            x[i] = self.limits[i].iswithin(x[i])
        return x
