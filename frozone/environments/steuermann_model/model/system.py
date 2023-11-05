import numpy as np


class ulb:
    """
    Method for declaring upper and lower bound


    Attributes
    ----------
    lower : int
        integer representing lowerbound
    upper : int
        integer representing upperbound
    """

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def iswithin(self, x) -> bool:
        """
        Validate if the param x is with upper and lower limit
        """
        if x is None:
            return 1
        elif self.lower > x:
            return self.lower
        elif self.upper < x:
            return self.upper
        else:
            None


class System:

    def __init__(self):
        self.pi = 3.141592653589793
        self.phase = 'CONE'
        self.limits = np.array([ulb(0.001, 120),  # Rf[mm]
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
                                ])


    def isvalidstate(self, x):

        for i in range(len(x)):
            l = self.limits[i].iswithin(x[i])
            if not l == None:
                x[i] = l
        return x
