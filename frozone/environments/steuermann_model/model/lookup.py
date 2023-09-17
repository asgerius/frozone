

class Lookup:
    """
    Create a lookup table with linear interpolation
    """
    def __init__(self, x, y):
        if not type(x) == list:
            raise AttributeError
        if not type(y) == list:
            raise AttributeError
        self.x = x
        self.y = y

    def GetValue(self, value) -> float:
        """

        :param value: input value for interpolation
        :return: value generated from interpolation
        """

        if value <= self.x[0]:
            return self.y[0]
        elif value >= self.x[-1]:
            return self.y[-1]
        else:
            for i in range(len(self.x)-1):
                if value == self.x[i]:
                    return self.y[i]
                elif self.x[i] < value < self.x[i + 1]:
                    x1 = self.x[i]
                    x2 = self.x[i+1]
                    y1 = self.y[i]
                    y2 = self.y[i+1]
                    m = (y2-y1)/(x2-x1)
                    h = float(y1 + m*(value-x1))
                    return h
