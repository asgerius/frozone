import enum
import random

import numpy as np
from tqdm import tqdm

from frozone.environments import Environment
from frozone.environments.steuermann_model.framework import simulate
from frozone.environments.steuermann_model.model.model import f


class Steuermann(Environment):

    dt = 6
    is_simulation = True

    class XLabels(enum.IntEnum):
        PolyRadius = 0
        CrystalRadius = 1
        UpperZone = 2
        LowerZone = 3
        MeltVolume = 4
        CrystalAngle = 5
        MeltNeckRadius = 6
        PolyAngle = 7

    class ULabels(enum.IntEnum):
        GeneratorVoltage = 0
        PolyPullRate = 1
        CrystalPullRate = 2

    class ZLabels(enum.IntEnum):
        Time = 0
        MeltingRate = 1
        CrystallizationRate = 2
        TdGeneratorVoltage = 3

    no_reference_variables = (XLabels.PolyAngle, )

    @classmethod
    def sample_init_process_vars(cls, n: int) -> np.ndarray:
        X = super().sample_init_process_vars(n)
        X[:, cls.XLabels.PolyRadius] = 50
        X[:, cls.XLabels.CrystalRadius] = 50
        X[:, cls.XLabels.UpperZone] = 7
        X[:, cls.XLabels.LowerZone] = 14
        X[:, cls.XLabels.MeltVolume] = 65
        X[:, cls.XLabels.CrystalAngle] = 0
        X[:, cls.XLabels.MeltNeckRadius] = 10
        X[:, cls.XLabels.PolyAngle] = 0
        return X

    @classmethod
    def sample_init_control_vars(cls, n: int) -> np.ndarray:
        U = np.empty((n, len(cls.ULabels)), dtype=cls.U_dtype)
        U[:, cls.ULabels.GeneratorVoltage] = 5
        U[:, cls.ULabels.PolyPullRate] = 3.8 / 60
        U[:, cls.ULabels.CrystalPullRate] = 3.8 / 60
        return U

    @classmethod
    def simulate(cls, n: int, iters: int, with_tqdm=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not cls.is_simulation:
            raise NotImplementedError("Environment %s cannot be simulated" % cls.__name__)

        X = np.empty((n, iters + 1, len(cls.XLabels)), dtype=cls.X_dtype)
        X[:, 0] = cls.sample_init_process_vars(n)

        U = np.empty((n, iters + 1, len(cls.ULabels)), dtype=cls.U_dtype)
        U[:, 0] = cls.sample_init_control_vars(n)

        Z = np.empty((n, iters + 1, len(cls.ZLabels)), dtype=cls.X_dtype)
        Z[..., cls.ZLabels.Time] = np.arange(0, iters + 1) * cls.dt
        Z[:, 0, cls.ZLabels.MeltingRate] = U[:, 0, cls.ULabels.PolyPullRate]
        Z[:, 0, cls.ZLabels.CrystallizationRate] = U[:, 0, cls.ULabels.CrystalPullRate]
        Z[:, 0, cls.ZLabels.TdGeneratorVoltage] = U[:, 0, cls.ULabels.GeneratorVoltage]

        S = np.empty((n, iters + 1, sum(cls.S_bin_count)), dtype=cls.S_dtype)

        response_iters = np.random.randint(iters // 15, iters // 4, n)
        response_vars = [random.choice(list(cls.ULabels)).value for _ in range(n)]
        response = np.random.uniform(0.93, 1.05, n)

        for i in tqdm(range(iters)) if with_tqdm else range(iters):
            U[:, i + 1] = U[:, 0]
            for j, response_iter in enumerate(response_iters):
                if i >= response_iter:
                    U[j, i + 1, response_vars[j]] *= response[j]

            X[:, i + 1], S[:, i + 1], Z[:, i + 1] = cls.forward(
                X[:, i], U[:, i],
                S[:, i], Z[:, i],
            )

        return X, U, S, Z

    @classmethod
    def forward(cls, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        assert n == len(U) == len(S) == len(Z), "Different lengths of input: X=%i, U=%i, S=%i, Z=%i" % (len(X), len(U), len(S), len(Z))

        Xnew = X.copy()
        Snew = S.copy()
        Znew = Z.copy()

        for i in range(n):
            x0 = np.array([
                X[i, cls.XLabels.PolyRadius],
                X[i, cls.XLabels.CrystalRadius],
                X[i, cls.XLabels.UpperZone],
                X[i, cls.XLabels.LowerZone],
                X[i, cls.XLabels.MeltVolume],
                Z[i, cls.ZLabels.MeltingRate],
                Z[i, cls.ZLabels.CrystallizationRate],
                X[i, cls.XLabels.CrystalAngle],
                Z[i, cls.ZLabels.TdGeneratorVoltage],
                X[i, cls.XLabels.MeltNeckRadius],
            ])

            x_pred = simulate(x0, U[i], Z[i, cls.ZLabels.Time], Z[i, cls.ZLabels.Time] + cls.dt, f, z=[X[i, cls.XLabels.PolyAngle]])

            Xnew[i, cls.XLabels.PolyRadius] = x_pred[0]
            Xnew[i, cls.XLabels.CrystalRadius] = x_pred[1]
            Xnew[i, cls.XLabels.UpperZone] = x_pred[2]
            Xnew[i, cls.XLabels.LowerZone] = x_pred[3]
            Xnew[i, cls.XLabels.MeltVolume] = x_pred[4]
            Xnew[i, cls.XLabels.CrystalAngle] = x_pred[7]
            Xnew[i, cls.XLabels.MeltNeckRadius] = x_pred[9]

            Znew[i, cls.ZLabels.Time] = Z[i, cls.ZLabels.Time] + cls.dt
            Znew[i, cls.ZLabels.MeltingRate] = x_pred[5]
            Znew[i, cls.ZLabels.CrystallizationRate] = x_pred[6]
            Znew[i, cls.ZLabels.TdGeneratorVoltage] = x_pred[8]

        return Xnew, Snew, Znew
