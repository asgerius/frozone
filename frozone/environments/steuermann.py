import enum
import random
import warnings

import numpy as np
from tqdm import tqdm

from frozone.data import squared_exponential_kernel
from frozone.environments import Environment
from frozone.environments.steuermann_model.framework import simulate
from frozone.environments.steuermann_model.model.model import f


warnings.filterwarnings("error")

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

    _lower = 0.9
    _upper = 1.1
    control_limits = {
        ULabels.GeneratorVoltage: (5 * _lower, 5 * _upper),
        ULabels.PolyPullRate: (3.8 / 60 * _lower, 3.8 / 60 * _upper),
        ULabels.CrystalPullRate: (3.8 / 60 * _lower, 3.8 / 60 * _upper),
    }

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
        U[:, ] = 5
        U[:, cls.ULabels.PolyPullRate] = 3.8 / 60
        U[:, cls.ULabels.CrystalPullRate] = 3.8 / 60
        return U

    @classmethod
    def simulate(cls, n: int, timesteps: int, with_tqdm=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not cls.is_simulation:
            raise NotImplementedError("Environment %s cannot be simulated" % cls.__name__)

        X = np.empty((n, timesteps, len(cls.XLabels)), dtype=cls.X_dtype)
        X[:, 0] = cls.sample_init_process_vars(n)

        U_offsets = {
            cls.ULabels.GeneratorVoltage: 5,
            cls.ULabels.PolyPullRate: 3.8 / 60,
            cls.ULabels.CrystalPullRate: 3.8 / 60,
        }
        U = np.zeros((n, timesteps, len(cls.ULabels)), dtype=cls.U_dtype)
        for i in range(n):
            for ulab, offset in U_offsets.items():
                if random.random() < 0.5:
                    # Sometimes, use constant values
                    U[i, :, ulab] = offset
                else:
                    # Other times, generate smooth data from a Gaussian process
                    num_start = np.random.randint(1, timesteps // 3)
                    l = np.random.uniform(100, 1000)
                    xs = np.arange(num_start, timesteps) * cls.dt
                    x0 = np.array([num_start - 1]) * cls.dt

                    K = squared_exponential_kernel(x0, x0, l)
                    Ks = squared_exponential_kernel(x0, xs, l)
                    Kss = squared_exponential_kernel(xs, xs, l)

                    covs = Kss - Ks.T @ np.linalg.pinv(K) @ Ks

                    U[i, :, ulab] = offset
                    U[i, num_start:, ulab] += 0.04 * offset * np.random.multivariate_normal(np.zeros(len(xs)), covs)

        U = cls.limit_control(U)

        Z = np.empty((n, timesteps, len(cls.ZLabels)), dtype=cls.X_dtype)
        Z[..., cls.ZLabels.Time] = np.arange(0, timesteps) * cls.dt
        Z[:, 0, cls.ZLabels.MeltingRate] = U[:, 0, cls.ULabels.PolyPullRate]
        Z[:, 0, cls.ZLabels.CrystallizationRate] = U[:, 0, cls.ULabels.CrystalPullRate]
        Z[:, 0, cls.ZLabels.TdGeneratorVoltage] = U[:, 0, cls.ULabels.GeneratorVoltage]

        S = np.empty((n, timesteps, sum(cls.S_bin_count)), dtype=cls.S_dtype)

        for i in tqdm(range(timesteps-1)) if with_tqdm else range(timesteps-1):
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
