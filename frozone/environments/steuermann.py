import enum
import sys

import numpy as np
from tqdm import tqdm

from frozone.environments import Environment
from frozone.environments.steuermann_model.framework import simulate
from frozone.environments.steuermann_model.model.model import f


class Steuermann(Environment):

    dt = 6
    is_simulation = True

    class XLabels(enum.IntEnum):
        PolyDia = 0
        CrystalDia = 1
        UpperZone = 2
        LowerZone = 3
        MeltVolume = 4
        MeltNeckDia = 5
        PolyAngle = 6
        CrystalAngle = 7

    class ULabels(enum.IntEnum):
        GeneratorVoltage = 0
        PolyPullRate = 1
        CrystalPullRate = 2

    class ZLabels(enum.IntEnum):
        Time = 0
        MeltingRate = 1
        CrystallizationRate = 2
        TdGeneratorVoltage = 3

    no_reference_variables = [XLabels.PolyDia, XLabels.MeltVolume,
                              XLabels.CrystalAngle, XLabels.MeltNeckDia, XLabels.PolyAngle]

    _lower = 0.8
    _upper = 1.2
    control_limits = {
        ULabels.GeneratorVoltage: (5 * _lower, 5 * _upper),
        ULabels.PolyPullRate:     (3.8 * _lower, 3.8 * _upper),
        ULabels.CrystalPullRate:  (3.8 * _lower, 3.8 * _upper),
    }

    units = {
        ("X", XLabels.PolyDia): "mm",
        ("X", XLabels.CrystalDia): "mm",
        ("X", XLabels.UpperZone): "mm",
        ("X", XLabels.LowerZone): "mm",
        ("X", XLabels.MeltVolume): "cm$^3$",
        ("X", XLabels.CrystalAngle): "deg",
        ("X", XLabels.MeltNeckDia): "mm",
        ("X", XLabels.PolyAngle): "deg",
        ("U", ULabels.GeneratorVoltage): "kV",
        ("U", ULabels.PolyPullRate): "mm/min",
        ("U", ULabels.CrystalPullRate): "mm/min",
    }

    @classmethod
    def sample_init_process_vars(cls, n: int) -> np.ndarray:
        X = super().sample_init_process_vars(n)
        X[:, cls.XLabels.PolyDia] = 100
        X[:, cls.XLabels.CrystalDia] = 100
        X[:, cls.XLabels.UpperZone] = 7
        X[:, cls.XLabels.LowerZone] = 14
        X[:, cls.XLabels.MeltVolume] = 65
        X[:, cls.XLabels.MeltNeckDia] = 20
        X[:, cls.XLabels.PolyAngle] = 0
        X[:, cls.XLabels.CrystalAngle] = 0
        return X

    @classmethod
    def sample_init_control_vars(cls, n: int) -> np.ndarray:
        U = np.empty((n, len(cls.ULabels)), dtype=cls.U_dtype)
        U[:, cls.ULabels.GeneratorVoltage] = 5
        U[:, cls.ULabels.PolyPullRate] = 3.8
        U[:, cls.ULabels.CrystalPullRate] = 3.8
        return U

    @classmethod
    def simulate(cls, n: int, timesteps: int, with_tqdm=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not cls.is_simulation:
            raise NotImplementedError("Environment %s cannot be simulated" % cls.__name__)

        X = np.empty((n, timesteps, len(cls.XLabels)), dtype=cls.X_dtype)
        X[:, 0] = cls.sample_init_process_vars(n)

        U = np.empty((n, timesteps, len(cls.ULabels)), dtype=cls.U_dtype)
        U[:, 0] = cls.sample_init_control_vars(n)
        U = np.stack([U[:, 0]] * timesteps, axis=1)

        Z = np.empty((n, timesteps, len(cls.ZLabels)), dtype=cls.X_dtype)
        Z[..., cls.ZLabels.Time] = cls.dt * np.arange(timesteps)
        Z[:, 0, cls.ZLabels.MeltingRate] = U[:, 0, cls.ULabels.PolyPullRate] / 60
        Z[:, 0, cls.ZLabels.CrystallizationRate] = U[:, 0, cls.ULabels.CrystalPullRate] / 60
        Z[:, 0, cls.ZLabels.TdGeneratorVoltage] = U[:, 0, cls.ULabels.GeneratorVoltage]

        S = np.empty((n, timesteps, sum(cls.S_bin_count)), dtype=cls.S_dtype)

        for j in range(2):
            start = j * 4800
            response_iters = np.random.randint(int((start + 3 * 60) // cls.dt), int((start + 10 * 60) // cls.dt), n)

            response_vars = np.random.randint(0, len(cls.ULabels), n)
            response = np.random.uniform(0.92, 1.08, n)
            for i in range(n):
                if response_iters[i] < timesteps:
                    U[i, response_iters[i]:, response_vars[i]] = response[i] * U[i, 0, response_vars[i]]

        for i in tqdm(range(timesteps-1), file=sys.stdout, disable=not with_tqdm):
            X[:, i + 1], S[:, i + 1], Z[:, i + 1] = cls.forward(
                X[:, i], U[:, i],
                S[:, i], Z[:, i],
            )

        return X, U, S, X[..., cls.reference_variables].copy(), Z

    @classmethod
    def forward(cls, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        assert n == len(U) == len(S) == len(Z), "Different lengths of input: X=%i, U=%i, S=%i, Z=%i" % (len(X), len(U), len(S), len(Z))

        Xnew = X.copy()
        Snew = S.copy()
        Znew = Z.copy()

        u_convertions = np.array([1, 60, 60])

        for i in range(n):
            x0 = np.array([
                X[i, cls.XLabels.PolyDia] / 2,
                X[i, cls.XLabels.CrystalDia] / 2,
                X[i, cls.XLabels.UpperZone],
                X[i, cls.XLabels.LowerZone],
                X[i, cls.XLabels.MeltVolume],
                Z[i, cls.ZLabels.MeltingRate],
                Z[i, cls.ZLabels.CrystallizationRate],
                X[i, cls.XLabels.CrystalAngle] * np.pi / 180,
                Z[i, cls.ZLabels.TdGeneratorVoltage],
                X[i, cls.XLabels.MeltNeckDia] / 2,
            ])
            x_pred = simulate(
                x0,
                U[i] / u_convertions,
                Z[i, cls.ZLabels.Time],
                Z[i, cls.ZLabels.Time] + cls.dt,
                f,
                z=[X[i, cls.XLabels.PolyAngle] * np.pi / 180],
            )

            Xnew[i, cls.XLabels.PolyDia] = x_pred[0] * 2
            Xnew[i, cls.XLabels.CrystalDia] = x_pred[1] * 2
            Xnew[i, cls.XLabels.UpperZone] = x_pred[2]
            Xnew[i, cls.XLabels.LowerZone] = x_pred[3]
            Xnew[i, cls.XLabels.MeltVolume] = x_pred[4]
            Xnew[i, cls.XLabels.CrystalAngle] = x_pred[7] * 180 / np.pi
            Xnew[i, cls.XLabels.MeltNeckDia] = x_pred[9] * 2

            Znew[i, cls.ZLabels.Time] = Z[i, cls.ZLabels.Time] + cls.dt
            Znew[i, cls.ZLabels.MeltingRate] = x_pred[5]
            Znew[i, cls.ZLabels.CrystallizationRate] = x_pred[6]
            Znew[i, cls.ZLabels.TdGeneratorVoltage] = x_pred[8]

        return Xnew, Snew, Znew
