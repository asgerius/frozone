import enum

import numpy as np
from tqdm import tqdm

from frozone.environments import Environment
from frozone.environments.steuermann_model.framework import simulate
from frozone.environments.steuermann_model.model.model import f


class Steuermann(Environment):

    dt = 6

    class XLabels(enum.IntEnum):
        PolyDia = 0
        CrystalDia = 1
        UpperZone = 2
        LowerZone = 3
        FullZone = 4
        MeltVolume = 5
        MeltNeckDia = 6
        PolyAngle = 7
        CrystalAngle = 8

    class ULabels(enum.IntEnum):
        GeneratorVoltage = 0
        PolyPullRate = 1
        CrystalPullRate = 2

    class ZLabels(enum.IntEnum):
        Time = 0
        MeltingRate = 1
        CrystallizationRate = 2
        TdGeneratorVoltage = 3

    no_reference_variables = [XLabels.PolyDia, XLabels.MeltVolume, XLabels.UpperZone, XLabels.LowerZone,
                              XLabels.CrystalAngle, XLabels.MeltNeckDia, XLabels.PolyAngle]

    units = {
        ("X", XLabels.PolyDia): "mm",
        ("X", XLabels.CrystalDia): "mm",
        ("X", XLabels.UpperZone): "mm",
        ("X", XLabels.LowerZone): "mm",
        ("X", XLabels.FullZone): "mm",
        ("X", XLabels.MeltVolume): "cm$^3$",
        ("X", XLabels.CrystalAngle): "deg",
        ("X", XLabels.MeltNeckDia): "mm",
        ("X", XLabels.PolyAngle): "deg",
        ("U", ULabels.GeneratorVoltage): "kV",
        ("U", ULabels.PolyPullRate): "mm/min",
        ("U", ULabels.CrystalPullRate): "mm/min",
    }

    _lower = 0.95
    _upper = 1.05

    @classmethod
    def sample_init_process_vars(cls, n: int) -> np.ndarray:
        X = super().sample_init_process_vars(n)
        X[:, cls.XLabels.PolyDia] = np.random.uniform(20 * cls._lower, 20 * cls._upper, n)
        X[:, cls.XLabels.CrystalDia] = X[:, cls.XLabels.PolyDia]
        X[:, cls.XLabels.UpperZone] = np.random.uniform(7 * cls._lower, 7 * cls._upper, n)
        X[:, cls.XLabels.LowerZone] = np.random.uniform(14 * cls._lower, 14 * cls._upper, n)
        X[:, cls.XLabels.FullZone] = X[:, cls.XLabels.UpperZone] + X[:, cls.XLabels.LowerZone]
        X[:, cls.XLabels.MeltVolume] = np.random.uniform(65 * cls._lower, 65 * cls._upper, n)
        X[:, cls.XLabels.MeltNeckDia] = np.random.uniform(20 * cls._lower, 20 * cls._upper, n)
        X[:, cls.XLabels.PolyAngle] = np.random.uniform(44.5, 45.5, n)  # Vary this one less
        X[:, cls.XLabels.CrystalAngle] = np.random.uniform(45.5, 45.5, n)
        return X

    @classmethod
    def sample_init_control_vars(cls, n: int) -> np.ndarray:
        lower = 0.95
        upper = 1.05
        U = np.empty((n, len(cls.ULabels)), dtype=cls.U_dtype)
        U[:, cls.ULabels.GeneratorVoltage] = np.random.uniform(4.5 * cls._lower, 4.5 * cls._upper, n)
        U[:, cls.ULabels.PolyPullRate] = np.random.uniform(1.2 * cls._lower, 1.2 * cls._upper, n)
        U[:, cls.ULabels.CrystalPullRate] = np.random.uniform(2.4 * cls._lower, 2.4 * cls._upper, n)
        return U

    @classmethod
    def init_hidden_vars(cls, U: np.ndarray) -> np.ndarray:
        Z = np.empty((len(U), len(cls.ZLabels)), dtype=cls.X_dtype)
        Z[:, cls.ZLabels.Time] = 0
        Z[:, cls.ZLabels.MeltingRate] = U[:, cls.ULabels.PolyPullRate] / 60
        Z[:, cls.ZLabels.CrystallizationRate] = U[:, cls.ULabels.CrystalPullRate] / 60
        Z[:, cls.ZLabels.TdGeneratorVoltage] = U[:, cls.ULabels.GeneratorVoltage]
        return Z

    @classmethod
    def simulate(cls, n: int, timesteps: int, with_tqdm=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        growth_time = 4800
        growth_iters = int(growth_time / cls.dt)
        init_weight = 1 - np.linspace(0, 1, growth_iters)
        end_weight = 1 - init_weight

        angled_poly_time = 660
        angled_poly_iters = int(angled_poly_time / cls.dt)

        X = np.empty((n, timesteps, len(cls.XLabels)), dtype=cls.X_dtype)
        X[:, 0] = cls.sample_init_process_vars(n)
        X[..., cls.XLabels.PolyAngle] = np.vstack(timesteps * [X[:, 0, cls.XLabels.PolyAngle]]).T
        X[:, angled_poly_iters:, cls.XLabels.PolyAngle] = 0
        X[..., cls.XLabels.PolyDia] = np.vstack(timesteps * [X[:, 0, cls.XLabels.PolyDia]]).T

        U = np.empty((n, timesteps, len(cls.ULabels)), dtype=cls.U_dtype)
        U[:, 0] = cls.sample_init_control_vars(n)
        max_generator_voltage = np.random.uniform(6 * cls._lower, 6 * cls._upper, n)
        U[:, :growth_iters, cls.ULabels.GeneratorVoltage] = np.outer(U[:, 0, cls.ULabels.GeneratorVoltage], init_weight) + np.outer(max_generator_voltage, end_weight)
        U[:, growth_iters:, cls.ULabels.GeneratorVoltage] = np.vstack((timesteps - growth_iters) * [max_generator_voltage]).T
        max_poly_pull_rate = np.random.uniform(2.4 * cls._lower, 2.4 * cls._upper, n)
        U[:, :growth_iters, cls.ULabels.PolyPullRate] = np.outer(U[:, 0, cls.ULabels.PolyPullRate], init_weight) + np.outer(max_poly_pull_rate, end_weight)
        U[:, growth_iters:, cls.ULabels.PolyPullRate] = np.vstack((timesteps - growth_iters) * [max_poly_pull_rate]).T
        U[..., cls.ULabels.CrystalPullRate] = np.vstack(timesteps * [U[:, 0, cls.ULabels.CrystalPullRate]]).T

        Z = np.empty((n, timesteps, len(cls.ZLabels)), dtype=cls.X_dtype)
        Z[:, 0] = cls.init_hidden_vars(U[:, 0])

        S = np.empty((n, timesteps, sum(cls.S_bin_count)), dtype=cls.S_dtype)

        for i in tqdm(range(timesteps-1), disable=not with_tqdm):
            next_poly_angles = X[:, i + 1, cls.XLabels.PolyAngle].copy()
            X[:, i + 1], S[:, i + 1], Z[:, i + 1] = cls.forward(
                X[:, i], U[:, i],
                S[:, i], Z[:, i],
            )
            X[:, i + 1, cls.XLabels.PolyAngle] = next_poly_angles

        return X, U, S, X[..., cls.reference_variables].copy(), Z

    @classmethod
    def forward(cls, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        assert n == len(U) == len(S) == len(Z), "Different lengths of input: X=%i, U=%i, S=%i, Z=%i" % (len(X), len(U), len(S), len(Z))

        Xnew = np.empty_like(X).copy()
        Znew = np.empty_like(Z).copy()

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
                U[i, cls.ULabels.PolyPullRate] / 60,
                U[i, cls.ULabels.CrystalPullRate] / 60,
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
            Xnew[i, cls.XLabels.FullZone] = Xnew[i, cls.XLabels.UpperZone] + Xnew[i, cls.XLabels.LowerZone]
            Xnew[i, cls.XLabels.MeltVolume] = x_pred[4]
            Xnew[i, cls.XLabels.MeltNeckDia] = x_pred[9] * 2
            Xnew[i, cls.XLabels.PolyAngle] = X[i, cls.XLabels.PolyAngle]
            Xnew[i, cls.XLabels.CrystalAngle] = x_pred[7] * 180 / np.pi

            Znew[i, cls.ZLabels.Time] = Z[i, cls.ZLabels.Time] + cls.dt
            Znew[i, cls.ZLabels.MeltingRate] = x_pred[5]
            Znew[i, cls.ZLabels.CrystallizationRate] = x_pred[6]
            Znew[i, cls.ZLabels.TdGeneratorVoltage] = x_pred[8]

        return Xnew, S.copy(), Znew
