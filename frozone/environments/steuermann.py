import enum

import numpy as np
from pelutils.ds.plots import exp_moving_avg
from tqdm import tqdm

from frozone.data import CONTROLLER_START
from frozone.environments import Environment
from frozone.environments.steuermann_model.framework import simulate
from frozone.environments.steuermann_model.model.model import f


class Steuermann(Environment):

    dt = 6
    is_simulation = True

    class XLabels(enum.IntEnum):
        PolyDia      = 0
        CrystalDia   = 1
        UpperZone    = 2
        LowerZone    = 3
        FullZone     = 4
        MeltVolume   = 5
        MeltNeckDia  = 6
        PolyAngle    = 7
        CrystalAngle = 8
        FullPolyDia  = 9

    class ULabels(enum.IntEnum):
        GeneratorVoltage = 0
        PolyPullRate     = 1
        CrystalPullRate  = 2

    class SLabels(enum.IntEnum):
        HasStarted = 0

    class ZLabels(enum.IntEnum):
        Time = 0
        MeltingRate = 1
        CrystallizationRate = 2
        TdGeneratorVoltage = 3

    no_reference_variables = [XLabels.PolyDia, XLabels.MeltVolume, XLabels.UpperZone, XLabels.LowerZone,
                              XLabels.CrystalAngle, XLabels.MeltNeckDia, XLabels.PolyAngle, XLabels.FullPolyDia]

    predefined_control = [ULabels.CrystalPullRate]

    S_bin_count = (1, )

    format = {
        ("X", XLabels.PolyDia):          "Poly diameter [mm]",
        ("X", XLabels.CrystalDia):       "Crystal diameter [mm]",
        ("X", XLabels.UpperZone):        "Upper zone height [mm]",
        ("X", XLabels.LowerZone):        "Lower zone height [mm]",
        ("X", XLabels.FullZone):         "Full zone height [mm]",
        ("X", XLabels.MeltVolume):       "Melt volume [cm$^3$]",
        ("X", XLabels.CrystalAngle):     "Crystal angle [deg]",
        ("X", XLabels.MeltNeckDia):      "Melt neck diameter [mm]",
        ("X", XLabels.PolyAngle):        "Poly angle [deg]",
        ("X", XLabels.FullPolyDia):      "Full poly diameter [mm]",
        ("U", ULabels.GeneratorVoltage): "Generator voltage [kV]",
        ("U", ULabels.PolyPullRate):     "Poly pull rate [mm$/$min]",
        ("U", ULabels.CrystalPullRate):  "Crystal pull rate [mm$/$min]",
    }

    _lower = 0.9
    _upper = 1.1

    _pull = 1.5

    control_limits = {
        ULabels.GeneratorVoltage: (1e-6, None),
        ULabels.PolyPullRate:     (1e-6, None),
        ULabels.CrystalPullRate:  (1e-6, None),
    }

    @classmethod
    def sample_init_process_vars(cls, n: int) -> np.ndarray:
        X = super().sample_init_process_vars(n)
        X[:, cls.XLabels.PolyDia] = np.random.uniform(20 * cls._lower, 20 * cls._upper, n)
        X[:, cls.XLabels.CrystalDia] = X[:, cls.XLabels.PolyDia]
        X[:, cls.XLabels.UpperZone] = np.random.uniform(7 * cls._lower, 7 * cls._upper, n)
        X[:, cls.XLabels.LowerZone] = np.random.uniform(12 * cls._lower, 12 * cls._upper, n)
        X[:, cls.XLabels.FullZone] = X[:, cls.XLabels.UpperZone] + X[:, cls.XLabels.LowerZone]
        X[:, cls.XLabels.MeltVolume] = np.random.uniform(65 * cls._lower, 65 * cls._upper, n)
        X[:, cls.XLabels.MeltNeckDia] = np.random.uniform(20 * cls._lower, 20 * cls._upper, n)
        X[:, cls.XLabels.PolyAngle] = 45
        X[:, cls.XLabels.CrystalAngle] = 45
        X[:, cls.XLabels.FullPolyDia] = 152.4
        return X

    @classmethod
    def sample_init_control_vars(cls, n: int) -> np.ndarray:
        U = np.empty((n, len(cls.ULabels)), dtype=cls.U_dtype)
        U[:, cls.ULabels.CrystalPullRate] = np.random.uniform(cls._pull * cls._lower, cls._pull * cls._upper, n)
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
    def simulate(cls, n: int, timesteps: int, with_tqdm=True, tqdm_position=0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        X = np.empty((n, timesteps, len(cls.XLabels)), dtype=cls.X_dtype)
        X[:, 0] = cls.sample_init_process_vars(n)
        X[..., cls.XLabels.PolyAngle] = np.vstack(timesteps * [X[:, 0, cls.XLabels.PolyAngle]]).T

        U = np.empty((n, timesteps, len(cls.ULabels)), dtype=cls.U_dtype)
        U[:, 0] = cls.sample_init_control_vars(n)
        U[..., cls.ULabels.CrystalPullRate] = np.vstack(timesteps * [U[:, 0, cls.ULabels.CrystalPullRate]]).T
        timevector = 60 * np.array([0,   40,  80,  120, 160, cls.dt * timesteps / 60])
        try:
            include_up_to = np.where(timevector > cls.dt * timesteps)[0][0]
        except IndexError:
            include_up_to = len(timevector)
        timevector = timevector[:include_up_to]
        gv = np.array([3,   4,   4.7, 5,   5,  5])[:include_up_to]
        ppr = np.array([1.5, 1.3, 2.5, 2.7, 2.7, 2.7])[:include_up_to]
        for i in range(n):
            x = np.linspace(0, timevector[-1], timesteps)
            U[i, :, cls.ULabels.GeneratorVoltage] = np.interp(x, timevector, gv * np.random.uniform(0.95, 1.05, include_up_to))
            U[i, :, cls.ULabels.PolyPullRate] = np.interp(x, timevector, ppr * np.random.uniform(0.95, 1.05, include_up_to))

            for uvar in cls.ULabels:
                alpha = 0.01
                U[i, :, uvar] = 0.5 * exp_moving_avg(U[i, :, uvar], alpha=alpha)[1] \
                              + 0.5 * exp_moving_avg(U[i, :, uvar], alpha=alpha, reverse=True)[1]

        U[..., cls.ULabels.PolyPullRate] *= np.random.uniform(0.95, 1.05)

        Z = np.empty((n, timesteps, len(cls.ZLabels)), dtype=cls.X_dtype)
        Z[:, 0] = cls.init_hidden_vars(U[:, 0])

        S = np.empty((n, timesteps, sum(cls.S_bin_count)), dtype=cls.S_dtype)

        constant_control_from = np.random.uniform(0.65, 0.8, n) * timesteps

        for i in tqdm(range(timesteps-1), disable=not with_tqdm, position=tqdm_position):
            X[:, i + 1], S[:, i + 1], Z[:, i + 1] = cls.forward(
                X[:, i], U[:, i],
                S[:, i], Z[:, i],
            )
            for j in range(n):
                if i >= constant_control_from[j]:
                    U[:, i + 1] = U[:, i]

        R = cls.get_reference_values(X)

        return X, U, S, R, Z

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

            Xnew[i, cls.XLabels.PolyDia]      = x_pred[0] * 2
            Xnew[i, cls.XLabels.CrystalDia]   = x_pred[1] * 2
            Xnew[i, cls.XLabels.UpperZone]    = x_pred[2]
            Xnew[i, cls.XLabels.LowerZone]    = x_pred[3]
            Xnew[i, cls.XLabels.FullZone]     = Xnew[i, cls.XLabels.UpperZone] + Xnew[i, cls.XLabels.LowerZone]
            Xnew[i, cls.XLabels.MeltVolume]   = x_pred[4]
            Xnew[i, cls.XLabels.MeltNeckDia]  = x_pred[9] * 2
            Xnew[i, cls.XLabels.PolyAngle]    = X[i, cls.XLabels.PolyAngle] if Xnew[i, cls.XLabels.PolyDia] < X[i, cls.XLabels.FullPolyDia] else 0
            Xnew[i, cls.XLabels.CrystalAngle] = x_pred[7] * 180 / np.pi
            Xnew[i, cls.XLabels.FullPolyDia]  = X[i, cls.XLabels.FullPolyDia]

            Znew[i, cls.ZLabels.Time] = Z[i, cls.ZLabels.Time] + cls.dt
            Znew[i, cls.ZLabels.MeltingRate] = x_pred[5]
            Znew[i, cls.ZLabels.CrystallizationRate] = x_pred[6]
            Znew[i, cls.ZLabels.TdGeneratorVoltage] = x_pred[8]

        return Xnew, S.copy(), Znew

    @classmethod
    def get_reference_values(cls, X: np.ndarray) -> np.ndarray:
        """ X should have shape n x timesteps x process variables. """
        num_sections = 5
        R = np.empty_like(X)[..., :len(cls.reference_variables)]

        controller_start_index = int(CONTROLLER_START // cls.dt)
        index = np.linspace(controller_start_index, X.shape[-2], num_sections + 1, dtype=int)
        for i, ref_var in enumerate(cls.reference_variables):
            for i_start, i_stop in zip(index[:-1], index[1:], strict=True):
                values = X[:, i_start:i_stop+1, ref_var]
                a = (values[:, -1] - values[:, 0]) / (i_stop - i_start)
                b = values[:, 0] - a * i_start
                R[:, i_start:i_stop, i] = (np.outer(a, np.arange(i_start, i_stop)).T + b).T

            constant_from = int(0.65 * X.shape[-2])
            for j in range(R[:, constant_from:].shape[1]):
                R[:, j + constant_from] = R[:, constant_from]
            for j in range(controller_start_index):
                R[:, j] = R[:, controller_start_index]

        crys_dia_var = cls.reference_variables.index(cls.XLabels.CrystalDia)
        for i in range(X.shape[0]):
            target = 1 * 203.2  # 8 inch target crystal. Change as necessary.
            R[i, :, crys_dia_var] = R[i, :, crys_dia_var] * target / R[i, -1, crys_dia_var]

        return R
