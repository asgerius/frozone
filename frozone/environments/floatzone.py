import enum

import numpy as np
from tqdm import tqdm

from frozone.data import PHASES, PHASE_TO_INDEX
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
    def init_hidden_vars(cls, U: np.ndarray) -> np.ndarray:
        Z = np.empty((len(U), len(cls.ZLabels)), dtype=cls.X_dtype)
        Z[:, cls.ZLabels.Time] = 0
        Z[:, cls.ZLabels.MeltingRate] = U[:, cls.ULabels.PolyPullRate] / 60
        Z[:, cls.ZLabels.CrystallizationRate] = U[:, cls.ULabels.CrystalPullRate] / 60
        Z[:, cls.ZLabels.TdGeneratorVoltage] = U[:, cls.ULabels.GeneratorVoltage]
        return Z

    @classmethod
    def simulate(cls, n: int, timesteps: int, with_tqdm=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        X = np.empty((n, timesteps, len(cls.XLabels)), dtype=cls.X_dtype)
        X[:, 0] = cls.sample_init_process_vars(n)

        U = np.empty((n, timesteps, len(cls.ULabels)), dtype=cls.U_dtype)
        U[:, 0] = cls.sample_init_control_vars(n)
        U = np.stack([U[:, 0]] * timesteps, axis=1)

        Z = np.empty((n, timesteps, len(cls.ZLabels)), dtype=cls.X_dtype)
        Z[:, 0] = cls.init_hidden_vars(U[:, 0])

        S = np.empty((n, timesteps, sum(cls.S_bin_count)), dtype=cls.S_dtype)

        for j in range(2):
            start = j * 4800
            response_iters = np.random.randint(int((start + 3 * 60) // cls.dt), int((start + 10 * 60) // cls.dt), n)

            response_vars = np.random.randint(0, len(cls.ULabels), n)
            response = np.random.uniform(0.92, 1.08, n)
            for i in range(n):
                if response_iters[i] < timesteps:
                    U[i, response_iters[i]:, response_vars[i]] = response[i] * U[i, 0, response_vars[i]]

        for i in tqdm(range(timesteps-1), disable=not with_tqdm):
            X[:, i + 1], S[:, i + 1], Z[:, i + 1] = cls.forward(
                X[:, i], U[:, i],
                S[:, i], Z[:, i],
            )

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

        return Xnew, S.copy(), Znew

class FloatZone(Environment):

    dt = 6

    class XLabels(enum.IntEnum):
        PolyDia         = 0
        CrystalDia      = 1
        UpperZone       = 2
        LowerZone       = 3
        FullZone        = 4
        MeltVolume      = 5
        MeltNeckDia     = 6
        PolyAngle       = 7
        CrystalAngle    = 8
        FullPolyDia     = 9
        GeneratorHeight = 10

    class ULabels(enum.IntEnum):
        GeneratorVoltage = 0
        PolyPullRate     = 1
        CrystalPullRate  = 2

    class SLabels(enum.IntEnum):
        GrowthState   = 0
        Machine       = 1
        SimulatedData = 2

    # Yet another instance of horrible, no good code
    ZLabels = Steuermann.ZLabels

    S_bin_count = (len(PHASE_TO_INDEX), 12, 1)

    no_reference_variables = [XLabels.PolyDia, XLabels.UpperZone, XLabels.LowerZone,
                              XLabels.MeltVolume, XLabels.MeltNeckDia, XLabels.PolyAngle,
                              XLabels.CrystalAngle, XLabels.FullPolyDia, XLabels.GeneratorHeight]

    predefined_control = [ULabels.CrystalPullRate]

    units = {
        ("X", XLabels.PolyDia): "mm",
        ("X", XLabels.CrystalDia): "mm",
        ("X", XLabels.UpperZone): "mm",
        ("X", XLabels.LowerZone): "mm",
        ("X", XLabels.FullZone): "mm",
        ("X", XLabels.MeltVolume): "cm$^3$",
        ("X", XLabels.MeltNeckDia): "mm",
        ("X", XLabels.PolyAngle): "deg",
        ("X", XLabels.CrystalAngle): "deg",
        ("X", XLabels.FullPolyDia): "mm",
        ("X", XLabels.GeneratorHeight): "mm",
        ("U", ULabels.GeneratorVoltage): "kV",
        ("U", ULabels.PolyPullRate): "mm/min",
        ("U", ULabels.CrystalPullRate): "mm/min",
    }

    @classmethod
    def init_hidden_vars(cls, U: np.ndarray) -> np.ndarray:
        return Steuermann.init_hidden_vars(cls.U_floatzone_to_steuermann(U))

    @classmethod
    def U_floatzone_to_steuermann(cls, U: np.ndarray) -> np.ndarray:
        U_st = np.empty((len(U), len(Steuermann.ULabels)), dtype=Steuermann.U_dtype)
        U_st[..., Steuermann.ULabels.GeneratorVoltage] = U[..., cls.ULabels.GeneratorVoltage]
        U_st[..., Steuermann.ULabels.PolyPullRate]     = U[..., cls.ULabels.PolyPullRate]
        U_st[..., Steuermann.ULabels.CrystalPullRate]  = U[..., cls.ULabels.CrystalPullRate]
        return U_st

    @classmethod
    def forward(cls, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        assert n == len(U) == len(S) == len(Z), "Different lengths of input: X=%i, U=%i, S=%i, Z=%i" % (len(X), len(U), len(S), len(Z))

        Xnew = X.copy()
        Znew = np.empty_like(Z).copy()

        X_st = np.empty((n, len(Steuermann.XLabels)), dtype=Steuermann.X_dtype)
        U_st = cls.U_floatzone_to_steuermann(U)
        Z_st = Z.copy()

        X_st[:, Steuermann.XLabels.PolyDia]      = X[:, cls.XLabels.PolyDia]
        X_st[:, Steuermann.XLabels.CrystalDia]   = X[:, cls.XLabels.CrystalDia]
        X_st[:, Steuermann.XLabels.UpperZone]    = X[:, cls.XLabels.UpperZone]
        X_st[:, Steuermann.XLabels.LowerZone]    = X[:, cls.XLabels.LowerZone]
        X_st[:, Steuermann.XLabels.MeltVolume]   = X[:, cls.XLabels.MeltVolume]
        X_st[:, Steuermann.XLabels.MeltNeckDia]  = X[:, cls.XLabels.MeltNeckDia]
        X_st[:, Steuermann.XLabels.PolyAngle]    = X[:, cls.XLabels.PolyAngle]
        X_st[:, Steuermann.XLabels.CrystalAngle] = X[:, cls.XLabels.CrystalAngle]

        Z_st[:, Steuermann.ZLabels.Time]                = Z[:, cls.ZLabels.Time]
        Z_st[:, Steuermann.ZLabels.MeltingRate]         = Z[:, cls.ZLabels.MeltingRate]
        Z_st[:, Steuermann.ZLabels.CrystallizationRate] = Z[:, cls.ZLabels.CrystallizationRate]
        Z_st[:, Steuermann.ZLabels.TdGeneratorVoltage]  = Z[:, cls.ZLabels.TdGeneratorVoltage]

        X_st, S_st, Z_st = Steuermann.forward(X_st, U_st, S, Z_st)

        Xnew[:, FloatZone.XLabels.PolyDia]      = X_st[:, Steuermann.XLabels.PolyDia]
        Xnew[:, FloatZone.XLabels.CrystalDia]   = X_st[:, Steuermann.XLabels.CrystalDia]
        Xnew[:, FloatZone.XLabels.UpperZone]    = X_st[:, Steuermann.XLabels.UpperZone]
        Xnew[:, FloatZone.XLabels.LowerZone]    = X_st[:, Steuermann.XLabels.LowerZone]
        full_zone = X_st[:, Steuermann.XLabels.UpperZone] + X_st[:, Steuermann.XLabels.LowerZone] + X[:, cls.XLabels.GeneratorHeight]
        Xnew[:, FloatZone.XLabels.FullZone]     = full_zone
        Xnew[:, FloatZone.XLabels.MeltVolume]   = X_st[:, Steuermann.XLabels.MeltVolume]
        Xnew[:, FloatZone.XLabels.MeltNeckDia]  = X_st[:, Steuermann.XLabels.MeltNeckDia]
        Xnew[:, FloatZone.XLabels.PolyAngle]    = X_st[:, Steuermann.XLabels.PolyAngle]
        Xnew[:, FloatZone.XLabels.CrystalAngle] = X_st[:, Steuermann.XLabels.CrystalAngle]

        Znew[:, Steuermann.ZLabels.Time]                = Z_st[:, Steuermann.ZLabels.Time]
        Znew[:, Steuermann.ZLabels.MeltingRate]         = Z_st[:, Steuermann.ZLabels.MeltingRate]
        Znew[:, Steuermann.ZLabels.CrystallizationRate] = Z_st[:, Steuermann.ZLabels.CrystallizationRate]
        Znew[:, Steuermann.ZLabels.TdGeneratorVoltage]  = Z_st[:, Steuermann.ZLabels.TdGeneratorVoltage]

        return Xnew, S.copy(), Znew

    def is_phase(phase: str | int, S: np.ndarray):
        if isinstance(phase, str):
            phase_num = next(phase_num for phase_num, phase_name in PHASES.items() if phase_name == phase)
        else:
            phase_num = phase
        index = PHASE_TO_INDEX[phase_num]
        return S[..., index] == 1
