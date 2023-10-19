import enum

import numpy as np

from frozone.data import PHASES, PHASE_TO_INDEX
from frozone.environments import Environment


class FloatZone(Environment):

    dt = 6
    is_simulation = False

    class XLabels(enum.IntEnum):
        PolyDia         = 0
        CrystalDia      = 1
        UpperZone       = 2
        LowerZone       = 3
        FullZone        = 4
        MeltVolume      = 5
        PolyAngle       = 6
        CrystalAngle    = 7
        # CrysAngleLeft   = 7
        # CrysAngleRight  = 8
        # MeltNeck        = 9
        # GrowthLine      = 10
        # PosPoly         = 11
        # PosCrys         = 12

    class ULabels(enum.IntEnum):
        GeneratorVoltage = 0
        PolyPullRate     = 1
        CrystalPullRate  = 2
        # PolyRotation    = 3
        # CrysRotation    = 3
        # CoilPosition    = 3

    class SLabels(enum.IntEnum):
        GrowthState     = 0
        Machine         = 1

    S_bin_count = (len(PHASE_TO_INDEX), 12)

    no_reference_variables = (XLabels.PolyDia, XLabels.UpperZone, XLabels.LowerZone,
                              XLabels.MeltVolume, XLabels.PolyAngle, XLabels.CrystalAngle)

    predefined_control: tuple[ULabels] = (ULabels.CrystalPullRate, )

    units = {
        ("X", XLabels.PolyDia): "mm",
        ("X", XLabels.CrystalDia): "mm",
        ("X", XLabels.UpperZone): "mm",
        ("X", XLabels.LowerZone): "mm",
        ("X", XLabels.FullZone): "mm",
        ("X", XLabels.MeltVolume): "cm$^3$",
        ("X", XLabels.PolyAngle): "deg",
        ("X", XLabels.CrystalAngle): "deg",
        ("U", ULabels.GeneratorVoltage): "kV",
        ("U", ULabels.PolyPullRate): "mm/min",
        ("U", ULabels.CrystalPullRate): "mm/min",
    }

    def is_phase(phase: str | int, S: np.ndarray):
        if isinstance(phase, str):
            phase_num = next(phase_num for phase_num, phase_name in PHASES.items() if phase_name == phase)
        else:
            phase_num = phase
        index = PHASE_TO_INDEX[phase_num]
        return S[..., index] == 1
