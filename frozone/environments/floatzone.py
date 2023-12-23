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
        FullPolyDia     = 8

    class ULabels(enum.IntEnum):
        GeneratorVoltage = 0
        PolyPullRate     = 1
        CrystalPullRate  = 2

    class SLabels(enum.IntEnum):
        GrowthState   = 0
        Machine       = 1

    S_bin_count = (len(PHASE_TO_INDEX), 12)

    no_reference_variables = [XLabels.PolyDia, XLabels.UpperZone, XLabels.LowerZone,
                              XLabels.MeltVolume, XLabels.PolyAngle,
                              XLabels.CrystalAngle, XLabels.FullPolyDia]

    predefined_control = [ULabels.CrystalPullRate]

    format = {
        ("X", XLabels.PolyDia):          "Poly diameter [mm]",
        ("X", XLabels.CrystalDia):       "Crystal diameter [mm]",
        ("X", XLabels.UpperZone):        "Upper zone height [mm]",
        ("X", XLabels.LowerZone):        "Lower zone height [mm]",
        ("X", XLabels.FullZone):         "Full zone height [mm]",
        ("X", XLabels.MeltVolume):       "Melt volume [cm$^3$]",
        ("X", XLabels.PolyAngle):        "Poly angle [deg]",
        ("X", XLabels.CrystalAngle):     "Crystal angle [deg]",
        ("X", XLabels.FullPolyDia):      "Full poly diameter [mm]",
        ("U", ULabels.GeneratorVoltage): "Generator voltage [kV]",
        ("U", ULabels.PolyPullRate):     "Poly pull rate [mm$/$min]",
        ("U", ULabels.CrystalPullRate):  "Crystal pull rate [mm$/$min]",
    }

    def is_phase(phase: str | int, S: np.ndarray):
        if isinstance(phase, str):
            phase_num = next(phase_num for phase_num, phase_name in PHASES.items() if phase_name == phase)
        else:
            phase_num = phase
        index = PHASE_TO_INDEX[phase_num]
        return S[..., index] == 1
