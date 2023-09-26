import enum

from frozone.environments import Environment


class FloatZone(Environment):

    dt = 6
    is_simulation = False

    class XLabels(enum.IntEnum):
        PolyDia         = 0
        CrysDia         = 1
        UpperZone       = 2
        LowerZone       = 3
        FullZone        = 4
        MeltVolume      = 5
        PolyAngle       = 6
        CrysAngle       = 7
        # CrysAngleLeft   = 7
        # CrysAngleRight  = 8
        # MeltNeck        = 9
        # GrowthLine      = 10
        # PosPoly         = 11
        # PosCrys         = 12

    class ULabels(enum.IntEnum):
        GenVoltage      = 0
        PolyPullRate    = 1
        CrysPullRate    = 2
        # PolyRotation    = 3
        # CrysRotation    = 3
        # CoilPosition    = 3

    class SLabels(enum.IntEnum):
        # PLCState        = 0
        # GrowthState     = 1
        # DropState       = 2
        Machine         = 0

    S_bin_count = (12, )

    history_only_variables = (XLabels.PolyAngle, )
