from __future__ import annotations

import multiprocessing as mp
import os
import random
import shutil
import sys
from glob import glob as glob  # glob
from pprint import pformat
from typing import Optional

import numpy as np
import pandas as pd
from pelutils import split_path, log, thousands_seperators
from tqdm import tqdm

from frozone.data import PHASE_TO_INDEX, DataSequence, PROCESSED_SUBDIR, RAW_SUBDIR, TEST_SUBDIR, TRAIN_SUBDIR, TRAIN_TEST_SPLIT
from frozone.environments import FloatZone


# These files all have something weird going on in them, so they are skipped
# See the logfile produced by data/analysis.log for details
BLACKLIST = {
    "M34/34_Automation_Archive_2018/34-1170_Automation.txt",
    "M34/34_Automation_Archive_2019/34-1720_Automation.txt",
    "M31/31-2466_Automation.txt",
    "M38/38_Automation_Archive_2021/38-1613_Automation.txt",
    "M31/31-2454_Automation.txt",
    "M41/41-2116_Automation.txt",
    "M41/41_Automation_Archive_2020/41-1277_Automation.txt",
    "M31/31_Automation_Archive_2021/31-1886_Automation.txt",
    "M33/33_Automation_Archive_2017/33-0882_Automation.txt",
    "M32/32-1228_Automation.txt",
    "M31/31-0943_Automation.txt",
    "M34/34_Automation_Archive_2018/34-1331_Automation.txt",
    "M32/32-1090_Automation.txt",
    "M43/43_Automation_Archive_2019/43-1176_Automation.txt",
    "M37/37_Automation_Archive_2022/37-0642_Automation.txt",
    "M33/33_Automation_Archive_2022/33-2362_Automation.txt",
    "M34/34_Automation_Archive_2017/34-0893_Automation.txt",
    "M37/37_Automation_Archive_2019/37-0021_Automation.txt",
    "M31/31-2578_Automation.txt",
    "M34/34_Automation_Archive_2019/34-1407_Automation.txt",
    "M36/36_Automation_Archive_2019/36-0962_Automation.txt",
    "M34/34_Automation_Archive_2016/34-0764_Automation.txt",
    "M41/41_Automation_Archive_2022/41-1796_Automation.txt",
    "M31/31-2633_Automation.txt",
    "M34/34_Automation_Archive_2017/34-1007_Automation.txt",
    "M36/36_Automation_Archive_2019/36-1037_Automation.txt",
    "M35/35_Automation_Archive_2022/35-0697_Automation.txt",
    "M36/36_Automation_Archive_2018/36-0523_Automation.txt",
    "M41/41_Automation_Archive_2020/41-1281_Automation.txt",
    "M32/32-2490_Automation.txt",
    "M32/32-1590_Automation.txt",
    "M32/32-1628_Automation.txt",
}
MACHINES = ("M31", "M32", "M33", "M34", "M35", "M36", "M37", "M38", "M41", "M43", "M52", "M54")

def get_phase_slices(fname: str, df: pd.DataFrame) -> dict[int, slice]:
    slices: dict[int, slice] = dict()

    current_phase = df.Growth_State_Act.values[-1]
    end_index = len(df)
    for index in range(len(df) - 2, -1, -1):
        phase = df.Growth_State_Act.values[index]
        if phase < current_phase:
            start_index = index + 1
            slices[current_phase] = slice(start_index, end_index)
            end_index = index + 1
            current_phase = phase
        elif phase > current_phase:
            break

    # Keep only interesting phases
    slices = { phase: slice_ for phase, slice_ in slices.items() if 8 <= phase <= 1024 }
    if len(slices) < 2:
        return dict()

    slices.pop(max(slices))
    assert all(8 <= phase <= 512 for phase in slices), f"{fname}\n{pformat(slices)}"

    sorted_phases = sorted(slices.keys())

    for phase0, phase1 in zip(sorted_phases[:-1], sorted_phases[1:], strict=True):
        slice0 = slices[phase0]
        slice1 = slices[phase1]
        assert slice0.start < slice0.stop, f"{fname}\n{pformat(slices)}"
        assert slice1.start < slice1.stop, f"{fname}\n{pformat(slices)}"
        assert slice0.stop == slice1.start, f"{fname}\n{pformat(slices)}"

    return slices

def parse_floatzone_df(fpath: str, df: pd.DataFrame, machine: str) -> Optional[DataSequence]:

    # Maps growth state to slice
    slices = get_phase_slices(fpath, df)
    if not slices:
        return

    sorted_phases = sorted(slices.keys())
    sorted_slices = [slices[phase] for phase in sorted_phases]
    offset = sorted_slices[0].start
    timesteps = sorted_slices[-1].stop - offset

    X = np.empty((timesteps, len(FloatZone.XLabels)), dtype=FloatZone.X_dtype)
    U = np.empty((timesteps, len(FloatZone.ULabels)), dtype=FloatZone.U_dtype)
    S = np.zeros((timesteps, sum(FloatZone.S_bin_count)), dtype=FloatZone.S_dtype)

    for phase, slice_ in zip(sorted_phases, sorted_slices):
        slice_data = slice(slice_.start - offset, slice_.stop - offset)

        X[slice_data, FloatZone.XLabels.PolyDia]      = df["PolyDia[mm]"].values[slice_]
        X[slice_data, FloatZone.XLabels.CrystalDia]   = df["CrysDia[mm]"].values[slice_]
        X[slice_data, FloatZone.XLabels.UpperZone]    = df["UpperZone[mm]"].values[slice_]
        X[slice_data, FloatZone.XLabels.LowerZone]    = df["LowerZone[mm]"].values[slice_]
        X[slice_data, FloatZone.XLabels.FullZone]     = df["FullZone[mm]"].values[slice_]
        if phase >= 512:
            # From the cone phase and onwards, the zone calculation is changed, so correct for this by adding the introduced offset
            X[slice_data, FloatZone.XLabels.UpperZone] += X[slice_data.start-1, FloatZone.XLabels.UpperZone] - X[slice_data.start, FloatZone.XLabels.UpperZone]
            X[slice_data, FloatZone.XLabels.LowerZone] += X[slice_data.start-1, FloatZone.XLabels.LowerZone] - X[slice_data.start, FloatZone.XLabels.LowerZone]
            X[slice_data, FloatZone.XLabels.FullZone]  += X[slice_data.start-1, FloatZone.XLabels.FullZone] - X[slice_data.start, FloatZone.XLabels.FullZone]
        X[slice_data, FloatZone.XLabels.MeltVolume]   = df["MeltVolume[mm3]"].values[slice_]
        X[slice_data, FloatZone.XLabels.PolyAngle]    = df["PolyAngle[deg]"].values[slice_]
        X[slice_data, FloatZone.XLabels.CrystalAngle] = df["CrysAngle[deg]"].values[slice_]
        # X[:, FloatZone.XLabels.MeltNeck]   = df_used["MeltNeck[mm]"].values
        # X[:, FloatZone.XLabels.GrowthLine] = df_used["GrowthLine[mm]"].values
        # X[:, FloatZone.XLabels.PosPoly]    = df_used["Pos_Poly[mm]"].values
        # X[:, FloatZone.XLabels.PosCrys]    = df_used["Pos_Crys[mm]"].values

        U[slice_data, FloatZone.ULabels.GeneratorVoltage]     = df["GenVoltage[kV]"].values[slice_]
        U[slice_data, FloatZone.ULabels.PolyPullRate]   = df["PolyPullRate[mm/min]"].values[slice_]
        U[slice_data, FloatZone.ULabels.CrystalPullRate]   = df["CrysPullRate[mm/min]"].values[slice_]
        # U[:, FloatZone.ULabels.PolyRotation]   = df["PolyRotation[rpm]"].values
        # U[:, FloatZone.ULabels.CrysRotation]   = df["CrysRotation[rpm]"].values
        # U[:, FloatZone.ULabels.CoilPosition]   = df["CoilPosition[mm]"].values

        S[slice_data, PHASE_TO_INDEX[phase]] = 1
        S[slice_data, len(PHASE_TO_INDEX) + MACHINES.index(machine)] = 1

    assert len(X) == len(U) == len(S), "%i %i %i" % (len(X), len(U), len(S))
    assert np.isnan(X).sum() == 0, f"{fpath}, NaN for X"
    assert np.isnan(U).sum() == 0, f"{fpath}, NaN for U"
    assert np.isnan(S).sum() == 0, f"{fpath}, NaN for S"

    return X, U, S

def process_floatzone_file(args: list[str]) -> str | None:
    raw_data_path, filepath = args
    with log.collect:
        full_path = os.path.join(raw_data_path, filepath)
        file_path_components = split_path(filepath)
        log("Loading %s" % full_path)
        df = pd.read_csv(full_path, quoting=3, delim_whitespace=True)
        machine = file_path_components[0]
        log("Parsing file")
        data = parse_floatzone_df(filepath, df, machine)
        if data is not None:
            X, U, S = data
            train_test_subdir = TRAIN_SUBDIR if random.random() < TRAIN_TEST_SPLIT else TEST_SUBDIR
            outpath = os.path.join(
                os.path.join(raw_data_path, os.path.pardir),
                PROCESSED_SUBDIR,
                train_test_subdir,
                *file_path_components[2:-1],
                os.path.splitext(file_path_components[-1])[0],
            )
            log("Saving file with sequence length %s" % thousands_seperators(len(X)))
            os.makedirs(os.path.split(outpath)[0], exist_ok=True)
            np.savez_compressed(outpath, X=X, U=U, S=S)

def process_floatzone_data(data_path: str, processes=None, max_files=None) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    log.configure(os.path.join(data_path, "process.log"), print_level=None)
    processes = processes or os.cpu_count()

    shutil.rmtree(os.path.join(data_path, PROCESSED_SUBDIR), ignore_errors=True)

    raw_data_path = os.path.join(data_path, RAW_SUBDIR)
    raw_files = glob(os.path.join(raw_data_path, "**", "*.txt"), recursive=True)
    random.shuffle(raw_files)
    raw_files = raw_files[:max_files]
    n_raw_total = len(raw_files)
    args = [(raw_data_path, file[len(raw_data_path)+1:]) for file in raw_files if not any(file.endswith(x) for x in BLACKLIST)]
    if max_files is None:
        assert 0 < n_raw_total == len(args) + len(BLACKLIST), f"{n_raw_total} == {len(args)} + {len(BLACKLIST)}"
    else:
        assert 0 < n_raw_total <= len(args)

    if processes > 1:
        with mp.Pool(processes=processes) as pool:
            tuple(tqdm(pool.imap(process_floatzone_file, args, chunksize=256), total=len(args)))
    else:
        for arg in tqdm(args, disable=False):
            process_floatzone_file(arg)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        data_path = "data-floatzone"
    else:
        data_path = sys.argv[1]
    process_floatzone_data(data_path, processes=None, max_files=None)
