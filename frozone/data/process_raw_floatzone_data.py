from __future__ import annotations

import multiprocessing as mp
import os
import random
import shutil
from collections import defaultdict
from glob import glob as glob  # glob

import numpy as np
import pandas as pd
from pelutils import split_path, log, thousands_seperators
from tqdm import tqdm

from frozone.data import Dataset, PHASES, PHASE_ORDER, PROCESSED_SUBDIR, RAW_SUBDIR, TEST_SUBDIR, TRAIN_SUBDIR, TRAIN_TEST_SPLIT
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

def get_phase_slice(phase_number: int, phase_array: np.ndarray) -> slice | None:
    phase_list = phase_array.tolist()
    try:
        next_phase = PHASE_ORDER[PHASE_ORDER.index(phase_number) + 1]
        end_index = phase_list.index(next_phase)
        offset = 1
        while phase_array[end_index - offset] == phase_number:
            offset += 1
        start_index = end_index - offset
        if not (start_index - 5 < end_index):
            return
        slice_ = slice(start_index + 1, end_index - 1)
        assert np.all(phase_array[slice_] == phase_number), "Not all correct phase in phase array"
        return slice_
    except ValueError:
        # Phase or next phase does not exist, so no slice
        return

def keep_automode_slice(automode: np.ndarray) -> slice:
    start, stop = None, None
    for i, is_automode in enumerate(automode):
        if is_automode and start is None:
            start = i
        elif not is_automode and start is not None:
            stop = i
            break

    return slice(start, stop)

def parse_floatzone_df(df: pd.DataFrame, machine: str) -> dict[int, Dataset]:

    # The minimum number of seconds a run should have been going for in order to be included
    min_seconds = {
        "Cone": 7000,
        "Full_Diameter": 7000,
    }

    data = dict()

    for phase_number, phase_name in PHASES.items():
        if phase_name not in min_seconds:
            # This case has not yet been considered, and so no data is made to save time
            continue

        slice = get_phase_slice(phase_number, df.Growth_State_Act.values)

        if slice is None or (slice.stop - slice.start) * 6 < min_seconds[phase_name]:
            continue

        # df_used = df.iloc[slice]

        # automode = df["Automode_GenVoltage"].values   & df["Automode_PolyPull"].values & \
        #            df["Automode_CrysPull"].values     & df["Automode_PolyRotation"].values & \
        #            df["Automode_CrysRotation"].values & df["Automode_CoilPos"].values & \
        #            df["External_Control"].values

        # # Only use data from when automation is active
        # # Values can be messy outside this range
        # use_slice = keep_automode_slice(automode)

        df_used = df.iloc[slice]
        n = len(df_used)

        try:
            assert (df_used.Growth_State_Act.values == phase_number).all(), "Not all same phase in df slice"
        except:
            breakpoint()

        X = np.empty((n, len(FloatZone.XLabels)), dtype=FloatZone.X_dtype)
        U = np.empty((n, len(FloatZone.ULabels)), dtype=FloatZone.U_dtype)
        S = np.zeros((n, sum(FloatZone.S_bin_count)), dtype=FloatZone.S_dtype)

        X[:, FloatZone.XLabels.PolyDia]    = df_used["PolyDia[mm]"].values
        X[:, FloatZone.XLabels.CrysDia]    = df_used["CrysDia[mm]"].values
        X[:, FloatZone.XLabels.UpperZone]  = df_used["UpperZone[mm]"].values
        X[:, FloatZone.XLabels.LowerZone]  = df_used["LowerZone[mm]"].values
        X[:, FloatZone.XLabels.FullZone]   = df_used["FullZone[mm]"].values
        X[:, FloatZone.XLabels.MeltVolume] = df_used["MeltVolume[mm3]"].values
        X[:, FloatZone.XLabels.PolyAngle]  = df_used["PolyAngle[deg]"].values
        X[:, FloatZone.XLabels.CrysAngle]  = df_used["CrysAngle[deg]"].values
        # X[:, FloatZone.XLabels.MeltNeck]   = df_used["MeltNeck[mm]"].values
        # X[:, FloatZone.XLabels.GrowthLine] = df_used["GrowthLine[mm]"].values
        # X[:, FloatZone.XLabels.PosPoly]    = df_used["Pos_Poly[mm]"].values
        # X[:, FloatZone.XLabels.PosCrys]    = df_used["Pos_Crys[mm]"].values

        U[:, FloatZone.ULabels.GenVoltage]     = df_used["GenVoltage[kV]"].values
        U[:, FloatZone.ULabels.PolyPullRate]   = df_used["PolyPullRate[mm/min]"].values
        U[:, FloatZone.ULabels.CrysPullRate]   = df_used["CrysPullRate[mm/min]"].values
        # U[:, FloatZone.ULabels.PolyRotation]   = df["PolyRotation[rpm]"].values
        # U[:, FloatZone.ULabels.CrysRotation]   = df["CrysRotation[rpm]"].values
        # U[:, FloatZone.ULabels.CoilPosition]   = df["CoilPosition[mm]"].values

        PLC_count = 0 #FloatZone.S_bin_count[FloatZone.SLabels.PLCState]
        # growth_state_index = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6,
        #                       256: 7, 512: 8, 1024: 9, 2048: 10, 4096: 11, 8192: 12}
        # assert len(growth_state_index) == FloatZone.S_bin_count[FloatZone.SLabels.GrowthState]
        # used_indices = set(growth_state_index.values())
        # assert all(i in used_indices for i in range(len(growth_state_index)))
        for i in range(n):
            # binary_str = f"{df_used['PLC_State_Act'].values[i]:032b}"
            # plc_is_true = [bool(int(b)) for b in binary_str]
            # S[i, :PLC_count][plc_is_true] = 1

            # S[i, PLC_count+growth_state_index[df["Growth_State_Act"].values[i]]] = 1

            S[i, PLC_count + MACHINES.index(machine)] = 1

        assert len(X) == len(U) == len(S), "%i %i %i" % (len(X), len(U), len(S))
        assert np.isnan(X).sum() == 0, "NaN for X"
        assert np.isnan(U).sum() == 0, "NaN for U"
        assert np.isnan(S).sum() == 0, "NaN for S"

        data[phase_number] = X, U, S

    return data

def process_floatzone_file(filepath: str) -> str | None:
    with log.collect:
        try:
            log("Loading %s" % filepath)
            df = pd.read_csv(filepath, quoting=3, delim_whitespace=True)
            path_components = split_path(filepath)
            machine = path_components[2]
            log("Parsing file")
            data = parse_floatzone_df(df, machine)
            for phase_number, data_sequence in data.items():
                train_test_subdir = TRAIN_SUBDIR if random.random() < TRAIN_TEST_SPLIT else TEST_SUBDIR
                outpath = os.path.join(
                    path_components[0],
                    PROCESSED_SUBDIR,
                    train_test_subdir,
                    PHASES[phase_number],
                    *path_components[2:-1],
                    os.path.splitext(path_components[-1])[0],
                )
                X, U, S = data_sequence
                log("Saving file with sequence length %s" % thousands_seperators(len(X)))
                os.makedirs(os.path.split(outpath)[0], exist_ok=True)
                np.savez_compressed(outpath, X=X, U=U, S=S)
        except Exception as e:
            return filepath + " " + str(e)

def process_floatzone_data(data_path: str, processes=None, max_files=None) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    log.configure(os.path.join(data_path, "process.log"), print_level=None)

    shutil.rmtree(os.path.join(data_path, PROCESSED_SUBDIR), ignore_errors=True)

    raw_files = glob(os.path.join(data_path, RAW_SUBDIR, "**", "*.txt"), recursive=True)
    random.shuffle(raw_files)
    raw_files = raw_files[:max_files]
    n_raw_total = len(raw_files)
    raw_files = [file for file in raw_files if os.path.join(*split_path(file)[2:]) not in BLACKLIST]
    if max_files is None:
        assert 0 < n_raw_total == len(raw_files) + len(BLACKLIST)
    else:
        assert 0 < n_raw_total <= len(raw_files)

    processes = processes or os.cpu_count()
    if processes > 1:
        with mp.Pool(processes=processes) as pool:
            results = tuple(tqdm(pool.imap(process_floatzone_file, raw_files, chunksize=256), total=len(raw_files)))
    else:
        results = list()
        for raw_file in tqdm(raw_files):
            result = process_floatzone_file(raw_file)
            if result:
                raise RuntimeError(result)
            results.append(result)

    errors = [error for error in results if error is not None]
    if errors:
        print("Errors occured when reading the following %i files" % len(errors))
        print("\n".join(errors))
    else:
        print("No errors occured")

if __name__ == "__main__":
    process_floatzone_data("data-floatzone", processes=None, max_files=None)
