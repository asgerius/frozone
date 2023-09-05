from __future__ import annotations

import multiprocessing as mp
import os
import random
import shutil
from glob import glob as glob  # glob

import numpy as np
import pandas as pd
from pelutils import split_path
from tqdm import tqdm

from frozone.data import PROCESSED_SUBDIR, RAW_SUBDIR, TEST_SUBDIR, TRAIN_SUBDIR, TRAIN_TEST_SPLIT
from frozone.environments import FloatZone


def keep_automode_slice(automode: np.ndarray) -> slice:
    start, stop = None, None
    for i, is_automode in enumerate(automode):
        if is_automode and start is None:
            start = i
        elif not is_automode and start is not None:
            stop = i
            break

    return slice(start, stop)

def parse_floatzone_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    automode = df["Automode_GenVoltage"].values   & df["Automode_PolyPull"].values & \
               df["Automode_CrysPull"].values     & df["Automode_PolyRotation"].values & \
               df["Automode_CrysRotation"].values & df["Automode_CoilPos"].values & \
               df["External_Control"].values

    # Only use data from when automation is active
    # Values can be messy outside this range
    use_slice = keep_automode_slice(automode)

    # assert (use_slice.stop - use_slice.start) / len(df) > 0.8 or len(df) < 2000, f"{use_slice.stop - use_slice.start} / {len(df)}"

    df = df.iloc[use_slice]
    n = len(df)

    X = np.empty((n, len(FloatZone.XLabels)), dtype=FloatZone.X_dtype)
    U = np.empty((n, len(FloatZone.ULabels)), dtype=FloatZone.U_dtype)
    S = np.zeros((n, sum(FloatZone.S_bin_count)), dtype=FloatZone.S_dtype)

    X[:, FloatZone.XLabels.PolyDia]    = df["PolyDia[mm]"].values
    X[:, FloatZone.XLabels.CrysDia]    = df["CrysDia[mm]"].values
    X[:, FloatZone.XLabels.UpperZone]  = df["UpperZone[mm]"].values
    X[:, FloatZone.XLabels.LowerZone]  = df["LowerZone[mm]"].values
    X[:, FloatZone.XLabels.FullZone]   = df["FullZone[mm]"].values
    X[:, FloatZone.XLabels.MeltVolume] = df["MeltVolume[mm3]"].values
    X[:, FloatZone.XLabels.PolyAngle]  = df["PolyAngle[deg]"].values
    if "CrysAngle_Left[deg]" in df.keys():
        # New runs in which left and right angles have been collected separately
        X[:, FloatZone.XLabels.CrysAngleLeft]  = df["CrysAngle_Left[deg]"].values
        X[:, FloatZone.XLabels.CrysAngleRight] = df["CrysAngle_Right[deg]"].values
    else:
        # Old runs in which left and right angles where averaged
        X[:, FloatZone.XLabels.CrysAngleLeft]  = df["CrysAngle[deg]"].values
        X[:, FloatZone.XLabels.CrysAngleRight] = df["CrysAngle[deg]"].values
    X[:, FloatZone.XLabels.MeltNeck]   = df["MeltNeck[mm]"].values
    X[:, FloatZone.XLabels.GrowthLine] = df["GrowthLine[mm]"].values
    X[:, FloatZone.XLabels.PosPoly]    = df["Pos_Poly[mm]"].values
    X[:, FloatZone.XLabels.PosCrys]    = df["Pos_Crys[mm]"].values

    U[:, FloatZone.ULabels.GenVoltage]     = df["GenVoltage[kV]"].values
    U[:, FloatZone.ULabels.PolyPullRate]   = df["PolyPullRate[mm/min]"].values
    U[:, FloatZone.ULabels.CrysPullRate]   = df["CrysPullRate[mm/min]"].values
    # U[:, FloatZone.ULabels.PolyRotation]   = df["PolyRotation[rpm]"].values
    # U[:, FloatZone.ULabels.CrysRotation]   = df["CrysRotation[rpm]"].values
    # U[:, FloatZone.ULabels.CoilPosition]   = df["CoilPosition[mm]"].values

    PLC_count = FloatZone.S_bin_count[FloatZone.SLabels.PLCState]
    growth_state_index = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6,
                          256: 7, 512: 8, 1024: 9, 2048: 10, 4096: 11, 8192: 12}
    assert len(growth_state_index) == FloatZone.S_bin_count[FloatZone.SLabels.GrowthState]
    used_indices = set(growth_state_index.values())
    assert all(i in used_indices for i in range(len(growth_state_index)))
    for i in range(n):
        binary_str = f"{df['PLC_State_Act'].values[i]:032b}"
        plc_is_true = [bool(int(b)) for b in binary_str]
        S[i, :PLC_count][plc_is_true] = 1

        S[i, PLC_count+growth_state_index[df["Growth_State_Act"].values[i]]] = 1

    assert len(X) == len(U) == len(S), "%i %i %i" % (len(X), len(U), len(S))
    assert np.isnan(X).sum() == 0, "NaN for X"
    assert np.isnan(U).sum() == 0, "NaN for U"
    assert np.isnan(S).sum() == 0, "NaN for S"

    return X, U, S

def process_floatzone_file(filepath: str) -> str | None:
    try:
        df = pd.read_csv(filepath, quoting=3, delim_whitespace=True)
        X, U, S = parse_floatzone_df(df)
        path_components = split_path(filepath)
        train_test_subdir = TRAIN_SUBDIR if random.random() < TRAIN_TEST_SPLIT else TEST_SUBDIR
        outpath = os.path.join(
            path_components[0],
            PROCESSED_SUBDIR,
            train_test_subdir,
            *path_components[2:-1],
            os.path.splitext(path_components[-1])[0],
        )
        os.makedirs(os.path.split(outpath)[0], exist_ok=True)
        np.savez_compressed(outpath, X=X, U=U, S=S)
    except Exception as e:
        return filepath + " " + str(e)

def process_floatzone_data(data_path: str, processes=mp.cpu_count()) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    shutil.rmtree(os.path.join(data_path, PROCESSED_SUBDIR), ignore_errors=True)

    raw_files = glob(os.path.join(data_path, RAW_SUBDIR, "**", "*.txt"), recursive=True)
    chunksize = 256
    with mp.Pool(processes=processes) as pool:
        results = tuple(tqdm(pool.imap(process_floatzone_file, raw_files, chunksize=chunksize), total=len(raw_files)))

    errors = [error for error in results if error is not None]
    print("Errors occured when reading the following files")
    print("\n".join(errors))

if __name__ == "__main__":
    process_floatzone_data("data-floatzone")
