import multiprocessing as mp
import os
import shutil
import traceback as tb
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
from pelutils import log
from tqdm import tqdm
from frozone.data.dataloader import load_data_files

from frozone.data import PROCESSED_SUBDIR, Metadata, list_processed_data_files
from frozone.data.dataloader import standardize
from frozone.environments import FloatZoneNNSim
from frozone.train import TrainConfig, TrainResults


warnings.filterwarnings("error")

@torch.inference_mode()
def simulate(metadata: Metadata, X: np.ndarray, U: np.ndarray, S: np.ndarray) -> torch.FloatTensor:

    X = np.expand_dims(X, axis=0)
    U = np.expand_dims(U, axis=0)
    S = np.expand_dims(S, axis=0)

    train_cfg = FloatZoneNNSim.train_cfg

    X[:, train_cfg.H:] = FloatZoneNNSim.forward_standardized_multiple(
        X[:, :train_cfg.H],
        U[:, :train_cfg.H],
        S[:, :train_cfg.H],
        S[:, train_cfg.H:],
        U[:, train_cfg.H:],
        X.shape[1] - train_cfg.H,
    )

    return X[0]

def generate(data_path: str, floatzone_data: str):

    shutil.rmtree(os.path.join(data_path, PROCESSED_SUBDIR), ignore_errors=True)

    log.section("Loading models")
    train_cfg = TrainConfig.load(FloatZoneNNSim.model_path)
    train_res = TrainResults.load(FloatZoneNNSim.model_path)
    FloatZoneNNSim.load(train_cfg, train_res)

    simulation_length = int(15000 / FloatZoneNNSim.dt)
    assert simulation_length >= train_cfg.H + train_cfg.F

    log.section("Loading data")
    npz_files = list_processed_data_files(floatzone_data, ".")
    dataset, used_files = load_data_files(npz_files, train_cfg, max_num_files=0)
    log(f"Loaded {len(dataset):,} runs")

    log.section("Standardizing data")
    standardize(FloatZoneNNSim, dataset, train_res)

    log.section("Generating simulated data")
    for (metadata, (X, U, S, R, Z)), used_file in zip(tqdm(dataset), used_files, strict=True):
        is_phase = FloatZoneNNSim.is_phase("Cone", S)
        if is_phase.sum() < train_cfg.F + train_cfg.H:
            continue
        X = X[is_phase]
        U = U[is_phase]
        S = S[is_phase]
        R = R[is_phase]
        if X.shape[0] < simulation_length:
            X_new = np.empty((simulation_length, len(FloatZoneNNSim.XLabels)), dtype=FloatZoneNNSim.X_dtype)
            U_new = np.empty((simulation_length, len(FloatZoneNNSim.ULabels)), dtype=FloatZoneNNSim.X_dtype)
            S_new = np.empty((simulation_length, sum(FloatZoneNNSim.S_bin_count)), dtype=FloatZoneNNSim.X_dtype)
            R_new = np.empty((simulation_length, len(FloatZoneNNSim.reference_variables)), dtype=FloatZoneNNSim.X_dtype)
            X_new[:X.shape[0]] = X
            U_new[:U.shape[0]] = U
            S_new[:S.shape[0]] = S
            R_new[:R.shape[0]] = R
            X_new[X.shape[0]:] = X[-1]
            U_new[U.shape[0]:] = U[-1]
            S_new[S.shape[0]:] = S[-1]
            R_new[R.shape[0]:] = R[-1]
        else:
            X_new = X[:simulation_length]
            U_new = U[:simulation_length]
            S_new = S[:simulation_length]
            R_new = R[:simulation_length]

        X, U, S, R = X_new, U_new, S_new, R_new
        assert X.shape[0] == simulation_length
        metadata.length = X.shape[0]

        X = simulate(metadata, X, U, S)
        X = X[:simulation_length] * train_res.std_x + train_res.mean_x
        U = U[:simulation_length] * train_res.std_u + train_res.mean_u
        S = S[:simulation_length]
        R = R[:simulation_length] * train_res.std_x[FloatZoneNNSim.reference_variables] + train_res.mean_x[FloatZoneNNSim.reference_variables]

        outpath = used_file.replace(f"/{os.path.basename(floatzone_data)}/", f"/{os.path.basename(data_path)}/")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        np.savez_compressed(
            outpath,
            metadata=metadata, X=X, U=U, S=S, R=R, Z=np.empty_like(X)[..., []],
        )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("floatzone_data")
    args = parser.parse_args()

    log.configure(os.path.join(args.data_path, "simulate.log"))

    with log.log_errors:
        generate(args.data_path, args.floatzone_data)
