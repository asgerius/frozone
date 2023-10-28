import os
import shutil
import warnings
from argparse import ArgumentParser

import numpy as np
from pelutils import log, LogLevels
from tqdm import tqdm
from frozone.data.dataloader import load_data_files

from frozone.data import PROCESSED_SUBDIR, SIMULATED_SUBDIR, Metadata, list_processed_data_files
from frozone.environments import FloatZone, Steuermann
from frozone.train import TrainConfig


warnings.filterwarnings("error")

def simulate(metadata: Metadata, X: np.ndarray, U: np.ndarray, S: np.ndarray) -> bool:

    assert FloatZone.dt == Steuermann.dt

    S[:, -1] = 1
    iters = metadata.length - 1

    X_st = np.empty((metadata.length, len(Steuermann.XLabels)), dtype=Steuermann.X_dtype)
    U_st = np.empty((metadata.length, len(Steuermann.ULabels)), dtype=Steuermann.U_dtype)
    Z_st = np.empty((metadata.length, len(Steuermann.ZLabels)), dtype=Steuermann.X_dtype)

    X_st[0, Steuermann.XLabels.PolyDia]      = X[0, FloatZone.XLabels.PolyDia]
    X_st[0, Steuermann.XLabels.CrystalDia]   = X[0, FloatZone.XLabels.CrystalDia]
    X_st[0, Steuermann.XLabels.UpperZone]    = X[0, FloatZone.XLabels.UpperZone]
    X_st[0, Steuermann.XLabels.LowerZone]    = X[0, FloatZone.XLabels.LowerZone]
    X_st[0, Steuermann.XLabels.MeltVolume]   = X[0, FloatZone.XLabels.MeltVolume]
    X_st[0, Steuermann.XLabels.MeltNeckDia]  = X[0, FloatZone.XLabels.MeltNeckDia]
    X_st[:, Steuermann.XLabels.PolyAngle]    = X[:, FloatZone.XLabels.PolyAngle]
    X_st[0, Steuermann.XLabels.CrystalAngle] = X[0, FloatZone.XLabels.CrystalAngle]

    U_st[:, Steuermann.ULabels.GeneratorVoltage] = U[:, FloatZone.ULabels.GeneratorVoltage]
    U_st[:, Steuermann.ULabels.PolyPullRate]     = U[:, FloatZone.ULabels.PolyPullRate]
    U_st[:, Steuermann.ULabels.CrystalPullRate]  = U[:, FloatZone.ULabels.CrystalPullRate]

    Z_st[:, Steuermann.ZLabels.Time] = Steuermann.dt * np.arange(metadata.length)
    Z_st[:, Steuermann.ZLabels.MeltingRate] = U[:, Steuermann.ULabels.PolyPullRate] / 60
    Z_st[:, Steuermann.ZLabels.CrystallizationRate] = U[:, Steuermann.ULabels.CrystalPullRate] / 60
    Z_st[:, Steuermann.ZLabels.TdGeneratorVoltage] = U[:, Steuermann.ULabels.GeneratorVoltage]

    for i in range(iters):
        try:
            X_st[[i+1]], _, Z_st[[i+1]] = Steuermann.forward(X_st[[i]], U_st[[i]], S[[i]], Z_st[[i]])
        except Exception as e:
            log.log_with_stacktrace(e, LogLevels.WARNING)
            return False
        X_st[i+1, Steuermann.XLabels.PolyDia] = X[i+1, FloatZone.XLabels.PolyDia]
        X_st[i+1, Steuermann.XLabels.PolyAngle] = X[i+1, FloatZone.XLabels.PolyAngle]

    X[:, FloatZone.XLabels.PolyDia] = X_st[:, Steuermann.XLabels.PolyDia]
    X[:, FloatZone.XLabels.CrystalDia] = X_st[:, Steuermann.XLabels.CrystalDia]
    X[:, FloatZone.XLabels.UpperZone] = X_st[:, Steuermann.XLabels.UpperZone]
    X[:, FloatZone.XLabels.LowerZone] = X_st[:, Steuermann.XLabels.LowerZone]
    full_zone = X_st[:, Steuermann.XLabels.UpperZone] + X_st[:, Steuermann.XLabels.LowerZone]
    full_zone_offset = full_zone[0] - X[0, FloatZone.XLabels.FullZone]
    X[:, FloatZone.XLabels.FullZone] = full_zone - full_zone_offset
    X[:, FloatZone.XLabels.MeltVolume] = X_st[:, Steuermann.XLabels.MeltVolume]
    X[:, FloatZone.XLabels.MeltNeckDia] = X_st[:, Steuermann.XLabels.MeltNeckDia]
    X[:, FloatZone.XLabels.PolyAngle] = X_st[:, Steuermann.XLabels.PolyAngle]
    X[:, FloatZone.XLabels.CrystalAngle] = X_st[:, Steuermann.XLabels.CrystalAngle]

    return True

def generate(data_path: str):

    shutil.rmtree(os.path.join(data_path, SIMULATED_SUBDIR), ignore_errors=True)

    tmp_train_cfg = TrainConfig(
        FloatZone.__name__, data_path, "Cone",
        history_window=FloatZone.dt, prediction_window=FloatZone.dt,
        batches=1, batch_size=1, lr=1, num_models=1,
        max_num_data_files=0, eval_size=0, loss_fn="l1", huber_delta=0,
        epsilon=0, augment_prob=0,
    )

    npz_files = list_processed_data_files(data_path, ".")
    log("Loading %i files" % len(npz_files))
    dataset, npz_files = load_data_files(npz_files, tmp_train_cfg, max_num_files=tmp_train_cfg.max_num_data_files)

    for (metadata, (X, U, S, R)), save_file in zip(tqdm(dataset, disable=True), npz_files, strict=True):
        assert FloatZone.is_phase("Cone", S).all()
        if not (X[:, FloatZone.XLabels.PolyDia] > 0).all():
            # TODO Dump these files, for they are probably indicative of mistakes
            log.warning("0 poly diameter detected", metadata.raw_file)
            continue
        success = simulate(metadata, X, U, S)
        if success:
            save_loc = save_file.replace(f"/{PROCESSED_SUBDIR}/", f"/{SIMULATED_SUBDIR}/")
            os.makedirs(os.path.split(save_loc)[0], exist_ok=True)
            np.savez(save_loc, metadata=metadata, X=X, U=U, S=S, R=R)
            log.debug("Simulation successfull", metadata.raw_file)
        else:
            log.warning("Simulation unsuccessfull", metadata.raw_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path")
    args = parser.parse_args()

    bad_files = [
        "/work3/s183912/floatzone/data-floatzone/Raw/M34/34_Automation_Archive_2018/34-1317_Automation.txt",
    ]

    generate(args.data_path)
