import multiprocessing as mp
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

def simulate(metadata: Metadata, X: np.ndarray, U: np.ndarray, S: np.ndarray, *, debug: bool) -> int:

    assert FloatZone.dt == Steuermann.dt

    S[:, -1] = 1
    iters = metadata.length - 1


    Z = np.empty((metadata.length, len(Steuermann.ZLabels)), dtype=Steuermann.X_dtype)

    Z[[0]] = FloatZone.init_hidden_vars(U[[0]])

    for i in range(iters):
        keeps = X[i+1, FloatZone.XLabels.PolyDia], X[i+1, FloatZone.XLabels.PolyAngle]
        try:
            X[[i+1]], _, Z[[i+1]] = FloatZone.forward(X[[i]], U[[i]], S[[i]], Z[[i]])
        except Exception as e:
            if debug:
                raise
            log.log_with_stacktrace(e, LogLevels.WARNING)
            return 1

        X[i+1, FloatZone.XLabels.PolyDia] = keeps[0]
        X[i+1, FloatZone.XLabels.PolyAngle] = keeps[1]

    return 0

def run_single_sim(args: tuple) -> int:
    with log.collect:
        (metadata, (X, U, S, R)), save_file, debug = args
        assert FloatZone.is_phase("Cone", S).all()
        if not (X[:, FloatZone.XLabels.PolyDia] > 0).all():
            log.warning("Code -1: %s" % metadata.raw_file)
            return -1
        result_code = simulate(metadata, X, U, S, debug=debug)
        if result_code == 0:
            save_loc = save_file.replace(f"/{PROCESSED_SUBDIR}/", f"/{SIMULATED_SUBDIR}/")
            os.makedirs(os.path.split(save_loc)[0], exist_ok=True)
            np.savez(save_loc, metadata=metadata, X=X, U=U, S=S, R=R)
        else:
            log.warning("Code %i: %s" % (result_code, metadata.raw_file))
        return result_code

def generate(data_path: str, *, debug: bool):

    shutil.rmtree(os.path.join(data_path, SIMULATED_SUBDIR), ignore_errors=True)

    tmp_train_cfg = TrainConfig(
        FloatZone.__name__, data_path, "Cone", use_sim_data=True,
        history_window=FloatZone.dt, prediction_window=FloatZone.dt,
        history_interp=1, prediction_interp=1,
        batches=1, batch_size=1, lr=1, num_models=1,
        max_num_data_files=10 if debug else 0, eval_size=0, loss_fn="l1", huber_delta=0,
        epsilon=0, augment_prob=0,
    )

    npz_files = list_processed_data_files(data_path, ".")
    log(f"Loading {tmp_train_cfg.max_num_data_files} from {len(npz_files):,} data files")
    dataset, npz_files = load_data_files(npz_files, tmp_train_cfg, max_num_files=tmp_train_cfg.max_num_data_files)
    log(f"Loaded {len(dataset):,} data files")

    args = [(data, save_file, debug) for data, save_file in zip(dataset, npz_files, strict=True)]
    if debug:
        results = list()
        for i, arg in enumerate(args):
            log(f"{i+1} / {len(args)}")
            results.append(run_single_sim(arg))
    else:
        with mp.Pool() as pool:
            results = list(tqdm(pool.imap(run_single_sim, args, chunksize=5), total=len(args)))

    results = np.array(results)
    log(
        *(f"Code {i}: {(results==i).sum():,}" for i in np.unique(results))
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log.configure(os.path.join(args.data_path, "simulate.log"), print_level=LogLevels.DEBUG if args.debug else None)

    with log.log_errors:
        generate(args.data_path, debug=args.debug)
