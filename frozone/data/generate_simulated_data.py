import math
import multiprocessing as mp
import os
import random
import shutil
import warnings
from argparse import ArgumentParser
from typing import Type

import numpy as np
from pelutils import log, set_seeds
from tqdm import tqdm

import frozone.environments as environments
from frozone.data import PROCESSED_SUBDIR, TEST_SUBDIR, TRAIN_SUBDIR, TRAIN_TEST_SPLIT, Metadata


warnings.filterwarnings("error")

def generate_in_process(args: tuple):
    env, num_simulations, timesteps, i = args
    set_seeds(i)  # np.random and multiprocessing is a vile combination
    X, U, S, R, Z = env.simulate(num_simulations, timesteps, with_tqdm=True, tqdm_position=i)
    assert len(X) == num_simulations
    log("Generated %i simulations from chunk %i" % (num_simulations, i))
    return X, U, S, R, Z

def generate(path: str, env: Type[environments.Environment], num_simulations: int, timesteps: int):

    processes = mp.cpu_count()
    sims_per_chunk = 64
    num_args = math.ceil(num_simulations / sims_per_chunk)
    args = [(env, sims_per_chunk, timesteps, i) for i in range(num_args)]

    log.section(f"Generating {num_simulations} runs in {num_args} chunks of size {sims_per_chunk}")
    with mp.Pool(processes) as pool:
        results = pool.map(generate_in_process, args)

    X = np.concatenate([r[0] for r in results], axis=0)[:num_simulations]
    U = np.concatenate([r[1] for r in results], axis=0)[:num_simulations]
    S = np.concatenate([r[2] for r in results], axis=0)[:num_simulations]
    R = np.concatenate([r[3] for r in results], axis=0)[:num_simulations]
    Z = np.concatenate([r[4] for r in results], axis=0)[:num_simulations]

    log.section("Adding noise")
    for xlab in env.XLabels:
        stds = X[..., xlab].std(axis=1)
        for i in range(num_simulations):
            X[i, :, xlab] += np.random.uniform(0, 0.02) * (stds[i] + 0.01) * np.random.randn(timesteps)

    shutil.rmtree(os.path.join(path, PROCESSED_SUBDIR), ignore_errors=True)

    log.section("Saving data to %i" % path)
    for i in tqdm(range(num_simulations)):
        train_test_subdir = TRAIN_SUBDIR if random.random() < TRAIN_TEST_SPLIT else TEST_SUBDIR
        outpath = os.path.join(
            path,
            PROCESSED_SUBDIR,
            train_test_subdir,
            f"{env.__name__}_{i}.npz",
        )
        os.makedirs(os.path.split(outpath)[0], exist_ok=True)
        metadata = Metadata(length=X.shape[1])
        np.savez_compressed(outpath, metadata=metadata, X=X[i], U=U[i], S=S[i], R=R[i], Z=Z[i])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("-n", "--num_simulations", type=int, default=20000)
    parser.add_argument("-t", "--time", type=float, default=20000)
    parser.add_argument("-e", "--env", default="Steuermann")
    args = parser.parse_args()

    log.configure(os.path.join(args.data_path, "generate.log"))
    with log.log_errors:
        env: Type[environments.Environment] = getattr(environments, args.env)
        generate(args.data_path, env, args.num_simulations, int(args.time / env.dt))
