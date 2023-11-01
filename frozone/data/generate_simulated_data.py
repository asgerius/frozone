import os
import random
import shutil
import warnings
from argparse import ArgumentParser
from typing import Type

import numpy as np
from tqdm import tqdm

import frozone.environments as environments
from frozone.data import PROCESSED_SUBDIR, TEST_SUBDIR, TRAIN_SUBDIR, TRAIN_TEST_SPLIT, Metadata


warnings.filterwarnings("error")

def generate(path: str, env: Type[environments.Environment], num_simulations: int, timesteps: int):
    X, U, S, R, Z = env.simulate(num_simulations, timesteps)

    for xlab in env.XLabels:
        stds = X[..., xlab].std(axis=1)
        for i in range(num_simulations):
            X[i, :, xlab] += np.random.uniform(0, 0.1) * (stds[i] + 0.01) * np.random.randn(timesteps)

    for ulab in env.ULabels:
        stds = U[..., ulab].std(axis=1)
        for i in range(num_simulations):
            U[i, :, ulab] += np.random.uniform(0, 0.1) * (stds[i] + 0.01) * np.random.randn(timesteps)

    shutil.rmtree(os.path.join(path, PROCESSED_SUBDIR), ignore_errors=True)

    for i in tqdm(range(num_simulations)):
        train_test_subdir = TRAIN_SUBDIR if random.random() < TRAIN_TEST_SPLIT else TEST_SUBDIR
        outpath = os.path.join(
            path,
            PROCESSED_SUBDIR,
            train_test_subdir,
            f"{env.__name__}_{i}.npz",
        )
        os.makedirs(os.path.split(outpath)[0], exist_ok=True)
        metadata = Metadata(length=len(X))
        np.savez_compressed(outpath, metadata=metadata, X=X[i], U=U[i], S=S[i], R=R[i])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("-n", "--num_simulations", type=int, default=20000)
    parser.add_argument("-t", "--time", type=float, default=10800)
    parser.add_argument("-e", "--env", default="Steuermann")
    args = parser.parse_args()

    env: Type[environments.Environment] = getattr(environments, args.env)
    generate(args.data_path, env, args.num_simulations, int(args.time / env.dt))
