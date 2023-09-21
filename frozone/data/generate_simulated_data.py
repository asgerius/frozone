import os
import random
import shutil
import sys
import warnings
from typing import Type

import numpy as np
from tqdm import tqdm

from frozone.data import PROCESSED_SUBDIR, TEST_SUBDIR, TRAIN_SUBDIR, TRAIN_TEST_SPLIT
from frozone.environments import Ball, Environment, Steuermann


warnings.filterwarnings("error")

def generate(path: str, env: Type[Environment], num_simulations: int, iters: int):
    X, U, S, Z = env.simulate(num_simulations, iters)

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
        np.savez_compressed(outpath, X=X[i], U=U[i], S=S[i])

if __name__ == "__main__":
    data_path = sys.argv[1]
    generate(data_path, Steuermann, 30000, int(3600 / Steuermann.dt))
