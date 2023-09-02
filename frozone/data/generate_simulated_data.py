import os
import random
import shutil
from typing import Type

import numpy as np
from tqdm import tqdm

from frozone.data import PROCESSED_SUBDIR, TEST_SUBDIR, TRAIN_SUBDIR, TRAIN_TEST_SPLIT
from frozone.environments import Ball, Environment


def generate(path: str, env: Type[Environment], num_simulations: int, iters: int):
    X, _, U, S = env.simulate(num_simulations, iters, 0.1)

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
    generate("data-ball", Ball, 20000, 5000)
