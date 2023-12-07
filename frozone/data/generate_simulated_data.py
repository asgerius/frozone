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
from frozone.data import PROCESSED_SUBDIR, TEST_SUBDIR, TRAIN_SUBDIR, TRAIN_TEST_SPLIT, Metadata, CONTROLLER_START


warnings.filterwarnings("error")

def preprend_constant(env: Type[environments.Environment], data: np.ndarray) -> np.ndarray:
    """ arr has shape n x timesteps x num vars or timesteps x num vars. """

    has_number_dimension = len(data.shape) == 3
    if not has_number_dimension:
        data = np.expand_dims(data, axis=0)

    filler = np.empty_like(data, shape=(data.shape[0], data.shape[1], data.shape[2] + 1))
    filler[..., :-1] = data

    arrs = list()
    for i in range(filler.shape[0]):
        arrs.append(list())
        for j in range(filler.shape[-1]):
            arr = filler[i, :, j]
            arr = np.concatenate([[arr[0]] * int(CONTROLLER_START // env.dt), arr])
            arrs[-1].append(arr)

    data = np.array(arrs).transpose(0, 2, 1)

    if not has_number_dimension:
        data = data[0]

    return data[..., :-1]

def generate_in_process(args: tuple):
    env, num_simulations, timesteps, i = args
    set_seeds(i)  # np.random and multiprocessing is a vile combination
    X, U, S, R, Z = env.simulate(num_simulations, timesteps, with_tqdm=True, tqdm_position=i)
    assert len(X) == num_simulations
    log("Generated %i simulations from chunk %i" % (num_simulations, i))
    X = preprend_constant(env, X)
    U = preprend_constant(env, U)
    S = preprend_constant(env, S)
    R = preprend_constant(env, R)
    Z = preprend_constant(env, Z)
    return X, U, S, R, Z

def generate(path: str, env: Type[environments.Environment], num_simulations: int, timesteps: int):

    processes = mp.cpu_count()
    sims_per_chunk = 32
    num_args = math.ceil(num_simulations / sims_per_chunk)
    args = [(env, sims_per_chunk, timesteps, i) for i in range(num_args)]

    log.section(f"Generating {num_simulations} runs in {num_args} chunks of size {sims_per_chunk}")
    with mp.Pool(processes) as pool:
        results = pool.map(generate_in_process, args)

    timesteps += int(CONTROLLER_START // env.dt)

    X = np.concatenate([r[0] for r in results], axis=0)[:num_simulations]
    U = np.concatenate([r[1] for r in results], axis=0)[:num_simulations]
    S = np.concatenate([r[2] for r in results], axis=0)[:num_simulations]
    R = np.concatenate([r[3] for r in results], axis=0)[:num_simulations]
    Z = np.concatenate([r[4] for r in results], axis=0)[:num_simulations]

    S[:, :int(CONTROLLER_START // env.dt), 0] = 0
    S[:, int(CONTROLLER_START // env.dt):, 0] = 1

    log.section("Adding noise")
    for xlab in env.XLabels:
        if xlab == environments.Steuermann.XLabels.FullPolyDia:
            continue
        stds = X[..., xlab].std(axis=1)
        for i in range(num_simulations):
            X[i, :, xlab] += np.random.uniform(0, 0.03) * (stds[i] + 0.01) * np.random.randn(timesteps)

    shutil.rmtree(os.path.join(path, PROCESSED_SUBDIR), ignore_errors=True)

    log.section("Saving data to %s" % path)
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
