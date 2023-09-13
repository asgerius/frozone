from __future__ import annotations

import random
import threading
import time
from copy import copy
from queue import Queue
from typing import Generator, Optional, Type

import numpy as np
import torch

import frozone.train
from frozone import device
from frozone.data import DataSequence, Dataset
from frozone.environments import Environment, FloatZone
from frozone.train import TrainConfig, TrainResults


def load_data_files(npz_files: list[str], train_cfg: Optional[TrainConfig], max_num_files = 0) -> Dataset:
    """ Loads data for an environment, which is returned as a list of (X, U, S) tuples, each of which
    is a numpy array of shape time steps x dimensionality. If max_num_files == 0, all files are used. """

    max_num_files = max_num_files or None

    if max_num_files:
        # Shuffle files when only loading a subset to get a more representative subset
        npz_files = copy(npz_files)
        random.shuffle(npz_files)

    sets = list()
    for npz_file in npz_files[:max_num_files]:
        arrs = np.load(npz_file)
        X, U, S = arrs["X"], arrs["U"], arrs["S"]
        if train_cfg and len(X) < train_cfg.H + train_cfg.F + 3:
            # Ignore files with too little data to be useful
            continue
        sets.append((X, U, S))

    return sets

def dataset_size(dataset: Dataset) -> int:
    return sum(len(X) for X, U, S in dataset)

def standardize(
    env: Type[Environment],
    dataset: Dataset,
    train_results: TrainResults,
) -> int:
    """ Calculates the feature-wise mean and standard deviations of X and U for the given data set. """

    if train_results.mean_x is None:

        sum_x = np.zeros(len(env.XLabels))
        sum_u = np.zeros(len(env.ULabels))
        sse_x = np.zeros(len(env.XLabels))
        sse_u = np.zeros(len(env.ULabels))

        # Calculate sum
        n = 0
        for X, U, S in dataset:
            sum_x += X.sum(axis=0)
            sum_u += U.sum(axis=0)
            n += len(X)

        mean_x = sum_x / n
        mean_u = sum_u / n

        # Calculate variance
        for X, U, S in dataset:
            X[...] = X - mean_x
            U[...] = U - mean_u

            sse_x += (X ** 2).sum(axis=0)
            sse_u += (U ** 2).sum(axis=0)

        std_x = np.sqrt(sse_x / (n - 1))
        std_u = np.sqrt(sse_u / (n - 1))

        train_results.mean_x = mean_x.astype(np.float32)
        train_results.std_x = std_x.astype(np.float32)
        train_results.mean_u = mean_u.astype(np.float32)
        train_results.std_u = std_u.astype(np.float32)

    else:

        for X, U, S in dataset:
            X[...] = X - train_results.mean_x
            U[...] = U - train_results.mean_u

    eps = 1e-6
    for i, (X, U, S) in enumerate(dataset):
        X[...] = X / (train_results.std_x + eps)
        U[...] = U / (train_results.std_u + eps)

        dataset[i] = tuple(data.astype(np.float16) for data in dataset[i])

def numpy_to_torch_device(*args: np.ndarray) -> list[torch.Tensor]:
    return [torch.from_numpy(x).to(device).float() for x in args]

def include_vector(env: Type[Environment], train_cfg: TrainConfig) -> np.ndarray:

    if env is FloatZone:
        xlabels = FloatZone.XLabels
        x_exclude = {
            "Cone": (xlabels.MeltNeck, )
        }[train_cfg.phase]
    else:
        x_exclude = tuple()
    # log("Excluding the the following process variables", [lab.name for lab in loss_x_exclude])
    x_include = np.ones(len(env.XLabels), dtype=np.float32)
    for xlab in x_exclude:
        x_include[xlab.value] = 0

    return x_include

def _start_dataloader_thread(
    env: Type[Environment],
    train_cfg: TrainConfig,
    dataset: Dataset,
    buffer: Queue[DataSequence],
):

    x_include = include_vector(env, train_cfg)

    def task():

        # Probability to select a given set is proportional to the amount of data in it
        p = np.array([len(X) for X, U, S in dataset]) / dataset_size(dataset)

        while frozone.train.is_doing_training:

            if buffer.qsize() >= train_cfg.num_models:
                # If full, wait a little and try again
                time.sleep(0.001)
                continue

            # These should not be changed to torch, as the sampling is apparently much faster in numpy
            Xh = np.empty((train_cfg.batch_size, train_cfg.H, len(env.XLabels)), dtype=np.float16)
            Uh = np.empty((train_cfg.batch_size, train_cfg.H, len(env.ULabels)), dtype=np.float16)
            Sh = np.empty((train_cfg.batch_size, train_cfg.H, sum(env.S_bin_count)), dtype=np.float16)
            Xf = np.empty((train_cfg.batch_size, train_cfg.F, len(env.XLabels)), dtype=np.float16)
            Uf = np.empty((train_cfg.batch_size, train_cfg.F, len(env.ULabels)), dtype=np.float16)
            Sf = np.empty((train_cfg.batch_size, train_cfg.F, sum(env.S_bin_count)), dtype=np.float16)

            set_index = np.random.choice(np.arange(len(dataset)), train_cfg.batch_size, replace=True)

            for i in range(train_cfg.batch_size):
                X, U, S = dataset[set_index[i]]
                start_iter = random.randint(0, len(X) - train_cfg.H - train_cfg.F - 1)

                Xh[i] = X[start_iter : start_iter + train_cfg.H]
                Uh[i] = U[start_iter : start_iter + train_cfg.H]
                Sh[i] = S[start_iter : start_iter + train_cfg.H]
                Xf[i] = X[start_iter + train_cfg.H : start_iter + train_cfg.H + train_cfg.F]
                Uf[i] = U[start_iter + train_cfg.H : start_iter + train_cfg.H + train_cfg.F]
                Sf[i] = S[start_iter + train_cfg.H : start_iter + train_cfg.H + train_cfg.F]

            buffer.put(numpy_to_torch_device(Xh * x_include, Uh, Sh, Xf * x_include, Uf, Sf))

    thread = threading.Thread(target=task, daemon=True)
    thread.start()

def dataloader(
    env: Type[Environment],
    train_cfg: TrainConfig,
    dataset: Dataset,
) -> Generator[tuple[torch.FloatTensor], None, None]:

    buffer = Queue(maxsize = 2 * train_cfg.num_models)

    _start_dataloader_thread(env, train_cfg, dataset, buffer)

    while True:
        yield buffer.get()
