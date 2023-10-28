from __future__ import annotations

import os
import random
import signal
import threading
import time
from copy import copy
from queue import Queue
from typing import Generator, Optional, Type

import numpy as np
import torch
from pelutils import TickTock, log, LogLevels, thousands_seperators

import frozone.train
from frozone import device_x, device_u, tensor_size
from frozone.data import DataSequence, Dataset, DatasetSim
from frozone.environments import Environment, FloatZone
from frozone.train import TrainConfig, TrainResults


EPS = 1e-6

def load_data_files(
    npz_files: list[str],
    train_cfg: Optional[TrainConfig],
    max_num_files=0,
    year=0,
    shuffle=True,
) -> tuple[Dataset, list[str]]:
    """ Loads data for an environment, which is returned as a list of (X, U, S, R) tuples, each of which
    is a numpy array of shape time steps x dimensionality. If max_num_files == 0, all files are used. """

    max_num_files = max_num_files or None
    npz_files = copy(npz_files)
    if shuffle:
        random.shuffle(npz_files)

    sets = list()
    used_files = list()
    for npz_file in npz_files:
        arrs = np.load(npz_file, allow_pickle=True)
        metadata, X, U, S, R = arrs["metadata"].item(), arrs["X"], arrs["U"], arrs["S"], arrs["R"]
        # print(npz_file, "\n", metadata.raw_file)
        if train_cfg and train_cfg.phase and train_cfg.get_env() is FloatZone:
            is_phase = FloatZone.is_phase(train_cfg.phase, S)
            X = X[is_phase]
            U = U[is_phase]
            S = S[is_phase]
            R = R[is_phase]
            metadata.length = len(X)
        if (train_cfg and metadata.length < train_cfg.H + train_cfg.F + 3) or metadata.date.year < year:
            # Ignore files with too little data to be useful
            # Also ignore too old data
            continue
        sets.append((metadata, (X, U, S, R)))
        used_files.append(npz_file)
        if len(sets) == max_num_files:
            return sets, used_files

    return sets, used_files

def dataset_size(dataset: Dataset) -> int:
    return sum(metadata.length for metadata, _ in dataset)

def standardize(
    env: Type[Environment],
    dataset: Dataset | DatasetSim,
    train_results: TrainResults,
) -> int:
    """ Calculates the feature-wise mean and standard deviations of X and U for the given data set. """

    if train_results.mean_x is None:
        dataset: Dataset

        sum_x = np.zeros(len(env.XLabels))
        sum_u = np.zeros(len(env.ULabels))
        sse_x = np.zeros(len(env.XLabels))
        sse_u = np.zeros(len(env.ULabels))

        # Calculate sum
        n = 0
        for _, (X, U, S, R) in dataset:
            sum_x += X.sum(axis=0)
            sum_u += U.sum(axis=0)
            n += len(X)

        mean_x = sum_x / n
        mean_u = sum_u / n

        # Calculate variance
        for _, (X, U, S, R) in dataset:
            X[...] = X - mean_x
            U[...] = U - mean_u

            sse_x += (X ** 2).sum(axis=0)
            sse_u += (U ** 2).sum(axis=0)

        std_x = np.sqrt(sse_x / (n - 1))
        std_u = np.sqrt(sse_u / (n - 1))

        train_results.mean_x = mean_x.astype(env.X_dtype)
        train_results.std_x = std_x.astype(env.X_dtype)
        train_results.mean_u = mean_u.astype(env.U_dtype)
        train_results.std_u = std_u.astype(env.U_dtype)

    else:

        for _, (X, U, S, R, *_) in dataset:
            X[...] = X - train_results.mean_x
            U[...] = U - train_results.mean_u

    for _, (X, U, S, R, *_) in dataset:
        X[...] = X / (train_results.std_x + EPS)
        U[...] = U / (train_results.std_u + EPS)
        R[...] = (R - train_results.mean_x[env.reference_variables]) / (train_results.std_x[env.reference_variables] + EPS)

def numpy_to_torch_device(*args: np.ndarray, device: torch.device) -> list[torch.Tensor]:
    return [torch.from_numpy(x).to(device).float() for x in args]

def _start_dataloader_thread(
    env: Type[Environment],
    train_cfg: TrainConfig,
    dataset: Dataset,
    buffer: Queue[DataSequence], *,
    train: bool,
):
    tt = TickTock()
    tt.tick()
    log_time_every = 30

    def task():
        nonlocal log_time_every

        while frozone.train.is_doing_training:

            if buffer.qsize() >= train_cfg.num_models:
                # If full, wait a little and try again
                time.sleep(0.001)
                continue

            tt.profile("Generate batch")

            # These should not be changed to torch, as the sampling is apparently much faster in numpy
            X = np.empty((train_cfg.batch_size, train_cfg.H + train_cfg.F, len(env.XLabels)), dtype=env.X_dtype)
            U = np.empty((train_cfg.batch_size, train_cfg.H + train_cfg.F, len(env.ULabels)), dtype=env.U_dtype)
            S = np.empty((train_cfg.batch_size, train_cfg.H + train_cfg.F, sum(env.S_bin_count)), dtype=env.S_dtype)

            set_index = np.random.choice(np.arange(len(dataset)), train_cfg.batch_size, replace=True)

            for i in range(train_cfg.batch_size):
                _, (X_seq, U_seq, S_seq, *_) = dataset[set_index[i]]
                start_iter = random.randint(0, len(X_seq) - train_cfg.H - train_cfg.F - 1)

                with tt.profile("Get slices"):
                    X[i] = X_seq[start_iter : start_iter + train_cfg.H + train_cfg.F]
                    U[i] = U_seq[start_iter : start_iter + train_cfg.H + train_cfg.F]
                    S[i] = S_seq[start_iter : start_iter + train_cfg.H + train_cfg.F]

            # if train:
            #     with tt.profile("Augment"):
            #         augment_data(X[i], U[i], train_cfg, tt)

            with tt.profile("To device"):
                X_dx, U_dx, S_dx = numpy_to_torch_device(X, U, S, device=device_x)
                X_du, U_du, S_du = numpy_to_torch_device(X, U, S, device=device_u)
            with tt.profile("Split and contiguous"):
                Xh_dx = X_dx[:, :train_cfg.H].contiguous()
                Uh_dx = U_dx[:, :train_cfg.H].contiguous()
                Sh_dx = S_dx[:, :train_cfg.H].contiguous()
                Xf_dx = X_dx[:, train_cfg.H:].contiguous()
                Uf_dx = U_dx[:, train_cfg.H:].contiguous()
                Sf_dx = S_dx[:, train_cfg.H:].contiguous()
                Xh_du = X_du[:, :train_cfg.H].contiguous()
                Uh_du = U_du[:, :train_cfg.H].contiguous()
                Sh_du = S_du[:, :train_cfg.H].contiguous()
                Xf_du = X_du[:, train_cfg.H:].contiguous()
                Uf_du = U_du[:, train_cfg.H:].contiguous()
                Sf_du = S_du[:, train_cfg.H:].contiguous()
            buffer.put((
                (Xh_dx, Uh_dx, Sh_dx, Xf_dx, Uf_dx, Sf_dx),
                (Xh_du, Uh_du, Sh_du, Xf_du, Uf_du, Sf_du),
            ))

            tt.end_profile()

            if tt.tock() > log_time_every and train:
                log("Batch time distribution", tt)
                log_time_every += 600
                tt.tick()

    def task_wrapper():
        try:
            task()
        except Exception as e:
            log.critical("Error occured in data loader thread")
            log.log_with_stacktrace(e, LogLevels.CRITICAL)
            frozone.train.is_doing_training = False
            os.kill(os.getpid(), signal.SIGKILL)

    thread = threading.Thread(target=task_wrapper, daemon=True)
    thread.start()

def dataloader(
    env: Type[Environment],
    train_cfg: TrainConfig,
    dataset: Dataset, *,
    train = False,
) -> Generator[tuple[tuple[torch.FloatTensor], tuple[torch.FloatTensor]], None, None]:

    buffer_size = 4
    buffer = Queue(maxsize=buffer_size)

    _start_dataloader_thread(env, train_cfg, dataset, buffer, train=train)

    is_first = True
    while True:
        batch = buffer.get()
        if not train and is_first:
            is_first = False
            size = sum(tensor_size(x) for x in batch[0])
            log(
                "Size of batch:  2 x %s b" % thousands_seperators(size),
                "Size of buffer: 2 x %s b" % thousands_seperators(size * buffer_size),
            )
        yield batch
