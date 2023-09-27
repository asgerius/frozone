import os
import random
from typing import Type

import numpy as np
import torch
from pelutils import TT, log, thousands_seperators
from pelutils.parser import Parser

import frozone.environments as environments
from frozone.data import TEST_SUBDIR, list_processed_data_files
from frozone.data.dataloader import Dataset, dataset_size, load_data_files, numpy_to_torch_device, standardize
from frozone.eval import ForwardConfig
from frozone.model.floatzone_network import FzNetwork
from frozone.plot.plot_forward import plot_forward
from frozone.train import TrainConfig, TrainResults


@torch.inference_mode()
def forward(
    path: str,
    env: Type[environments.Environment],
    models: list[FzNetwork],
    dataset: Dataset,
    train_cfg: TrainConfig,
    train_results: TrainResults,
    forward_cfg: ForwardConfig,
):
    sequence_length = train_cfg.H + train_cfg.F
    timesteps = forward_cfg.num_sequences * sequence_length
    log("Filtering dataset to only include files with at least %s data points" % thousands_seperators(timesteps))
    dataset = [(X, U, S) for X, U, S in dataset if len(X) >= timesteps]
    random.shuffle(dataset)
    log("%s files after filtering" % thousands_seperators(len(dataset)))

    log("Sampling start indices")
    X_true = np.empty((forward_cfg.num_samples, timesteps, len(env.XLabels)), dtype=env.X_dtype)
    U_true = np.empty((forward_cfg.num_samples, timesteps, len(env.ULabels)), dtype=env.U_dtype)
    S_true = np.empty((forward_cfg.num_samples, timesteps, sum(env.S_bin_count)), dtype=env.S_dtype)
    for i, (X, U, S) in enumerate(dataset[:forward_cfg.num_samples]):
        start_index = np.random.randint(0, len(X) - timesteps)
        X_true[i] = X[start_index : start_index + timesteps]
        U_true[i] = U[start_index : start_index + timesteps]
        S_true[i] = S[start_index : start_index + timesteps]

    X_pred = np.stack([X_true] * train_cfg.num_models, axis=1)
    U_pred = np.stack([U_true] * train_cfg.num_models, axis=1)
    X_true, U_true, S_true, X_pred, U_pred = numpy_to_torch_device(X_true, U_true, S_true, X_pred, U_pred)

    log("Running forward estimation")
    TT.profile("Inference")
    for i, model in enumerate(models):
        for j in range(forward_cfg.num_sequences):
            seq_start = j * sequence_length
            seq_mid = j * sequence_length + train_cfg.H
            seq_end = (j + 1) * sequence_length
            if model.config.has_dynamics:
                _, X_pred[:, i, seq_mid:seq_end], _ = model(
                    X_true[:, seq_start:seq_mid],
                    U_true[:, seq_start:seq_mid],
                    S_true[:, seq_start:seq_mid],
                    S_true[:, seq_mid:seq_end],
                    Uf = U_true[:, seq_mid:seq_end],
                )
            if model.config.has_control:
                _, _, U_pred[:, i, seq_mid:seq_end] = model(
                    X_true[:, seq_start:seq_mid],
                    U_true[:, seq_start:seq_mid],
                    S_true[:, seq_start:seq_mid],
                    S_true[:, seq_mid:seq_end],
                    Xf = X_true[:, seq_mid:seq_end],
                )
    TT.end_profile()

    with TT.profile("Unstandardize"):
        X_true = X_true.cpu().numpy() * train_results.std_x + train_results.mean_x
        U_true = U_true.cpu().numpy() * train_results.std_u + train_results.mean_u
        X_pred = X_pred.cpu().numpy() * train_results.std_x + train_results.mean_x
        U_pred = U_pred.cpu().numpy() * train_results.std_u + train_results.mean_u

    if not models[0].config.has_dynamics:
        X_pred = None
    if not models[0].config.has_control:
        U_pred = None

    log("Plotting samples")
    with TT.profile("Plot"):
        plot_forward(path, env, train_cfg, train_results, forward_cfg, X_true, U_true, X_pred, U_pred)

if __name__ == "__main__":
    parser = Parser()
    job = parser.parse_args()

    log.configure(os.path.join(job.location, "forward.log"))

    with log.log_errors:
        log.section("Loading stuff to run forward evaluation")

        log("Loading config and results")
        train_cfg = TrainConfig.load(job.location)
        train_results = TrainResults.load(job.location)
        forward_cfg = ForwardConfig(num_samples=5, num_sequences=2)
        forward_cfg.save(job.location)

        env = train_cfg.get_env()

        log("Loading models")
        with TT.profile("Load model", hits=train_cfg.num_models):
            models = [FzNetwork.load(job.location, i).eval() for i in range(train_cfg.num_models)]

        log("Loading data")
        with TT.profile("Load data"):
            test_npz_files = list_processed_data_files(train_cfg.data_path, TEST_SUBDIR, train_cfg.phase)
            test_dataset = load_data_files(test_npz_files, train_cfg, max_num_files = 2 * forward_cfg.num_samples)
            log(
                "Loaded dataset",
                "Used test files:  %s" % thousands_seperators(len(test_dataset)),
                "Test data points: %s" % thousands_seperators(dataset_size(test_dataset)),
            )

        log("Standardizing data")
        with TT.profile("Load data"):
            standardize(env, test_dataset, train_results)

        log.section("Running forward evaluation")
        with TT.profile("Forward"):
            forward(
                job.location,
                env,
                models,
                test_dataset,
                train_cfg,
                train_results,
                forward_cfg,
            )

        log(TT)
