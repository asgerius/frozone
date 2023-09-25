import os
from typing import Type

import numpy as np
import torch
from pelutils import TT, log, thousands_seperators
from pelutils.parser import Flag, Parser

import frozone.environments as environments
from frozone.data import PROCESSED_SUBDIR, TEST_SUBDIR, TRAIN_SUBDIR, list_processed_data_files
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

    predict_steps = int(forward_cfg.prediction_window // env.dt)
    timesteps = train_cfg.H + train_cfg.F + predict_steps - 1
    log("Filtering dataset to only include files with at least %s data points" % thousands_seperators(timesteps))
    dataset = [(X, U, S) for X, U, S in dataset if len(X) >= timesteps]
    log("%s files after filtering" % thousands_seperators(len(dataset)))

    log("Sampling %i cases from the dataset" % forward_cfg.num_samples)
    dataset = [dataset[i] for i in np.random.choice(len(dataset), forward_cfg.num_samples, replace=True)]

    log("Sampling start indices")
    X_true = np.empty((forward_cfg.num_samples, timesteps, len(env.XLabels)), dtype=np.float32)
    U_true = np.empty((forward_cfg.num_samples, timesteps, len(env.ULabels)), dtype=np.float32)
    S_true = np.empty((forward_cfg.num_samples, timesteps, sum(env.S_bin_count)), dtype=np.float32)
    for i, (X, U, S) in enumerate(dataset):
        start_index = np.random.randint(0, len(X) - timesteps)
        X_true[i] = X[start_index : start_index + timesteps]
        U_true[i] = U[start_index : start_index + timesteps]
        S_true[i] = S[start_index : start_index + timesteps]
    X_pred_m = np.stack([X_true[:, : train_cfg.H + predict_steps]] * train_cfg.num_models, axis=2)
    X_pred_p = np.stack([X_true[:, : train_cfg.H + train_cfg.F]] * train_cfg.num_models, axis=2)
    X_pred_i = X_pred_m.copy()

    X_pred_m, X_pred_p, X_pred_i, U_true, S_true = numpy_to_torch_device(X_pred_m, X_pred_p, X_pred_i, U_true, S_true)

    log("Running forward estimation")
    for i in range(predict_steps):
        TT.profile("Time step")
        log.debug("Step %i / %i" % (i, predict_steps))
        for j, model in enumerate(models):
            X_pred_m[:, train_cfg.H + i, j] = model(
                X_pred_m[:, i : i + train_cfg.H].mean(dim=2),
                U_true[:, i : i + train_cfg.H],
                S_true[:, i : i + train_cfg.H],
                S_true[:, i + train_cfg.H : i + train_cfg.H + train_cfg.F],
                Xf = None,
                Uf = U_true[:, i + train_cfg.H : i + train_cfg.H + train_cfg.F],
            )[1][:, 0]
            X_pred_i[:, train_cfg.H + i, j] = model(
                X_pred_i[:, i : i + train_cfg.H, j],
                U_true[:, i : i + train_cfg.H],
                S_true[:, i : i + train_cfg.H],
                S_true[:, i + train_cfg.H : i + train_cfg.H + train_cfg.F],
                Xf = None,
                Uf = U_true[:, i + train_cfg.H : i + train_cfg.H + train_cfg.F],
            )[1][:, 0]

            if i == 0:
                X_pred_p[:, train_cfg.H:, j] = model(
                    X_pred_m[:, :train_cfg.H].mean(dim=2),
                    U_true[:, :train_cfg.H],
                    S_true[:, :train_cfg.H],
                    S_true[:, train_cfg.H : train_cfg.H + train_cfg.F],
                    Xf = None,
                    Uf = U_true[:, train_cfg.H : train_cfg.H + train_cfg.F],
                )[1]

        TT.end_profile()

    with TT.profile("Unstandardize"):
        X_true = X_true * train_results.std_x + train_results.mean_x
        X_pred_m = X_pred_m.cpu().numpy() * train_results.std_x + train_results.mean_x
        X_pred_p = X_pred_p.cpu().numpy() * train_results.std_x + train_results.mean_x
        X_pred_i = X_pred_i.cpu().numpy() * train_results.std_x + train_results.mean_x
        U_true = U_true.cpu().numpy() * train_results.std_u + train_results.mean_u

    log("Plotting samples")
    with TT.profile("Plot"):
        plot_forward(path, env, train_cfg, train_results, forward_cfg, X_true, X_pred_m, X_pred_p, X_pred_i, U_true)

if __name__ == "__main__":
    parser = Parser()
    job = parser.parse_args()

    log.configure(os.path.join(job.location, "forward.log"))

    with log.log_errors:
        log.section("Loading stuff to run forward evaluation")

        log("Loading config and results")
        train_cfg = TrainConfig.load(job.location)
        train_results = TrainResults.load(job.location)

        env = train_cfg.get_env()

        log("Loading models")
        with TT.profile("Load model", hits=train_cfg.num_models):
            models = [FzNetwork.load(job.location, i).eval() for i in range(train_cfg.num_models)]

        log("Loading data")
        with TT.profile("Load data"):
            test_npz_files = list_processed_data_files(train_cfg.data_path, TEST_SUBDIR, train_cfg.phase)
            test_dataset = load_data_files(test_npz_files, train_cfg)
            log(
                "Loaded dataset",
                "Used test files:  %s" % thousands_seperators(len(test_dataset)),
                "Test data points: %s" % thousands_seperators(dataset_size(test_dataset)),
            )

        log("Standardizing data")
        with TT.profile("Load data"):
            standardize(env, test_dataset, train_results)

        log.section("Running forward evaluation")
        with TT.profile("Forward evaluation"):
            forward(
                job.location,
                env,
                models,
                test_dataset,
                train_cfg,
                train_results,
                ForwardConfig(num_samples=5, prediction_window=150),
            )

        log(TT)
