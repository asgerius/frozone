import os
import random
from datetime import datetime
from typing import Type

import numpy as np
import torch
from pelutils import TT, log, LogLevels, thousands_seperators
from pelutils.parser import Parser

import frozone.environments as environments
from frozone import cuda_sync, device_x, device_u
from frozone.data import TEST_SUBDIR, list_processed_data_files
from frozone.data.dataloader import Dataset, dataset_size, load_data_files, numpy_to_torch_device, standardize
from frozone.eval import ForwardConfig
from frozone.model.floatzone_network import FzNetwork
from frozone.plot.plot_forward import plot_forward
from frozone.train import TrainConfig, TrainResults


def forward(
    path: str,
    env: Type[environments.Environment],
    models: list[tuple[FzNetwork, FzNetwork]],
    dataset: Dataset,
    train_cfg: TrainConfig,
    train_results: TrainResults,
    forward_cfg: ForwardConfig,
):

    for dm, cm in models:
        dm.requires_grad_(False)
        cm.requires_grad_(False).to(device_x)

    sequence_length = train_cfg.H + train_cfg.F
    timesteps = forward_cfg.num_sequences * sequence_length
    log("Filtering dataset to only include files with at least %s data points" % thousands_seperators(timesteps))
    dataset = [data for data in dataset if data[0].length >= timesteps][:forward_cfg.num_samples]
    forward_cfg.num_samples = len(dataset)  # This is no good, very terrible code
    random.shuffle(dataset)
    log("%s files after filtering" % thousands_seperators(len(dataset)))

    log("Sampling start indices")
    X_true = np.empty((forward_cfg.num_samples, timesteps, len(env.XLabels)), dtype=env.X_dtype)
    U_true = np.empty((forward_cfg.num_samples, timesteps, len(env.ULabels)), dtype=env.U_dtype)
    S_true = np.empty((forward_cfg.num_samples, timesteps, sum(env.S_bin_count)), dtype=env.S_dtype)
    R_true = np.empty((forward_cfg.num_samples, timesteps, len(env.reference_variables)), dtype=env.X_dtype)
    metadatas = list()
    for i, (metadata, (X, U, S, R)) in enumerate(dataset[:forward_cfg.num_samples]):
        metadatas.append(metadata)
        start_index = 0  # np.random.randint(0, len(X) - timesteps)
        X_true[i] = X[start_index : start_index + timesteps]
        U_true[i] = U[start_index : start_index + timesteps]
        S_true[i] = S[start_index : start_index + timesteps]
        R_true[i] = R[start_index : start_index + timesteps]

    X_pred = np.stack([X_true] * train_cfg.num_models, axis=1)
    U_pred = np.stack([U_true] * train_cfg.num_models, axis=1)
    U_pred_opt = U_true.copy()
    U_pred_ref = U_true.copy()
    X_true, U_true, S_true, X_pred, U_pred, U_pred_ref, R_true = numpy_to_torch_device(
        X_true, U_true, S_true, X_pred, U_pred, U_pred_ref, R_true, device=device_x,
    )
    # This is just for testing purposes
    # R_true = torch.rand_like(R_true)

    X_true_smooth = X_true.clone()
    U_true_smooth = U_true.clone()

    log("Running forward predictions")
    TT.profile("Inference")
    for j in range(forward_cfg.num_sequences):
        seq_start = j * sequence_length
        seq_mid = j * sequence_length + train_cfg.H
        seq_end = (j + 1) * sequence_length

        U_pred_opt[:, seq_mid:seq_end] = 0
        U_pred_ref[:, seq_mid:seq_end] = 0

        for i, (dynamics_model, control_model) in enumerate(models):

            if i == 0:
                X_true_smooth[:, seq_start:seq_mid] = dynamics_model.smoothen_h(X_true[:, seq_start:seq_mid])
                X_true_smooth[:, seq_mid:seq_end] = dynamics_model.smoothen_f(X_true[:, seq_mid:seq_end])
                U_true_smooth[:, seq_start:seq_mid] = control_model.smoothen_h(U_true[:, seq_start:seq_mid])
                U_true_smooth[:, seq_mid:seq_end] = control_model.smoothen_f(U_true[:, seq_mid:seq_end])

            with TT.profile("Dynamics"), torch.inference_mode():
                X_pred[:, i, seq_mid:seq_end] = dynamics_model(
                    X_true[:, seq_start:seq_mid],
                    U_true[:, seq_start:seq_mid],
                    S_true[:, seq_start:seq_mid],
                    Sf = S_true[:, seq_mid:seq_end],
                    Uf = U_true[:, seq_mid:seq_end],
                )
                cuda_sync()

            with TT.profile("Control"), torch.inference_mode():
                U_pred[:, i, seq_mid:seq_end] = control_model(
                    X_true[:, seq_start:seq_mid],
                    U_true[:, seq_start:seq_mid],
                    S_true[:, seq_start:seq_mid],
                    Sf = S_true[:, seq_mid:seq_end],
                    Xf = X_true[:, seq_mid:seq_end, env.reference_variables],
                )
                cuda_sync()

            with TT.profile("Control (optimized)"), torch.inference_mode(False):

                for k in range(forward_cfg.num_samples):

                    Uf = U_pred[[k], i, seq_mid:seq_end].clone()
                    Uf.requires_grad_()
                    optimizer = torch.optim.AdamW([Uf], lr=forward_cfg.step_size)

                    for _ in range(forward_cfg.opt_steps):
                        Xf = dynamics_model(
                            X_true[[k], seq_start:seq_mid],
                            U_true[[k], seq_start:seq_mid],
                            S_true[[k], seq_start:seq_mid],
                            Sf = S_true[[k], seq_mid:seq_end],
                            Uf = Uf,
                        )

                        loss = dynamics_model.loss(X_true[[k], seq_mid:seq_end], Xf)
                        loss.backward()

                        optimizer.step()
                        optimizer.zero_grad()

                    U_pred_opt[[k], seq_mid:seq_end] += Uf.detach().cpu().numpy() / train_cfg.num_models

                cuda_sync()

            with TT.profile("Control (reference)"), torch.inference_mode():
                U_pred_ref[:, seq_mid:seq_end] += control_model(
                    X_true[:, seq_start:seq_mid],
                    U_true[:, seq_start:seq_mid],
                    S_true[:, seq_start:seq_mid],
                    Sf = S_true[:, seq_mid:seq_end],
                    Xf = R_true[:, seq_mid:seq_end],
                ).squeeze() / train_cfg.num_models
                cuda_sync()

    TT.end_profile()

    with TT.profile("Unstandardize"):
        X_true = X_true.cpu().numpy() * train_results.std_x + train_results.mean_x
        U_true = U_true.cpu().numpy() * train_results.std_u + train_results.mean_u
        X_true_smooth = X_true_smooth.cpu().numpy() * train_results.std_x + train_results.mean_x
        U_true_smooth = U_true_smooth.cpu().numpy() * train_results.std_u + train_results.mean_u
        X_pred = X_pred.cpu().numpy() * train_results.std_x + train_results.mean_x
        U_pred = U_pred.cpu().numpy() * train_results.std_u + train_results.mean_u
        U_pred_opt = U_pred_opt * train_results.std_u + train_results.mean_u
        U_pred_ref = U_pred_ref.cpu().numpy() * train_results.std_u + train_results.mean_u
        R_true = R_true.cpu().numpy() * train_results.std_x[env.reference_variables] + train_results.mean_x[env.reference_variables]

    log("Plotting samples")
    with TT.profile("Plot"):
        plot_forward(
            path, env,
            train_cfg, train_results, forward_cfg, metadatas,
            X_true, U_true, X_true_smooth, U_true_smooth,
            X_pred, U_pred, U_pred_opt, U_pred_ref, R_true,
        )

    for dm, cm in models:
        dm.requires_grad_(True)
        cm.requires_grad_(True).to(device_u)

if __name__ == "__main__":
    parser = Parser()
    job = parser.parse_args()

    log.configure(os.path.join(job.location, "forward.log"), print_level=LogLevels.DEBUG)

    with log.log_errors:
        log.section("Loading stuff to run forward evaluation")

        log("Loading config and results")
        train_cfg = TrainConfig.load(job.location)
        train_results = TrainResults.load(job.location)
        env = train_cfg.get_env()
        forward_cfg = ForwardConfig(num_samples=5, num_sequences=3 if env is environments.FloatZone else 1, opt_steps=0, step_size=1e-2)
        forward_cfg.save(job.location)

        log("Loading models")
        with TT.profile("Load model", hits=train_cfg.num_models):
            models = [FzNetwork.load(job.location, i) for i in range(train_cfg.num_models)]
            for dm, cm in models:
                dm.eval()
                cm.eval()

        log("Loading data")
        with TT.profile("Load data"):
            test_npz_files = list_processed_data_files(train_cfg.data_path, TEST_SUBDIR)
            test_dataset, _ = load_data_files(test_npz_files, train_cfg, max_num_files=3*forward_cfg.num_samples, year=datetime.now().year)
            log(
                "Loaded dataset",
                "Used test files:  %s" % thousands_seperators(len(test_dataset)),
                "Test data points: %s" % thousands_seperators(dataset_size(test_dataset)),
            )

        log("Standardizing data")
        with TT.profile("Standardize"):
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
