import os
import random
from typing import Type

import numpy as np
import torch
from pelutils import TT, log, thousands_seperators
from pelutils.parser import Parser

import frozone.environments as environments
from frozone.data.dataloader import DatasetSim, numpy_to_torch_device, standardize
from frozone.eval import SimulationConfig
from frozone.model.floatzone_network import FzNetwork
from frozone.train import TrainConfig, TrainResults


@torch.inference_mode()
def simulated_control(
    path: str,
    env: Type[environments.Environment],
    models: list[FzNetwork],
    dataset: DatasetSim,
    train_cfg: TrainConfig,
    train_results: TrainResults,
    simulation_cfg: SimulationConfig,
):
    X_true = np.empty((simulation_cfg.num_samples, timesteps, len(env.XLabels)), dtype=env.X_dtype)
    U_true = np.empty((simulation_cfg.num_samples, timesteps, len(env.ULabels)), dtype=env.U_dtype)
    S_true = np.empty((simulation_cfg.num_samples, timesteps, sum(env.S_bin_count)), dtype=env.S_dtype)
    Z_true = np.empty((simulation_cfg.num_samples, timesteps, len(env.ZLabels)), dtype=env.X_dtype)

    for i in simulation_cfg.num_samples:
        X_true[i], U_true[i], S_true[i], Z_true[i] = dataset[i]

    X_pred_by_model = np.stack([X_true] * train_cfg.num_models, axis=1)
    U_pred_by_model = np.stack([U_true] * train_cfg.num_models, axis=1)
    S_pred_by_model = np.stack([S_true] * train_cfg.num_models, axis=1)
    Z_pred_by_model = np.stack([Z_true] * train_cfg.num_models, axis=1)
    X_pred = X_true.copy()
    U_pred = U_true.copy()
    S_pred = S_true.copy()
    Z_pred = Z_true.copy()

    for i in simulation_cfg.simulation_steps(env):
        seq_start = i
        seq_mid = i + train_cfg.H
        seq_end = seq_end + train_cfg.F

        U_pred[:, seq_mid] = 0

        for j, model in enumerate(models):
            # Run for each model individually
            U_pred_by_model[:, j, seq_mid] = model(*numpy_to_torch_device(
                    X_pred_by_model[:, j, seq_start:seq_mid],
                    U_pred_by_model[:, j, seq_start:seq_mid],
                    S_pred_by_model[:, j, seq_start:seq_mid],
                    S_true[:, seq_mid:seq_end],
                ),
                Xf = numpy_to_torch_device(X_true[:, seq_mid:seq_end]),
            )[2][:, 0].cpu().numpy()

            X_pred_by_model[:, seq_mid], S_pred_by_model[:, seq_mid], Z_pred_by_model[:, seq_mid] = env.forward(
                X_pred_by_model[:, j, seq_mid-1],
                U_pred_by_model[:, j, seq_mid-1],
                S_pred_by_model[:, j, seq_mid-1],
                Z_pred_by_model[:, j, seq_mid-1],
            )

            # Ensemble prediction
            U_pred[:, seq_mid] += model(*numpy_to_torch_device(
                    X_pred[:, seq_start:seq_mid],
                    U_pred[:, seq_start:seq_mid],
                    S_pred[:, seq_start:seq_mid],
                    S_true[:, seq_mid:seq_end],
                ),
                Xf = numpy_to_torch_device(X_true[:, seq_mid:seq_end]),
            )[2][:, 0] / train_cfg.num_models

        X_pred[:, seq_mid], S_pred[:, seq_mid], Z_pred[:, seq_mid] = env.forward(
            X_pred[:, seq_mid-1],
            U_pred[:, seq_mid-1],
            S_pred[:, seq_mid-1],
            Z_pred[:, seq_mid-1],
        )


if __name__ == "__main__":
    parser = Parser()
    job = parser.parse_args()

    log.configure(os.path.join(job.location, "simulated-controller.log"))

    with log.log_errors:
        log.section("Loading stuff to run simulation")

        log("Loading config and results")
        train_cfg = TrainConfig.load(job.location)
        train_results = TrainResults.load(job.location)
        simulation_cfg = SimulationConfig(5, 500)
        simulation_cfg.save(job.location)

        env = train_cfg.get_env()
        assert env.is_simulation, "Loaded environment %s is not a simulation" % env.__name__

        log("Loading models")
        with TT.profile("Load model", hits=train_cfg.num_models):
            models = [FzNetwork.load(job.location, i).eval() for i in range(train_cfg.num_models)]
            assert all(model.config.has_control for model in models), "One or more of the models are not trained for control"

        log("Simulating data")
        timesteps = train_cfg.H + train_cfg.F + simulation_cfg.simulation_steps(env)
        X_all, U_all, S_all, Z_all = env.simulate(simulation_cfg.num_samples, timesteps-1, with_tqdm=False)
        dataset = [(X, U, S, Z) for X, U, S, Z in zip(X_all, U_all, S_all, Z_all)]

        log("Standardizing data")
        with TT.profile("Load data"):
            standardize(env, dataset, train_results)

        log.section("Running forward evaluation")
        with TT.profile("Forward"):
            simulated_control(
                job.location,
                env,
                models,
                dataset,
                train_cfg,
                train_results,
                simulation_cfg,
            )

        log(TT)
