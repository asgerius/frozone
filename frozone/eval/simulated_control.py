import math
import os
from datetime import datetime
from typing import Type

import numpy as np
import torch
from pelutils import TT, log
from pelutils.parser import Parser
from tqdm import tqdm

import frozone.environments as environments
from frozone.data import TEST_SUBDIR, Dataset, list_processed_data_files
from frozone.data.dataloader import dataset_size, load_data_files, numpy_to_torch_device, standardize
from frozone.eval import SimulationConfig
from frozone.model.floatzone_network import FzNetwork, interpolate
from frozone.plot.plot_simulated_control import plot_simulated_control
from frozone.train import TrainConfig, TrainResults


class ControllerStrategies:

    def __init__(
        self,
        models: list[tuple[FzNetwork, FzNetwork]],
        X_true: np.ndarray,
        S_true: np.ndarray,
        R_true: np.ndarray,
        train_cfg: TrainConfig,
        train_results: TrainResults,
        simulation_cfg: SimulationConfig,
    ):
        self.models = models
        self.X_true = X_true
        self.S_true = S_true
        self.X_true_d, self.S_true_d, self.R_true_d = numpy_to_torch_device(X_true, S_true, R_true)
        self.X_true_d.unsqueeze_(0)
        self.S_true_d.unsqueeze_(0)
        self.R_true_d.unsqueeze_(0)
        self.train_cfg = train_cfg
        self.train_results = train_results
        self.simulation_cfg = simulation_cfg
        self.env = self.train_cfg.get_env()
        self.control_interval = simulation_cfg.control_every_steps(self.env, train_cfg)

    def sequences(self, step: int) -> tuple[int, int, int, int]:
        seq_start = step
        seq_mid = seq_start + self.train_cfg.H
        seq_control = seq_mid + self.control_interval
        seq_end = seq_mid + self.train_cfg.F
        return seq_start, seq_mid, seq_control, seq_end

    @torch.inference_mode()
    def step_single_model(self, step: int, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray):
        seq_start, seq_mid, seq_control, seq_end = self.sequences(step)

        for j, (dynamics_model, control_model) in enumerate(self.models):
            Xh, Uh, Sh = numpy_to_torch_device(
                X[[j], seq_start:seq_mid],
                U[[j], seq_start:seq_mid],
                S[[j], seq_start:seq_mid],
            )
            U[j, seq_mid:seq_control, self.env.predicted_control] = interpolate(self.train_cfg.F, control_model(
                interpolate(self.train_cfg.Hi, Xh, self.train_cfg, h=True),
                interpolate(self.train_cfg.Hi, Uh, self.train_cfg, h=True),
                interpolate(self.train_cfg.Hi, Sh, self.train_cfg, h=True),
                Sf = interpolate(self.train_cfg.Fi, self.S_true_d[:, seq_mid:seq_end]),
                Xf = interpolate(self.train_cfg.Fi, self.R_true_d[:, seq_mid:seq_end]),
            )).cpu().numpy()[0, :self.control_interval, self.env.predicted_control]

            if self.env is environments.FloatZoneNNSim:
                X[[j], seq_mid:seq_control] = environments.FloatZoneNNSim.forward_standardized_multiple(
                    X[[j], seq_start:seq_mid],
                    U[[j], seq_start:seq_mid],
                    S[[j], seq_start:seq_mid],
                    S[[j], seq_mid:seq_end],
                    U[[j], seq_mid:seq_end],
                    self.simulation_cfg.control_every_steps(self.env, self.train_cfg),
                )
            else:
                for i in range(self.control_interval):
                    X[[j], seq_mid + i], S[[j], seq_mid + i], Z[[j], seq_mid + i] = self.env.forward_standardized(
                        X[[j], seq_mid+i-1],
                        U[[j], seq_mid+i-1],
                        S[[j], seq_mid+i-1],
                        Z[[j], seq_mid+i-1],
                        self.train_results,
                    )

    @torch.inference_mode()
    def step_ensemble(self, step: int, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray):
        seq_start, seq_mid, seq_control, seq_end = self.sequences(step)

        U[:, seq_mid:seq_control] = 0
        Xh, Uh, Sh = numpy_to_torch_device(
            X[:, seq_start:seq_mid],
            U[:, seq_start:seq_mid],
            S[:, seq_start:seq_mid],
        )

        for dynamics_model, control_model in self.models:
            U[:, seq_mid:seq_control] += control_model(
                Xh, Uh, Sh,
                self.S_true_d[:, seq_mid:seq_end],
                Xf = self.get_target_Xf(
                    Xh[..., self.env.reference_variables],
                    self.X_true_d[:, seq_mid:seq_end, self.env.reference_variables],
                ),
            )[:, :self.control_interval].cpu().numpy() / self.train_cfg.num_models

        U[:, seq_mid:seq_control] = self.env.limit_control(U[:, seq_mid:seq_control], mean=self.train_results.mean_u, std=self.train_results.std_u)

        if self.env is environments.FloatZoneNNSim:
            pass
        else:
            for i in range(self.control_interval):
                X[:, seq_mid + i], S[:, seq_mid + i], Z[:, seq_mid + i] = self.env.forward_standardized(
                    X[:, seq_mid+i-1],
                    U[:, seq_mid+i-1],
                    S[:, seq_mid+i-1],
                    Z[:, seq_mid+i-1],
                    self.train_results,
                )

    @torch.inference_mode(False)
    def step_optimized_ensemble(self, step: int, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray):
        seq_start, seq_mid, seq_control, seq_end = self.sequences(step)

        U[:, seq_mid:seq_end] = 0

        for i in range(self.simulation_cfg.num_samples):

            Xh, Uh, Sh = numpy_to_torch_device(
                X[[i], seq_start:seq_mid],
                U[[i], seq_start:seq_mid],
                S[[i], seq_start:seq_mid],
            )
            for dynamics_model, control_model in self.models:
                with torch.no_grad():
                    target_Xf = self.get_target_Xf(Xh, self.X_true_d[[i], seq_mid:seq_end])
                    Uf = control_model(
                        Xh, Uh, Sh,
                        Sf = self.S_true_d[[i], seq_mid:seq_end],
                        Xf = target_Xf[..., self.env.reference_variables],
                    )
                Uf.requires_grad_()
                optimizer = torch.optim.AdamW([Uf], lr=self.simulation_cfg.step_size)

                for _ in range(self.simulation_cfg.opt_steps):
                    Xf = dynamics_model(
                        Xh, Uh, Sh,
                        Sf = self.S_true_d[[i], seq_mid:seq_end],
                        Uf = Uf,
                    )

                    loss = dynamics_model.loss(target_Xf, Xf)
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                U[[i], seq_mid:seq_end] += Uf.detach().cpu().numpy() / self.train_cfg.num_models

        U[:, seq_mid:seq_control] = self.env.limit_control(U[:, seq_mid:seq_control], mean=self.train_results.mean_u, std=self.train_results.std_u)

        if self.env is environments.FloatZoneNNSim:
            pass
        else:
            for i in range(self.control_interval):
                X[:, seq_mid + i], S[:, seq_mid + i], Z[:, seq_mid + i] = self.env.forward_standardized(
                    X[:, seq_mid+i-1],
                    U[:, seq_mid+i-1],
                    S[:, seq_mid+i-1],
                    Z[:, seq_mid+i-1],
                    self.train_results,
                )

def simulated_control(
    path: str,
    env: Type[environments.Environment],
    models: list[tuple[FzNetwork, FzNetwork]],
    dataset: Dataset,
    train_cfg: TrainConfig,
    train_results: TrainResults,
    simulation_cfg: SimulationConfig, *,
    with_tqdm = False,
):
    simulation_cfg.save(path)

    for dm, cm in models:
        dm.eval().requires_grad_(False)
        cm.eval().requires_grad_(False)

    log("Simulating data")
    control_interval = simulation_cfg.control_every_steps(env, train_cfg)

    for i, (metadata, (X, U, S, R, Z)) in enumerate(dataset[:simulation_cfg.num_samples]):
        log(f"Sample {i+1:,} / {simulation_cfg.num_samples:,}")
        timesteps = X.shape[-2]
        X_true = X.copy()
        U_true = U.copy()
        S_true = S.copy()
        R_true = R.copy()
        Z_true = Z.copy()

        X_pred_by_model = np.stack([X_true] * train_cfg.num_models, axis=0)
        U_pred_by_model = np.stack([U_true] * train_cfg.num_models, axis=0)
        S_pred_by_model = np.stack([S_true] * train_cfg.num_models, axis=0)
        Z_pred_by_model = np.stack([Z_true] * train_cfg.num_models, axis=0)

        X_pred = X_true.copy()
        U_pred = U_true.copy()
        S_pred = S_true.copy()
        Z_pred = Z_true.copy()
        X_pred_opt = X_true.copy()
        U_pred_opt = U_true.copy()
        S_pred_opt = S_true.copy()
        Z_pred_opt = Z_true.copy()

        controller_strategies = ControllerStrategies(models, X_true, S_true, R_true, train_cfg, train_results, simulation_cfg)
        r = range((timesteps - train_cfg.H) // control_interval)

        TT.profile("Control")
        for j in tqdm(r, disable=not with_tqdm):
            with TT.profile("By model", hits=train_cfg.num_models):
                controller_strategies.step_single_model(j * control_interval, X_pred_by_model, U_pred_by_model, S_pred_by_model, Z_pred_by_model)
            # with TT.profile("Ensemble", hits = simulation_cfg.num_samples):
            #     controller_strategies.step_ensemble(j * control_interval, X_pred, U_pred, S_pred, Z_pred)
            # with TT.profile("Ensemble optimized", hits = simulation_cfg.num_samples):
                # controller_strategies.step_optimized_ensemble(j * control_interval, X_pred_opt, U_pred_opt, S_pred_opt, Z_pred_opt)

        TT.end_profile()

        with TT.profile("Unstandardize"):
            X_true = X_true * train_results.std_x + train_results.mean_x
            U_true = U_true * train_results.std_u + train_results.mean_u
            R_true = R_true * train_results.std_x[env.reference_variables] + train_results.mean_x[env.reference_variables]
            X_pred_by_model = X_pred_by_model * train_results.std_x + train_results.mean_x
            U_pred_by_model = U_pred_by_model * train_results.std_u + train_results.mean_u
            X_pred = X_pred * train_results.std_x + train_results.mean_x
            U_pred = U_pred * train_results.std_u + train_results.mean_u
            X_pred_opt = X_pred_opt * train_results.std_x + train_results.mean_x
            U_pred_opt = U_pred_opt * train_results.std_u + train_results.mean_u

        control_end = r.stop * control_interval + train_cfg.H
        with TT.profile("Plot"):
            plot_simulated_control(
                path = path,
                env = env,
                train_cfg = train_cfg,
                train_results = train_results,
                simulation_cfg = simulation_cfg,
                X_true = X_true,
                U_true = U_true,
                R_true = R_true,
                X_pred = X_pred[:control_end],
                U_pred = U_pred[:control_end],
                X_pred_opt = X_pred_opt[:control_end],
                U_pred_opt = U_pred_opt[:control_end],
                X_pred_by_model = X_pred_by_model[:, :control_end],
                U_pred_by_model = U_pred_by_model[:, :control_end],
                sample_no = i,
            )

    for dm, cm in models:
        dm.train().requires_grad_(True)
        cm.train().requires_grad_(True)

if __name__ == "__main__":
    parser = Parser()
    job = parser.parse_args()

    log.configure(os.path.join(job.location, "simulated-controller.log"))

    with log.log_errors:
        log.section("Loading stuff to run simulation")

        log("Loading config and results")
        train_cfg = TrainConfig.load(job.location)
        train_results = TrainResults.load(job.location)

        env = train_cfg.get_env()
        if env is environments.FloatZoneNNSim:
            env.load(TrainConfig.load(env.model_path), TrainResults.load(env.model_path))

        simulation_cfg = SimulationConfig(5, train_cfg.prediction_window, 5, 2e-2)

        log("Loading models")
        with TT.profile("Load model", hits=train_cfg.num_models):
            models = [FzNetwork.load(job.location, i) for i in range(train_cfg.num_models)]

        log("Loading data")
        with TT.profile("Load data"):
            test_npz_files = list_processed_data_files(train_cfg.data_path, TEST_SUBDIR)
            test_dataset, _ = load_data_files(test_npz_files, train_cfg, max_num_files=simulation_cfg.num_samples, year=datetime.now().year)
            log(
                "Loaded dataset",
                f"Used test files:  {len(test_dataset):,}",
                f"Test data points: {dataset_size(test_dataset):,}",
            )

        log("Standardizing data")
        with TT.profile("Standardize"):
            standardize(env, test_dataset, train_results)

        log.section("Running simulated control")
        with TT.profile("Simulate control"):
            simulated_control(
                job.location,
                env,
                models,
                test_dataset,
                train_cfg,
                train_results,
                simulation_cfg,
                with_tqdm=True,
            )

        log(TT)
