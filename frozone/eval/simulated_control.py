import os
import shutil
from pprint import pformat
from typing import Type

import numpy as np
import torch
from pelutils import TT, log, Table, HardwareInfo
from pelutils.parser import Option, Parser
from pelutils.ds.stats import z
from tqdm import tqdm

import frozone.environments as environments
from frozone import cuda_sync
from frozone.data import TEST_SUBDIR, Dataset, list_processed_data_files, CONTROLLER_START
from frozone.data.dataloader import dataset_size, load_data_files, numpy_to_torch_device, standardize
from frozone.eval import SimulationConfig
from frozone.model.floatzone_network import FzNetwork, interpolate
from frozone.plot.plot_simulated_control import plot_simulated_control, plot_error, _plot_folder as sim_plot_folder
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

    def get_sequence(self, step: int) -> tuple[int, int, int, int]:
        seq_start = step
        seq_mid = seq_start + self.train_cfg.H
        seq_control = seq_mid + self.control_interval
        seq_end = seq_mid + self.train_cfg.F
        return seq_start, seq_mid, seq_control, seq_end

    @torch.inference_mode()
    def step_single_model(self, step: int, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray):
        seq_start, seq_mid, seq_control, seq_end = self.get_sequence(step)

        for j, (dynamics_model, control_model) in enumerate(self.models):
            Xh, Uh, Sh = numpy_to_torch_device(
                X[[j], seq_start:seq_mid],
                U[[j], seq_start:seq_mid],
                S[[j], seq_start:seq_mid],
            )
            if step == 0:
                # GPU warmup
                control_model(
                    interpolate(self.train_cfg.Hi, Xh, self.train_cfg, h=True),
                    interpolate(self.train_cfg.Hi, Uh, self.train_cfg, h=True),
                    interpolate(self.train_cfg.Hi, Sh, self.train_cfg, h=True),
                    Sf = interpolate(self.train_cfg.Fi, self.S_true_d[:, seq_mid:seq_end]),
                    Xf = interpolate(self.train_cfg.Fi, self.R_true_d[:, seq_mid:seq_end]),
                )
                cuda_sync()
            with TT.profile("Predict"):
                U[j, seq_mid:seq_control, self.env.predicted_control] = interpolate(self.train_cfg.F, control_model(
                    interpolate(self.train_cfg.Hi, Xh, self.train_cfg, h=True),
                    interpolate(self.train_cfg.Hi, Uh, self.train_cfg, h=True),
                    interpolate(self.train_cfg.Hi, Sh, self.train_cfg, h=True),
                    Sf = interpolate(self.train_cfg.Fi, self.S_true_d[:, seq_mid:seq_end]),
                    Xf = interpolate(self.train_cfg.Fi, self.R_true_d[:, seq_mid:seq_end]),
                )).cpu().numpy()[0, :self.control_interval, self.env.predicted_control]
                cuda_sync()

            with TT.profile("Limit control"):
                U[[j], seq_mid:seq_control] = self.env.limit_control(U[[j], seq_mid:seq_control], mean=self.train_results.mean_u, std=self.train_results.std_u)

            with TT.profile("Forward"):
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
        seq_start, seq_mid, seq_control, seq_end = self.get_sequence(step)

        U[0, seq_mid:seq_control, self.env.predicted_control] = 0
        Xh, Uh, Sh = numpy_to_torch_device(
            X[[0], seq_start:seq_mid],
            U[[0], seq_start:seq_mid],
            S[[0], seq_start:seq_mid],
        )

        for dynamics_model, control_model in self.models:
            with TT.profile("Predict"):
                U[0, seq_mid:seq_control, self.env.predicted_control] += interpolate(self.train_cfg.F, control_model(
                    interpolate(self.train_cfg.Hi, Xh, self.train_cfg, h=True),
                    interpolate(self.train_cfg.Hi, Uh, self.train_cfg, h=True),
                    interpolate(self.train_cfg.Hi, Sh, self.train_cfg, h=True),
                    Sf = interpolate(self.train_cfg.Fi, self.S_true_d[:, seq_mid:seq_end]),
                    Xf = interpolate(self.train_cfg.Fi, self.R_true_d[:, seq_mid:seq_end]),
                )).cpu().numpy()[0, :self.control_interval, self.env.predicted_control] / self.train_cfg.num_models
                cuda_sync()

        with TT.profile("Limit control"):
            U[:, seq_mid:seq_control] = self.env.limit_control(U[:, seq_mid:seq_control], mean=self.train_results.mean_u, std=self.train_results.std_u)

        with TT.profile("Forward"):
            if self.env is environments.FloatZoneNNSim:
                X[[0], seq_mid:seq_control] = environments.FloatZoneNNSim.forward_standardized_multiple(
                    X[[0], seq_start:seq_mid],
                    U[[0], seq_start:seq_mid],
                    S[[0], seq_start:seq_mid],
                    S[[0], seq_mid:seq_end],
                    U[[0], seq_mid:seq_end],
                    self.simulation_cfg.control_every_steps(self.env, self.train_cfg),
                )
            else:
                for i in range(self.control_interval):
                    X[[0], seq_mid + i], S[[0], seq_mid + i], Z[[0], seq_mid + i] = self.env.forward_standardized(
                        X[[0], seq_mid+i-1],
                        U[[0], seq_mid+i-1],
                        S[[0], seq_mid+i-1],
                        Z[[0], seq_mid+i-1],
                        self.train_results,
                    )

    @torch.inference_mode(False)
    def step_optimized_ensemble(self, step: int, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray):
        seq_start, seq_mid, seq_control, seq_end = self.get_sequence(step)

        U[0, seq_mid:seq_end, self.env.predicted_control] = 0

        Xh, Uh, Sh = numpy_to_torch_device(
            X[[0], seq_start:seq_mid],
            U[[0], seq_start:seq_mid],
            S[[0], seq_start:seq_mid],
        )
        Xh_interp = interpolate(self.train_cfg.Hi, Xh, self.train_cfg, h=True)
        Uh_interp = interpolate(self.train_cfg.Hi, Uh, self.train_cfg, h=True)
        Sh_interp = interpolate(self.train_cfg.Hi, Sh, self.train_cfg, h=True)
        Sf_interp = interpolate(self.train_cfg.Fi, self.S_true_d[[0], seq_mid:seq_end])
        Rf_interp = interpolate(self.train_cfg.Fi, self.R_true_d[[0], seq_mid:seq_end])
        Xf_interp = interpolate(self.train_cfg.Fi, self.X_true_d[[0], seq_mid:seq_end])
        Xf_interp[..., self.env.reference_variables] = Rf_interp

        for dynamics_model, control_model in self.models:
            with TT.profile("Predict"), torch.no_grad():
                Uf = control_model(
                    Xh_interp, Uh_interp, Sh_interp,
                    Sf = Sf_interp,
                    Xf = Rf_interp
                )
                cuda_sync()
            Uf.requires_grad_()
            optimizer = torch.optim.AdamW([Uf], lr=self.simulation_cfg.step_size)

            for _ in range(self.simulation_cfg.opt_steps):
                with TT.profile("Step"):
                    Xf = dynamics_model(
                        Xh_interp, Uh_interp, Sh_interp,
                        Sf = Sf_interp,
                        Uf = Uf,
                    )

                    loss = dynamics_model.loss(Xf_interp, Xf)
                    loss.backward()
                    Uf.grad[..., self.env.predefined_control] = 0

                    optimizer.step()
                    optimizer.zero_grad()

                    cuda_sync()

            U[0, seq_mid:seq_control, self.env.predicted_control] += interpolate(self.train_cfg.F, Uf) \
                .detach() \
                .cpu() \
                .numpy()[0, :self.control_interval, self.env.predicted_control] / self.train_cfg.num_models

        with TT.profile("Limit control"):
            U[:, seq_mid:seq_control] = self.env.limit_control(U[:, seq_mid:seq_control], mean=self.train_results.mean_u, std=self.train_results.std_u)

        with TT.profile("Forward"):
            if self.env is environments.FloatZoneNNSim:
                X[[0], seq_mid:seq_control] = environments.FloatZoneNNSim.forward_standardized_multiple(
                    X[[0], seq_start:seq_mid],
                    U[[0], seq_start:seq_mid],
                    S[[0], seq_start:seq_mid],
                    S[[0], seq_mid:seq_end],
                    U[[0], seq_mid:seq_end],
                    self.simulation_cfg.control_every_steps(self.env, self.train_cfg),
                )
            else:
                for i in range(self.control_interval):
                    X[:, seq_mid + i], S[:, seq_mid + i], Z[:, seq_mid + i] = self.env.forward_standardized(
                        X[:, seq_mid+i-1],
                        U[:, seq_mid+i-1],
                        S[:, seq_mid+i-1],
                        Z[:, seq_mid+i-1],
                        self.train_results,
                    )

def simulate_control(
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
    shutil.rmtree(os.path.join(path, sim_plot_folder), ignore_errors=True)
    os.mkdir(os.path.join(path, sim_plot_folder))

    for dm, cm in models:
        dm.eval().requires_grad_(False)
        cm.eval().requires_grad_(False)

    log("Simulating data", simulation_cfg)
    control_interval = simulation_cfg.control_every_steps(env, train_cfg)
    control_start_step = int(CONTROLLER_START // env.dt)

    # Each array has shape 4 x timesteps x number of reference values
    # First row is the reference values
    # Second is the values under a single model
    # Third is under ensemble
    # Fourth is under optimization
    results = list()

    for i, (metadata, (X, U, S, R, Z)) in enumerate(tqdm(dataset[:simulation_cfg.num_samples], position=0, disable=not with_tqdm)):
        log.debug(f"Sample {i+1:,} / {simulation_cfg.num_samples:,}")
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

        X_pred = np.expand_dims(X_true.copy(), axis=0)
        U_pred = np.expand_dims(U_true.copy(), axis=0)
        S_pred = np.expand_dims(S_true.copy(), axis=0)
        Z_pred = np.expand_dims(Z_true.copy(), axis=0)
        X_pred_opt = np.expand_dims(X_true.copy(), axis=0)
        U_pred_opt = np.expand_dims(U_true.copy(), axis=0)
        S_pred_opt = np.expand_dims(S_true.copy(), axis=0)
        Z_pred_opt = np.expand_dims(Z_true.copy(), axis=0)

        controller_strategies = ControllerStrategies(models, X_true, S_true, R_true, train_cfg, train_results, simulation_cfg)
        control_step_range = range((timesteps - control_start_step) // control_interval)

        TT.profile("Control")
        try:
            for j in tqdm(control_step_range, position=1, disable=not with_tqdm):
                with TT.profile("By model"):
                    controller_strategies.step_single_model(j * control_interval + control_start_step - train_cfg.H, X_pred_by_model, U_pred_by_model, S_pred_by_model, Z_pred_by_model)
                with TT.profile("Ensemble"):
                    controller_strategies.step_ensemble(j * control_interval + control_start_step - train_cfg.H, X_pred, U_pred, S_pred, Z_pred)
                with TT.profile("Ensemble optimized"):
                    controller_strategies.step_optimized_ensemble(j * control_interval + control_start_step - train_cfg.H, X_pred_opt, U_pred_opt, S_pred_opt, Z_pred_opt)
        # except Exception as e:
        #     log.error("Simulation %i failed" % i)
        #     log.log_with_stacktrace(e)
        #     continue
        finally:
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

        # Undo extra dimension
        X_pred = X_pred[0]
        U_pred = U_pred[0]
        S_pred = S_pred[0]
        Z_pred = Z_pred[0]
        X_pred_opt = X_pred_opt[0]
        U_pred_opt = U_pred_opt[0]
        S_pred_opt = S_pred_opt[0]
        Z_pred_opt = Z_pred_opt[0]

        control_end = control_step_range.stop * control_interval + control_start_step
        if i < 10:
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

        results.append(np.stack([
            R_true[:control_end],
            *X_pred_by_model[:, :control_end, env.reference_variables],
            X_pred[:control_end, env.reference_variables],
            X_pred_opt[:control_end, env.reference_variables],
        ], axis=0))

    results = np.array(results)
    np.save(os.path.join(path, "simulation-results.npy"), results)

    control_methods = (*("Model %i" % (i + 1) for i in range(train_cfg.num_models)), "Ensemble", "Optimized ensemble")
    # Error has shape num samples x control methods x time steps x reference variables
    error = np.abs(np.stack([
        results[:, 0] - results[:, i+1] for i in range(len(control_methods))
    ], axis=1))
    sorted_error = np.sort(error, axis=0)
    error_calcs = dict()
    for i, rlab in enumerate(env.reference_variables):
        error_table = Table()
        error_table.add_header(["", "error_mean", "error_80", "error_100"])
        error_calcs[rlab] = dict()
        for j, control_method in enumerate(control_methods):
            error_calcs[rlab][control_method] = {
                "error_mean": error[:, j, :, i].mean(axis=0),
                "error_80":   sorted_error[int(0.8 * len(error)), j, :, i],
                "error_100":  sorted_error[-1, j, :, i],
            }
            error_table.add_row([
                control_method,
                "%.3f +/- %.4f" % (
                    error_calcs[rlab][control_method]["error_mean"][control_start_step:].mean(),
                    z() * error_calcs[rlab][control_method]["error_mean"][control_start_step:].std(ddof=1) / np.sqrt(results.shape[-2] - CONTROLLER_START),
                ),
                "%.3f +/- %.4f" % (
                    error_calcs[rlab][control_method]["error_80"][control_start_step:].mean(),
                    z() * error_calcs[rlab][control_method]["error_80"][control_start_step:].std(ddof=1) / np.sqrt(results.shape[-2] - CONTROLLER_START),
                ),
                "%.3f +/- %.4f" % (
                    error_calcs[rlab][control_method]["error_100"][control_start_step:].mean(),
                    z() * error_calcs[rlab][control_method]["error_100"][control_start_step:].std(ddof=1) / np.sqrt(len(results)),
                ),
            ], [1, 0, 0, 0])

        log(f"Loss statistics for {env.format_label(rlab)}", error_table)

    np.savez_compressed(os.path.join(path, "simulation-results.npz"), results=results, error=error, error_calcs=error_calcs)

    with TT.profile("Plot error"):
        plot_error(path, env, train_cfg, simulation_cfg, error_calcs)

    for dm, cm in models:
        dm.train().requires_grad_(True)
        cm.train().requires_grad_(True)

if __name__ == "__main__":
    parser = Parser(
        Option("num-simulations", default=None, type=int),
        Option("control-window", default=None, type=int),
        Option("opt-steps", default=15),
        Option("step-size", default=1e-3),
    )
    job = parser.parse_args()
    job.location = os.path.join(job.location, job.name)
    job.prepare_directory()

    log.configure(os.path.join(job.location, "simulated-controller.log"))

    with log.log_errors:
        job.prepare_directory()
        log.section("Job %s" % job.name, vars(job))
        log.log_repo()
        log(HardwareInfo.string())

        log.section("Loading stuff to run simulation")
        log(pformat(vars(job)))

        log("Loading config and results")
        traindir = os.path.join(job.location, os.path.pardir)
        train_cfg = TrainConfig.load(traindir)
        job.control_window = job.control_window or train_cfg.prediction_window
        assert 0 < job.control_window <= train_cfg.prediction_window
        train_results = TrainResults.load(traindir)

        env = train_cfg.get_env()
        if env is environments.FloatZoneNNSim:
            env.load(TrainConfig.load(env.model_path), TrainResults.load(env.model_path))

        log("Loading models")
        with TT.profile("Load model", hits=train_cfg.num_models):
            models = [FzNetwork.load(traindir, i) for i in range(train_cfg.num_models)]

        log("Loading data")
        test_npz_files = list_processed_data_files(train_cfg.data_path, TEST_SUBDIR)
        simulation_cfg = SimulationConfig(
            job.num_simulations or len(test_npz_files),
            job.control_window,
            job.opt_steps,
            job.step_size,
        )
        with TT.profile("Load data"):
            test_dataset, _ = load_data_files(test_npz_files, train_cfg, max_num_files=simulation_cfg.num_samples)
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
            simulate_control(
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
