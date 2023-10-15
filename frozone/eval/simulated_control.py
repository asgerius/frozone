import os
from typing import Type

import numpy as np
import torch
from pelutils import TT, log
from pelutils.parser import Parser
from tqdm import tqdm

import frozone.environments as environments
from frozone import device_x, device_u
from frozone.data.dataloader import numpy_to_torch_device, standardize
from frozone.eval import SimulationConfig
from frozone.model.floatzone_network import FzNetwork
from frozone.plot.plot_simulated_control import plot_simulated_control
from frozone.train import TrainConfig, TrainResults


class ControllerStrategies:

    def __init__(
        self,
        models: list[tuple[FzNetwork, FzNetwork]],
        X_true: np.ndarray,
        S_true: np.ndarray,
        train_cfg: TrainConfig,
        train_results: TrainResults,
        simulation_cfg: SimulationConfig,
    ):
        self.models = models
        self.X_true = X_true
        self.S_true = S_true
        self.X_true_d, self.S_true_d = numpy_to_torch_device(X_true, S_true, device=device_x)
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

    def get_target_Xf(self, Xh: torch.FloatTensor, Xf_ref: torch.FloatTensor) -> torch.FloatTensor:
        """ Returns the target, which is a linear mix of Xh and self.X_true. """
        steps = self.simulation_cfg.correction_steps(self.env, self.train_cfg)
        # 0 For only xH and 1 for X_true
        step_weight = torch.linspace(0, 1, steps + 1, device=Xh.device)[1:]

        XH = torch.stack([Xh[:, -1]] * steps, dim=-1)
        X_target = Xf_ref.clone()
        X_target[:, :steps] = ((1 - step_weight) * XH + step_weight * X_target[:, :steps].permute(0, 2, 1)).permute(0, 2, 1)

        return X_target

    @torch.inference_mode()
    def step_single_model(self, step: int, X: np.ndarray, U: np.ndarray, S: np.ndarray, Z: np.ndarray):
        seq_start, seq_mid, seq_control, seq_end = self.sequences(step)

        for j, (dynamics_model, control_model) in enumerate(self.models):
            Xh, Uh, Sh = numpy_to_torch_device(
                X[:, j, seq_start:seq_mid],
                U[:, j, seq_start:seq_mid],
                S[:, j, seq_start:seq_mid],
                device = device_x,
            )
            U[:, j, seq_mid:seq_control] = control_model(
                Xh, Uh, Sh,
                Sf = self.S_true_d[:, seq_mid:seq_end],
                Xf = self.get_target_Xf(Xh, self.X_true_d[:, seq_mid:seq_end]),
            )[:, :self.control_interval].cpu().numpy()

            U[:, j, seq_mid] = self.env.limit_control(U[:, j, seq_mid], mean=self.train_results.mean_u, std=self.train_results.std_u)

            for i in range(self.control_interval):
                X[:, j, seq_mid + i], S[:, j, seq_mid + i], Z[:, j, seq_mid + i] = self.env.forward_standardized(
                    X[:, j, seq_mid+i-1],
                    U[:, j, seq_mid+i-1],
                    S[:, j, seq_mid+i-1],
                    Z[:, j, seq_mid+i-1],
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
            device = device_x,
        )

        for dynamics_model, control_model in self.models:
            U[:, seq_mid:seq_control] += control_model(
                Xh, Uh, Sh,
                self.S_true_d[:, seq_mid:seq_end],
                Xf = self.get_target_Xf(Xh, self.X_true_d[:, seq_mid:seq_end]),
            )[:, :self.control_interval].cpu().numpy() / self.train_cfg.num_models

        U[:, seq_mid:seq_control] = self.env.limit_control(U[:, seq_mid:seq_control], mean=self.train_results.mean_u, std=self.train_results.std_u)

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
                device = device_x,
            )
            for dynamics_model, control_model in self.models:
                with torch.no_grad():
                    target_Xf = self.get_target_Xf(Xh, self.X_true_d[[i], seq_mid:seq_end])
                    Uf = control_model(
                        Xh, Uh, Sh,
                        Sf = self.S_true_d[[i], seq_mid:seq_end],
                        Xf = target_Xf,
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
    train_cfg: TrainConfig,
    train_results: TrainResults,
    simulation_cfg: SimulationConfig, *,
    with_tqdm = False,
):
    simulation_cfg.save(path)

    for dm, cm in models:
        dm.eval().requires_grad_(False)
        cm.eval().requires_grad_(False).to(device_x)

    log("Simulating data")
    offset_steps = 0  # int(1000 / env.dt)
    timesteps = train_cfg.H + simulation_cfg.simulation_steps(env) + train_cfg.F - simulation_cfg.control_every_steps(env, train_cfg)
    X_all, U_all, S_all, Z_all = env.simulate(simulation_cfg.num_samples, timesteps + offset_steps, with_tqdm=False)
    dataset = [(X[offset_steps:], U[offset_steps:], S[offset_steps:], Z[offset_steps:]) for X, U, S, Z in zip(X_all, U_all, S_all, Z_all)]

    log("Standardizing data")
    with TT.profile("Standardize"):
        standardize(env, dataset, train_results)

    X_true = np.empty((simulation_cfg.num_samples, timesteps, len(env.XLabels)), dtype=env.X_dtype)
    U_true = np.empty((simulation_cfg.num_samples, timesteps, len(env.ULabels)), dtype=env.U_dtype)
    S_true = np.empty((simulation_cfg.num_samples, timesteps, sum(env.S_bin_count)), dtype=env.S_dtype)
    Z_true = np.empty((simulation_cfg.num_samples, timesteps, len(env.ZLabels)), dtype=env.X_dtype)

    for i in range(simulation_cfg.num_samples):
        X_true[i], U_true[i], S_true[i], Z_true[i] = dataset[i]

    X_pred_by_model = np.stack([X_true] * train_cfg.num_models, axis=1)
    U_pred_by_model = np.stack([U_true] * train_cfg.num_models, axis=1)
    S_pred_by_model = np.stack([S_true] * train_cfg.num_models, axis=1)
    Z_pred_by_model = np.stack([Z_true] * train_cfg.num_models, axis=1)
    X_pred = X_true.copy()
    U_pred = U_true.copy()
    S_pred = S_true.copy()
    Z_pred = Z_true.copy()
    X_pred_opt = X_true.copy()
    U_pred_opt = U_true.copy()
    S_pred_opt = S_true.copy()
    Z_pred_opt = Z_true.copy()

    controller_strategies = ControllerStrategies(models, X_true, S_true, train_cfg, train_results, simulation_cfg)

    log("Running simulation")
    TT.profile("Step", hits=simulation_cfg.simulation_steps(env))
    control_interval = simulation_cfg.control_every_steps(env, train_cfg)
    r = range(simulation_cfg.simulation_steps(env) // control_interval)
    for i in tqdm(r) if with_tqdm else r:
        with TT.profile("By model", hits = simulation_cfg.num_samples * train_cfg.num_models):
            controller_strategies.step_single_model(i * control_interval, X_pred_by_model, U_pred_by_model, S_pred_by_model, Z_pred_by_model)
        with TT.profile("Ensemble", hits = simulation_cfg.num_samples):
            controller_strategies.step_ensemble(i * control_interval, X_pred, U_pred, S_pred, Z_pred)
        with TT.profile("Ensemble optimized", hits = simulation_cfg.num_samples):
            controller_strategies.step_optimized_ensemble(i * control_interval, X_pred_opt, U_pred_opt, S_pred_opt, Z_pred_opt)

    TT.end_profile()

    with TT.profile("Unstandardize"):
        X_true = X_true * train_results.std_x + train_results.mean_x
        U_true = U_true * train_results.std_u + train_results.mean_u
        X_pred = X_pred * train_results.std_x + train_results.mean_x
        U_pred = U_pred * train_results.std_u + train_results.mean_u
        X_pred_opt = X_pred_opt * train_results.std_x + train_results.mean_x
        U_pred_opt = U_pred_opt * train_results.std_u + train_results.mean_u
        X_pred_by_model = X_pred_by_model * train_results.std_x + train_results.mean_x
        U_pred_by_model = U_pred_by_model * train_results.std_u + train_results.mean_u

    log("Plotting samples")
    with TT.profile("Plot"):
        plot_simulated_control(
            path = path,
            env = env,
            train_cfg = train_cfg,
            train_results = train_results,
            simulation_cfg = simulation_cfg,
            X_true = X_true,
            U_true = U_true,
            X_pred = X_pred,
            U_pred = U_pred,
            X_pred_opt = X_pred_opt,
            U_pred_opt = U_pred_opt,
            X_pred_by_model = X_pred_by_model,
            U_pred_by_model = U_pred_by_model,
        )

    for dm, cm in models:
        dm.train().requires_grad_(True)
        cm.train().requires_grad_(True).to(device_u)

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
        assert env.is_simulation, "Loaded environment %s is not a simulation" % env.__name__

        simulation_cfg = SimulationConfig(5, train_cfg.prediction_window, train_cfg.prediction_window, env.dt, 5, 2e-2)

        log("Loading models")
        with TT.profile("Load model", hits=train_cfg.num_models):
            models = [FzNetwork.load(job.location, i) for i in range(train_cfg.num_models)]

        log.section("Running simulated control")
        with TT.profile("Simulate control"):
            simulated_control(
                job.location,
                env,
                models,
                train_cfg,
                train_results,
                simulation_cfg,
                with_tqdm=True,
            )

        log(TT)
