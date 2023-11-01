import math
from typing import Type

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from pelutils import TT, JobDescription, log, thousands_seperators, HardwareInfo

import frozone.environments as environments
from frozone import device, amp_context, cuda_sync
from frozone.data import list_processed_data_files, PROCESSED_SUBDIR
from frozone.data.dataloader import dataloader, dataset_size, load_data_files, standardize
from frozone.data.process_raw_floatzone_data import TEST_SUBDIR, TRAIN_SUBDIR
from frozone.eval import ForwardConfig, SimulationConfig
from frozone.eval.forward import forward
from frozone.eval.simulated_control import simulated_control
from frozone.model.floatzone_network import FzConfig, FzNetwork
from frozone.plot.plot_train import plot_loss, plot_lr
from frozone.train import TrainConfig, TrainResults


def build_lr_scheduler(config: TrainConfig, optimizer: torch.optim.Optimizer):
    return lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = config.lr,
        epochs = config.batches,
        steps_per_epoch = 1,
    )

def train(job: JobDescription):

    log(HardwareInfo.string())

    train_cfg = TrainConfig(
        env = job.env,
        data_path = job.data_path,
        phase = job.phase,
        history_window = job.history_window,
        prediction_window = job.prediction_window,
        history_interp = job.history_interp,
        prediction_interp = job.prediction_interp,
        batches = job.batches,
        batch_size = job.batch_size,
        lr = job.lr,
        num_models = job.num_models,
        max_num_data_files = job.max_num_data_files,
        eval_size = job.eval_size,
        loss_fn = job.loss_fn,
        huber_delta = job.huber_delta,
        epsilon = job.epsilon,
        augment_prob = job.augment_prob,
    )
    train_results = TrainResults.empty(train_cfg.num_models)

    log("Training configuration", train_cfg)

    env: Type[environments.Environment] = getattr(environments, train_cfg.env)
    log("Got environment %s" % env.__name__)

    train_npz_files = list_processed_data_files(train_cfg.data_path, TRAIN_SUBDIR, where=PROCESSED_SUBDIR)
    test_npz_files = list_processed_data_files(train_cfg.data_path, TEST_SUBDIR, where=PROCESSED_SUBDIR)
    log(
        "Found data files",
        "Train: %s (%.2f %%)" % (thousands_seperators(len(train_npz_files)), 100 * len(train_npz_files) / (len(train_npz_files) + len(test_npz_files))),
        "Test:  %s (%.2f %%)" % (thousands_seperators(len(test_npz_files)), 100 * len(test_npz_files) / (len(train_npz_files) + len(test_npz_files))),
    )
    log("Loading data")
    with TT.profile("Load data"):
        train_dataset, _ = load_data_files(train_npz_files, train_cfg, max_num_files=train_cfg.max_num_data_files)
        test_dataset, _ = load_data_files(test_npz_files, train_cfg, max_num_files=train_cfg.max_num_data_files)
    log(
        "Loaded datasets",
        "Used train files:  %s" % thousands_seperators(len(train_dataset)),
        "Used test files:   %s" % thousands_seperators(len(test_dataset)),
        "Train data points: %s" % thousands_seperators(dataset_size(train_dataset)),
        "Test data points:  %s" % thousands_seperators(dataset_size(test_dataset)),
    )

    log("Standardizing data")
    with TT.profile("Standardize"):
        standardize(env, train_dataset, train_results)
        standardize(env, test_dataset, train_results)
    train_dataloader = dataloader(env, train_cfg, train_dataset, train=True)
    test_dataloader = dataloader(env, train_cfg, test_dataset)

    log("Building %i models" % train_cfg.num_models)
    models: list[tuple[FzNetwork, FzNetwork]] = list()
    optimizers: list[tuple[torch.optim.Optimizer, torch.optim.Optimizer]] = list()
    schedulers: list[tuple[lr_scheduler.LRScheduler, lr_scheduler.LRScheduler]] = list()

    model_config = FzConfig(
        dx = len(env.XLabels),
        du = len(env.ULabels),
        dz = job.dz,
        ds = sum(env.S_bin_count),
        dr = len(env.reference_variables),
        Hi = train_cfg.Hi,
        Fi = train_cfg.Fi,
        encoder_name = job.encoder_name,
        decoder_name = job.decoder_name,
        fc_layer_num = job.fc_layer_num,
        fc_layer_size = job.fc_layer_size,
        t_layer_num = job.t_layer_num,
        t_nhead = job.t_nhead,
        t_d_feedforward = job.t_d_feedforward,
        alpha = job.alpha,
        dropout = job.dropout,
        activation_fn = job.activation_fn,
    )
    log("Got model configuration", model_config)

    for i in range(train_cfg.num_models):
        dynamics_model = FzNetwork(model_config, train_cfg, for_control=False).to(device)
        control_model = FzNetwork(model_config, train_cfg, for_control=True).to(device)

        models.append((dynamics_model, control_model))
        optimizers.append((
            torch.optim.AdamW(dynamics_model.parameters(), lr=train_cfg.lr),
            torch.optim.AdamW(control_model.parameters(), lr=train_cfg.lr),
        ))
        schedulers.append((
            build_lr_scheduler(train_cfg, optimizers[-1][0]),
            build_lr_scheduler(train_cfg, optimizers[-1][1]),
        ))

        if i == 0:
            log.debug("Built models", dynamics_model, control_model)
            log(
                "Number of dynamics model parameters",
                "Encoder (H): %s" % thousands_seperators(dynamics_model.Eh.numel()),
                "Encoder (F): %s" % thousands_seperators(dynamics_model.Ef.numel()),
                "Decoder:     %s" % thousands_seperators(dynamics_model.D.numel()),
                "Total:       %s" % thousands_seperators(dynamics_model.numel()),
                "Number of control model parameters",
                "Encoder (H): %s" % thousands_seperators(control_model.Eh.numel()),
                "Encoder (F): %s" % thousands_seperators(control_model.Ef.numel()),
                "Decoder:     %s" % thousands_seperators(control_model.D.numel()),
                "Total:       %s" % thousands_seperators(control_model.numel()),
            )

    log_every = 100
    checkpoint_every = 300
    rare_checkpoint_every = 20
    rare_checkpoint_counter = 1

    def checkpoint(batch_no: int):
        """ Performs checkpoint operations such as saving model progress, plotting, evaluation, etc. """
        nonlocal rare_checkpoint_counter

        log("Doing checkpoint at batch %i / %i" % (batch_no, train_cfg.batches))
        train_results.checkpoints.append(batch_no)

        is_rare_checkpoint = rare_checkpoint_counter == 0 or batch_no == checkpoint_every or batch_no == train_cfg.batches

        with TT.profile("Checkpoint"):

            # Evaluate
            for dynamics_model, control_model in models:
                dynamics_model.eval()
                control_model.eval()

            with TT.profile("Evalutate"):
                test_loss_x = np.empty((train_cfg.num_eval_batches, train_cfg.num_models))
                test_loss_u = np.empty((train_cfg.num_eval_batches, train_cfg.num_models))

                ensemble_loss_x = np.empty(train_cfg.num_eval_batches)
                ensemble_loss_u = np.empty(train_cfg.num_eval_batches)

                for j in range(train_cfg.num_eval_batches):
                    Xh, Uh, Sh, Xf, Uf, Sf = next(train_dataloader)
                    Xf_pred_mean = torch.zeros_like(Xf)
                    Uf_pred_mean = torch.zeros_like(Uf)
                    for k, (dynamics_model, control_model) in enumerate(models):
                        with torch.inference_mode(), TT.profile("Forward", hits=train_cfg.num_models), amp_context():
                            Xf_pred = dynamics_model(Xh, Uh, Sh, Sf, Uf=Uf)
                            Uf_pred = control_model(Xh, Uh, Sh, Sf, Xf=Xf[..., env.reference_variables])
                            Xf_pred_mean += Xf_pred
                            Uf_pred_mean += Uf_pred
                            test_loss_x[j, k] = dynamics_model.loss(Xf, Xf_pred).item()
                            test_loss_u[j, k] = control_model.loss(Uf, Uf_pred).item()

                    Xf_pred_mean /= train_cfg.num_models
                    Uf_pred_mean /= train_cfg.num_models

                    # This is bad code
                    ensemble_loss_x[j] = dynamics_model.loss(Xf, Xf_pred_mean).item()
                    ensemble_loss_u[j] = control_model.loss(Uf, Uf_pred_mean).item()

                for k in range(train_cfg.num_models):
                    train_results.test_loss_x[k].append(test_loss_x[:, k].mean())
                    train_results.test_loss_u[k].append(test_loss_u[:, k].mean())
                    train_results.test_loss_x_std[k].append(test_loss_x[:, k].std(ddof=1) * math.sqrt(train_cfg.num_eval_batches))
                    train_results.test_loss_u_std[k].append(test_loss_u[:, k].std(ddof=1) * math.sqrt(train_cfg.num_eval_batches))

                train_results.ensemble_loss_x.append(ensemble_loss_x.mean())
                train_results.ensemble_loss_u.append(ensemble_loss_u.mean())

            log(
                "Mean test loss X: %.6f" % np.array(train_results.test_loss_x)[:, -1].mean(),
                "Mean test loss U: %.6f" % np.array(train_results.test_loss_u)[:, -1].mean(),
                "Ensemble loss X:  %.6f" % train_results.ensemble_loss_x[-1],
                "Ensemble loss U:  %.6f" % train_results.ensemble_loss_u[-1],
            )

            # Plot training stuff
            if is_rare_checkpoint:
                log("Plotting training progress")
                plot_loss(job.location, train_cfg, train_results)
                plot_lr(job.location, train_cfg, train_results)
            rare_checkpoint_counter = (rare_checkpoint_counter + 1) % rare_checkpoint_every

            if batch_no == train_cfg.batches:
                with TT.profile("Forward evalutation"):
                    forward(
                        job.location,
                        env,
                        models,
                        test_dataset,
                        train_cfg,
                        train_results,
                        ForwardConfig(num_samples=5, num_sequences=3 if env is environments.FloatZone else 1, opt_steps=5, step_size=3e-4),
                    )

                with TT.profile("Simulate control"):
                    simulated_control(
                        job.location,
                        env,
                        models,
                        test_dataset,
                        train_cfg,
                        train_results,
                        SimulationConfig(5, train_cfg.prediction_window, train_cfg.prediction_window, env.dt, 0, 2e-2),
                    )

            for dynamics_model, control_model in models:
                dynamics_model.train()
                control_model.train()

            # Save training progress
            train_cfg.save(job.location)
            train_results.save(job.location)
            for i in range(train_cfg.num_models):
                models[i][0].save(job.location, i)
                models[i][1].save(job.location, i)

        if is_rare_checkpoint:
            log("Training time distribution", TT)

            if batch_no > 0:
                total_examples = train_cfg.num_models * train_cfg.batch_size * batch_no
                batch_profile = next(p for p in TT.profiles if p.name == "Batch")
                total_train_time = batch_profile.sum()
                time_per_batch = total_train_time / batch_no
                remaining_time = (train_cfg.batches - batch_no) * time_per_batch
                log(
                    "Total training time:    %s s" % thousands_seperators(round(total_train_time)),
                    "Total examples seen:    %i x %s" % (train_cfg.num_models, thousands_seperators(total_examples // train_cfg.num_models)),
                    "Average:                %s examples / s" % thousands_seperators(round(total_examples / total_train_time)),
                    "Approx. remaining time: %s s" % thousands_seperators(round(remaining_time)),
                )

    log.section("Beginning training loop", "Training for %s batches" % thousands_seperators(train_cfg.batches))
    for i in range(train_cfg.batches):
        is_checkpoint = i % checkpoint_every == 0
        do_log = i == 0 or i == train_cfg.batches - 1 or i % log_every == 0

        if is_checkpoint:
            checkpoint(i)

        if do_log:
            log.debug("Batch %i / %i" % (i, train_cfg.batches))

        TT.profile("Batch")

        train_results.lr.append(schedulers[0][0].get_last_lr())

        for j in range(train_cfg.num_models):
            dynamics_model, control_model = models[j]
            dynamics_optimizer, control_optimizer = optimizers[j]
            dynamics_scheduler, control_scheduler = schedulers[j]

            with TT.profile("Get data"):
                Xh, Uh, Sh, Xf, Uf, Sf = next(test_dataloader)
            with TT.profile("Step"), amp_context():
                Xf_pred = dynamics_model(Xh, Uh, Sh, Sf, Uf=Uf)
                Uf_pred = control_model(Xh, Uh, Sh, Sf, Xf=Xf[..., env.reference_variables])

                loss_x = dynamics_model.loss(Xf, Xf_pred)
                loss_u = control_model.loss(Uf, Uf_pred)

                loss_x.backward()
                loss_u.backward()

                dynamics_optimizer.step()
                control_optimizer.step()

                dynamics_optimizer.zero_grad()
                control_optimizer.zero_grad()

                dynamics_scheduler.step()
                control_scheduler.step()

                cuda_sync()

            train_results.train_loss_x[j].append(loss_x.item())
            train_results.train_loss_u[j].append(loss_u.item())

        if do_log:
            log.debug(
                "Loss X: %.6f" % (sum(train_results.train_loss_x[j][-1] for j in range(train_cfg.num_models)) / train_cfg.num_models),
                "Loss U: %.6f" % (sum(train_results.train_loss_u[j][-1] for j in range(train_cfg.num_models)) / train_cfg.num_models),
            )

        TT.end_profile()

    checkpoint(train_cfg.batches)
