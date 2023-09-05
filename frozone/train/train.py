import math
import os
from glob import glob as glob  # glob
from typing import Type

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from pelutils import TT, JobDescription, log, thousands_seperators

import frozone.environments as environments
from frozone import device
from frozone.data.dataloader import dataloader, load_data_files, standardize
from frozone.data.process_raw_floatzone_data import PROCESSED_SUBDIR, TEST_SUBDIR, TRAIN_SUBDIR
from frozone.model.floatzone_network import FzConfig, FzNetwork
from frozone.plot.plot_train import plot_loss, plot_lr
from frozone.train import TrainConfig, TrainResults


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def build_lr_scheduler(config: TrainConfig, optimizer: torch.optim.Optimizer):
    return lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = config.lr,
        epochs = config.batches,
        steps_per_epoch = 1,
    )

def train(job: JobDescription):

    train_cfg = TrainConfig(
        env = job.env,
        data_path = job.data_path,
        dt = job.dt,
        history_window = job.history_window,
        prediction_window = job.prediction_window,
        batches = job.batches,
        batch_size = job.batch_size,
        lr = job.lr,
        num_models = job.num_models,
        max_num_data_files = job.max_num_data_files,
        eval_size = job.eval_size,
        alpha = job.alpha,
        epsilon = job.epsilon,
    )
    train_results = TrainResults.empty(train_cfg.num_models)

    env: Type[environments.Environment] = getattr(environments, train_cfg.env)
    log("Got environment %s" % env.__name__)

    log("Building %i models" % train_cfg.num_models)
    models: list[FzNetwork] = list()
    optimizers: list[torch.optim.Optimizer] = list()
    schedulers: list[lr_scheduler.LRScheduler] = list()
    for i in range(train_cfg.num_models):
        model = FzNetwork(FzConfig(
            Dx = len(env.XLabels),
            Du = len(env.ULabels),
            Dz = job.Dz,
            Ds = sum(env.S_bin_count),
            H = train_cfg.H,
            F = train_cfg.F,
            history_encoder_name  = job.history_encoder,
            target_encoder_name   = job.target_encoder,
            dynamics_network_name = job.dynamics_network,
            control_network_name  = job.control_network,
            mode = 0 if 0 < train_cfg.alpha < 1 else 1 if train_cfg.alpha == 0 else 2,
            fc_layer_num = job.fc_layer_num,
            fc_layer_size = job.fc_layer_size,
            resnext_cardinality = job.resnext_cardinality,
            dropout_p = job.dropout_p,
            activation_fn = job.activation_fn,
        )).to(device)
        models.append(model)
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, foreach=True))
        schedulers.append(build_lr_scheduler(train_cfg, optimizers[-1]))
        if i == 0:
            log("Got model configuration", model.config)
            log.debug("Built model", model)
            log(
                "Number of model parameters",
                "History encoder:  %s" % thousands_seperators(model.history_encoder.numel()),
                "Target encoder:   %s" % thousands_seperators(model.target_encoder.numel()),
                "Dynamics network: %s" % thousands_seperators(model.dynamics_network.numel() if model.config.has_dynamics_network else 0),
                "Control network:  %s" % thousands_seperators(model.control_network.numel() if model.config.has_control_network else 0),
                "Total:            %s" % thousands_seperators(model.numel()),
                sep="   \n",
            )

    loss_fn = torch.nn.L1Loss()

    train_npz_files = glob(os.path.join(train_cfg.data_path, PROCESSED_SUBDIR, TRAIN_SUBDIR, "**", "*.npz"), recursive=True)
    test_npz_files = glob(os.path.join(train_cfg.data_path, PROCESSED_SUBDIR, TEST_SUBDIR, "**", "*.npz"), recursive=True)
    log(
        "Found data files",
        "Train: %s (%.2f %%)" % (thousands_seperators(len(train_npz_files)), 100 * len(train_npz_files) / (len(train_npz_files) + len(test_npz_files))),
        "Test:  %s (%.2f %%)" % (thousands_seperators(len(test_npz_files)), 100 * len(test_npz_files) / (len(train_npz_files) + len(test_npz_files))),
        sep="    \n",
    )
    log("Loading data")
    with TT.profile("Load data"):
        train_dataset = load_data_files(train_npz_files, train_cfg, max_num_files=8000)
        test_dataset = load_data_files(test_npz_files, train_cfg)
    log("Standardizing data")
    with TT.profile("Standardize"):
        standardize(env, train_dataset, train_results)
        standardize(env, test_dataset, train_results)
    train_dataloader = dataloader(env, train_cfg, train_dataset)
    test_dataloader = dataloader(env, train_cfg, test_dataset)

    rare_checkpoint_counter = 1
    @torch.inference_mode()
    def checkpoint(batch_no: int):
        """ Performs checkpoint operations such as saving model progress, plotting, evaluation, etc. """
        nonlocal rare_checkpoint_counter

        log("Doing checkpoint at batch %i" % batch_no)
        train_results.checkpoints.append(batch_no)

        is_rare_checkpoint = rare_checkpoint_counter == 0 or batch_no == train_cfg.batches

        with TT.profile("Checkpoint"):
            model = models[0]
            model.eval()

            # Evaluate
            with TT.profile("Evalutate"):
                test_loss_x = np.empty(train_cfg.num_eval_batches)
                test_loss_u = np.empty(train_cfg.num_eval_batches)
                test_loss = np.empty(train_cfg.num_eval_batches)
                for j in range(train_cfg.num_eval_batches):
                    Xh, Uh, Sh, Xf, Sf, u, s = next(test_dataloader)
                    with TT.profile("Forward"):
                        _, pred_x, pred_u = model(Xh, Uh, Sh, Xf, Sf, u, s)
                    test_loss_x[j] = loss_fn(Xf[:, 1], pred_x).item() if pred_x is not None else torch.tensor(0)
                    test_loss_u[j] = loss_fn(u, pred_u).item() if pred_u is not None else torch.tensor(0)
                    test_loss[j] = (1 - train_cfg.alpha) * test_loss_x[j] + train_cfg.alpha * test_loss_u[j]
                    assert not np.isnan(test_loss[j])
                train_results.test_loss_x.append(test_loss_x.mean())
                train_results.test_loss_u.append(test_loss_u.mean())
                train_results.test_loss.append(test_loss.mean())
                log(
                    "Test loss X: %.6f" % train_results.test_loss_x[-1],
                    "Test loss U: %.6f" % train_results.test_loss_u[-1],
                    "Test loss:   %.6f" % train_results.test_loss[-1],
                )
                train_results.test_loss_x_std.append(test_loss_x.std(ddof=1) / math.sqrt(train_cfg.num_eval_batches))
                train_results.test_loss_u_std.append(test_loss_u.std(ddof=1) / math.sqrt(train_cfg.num_eval_batches))
                train_results.test_loss_std.append(test_loss.std(ddof=1) / math.sqrt(train_cfg.num_eval_batches))

            # Plot training stuff
            if is_rare_checkpoint:
                log("Plotting training progress")
                plot_loss(job.location, train_cfg, train_results)
                plot_lr(job.location, train_cfg, train_results)
            rare_checkpoint_counter = (rare_checkpoint_counter + 1) % 5

            model.train()

            # Save training progress
            train_cfg.save(job.location)
            train_results.save(job.location)
            for i in range(train_cfg.num_models):
                models[i].save(job.location, i)

        if is_rare_checkpoint:
            log(TT)

    log.section("Beginning training loop", "Training for %s batches" % thousands_seperators(train_cfg.batches))
    for i in range(train_cfg.batches):
        is_checkpoint = i % 1000 == 0
        do_log = i == 0 or i == train_cfg.batches - 1 or i % 100 == 0

        if is_checkpoint:
            checkpoint(i)

        if do_log:
            log.debug("Batch %i / %i" % (i, train_cfg.batches))

        TT.profile("Batch")

        train_results.lr.append(schedulers[0].get_last_lr())

        for j in range(train_cfg.num_models):
            model = models[j]
            optimizer = optimizers[j]
            scheduler = schedulers[j]

            with TT.profile("Get data"):
                Xh, Uh, Sh, Xf, Sf, u, s = next(train_dataloader)
            with TT.profile("Forward"):
                _, pred_x, pred_u = model(Xh, Uh, Sh, Xf, Sf, u, s)
                cuda_sync()

            with TT.profile("Backward"):
                loss_x = loss_fn(Xf[:, 1], pred_x) if pred_x is not None else torch.tensor(0)
                loss_u = loss_fn(u, pred_u) if pred_u is not None else torch.tensor(0)
                loss = (1 - train_cfg.alpha) * loss_x + train_cfg.alpha * loss_u
                assert not torch.isnan(loss).isnan().any()
                train_results.train_loss_x[j].append(loss_x.item())
                train_results.train_loss_u[j].append(loss_u.item())
                train_results.train_loss[j].append(loss.item())
                loss.backward()
                cuda_sync()

            with TT.profile("Step"):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                cuda_sync()

        if do_log:
            log.debug("Loss: %.6f" % (sum(train_results.train_loss[j][-1] for j in range(train_cfg.num_models)) / train_cfg.num_models))

        TT.end_profile()

    checkpoint(train_cfg.batches)
