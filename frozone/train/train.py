import math
import os
from glob import glob as glob  # glob
from typing import Type

import numpy as np
import torch
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
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = config.lr,
        epochs = TrainConfig.batches,
        steps_per_epoch = 1,
    )

def train(job: JobDescription):

    train_cfg = TrainConfig()
    train_results = TrainResults.empty()

    env: Type[environments.Environment] = getattr(environments, train_cfg.env)
    log("Got environment %s" % env.__name__)

    model = FzNetwork(FzConfig(
        Dx = len(env.XLabels),
        Du = len(env.ULabels),
        Dz = 500,
        Ds = sum(env.S_bin_count),
        H = train_cfg.H,
        F = train_cfg.F,
        history_encoder_name  = "FullyConnected",
        target_encoder_name   = "FullyConnected",
        dynamics_network_name = "FullyConnected",
        control_network_name  = "FullyConnected",
        fc_layer_num = 3,
        fc_layer_size = 400,
        resnext_cardinality = 2,
        dropout_p = 0.0,
    )).to(device)
    log.debug("Built model", model)
    log(
        "Number of model parameters",
        "History encoder:  %s" % thousands_seperators(model.history_encoder.numel()),
        "Target encoder:   %s" % thousands_seperators(model.target_encoder.numel()),
        "Dynamics network: %s" % thousands_seperators(model.dynamics_network.numel()),
        "Control network:  %s" % thousands_seperators(model.control_network.numel()),
        "Total:            %s" % thousands_seperators(model.numel()),
        sep="   \n",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    scheduler = build_lr_scheduler(train_cfg, optimizer)
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

    plot_counter = 1
    @torch.inference_mode()
    def checkpoint(batch_no: int):
        """ Performs checkpoint operations such as saving model progress, plotting, evaluation, etc. """
        nonlocal plot_counter

        log("Doing checkpoint at batch %i" % batch_no)
        train_results.checkpoints.append(batch_no)

        with TT.profile("Checkpoint"):
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
                    test_loss_x[j] = loss_fn(Xf[:, 1], pred_x).item()
                    test_loss_u[j] = loss_fn(u, pred_u).item()
                    test_loss[j] = (1 - train_cfg.alpha) * test_loss_x[j] + train_cfg.alpha * test_loss_u[j]
                train_results.test_loss_x.append(test_loss_x.mean())
                train_results.test_loss_u.append(test_loss_u.mean())
                train_results.test_loss.append(test_loss.mean())
                log(
                    "Test loss X: %.6f" % train_results.test_loss_x[-1],
                    "Test loss U: %.6f" % train_results.test_loss_u[-1],
                    "Test loss:   %.6f" % train_results.test_loss[-1],
                )
                train_results.test_loss_x_std.append(test_loss_x.std(ddof=1) * math.sqrt(train_cfg.eval_size))
                train_results.test_loss_u_std.append(test_loss_u.std(ddof=1) * math.sqrt(train_cfg.eval_size))
                train_results.test_loss_std.append(test_loss.std(ddof=1) * math.sqrt(train_cfg.eval_size))

            # Plot training stuff
            if plot_counter == 0 or batch_no == train_cfg.batches:
                log("Plotting training progress")
                plot_loss(job.location, train_cfg, train_results)
                plot_lr(job.location, train_cfg, train_results)
            plot_counter = (plot_counter + 1) % 5

            model.train()

            # Save training progress
            train_cfg.save(job.location)
            train_results.save(job.location)
            model.save(job.location)

    log.section("Beginning training loop", "Training for %s batches" % thousands_seperators(train_cfg.batches))
    for i in range(train_cfg.batches):
        is_checkpoint = i % 1000 == 0
        do_log = i == 0 or i == train_cfg.batches - 1 or i % 100 == 0

        if is_checkpoint:
            checkpoint(i)

        if do_log:
            log.debug("Batch %i / %i" % (i, train_cfg.batches))

        TT.profile("Batch")

        train_results.lr.append(scheduler.get_last_lr())

        Xh, Uh, Sh, Xf, Sf, u, s = next(train_dataloader)
        with TT.profile("Forward"):
            _, pred_x, pred_u = model(Xh, Uh, Sh, Xf, Sf, u, s)
            cuda_sync()
        loss_x = loss_fn(Xf[:, 1], pred_x)
        loss_u = loss_fn(u, pred_u)
        loss = (1 - train_cfg.alpha) * loss_x + train_cfg.alpha * loss_u
        with TT.profile("Backward"):
            loss.backward()
            cuda_sync()
        with TT.profile("Step"):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            cuda_sync()

        if do_log:
            log.debug("Loss: %.6f" % loss.item())
        train_results.train_loss_x.append(loss_x.item())
        train_results.train_loss_u.append(loss_u.item())
        train_results.train_loss.append(loss.item())

        TT.end_profile()

    checkpoint(train_cfg.batches)

    log(TT)
