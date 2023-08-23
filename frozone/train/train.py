import math
from typing import Type

import numpy as np
import torch
from pelutils import TT, JobDescription, log, thousands_seperators

import frozone.environments as environments
from frozone import device
from frozone.data.dataloader import simulation_dataloader as create_dataloader
from frozone.model.floatzone_network import FzConfig, FzNetwork
from frozone.plot.plot_train import plot_loss
from frozone.train import TrainConfig, TrainResults


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def train(job: JobDescription):

    train_cfg = TrainConfig()
    train_results = TrainResults.empty()

    env: Type[environments.Environment] = getattr(environments, train_cfg.env)
    log("Got environment %s" % env.__name__)

    model = FzNetwork(FzConfig(
        Dx = len(env.XLabels),
        Du = len(env.ULabels),
        Dz = 800,
        Ds = sum(env.S_bin_count),
        H = train_cfg.H,
        F = train_cfg.F,
        history_encoder_name  = "FullyConnected",
        target_encoder_name   = "FullyConnected",
        dynamics_network_name = "FullyConnected",
        control_network_name  = "FullyConnected",
        fc_hidden_num = 3,
        fc_hidden_size = 500,
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

    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.MSELoss()

    train_dataloader = create_dataloader(env, train_cfg)
    test_dataloader = create_dataloader(env, train_cfg)

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
                log("Test loss: %.6f" % train_results.test_loss[-1])
                train_results.test_loss_x_std.append(test_loss_x.std(ddof=1) * math.sqrt(train_cfg.eval_size))
                train_results.test_loss_u_std.append(test_loss_u.std(ddof=1) * math.sqrt(train_cfg.eval_size))
                train_results.test_loss_std.append(test_loss.std(ddof=1) * math.sqrt(train_cfg.eval_size))

            # Plot training stuff
            if plot_counter == 0 or batch_no == train_cfg.batches:
                log("Plotting training progress")
                plot_loss(job.location, train_cfg, train_results)
            plot_counter = (plot_counter + 1) % 10

            model.train()

            # Save training progress
            train_cfg.save(job.location)
            train_results.save(job.location)
            model.save(job.location)

    log.section("Beginning training loop", "Training for %s batches" % thousands_seperators(train_cfg.batches))
    for i in range(train_cfg.batches):
        is_checkpoint = i % 500 == 0
        do_log = i == 0 or i == train_cfg.batches - 1 or i % 100 == 0

        if is_checkpoint:
            checkpoint(i)

        if do_log:
            log.debug("Batch %i / %i" % (i, train_cfg.batches))

        TT.profile("Batch")

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
            optim.step()
            optim.zero_grad()
            cuda_sync()

        if do_log:
            log.debug("Loss: %.6f" % loss.item())
        train_results.train_loss_x.append(loss_x.item())
        train_results.train_loss_u.append(loss_u.item())
        train_results.train_loss.append(loss.item())

        TT.end_profile()

    checkpoint(train_cfg.batches)

    log(TT)
