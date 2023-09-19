import math
import os
from typing import Type

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from pelutils import TT, JobDescription, log, thousands_seperators, HardwareInfo

import frozone.environments as environments
from frozone import device, amp_context
from frozone.data import list_processed_data_files
from frozone.data.dataloader import dataloader, dataset_size, history_only_vector, load_data_files, standardize
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

    log(HardwareInfo.string())

    train_cfg = TrainConfig(
        env = job.env,
        data_path = job.data_path,
        phase = job.phase,
        history_window = job.history_window,
        prediction_window = job.prediction_window,
        batches = job.batches,
        batch_size = job.batch_size,
        lr = job.lr,
        num_models = job.num_models,
        max_num_data_files = job.max_num_data_files,
        eval_size = job.eval_size,
        loss_fn = job.loss_fn,
        huber_delta = job.huber_delta,
        alpha = job.alpha,
        epsilon = job.epsilon,
    )
    train_results = TrainResults.empty(train_cfg.num_models)

    log("Training configuration", train_cfg)

    env: Type[environments.Environment] = getattr(environments, train_cfg.env)
    log("Got environment %s" % env.__name__)

    train_npz_files = list_processed_data_files(train_cfg.data_path, TRAIN_SUBDIR, train_cfg.phase)
    test_npz_files = list_processed_data_files(train_cfg.data_path, TEST_SUBDIR, train_cfg.phase)
    log(
        "Found data files",
        "Train: %s (%.2f %%)" % (thousands_seperators(len(train_npz_files)), 100 * len(train_npz_files) / (len(train_npz_files) + len(test_npz_files))),
        "Test:  %s (%.2f %%)" % (thousands_seperators(len(test_npz_files)), 100 * len(test_npz_files) / (len(train_npz_files) + len(test_npz_files))),
    )
    log("Loading data")
    with TT.profile("Load data"):
        train_dataset = load_data_files(train_npz_files, train_cfg, max_num_files=train_cfg.max_num_data_files)
        test_dataset = load_data_files(test_npz_files, train_cfg)
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
    train_dataloader = dataloader(env, train_cfg, train_dataset)
    test_dataloader = dataloader(env, train_cfg, test_dataset)

    log("Building %i models" % train_cfg.num_models)
    models: list[FzNetwork] = list()
    optimizers: list[torch.optim.Optimizer] = list()
    schedulers: list[lr_scheduler.LRScheduler] = list()
    for i in range(train_cfg.num_models):
        model = FzNetwork(FzConfig(
            dx = len(env.XLabels),
            du = len(env.ULabels),
            dz = job.dz,
            ds = sum(env.S_bin_count),
            H = train_cfg.H,
            F = train_cfg.F,
            encoder_name = job.encoder_name,
            decoder_name = job.decoder_name,
            mode = 0 if 0 < train_cfg.alpha < 1 else 1 if train_cfg.alpha == 0 else 2,
            fc_layer_num = job.fc_layer_num,
            fc_layer_size = job.fc_layer_size,
            t_layer_num = job.t_layer_num,
            t_nhead = job.t_nhead,
            t_d_feedforward = job.t_d_feedforward,
            dropout = job.dropout,
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
                "History encoder:  %s" % thousands_seperators(model.Eh.numel()),
                "Future encoder X: %s" % thousands_seperators(model.Ex.numel() if model.config.has_control else 0),
                "Future encoder U: %s" % thousands_seperators(model.Eu.numel() if model.config.has_dynamics else 0),
                "Decoder X:        %s" % thousands_seperators(model.Dx.numel() if model.config.has_dynamics else 0),
                "Decoder U:        %s" % thousands_seperators(model.Du.numel() if model.config.has_control else 0),
                "Total:            %s" % thousands_seperators(model.numel()),
            )

    if train_cfg.loss_fn == "l1":
        loss_fn = torch.nn.L1Loss(reduction="none")
    elif train_cfg.loss_fn == "l2":
        loss_fn = torch.nn.MSELoss(reduction="none")
    elif train_cfg.loss_fn == "huber":
        _loss_fn = torch.nn.HuberLoss(reduction="none", delta=0.05)
        loss_fn = lambda target, input: 1 / train_cfg.huber_delta * _loss_fn(target, input)
    loss_weight = torch.ones(train_cfg.F, device=device)
    loss_weight = loss_weight / loss_weight.sum()

    future_include_weights = history_only_vector(env, train_cfg)
    future_include_weights = torch.from_numpy(future_include_weights).to(device) * len(future_include_weights) / future_include_weights.sum()

    def loss_fn_x(x_target: torch.FloatTensor, x_pred: torch.FloatTensor) -> torch.FloatTensor:
        loss: torch.FloatTensor = loss_fn(x_target, x_pred).mean(dim=0)
        return (loss.T @ loss_weight * future_include_weights).mean()
    def loss_fn_u(u_target: torch.FloatTensor, u_pred: torch.FloatTensor) -> torch.FloatTensor:
        return loss_fn(u_target, u_pred).mean()

    rare_checkpoint_counter = 1
    @torch.inference_mode()
    def checkpoint(batch_no: int):
        """ Performs checkpoint operations such as saving model progress, plotting, evaluation, etc. """
        nonlocal rare_checkpoint_counter

        log("Doing checkpoint at batch %i" % batch_no)
        train_results.checkpoints.append(batch_no)

        is_rare_checkpoint = rare_checkpoint_counter == 0 or batch_no == train_cfg.batches

        with TT.profile("Checkpoint"):

            # Evaluate
            for j, model in enumerate(models):
                model.eval()

                with TT.profile("Evalutate"):
                    test_loss_x = np.empty(train_cfg.num_eval_batches)
                    test_loss_u = np.empty(train_cfg.num_eval_batches)
                    test_loss = np.empty(train_cfg.num_eval_batches)
                    for k in range(train_cfg.num_eval_batches):
                        Xh, Uh, Sh, Xf, Uf, Sf = next(test_dataloader)
                        with TT.profile("Forward"), amp_context():
                            _, Xf_pred, Uf_pred = model(Xh, Uh, Sh, Sf, Xf=Xf, Uf=Uf)
                        test_loss_x[k] = loss_fn_x(Xf, Xf_pred).item() if Xf_pred is not None else torch.tensor(0)
                        test_loss_u[k] = loss_fn_u(Uf, Uf_pred).item() if Uf_pred is not None else torch.tensor(0)
                        test_loss[k] = (1 - train_cfg.alpha) * test_loss_x[k] + train_cfg.alpha * test_loss_u[k]
                        assert not np.isnan(test_loss[k])
                    train_results.test_loss_x[j].append(test_loss_x.mean())
                    train_results.test_loss_u[j].append(test_loss_u.mean())
                    train_results.test_loss[j].append(test_loss.mean())
                    train_results.test_loss_x_std[j].append(test_loss_x.std(ddof=1) * math.sqrt(train_cfg.num_eval_batches))
                    train_results.test_loss_u_std[j].append(test_loss_u.std(ddof=1) * math.sqrt(train_cfg.num_eval_batches))
                    train_results.test_loss_std[j].append(test_loss.std(ddof=1) * math.sqrt(train_cfg.num_eval_batches))

                model.train()

            log(
                "Test loss X: %.6f" % np.array(train_results.test_loss_x)[:, -1].mean(),
                "Test loss U: %.6f" % np.array(train_results.test_loss_u)[:, -1].mean(),
                "Test loss:   %.6f" % np.array(train_results.test_loss)[:, -1].mean(),
            )

            # Plot training stuff
            if is_rare_checkpoint:
                log("Plotting training progress")
                plot_loss(job.location, train_cfg, train_results)
                plot_lr(job.location, train_cfg, train_results)
            rare_checkpoint_counter = (rare_checkpoint_counter + 1) % 5

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
                Xh, Uh, Sh, Xf, Uf, Sf = next(train_dataloader)
            with TT.profile("Forward"), amp_context():
                _, Xf_pred, Uf_pred = model(Xh, Uh, Sh, Sf, Xf=Xf, Uf=Uf)
                loss_x = loss_fn_x(Xf, Xf_pred) if Xf_pred is not None else torch.tensor(0)
                loss_u = loss_fn_u(Uf, Uf_pred) if Uf_pred is not None else torch.tensor(0)
                loss = (1 - train_cfg.alpha) * loss_x + train_cfg.alpha * loss_u
                assert not torch.isnan(loss).isnan().any()
                train_results.train_loss_x[j].append(loss_x.item())
                train_results.train_loss_u[j].append(loss_u.item())
                train_results.train_loss[j].append(loss.item())
                cuda_sync()

            with TT.profile("Backward"):
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

    total_examples = train_cfg.num_models * train_cfg.batch_size * train_cfg.batches
    batch_profile = next(p for p in TT.profiles if p.name == "Batch")
    total_train_time = batch_profile.sum()
    log(
        "Total training time: %s s" % thousands_seperators(round(total_train_time)),
        "Total examples seen: %i x %s" % (train_cfg.num_models, thousands_seperators(total_examples // train_cfg.num_models)),
        "Average:             %s examples / s" % thousands_seperators(round(total_examples / total_train_time)),
    )
