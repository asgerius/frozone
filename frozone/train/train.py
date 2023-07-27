from typing import Type

import torch
from pelutils import TT, JobDescription, log, thousands_seperators

import frozone.simulations as simulations
from frozone import device
from frozone.data.dataloader import simulation_dataloader as create_dataloader
from frozone.model import Frozone
from frozone.plot.plot_train import plot_loss
from frozone.train import TrainConfig, TrainResults


def train(job: JobDescription):

    train_cfg = TrainConfig()
    train_results = TrainResults.empty()

    env: Type[simulations.Simulation] = getattr(simulations, train_cfg.env)
    log("Got environment %s" % env.__name__)

    model = Frozone(Frozone.Config(
        len(env.ProcessVariables),
        len(env.ControlVariables),
        train_cfg.history_window_steps, train_cfg.predict_window_steps, 3, 100,
    )).to(device)
    log("Built model", model)
    log("Number of model parameters: %s" % thousands_seperators(model.numel()))

    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.MSELoss()

    train_dataloader = create_dataloader(env, train_cfg)
    test_dataloader = create_dataloader(env, train_cfg)

    def checkpoint(batch_no: int):
        """ Performs checkpoint operations such as saving model progress, plotting, evaluation, etc. """
        log("Doing checkpoint at batch %i" % batch_no)
        train_results.checkpoints.append(batch_no)

        with TT.profile("Checkpoint"), torch.inference_mode():
            model.eval()

            # Evaluate
            with TT.profile("Evalutate"):
                history_process, history_control, target_process, target_control = next(test_dataloader)
                with TT.profile("Forward"):
                    predicted_control = model(history_process, history_control, target_process)
                loss = loss_fn(target_control, predicted_control)
                log("Test loss: %.6f" % loss.item())
                train_results.test_loss.append(loss.item())

            # Plot
            plot_loss(job.location, train_cfg, train_results)

            model.train()

            # Save training progress
            train_cfg.save(job.location)
            train_results.save(job.location)
            model.save(job.location)

    log.section("Beginning training loop")
    for i in range(train_cfg.batches):
        is_checkpoint = i % 500 == 0
        do_log = i == 0 or i == train_cfg.batches - 1 or i % 100 == 0

        if is_checkpoint:
            checkpoint(i)

        if do_log:
            log.debug("Batch %i / %i" % (i, train_cfg.batches))

        TT.profile("Batch")

        history_process, history_control, target_process, target_control = next(train_dataloader)
        with TT.profile("Forward"):
            predicted_control = model(history_process, history_control, target_process)
        loss = loss_fn(target_control, predicted_control)
        with TT.profile("Backward"):
            loss.backward()
        with TT.profile("Step"):
            optim.step()
            optim.zero_grad()

        if do_log:
            log.debug("Loss: %.6f" % loss.item())
        train_results.train_loss.append(loss.item())

        TT.end_profile()

    checkpoint(train_cfg.batches)

    print(TT)
