from typing import Type

import torch
from pelutils import TT, JobDescription, log, thousands_seperators

import frozone.simulations as simulations
from frozone import device
from frozone.data.dataloader import simulation_dataloader as create_dataloader
from frozone.model import FFFrozone
from frozone.plot.plot_train import plot_loss
from frozone.train import TrainConfig, TrainResults


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def train(job: JobDescription):

    train_cfg = TrainConfig()
    train_results = TrainResults.empty()

    env: Type[simulations.Simulation] = getattr(simulations, train_cfg.env)
    log("Got environment %s" % env.__name__)

    model = FFFrozone(FFFrozone.Config(
        len(env.ProcessVariables),
        len(env.ControlVariables),
        1000,
        train_cfg.history_steps, train_cfg.predict_steps, 3, 300,
    )).to(device)
    log("Built model", model)
    log(
        "Number of model parameters",
        "Latent:  %s" % thousands_seperators(model.latent_model.numel()),
        "Control: %s" % thousands_seperators(model.control_model.numel()),
        "Process: %s" % thousands_seperators(model.process_model.numel()),
        "Total:   %s" % thousands_seperators(model.numel()),
        sep="   \n"
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
                XH, UH, XF, UF = next(test_dataloader)
                assert torch.all(UH >= 0)
                assert torch.all(UH <= 1)
                assert torch.all(UF >= 0)
                assert torch.all(UF <= 1)
                with TT.profile("Forward"):
                    _, pred_UF, pred_XF = model(XH, UH, XF, UF)
                loss_x = loss_fn(XF, pred_XF)
                loss_u = loss_fn(UF, pred_UF)
                loss = (1 - train_cfg.alpha) * loss_x + train_cfg.alpha * loss_u
                log("Test loss: %.6f" % loss.item())
                train_results.test_loss_x.append(loss_x.item())
                train_results.test_loss_u.append(loss_u.item())
                train_results.test_loss.append(loss.item())

            # Plot training stuff
            if plot_counter == 0 or batch_no == train_cfg.batches:
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

        XH, UH, XF, UF = next(train_dataloader)
        with TT.profile("Forward"):
            _, pred_UF, pred_XF = model(XH, UH, XF, UF)
            cuda_sync()
        loss_x = loss_fn(XF, pred_XF)
        loss_u = loss_fn(UF, pred_UF)
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

    print(TT)
