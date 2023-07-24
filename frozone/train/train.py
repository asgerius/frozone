import torch
from pelutils import TT

from frozone import device
from frozone.data.dataloader import simulation_dataloader as create_dataloader
from frozone.model import Frozone
from frozone.simulations import Ball


def train():
    batches = 10000
    batch_size = 100

    # Windows are given in seconds
    history_window = 10
    prediction_window = 5

    dt = 0.1
    history_window_steps = int(history_window / dt)
    predict_window_steps = int(prediction_window / dt)

    simulation = Ball
    model = Frozone(Frozone.Config(
        len(simulation.ProcessVariables),
        len(simulation.ControlVariables),
        0.1, history_window_steps, predict_window_steps, 3, 100,
    )).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.MSELoss()

    train_dataloader = create_dataloader(simulation, batch_size, model)
    test_dataloader = create_dataloader(simulation, batch_size, model)

    for i in range(batches):
        if i % 500 == 0:
            with torch.inference_mode(), TT.profile("Evaluate"):
                model.eval()
                history_process, history_control, target_process, target_control = next(test_dataloader)
                with TT.profile("Forward"):
                    predicted_control = model(history_process, history_control, target_process)
                loss = loss_fn(target_control, predicted_control)
                print(i, "TEST LOSS %.6f" % loss.item())
                model.train()

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

        if i % 100 == 0:
            print(i, "%.6f" % loss.item())

        TT.end_profile()


    model.save('out')

    print(TT)
