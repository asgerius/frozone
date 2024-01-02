import contextlib
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.cuda.amp as amp
from pelutils import TT, except_keys
from tqdm import tqdm

from frozone import cuda_sync
from frozone.data import list_processed_data_files, Dataset, TEST_SUBDIR
from frozone.data.dataloader import numpy_to_torch_device, load_data_files, standardize
from frozone.environments import Steuermann
from frozone.model.floatzone_network import interpolate, FzNetwork
from frozone.train import TrainConfig, TrainResults


def sample_model_input(
    train_cfg: TrainConfig,
    dataset: Dataset,
) -> dict[str, torch.FloatTensor]:
    """ Returns input to a control model. """

    metadata, (X, U, S, R, Z) = random.choice(dataset)

    seq_len = train_cfg.H + train_cfg.F
    seq_start = np.random.randint(0, X.shape[0] - seq_len)
    seq_mid = seq_start + train_cfg.H
    seq_end = seq_start + seq_len

    Xh, Uh, Sh, Sf, Rf, Uf = numpy_to_torch_device(
        np.expand_dims(X[seq_start:seq_mid], 0),
        np.expand_dims(U[seq_start:seq_mid], 0),
        np.expand_dims(S[seq_start:seq_mid], 0),
        np.expand_dims(S[seq_mid:seq_end], 0),
        np.expand_dims(R[seq_mid:seq_end], 0),
        np.expand_dims(U[seq_mid:seq_end], 0),
    )

    return {
        "Xh": interpolate(train_cfg.Hi, Xh, train_cfg, h=True),
        "Uh": interpolate(train_cfg.Hi, Uh, train_cfg, h=True),
        "Sh": interpolate(train_cfg.Hi, Sh, train_cfg, h=True),
        "Sf": interpolate(train_cfg.Fi, Sf),
        "Xf": interpolate(train_cfg.Fi, Rf),
        "Uf": interpolate(train_cfg.Fi, Uf),
    }

@torch.inference_mode()
def default(train_cfg: TrainConfig, control_model: FzNetwork, input: dict[str, torch.FloatTensor]) -> torch.FloatTensor:
    return interpolate(train_cfg.F, control_model(**input))

@torch.inference_mode()
def with_amp(train_cfg: TrainConfig, control_model: FzNetwork, input: dict[str, torch.FloatTensor]) -> torch.FloatTensor:
    with amp.autocast():
        return interpolate(train_cfg.F, control_model(**input))

@torch.inference_mode()
def torch_compile(train_cfg: TrainConfig, control_model: FzNetwork, input: dict[str, torch.FloatTensor]) -> torch.FloatTensor:
    return interpolate(train_cfg.F, control_model(**input))

def backprop(train_cfg: TrainConfig, dynamics_model: FzNetwork, control_model, input: dict[str, torch.FloatTensor]):

    with TT.profile("Predict"), torch.no_grad():
        Uf = control_model(**except_keys(input, ["Uf"]))
        cuda_sync()
    Uf.requires_grad_()
    optimizer = torch.optim.AdamW([Uf], lr=1e-3)

    for _ in range(3):
        with TT.profile("Step"):
            Xf = dynamics_model(Uf=Uf, **except_keys(input, ["Xf", "Uf"]))

            loss = dynamics_model.loss(Xf, Xf)
            loss.backward()
            Uf.grad[..., Steuermann.predefined_control] = 0

            optimizer.step()
            optimizer.zero_grad()

            cuda_sync()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("path")
    args = parser.parse_args()

    train_cfg = TrainConfig.load(args.path)
    train_results = TrainResults.load(args.path)
    dynamics_network, control_network = FzNetwork.load(args.path, 0)
    dynamics_network.eval()
    control_network.eval()

    compiled_networks = dict()
    for backend in "inductor",:
        print("Compiling with backend %s" % backend)
        compiled_networks[backend] = torch.compile(control_network, fullgraph=True, dynamic=False, backend=backend)

    warmup = 10
    n = 10000
    data_files = list_processed_data_files(args.data, TEST_SUBDIR)
    dataset, _ = load_data_files(data_files, train_cfg, max_num_files=10)

    standardize(Steuermann, dataset, train_results)

    for i in tqdm(range(n + warmup)):
        input = sample_model_input(train_cfg, dataset)
        outputs: dict[str, torch.FloatTensor] = dict()

        cuda_sync()
        with TT.profile("Default") if i >= warmup else contextlib.ExitStack():
            outputs["default"] = default(train_cfg, control_network, except_keys(input, ["Uf"]))
            cuda_sync()
        with TT.profile("AMP") if i >= warmup else contextlib.ExitStack():
            outputs["amp"] = with_amp(train_cfg, control_network, except_keys(input, ["Uf"]))
            cuda_sync()
        for backend, compiled_network in compiled_networks.items():
            with TT.profile("Compiled %s" % backend) if i >= warmup else contextlib.ExitStack():
                outputs["compiled-%s" % backend] = torch_compile(train_cfg, compiled_network, input)
                cuda_sync()

        with TT.profile("Step") if i >= warmup else contextlib.ExitStack():
            backprop(train_cfg, dynamics_network, control_network, input)
            cuda_sync()

    print(TT)
