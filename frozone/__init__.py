import contextlib

import numpy as np
import torch
import torch.cuda.amp as amp


if torch.cuda.device_count() == 0:
    device_x = device_u = torch.device("cpu")
elif torch.cuda.device_count() == 1:
    device_x = device_u = torch.device("cuda")
else:
    device_x = torch.device("cuda:0")
    device_u = torch.device("cuda:1")

def amp_context():
    return amp.autocast() if torch.cuda.is_available() else contextlib.ExitStack()

def tensor_size(x: np.ndarray | torch.Tensor) -> int:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.element_size() * x.numel()

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_x)
        torch.cuda.synchronize(device_u)
