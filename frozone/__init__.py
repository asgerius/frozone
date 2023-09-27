import contextlib

import numpy as np
import torch
import torch.cuda.amp as amp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def amp_context():
    return amp.autocast() if torch.cuda.is_available() else contextlib.ExitStack()

def tensor_size(x: np.ndarray | torch.Tensor) -> int:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.element_size() * x.numel()
