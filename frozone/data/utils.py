import numpy as np
import torch

from frozone import device


def numpy_to_torch_device(*arrays: np.ndarray) -> list[torch.Tensor]:
    return [torch.from_numpy(x).to(device).float() for x in arrays]
