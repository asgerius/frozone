import contextlib

import torch
import torch.cuda.amp as amp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def amp_context():
    return amp.autocast() if torch.cuda.is_available() else contextlib.ExitStack()
