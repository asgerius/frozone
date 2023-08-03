from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import Any, Optional, Type

import torch
import torch.nn as nn
from pelutils import DataStorage

from frozone import device


class Frozone(nn.Module, abc.ABC):

    @dataclass
    class Config(DataStorage):

        D: int  # Number of process variables
        d: int  # Number of control variables
        K: int  # Number of latent variables

        h: int  # Number of history steps
        f: int  # Number of prediction steps

        def __post_init__(self):
            assert self.D > 0
            assert self.d > 0
            assert self.h > 0
            assert self.f > 0

    latent_model: Frozone
    control_model: Frozone
    process_model: Frozone

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

    @abc.abstractmethod
    def forward(
        self,
        XH: torch.FloatTensor,
        UH: torch.FloatTensor,
        XF: Optional[torch.FloatTensor] = None,
        UF: Optional[torch.FloatTensor] = None,
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> torch.FloatTensor:
        """ This method implementation does not change anything, but it adds half-assed type support for forward calls. """
        return super().__call__(*args, **kwds)

    def numel(self) -> int:
        """ Number of model parameters. Further docs here: https://pokemondb.net/pokedex/numel """
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def _state_dict_file_name(cls) -> str:
        return cls.__name__ + ".pt"

    def save(self, path: str):
        torch.save(self.state_dict(), os.path.join(path, self._state_dict_file_name()))
        self.config.save(path, self.__class__.__name__ + "_" + self.Config.__name__)

    @classmethod
    def load(cls, path: str) -> Frozone:
        config = cls.Config.load(path, cls.__name__ + "_" + cls.Config.__name__)
        model = cls(config).to(device)
        model.load_state_dict(torch.load(os.path.join(path, cls._state_dict_file_name()), map_location=device))
        return model

from .ff import FFFrozone
