from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import Optional, Type

import torch
import torch.nn as nn
from pelutils import DataStorage

from frozone import device


class Frozone(nn.Module, abc.ABC):

    Config: Type[DataStorage]

    latent_model: Frozone
    control_model: Frozone
    process_model: Frozone

    def __init__(self, config: DataStorage):
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

    def numel(self) -> int:
        """ Number of model parameters. Further docs here: https://pokemondb.net/pokedex/numel """
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def _state_dict_file_name(cls) -> str:
        return cls.__name__ + ".pt"

    def save(self, path: str):
        torch.save(self.state_dict(), os.path.join(path, self._state_dict_file_name()))
        self.config.save(path)

    @classmethod
    def load(cls, path: str) -> Frozone:
        config = cls.Config.load(path, cls.__name__)
        model = cls(config).to(device)
        model.load_state_dict(torch.load(os.path.join(path, cls._state_dict_file_name()), map_location=device))
        return model

from .ff import FFFrozone
