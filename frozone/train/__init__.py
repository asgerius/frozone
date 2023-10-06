from __future__ import annotations

import math
from dataclasses import dataclass
from pprint import pformat
from typing import Callable, Optional, Type

import numpy as np
import torch
from pelutils import DataStorage

import frozone.environments as environments
from frozone import device


# This global variable is used to indicate to threads whether or not a training is running.
# If it is false, threads should stop within a very short amount of time.
is_doing_training = False

def history_only_weights(env: Type[environments.Environment]) -> np.ndarray:

    x_future_include = np.ones(len(env.XLabels), dtype=np.float32)
    for xlab in env.no_reference_variables:
        x_future_include[xlab] = 0

    return x_future_include

def get_loss_fns(env: Type[environments.Environment], train_cfg: TrainConfig) -> tuple[Callable, Callable]:
    if train_cfg.loss_fn == "l1":
        loss_fn = torch.nn.L1Loss(reduction="none")
    elif train_cfg.loss_fn == "l2":
        loss_fn = torch.nn.MSELoss(reduction="none")
    elif train_cfg.loss_fn == "huber":
        _loss_fn = torch.nn.HuberLoss(reduction="none", delta=train_cfg.huber_delta)
        loss_fn = lambda target, input: 1 / train_cfg.huber_delta * _loss_fn(target, input)
    loss_weight = torch.ones(train_cfg.F, device=device)
    loss_weight = loss_weight / loss_weight.sum()

    future_include_weights = history_only_weights(env)
    future_include_weights = torch.from_numpy(future_include_weights).to(device) * len(future_include_weights) / future_include_weights.sum()

    def loss_fn_x(x_target: torch.FloatTensor, x_pred: torch.FloatTensor) -> torch.FloatTensor:
        loss: torch.FloatTensor = loss_fn(x_target, x_pred).mean(dim=0)
        return (loss.T @ loss_weight * future_include_weights).mean()
    def loss_fn_u(u_target: torch.FloatTensor, u_pred: torch.FloatTensor) -> torch.FloatTensor:
        loss: torch.FloatTensor = loss_fn(u_target, u_pred).mean(dim=0)
        return (loss.T @ loss_weight).mean()

    return loss_fn_x, loss_fn_u

@dataclass
class TrainConfig(DataStorage):

    # Environment should match name of a class in simulations
    env:                str
    data_path:          str
    phase:              Optional[str]

    # Windows are given in seconds
    history_window:     float
    prediction_window:  float

    # Training stuff
    batches:            int
    batch_size:         int
    lr:                 float
    num_models:         int
    # Dataset stuff
    max_num_data_files: int
    eval_size:          int

    # Loss stuff
    loss_fn:            str
    huber_delta:        float

    # Data augmentation
    # Standard deviation of the generated noise
    # This is given as a multiplier to the feature-wise standard deviation in the data
    epsilon:            float
    augment_prob:       float

    @property
    def H(self) -> int:
        return int(self.history_window / self.get_env().dt)

    @property
    def F(self) -> int:
        return int(self.prediction_window / self.get_env().dt)

    @property
    def num_eval_batches(self) -> int:
        return math.ceil(self.eval_size / self.batch_size)

    def get_env(self) -> Type[environments.Environment]:
        return getattr(environments, self.env)

    def __str__(self) -> str:
        return pformat(vars(self))

@dataclass
class TrainResults(DataStorage):

    # Feature-wise means and standard deviations of training data
    mean_x: Optional[np.ndarray]
    std_x:  Optional[np.ndarray]
    mean_u: Optional[np.ndarray]
    std_u:  Optional[np.ndarray]

    checkpoints:        list[int]  # Batches before which there was a checkpoint
    train_loss_x:       list[list[float]]  # Models outermost, batches innermost
    train_loss_u:       list[list[float]]
    test_loss_x:        list[list[float]]
    test_loss_u:        list[list[float]]
    test_loss_x_std:    list[list[float]]
    test_loss_u_std:    list[list[float]]
    ensemble_loss_x:    list[float]
    ensemble_loss_u:    list[float]
    lr:                 list[float]

    @classmethod
    def empty(cls, num_models: int) -> TrainResults:
        return TrainResults(
            # Mean and standard deviation
            None, None, None, None,
            # Everything else
            list(), *([list() for _ in range(num_models)] for _ in range(6)),
            list(), list(), list(),
        )
