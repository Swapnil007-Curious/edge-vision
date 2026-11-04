"""Exponential moving average of model weights, for the v2 training recipe."""

import copy
from typing import Generic, TypeVar

import torch
from torch import nn

ModuleT = TypeVar("ModuleT", bound=nn.Module)


class ModelEMA(Generic[ModuleT]):
    """A frozen copy of a model whose parameters track ema = decay * ema + (1 - decay) * live
    after every optimizer step. Buffers (BatchNorm running statistics, feature standardisation)
    are copied from the live model at each update. The copy stays in eval mode: it is only
    evaluated and saved."""

    def __init__(self, model: ModuleT, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.module: ModuleT = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for average, live in zip(self.module.parameters(), model.parameters(), strict=True):
            average.lerp_(live.detach(), 1.0 - self.decay)
        for stored, current in zip(self.module.buffers(), model.buffers(), strict=True):
            stored.copy_(current)
        self.updates += 1
