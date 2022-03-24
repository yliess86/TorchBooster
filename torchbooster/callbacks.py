"""callbacks.py

Callbacks utilities.
The Module provides simple utilities for running callbacks
during training of PyTorch models.
"""
from __future__ import annotations

from pathlib import Path
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torchbooster.scheduler import BaseScheduler
from typing import (Any, Union)

import torch


class BaseCallback:
    """Base Callback
    
    Attributes
    ----------
    current: int
        current step of the training
        incremented when update is called
    """

    def __init__(self) -> None:
        super().__init__()
        self.current = 0

    def __call__(self, *args, **kwargs) -> None:
        self.current += 1
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError("Method 'update' is not implemented.")


StateDictable = Union[Module, DistributedDataParallel, Optimizer, BaseScheduler, GradScaler]
def try_extract_state_dict(value: Union[Any, StateDictable]) -> Union[Any, dict(str, Any)]:
    """Try Extract State Dict

    Handle State Dict extraction.
    No more `DDP.module.state_dict()`!
    
    Parameters
    ----------
    value: Any | StateDictable
        any pickable value or pytorch entity that can
        extract a state dict

    Returns
    -------
    state_dict: Any | dict(str, Any)
        value if no state dict extracted else
        state dict
    """
    if (
        isinstance(value, Module       ) or
        isinstance(value, Optimizer    ) or
        isinstance(value, BaseScheduler) or
        isinstance(value, GradScaler   )
    ):
        return value.state_dict()
    
    if isinstance(value, DistributedDataParallel):
        return value.module.state_dict()

    return value


class SaveCallback(BaseCallback):
    """Save Callback
    
    Parameters
    ----------
    every, n_iter: int
        save evert n iterations for a total of n_iter
    root: Path
        path to the root folder where to save the `.pt` file
    prefix:
        name prefix for the `.pt` file
        the file is save as `root/prefix_001.pt` for n_iter 100,
        `root/prefix_01.pt` for n_iter 10

    Attributes
    ----------
    path: Path
        build `.pt` file path from parameters
    """
    
    def __init__(
        self,
        every: int,
        n_iter: int,
        root: Path,
        prefix: str,
    ) -> None:
        super().__init__()
        self.every = every
        self.n_iter = n_iter
        self.root = root
        self.prefix = prefix

    @property
    def path(self) -> Path:
        n = len(str(self.n_iter))
        file = f"{self.prefix}_{self.current:0{n}d}.pt"
        return Path(self.root, file)

    def update(self, **kwargs) -> None:
        """Update
        
        Parameters
        ----------
        kwargs: dict(str, Any | StateDictable)
            value to be saved in the `.pt` file
            if update(model, optim, epoch) is called,
            the object to save will be of the form
            `{ mode: model.state_dict(), optim: optim.state_dict(), epoch: epoch }`
        """
        if self.current % self.every == 0:
            torch.save({
                key: try_extract_state_dict(value)
                for key, value in kwargs.items()
            }, self.path)


__all__ = [
    BaseCallback,
    SaveCallback,
]