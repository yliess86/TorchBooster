"""utils.py

Training utilities.
The Module provides simple training utilities
to reduce boilerplate code when implementing
training code for PyTorch.
"""
from __future__ import annotations
from collections import namedtuple

from itertools import chain
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torchbooster.scheduler import BaseScheduler
from typing import (Any, Dict, Iterator, List, NamedTuple, Tuple, TypeVar, Union)

import numpy as np
import os
import random
import torch


def boost(enable: bool = True) -> None:
    """Boost
    
    Enable cudnn benchmark to optimize cuda kernel parameters.
    Distable checks and profilers for autograd.

    Parameters
    ----------
    enable: bool (dault: True)
        enable if True or disable if False
    """
    torch.backends.cudnn.benchmark = enable
    torch.autograd.profiler.profile(enabled=not enable)
    torch.autograd.profiler.emit_nvtx(enabled=not enable)
    torch.autograd.set_detect_anomaly(mode=not enable)


def seed(value: int) -> None:
    """Seed
    
    Set seed for random, numpy, and pytorch pseudo-random generators (PGN).

    Parameters
    ----------
    value: int
        seed value to pass to every PGN
    """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def freeze(module: Module) -> Module:
    """Freeze
    
    Freeze module's parameters (grad to False).

    Parameters
    ----------
    module: Module
        module to freeze

    Returns
    -------
    module: Module
        module with freezed parameters
    """
    for param in module.parameters():
        param.requires_grad = False
    return module


def detach(*tensors: Iterator[Tensor]) -> Union[Tensor, Iterator[Tensor]]:
    """Detach
    
    Detach tensors.

    Parameters
    ----------
    tensors: Tensor | Iterator[Tensor]
        tensor or tensors to be detached

    Returns
    -------
    tensors: Tensor | Iterator[Tensor]
        detach tensor or tensors
    """
    if len(tensors) == 1: return tensors[0].detach()
    return (t.detach() for t in tensors)


def iter_loader(loader: DataLoader) -> Iterator[int, Any]:
    """Iter Loader
    
    Infinite iterator for DataLoader.
    Enable the expression of training by num_iter instead of num_epochs.
    https://twitter.com/karpathy/status/1508437725514510336

    Parameters
    ----------
    loader: DataLoader
        data loader

    Returns
    -------
    iterator: Iterator[int, Any]
        iterator wrapping the dataloader
        returns the current epoch and the dataloader next output
    """
    iterator = iter(loader)
    epoch = 0

    while True:
        try: yield epoch, next(iterator)
        except StopIteration:
            iterator = iter(loader)
            epoch += 1
            yield epoch, next(iterator)


def isinstance_namedtuple(obj) -> bool:
    return (
        isinstance(obj, tuple) and
        hasattr(obj, '_asdict') and
        hasattr(obj, '_fields')
    )

Tensorable = TypeVar('Tensorable', Tuple[Any], List[Any], Dict[str, Any])
Tensored   = TypeVar('Tensored',  List[Tensor], Dict[str, Tensor])
Device = Union[str, torch.device]
def to_tensor(data: Tensorable, dtype: torch.dtype = torch.float32, device: Device = "cpu") -> Tensored:
    """to_tensor
    Converts data to a pytorch tensor

    Parameters
    ----------
    data : Tensorable
        The input data containing data to be transformed to a tensor
    dtype : torch.dtype, optional
        The dtype of the returned tensor, by default torch.float32
    device : Device, optional
        The device to put the tensor on, by default "cpu"

    Returns
    -------
    Tensored
        The tensored data with the same shape as data but transformed to torch.Tensor
    """
    def tensor(element):
        return torch.tensor(element, device=device)

    if isinstance(data, list):
        if len(data) == 1:
            return tensor(data[0])
        return tensor(data)
    if hasattr(data, '__dict__'):#isinstance(data, dict):
        if hasattr(data, "copy"): # work on a copy
            data = data.copy()
        for k,v in data.items():
            data[k] = tensor(v)
        return data
    if isinstance_namedtuple(data):
        a = [tensor(elem) for elem in data._asdict().values()]
        return data.__class__(*a)
        #(k: tensor(v) for k, v in data.item())
    return data


def stack_dictionaries(data: List[Dict[str, Tensor]], dim: int = 0) -> Dict[str, Tensor]:
    """stack_dictionaries
    Stack given dictionaries of str, Tensor type and returns a single dictionaries with Tensor stacked 

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The input dictionaries with the same keys and tensor of same shape

    Returns
    -------
    Dict[str, Tensor]
        The stacked dictionary
    """
    dic = {k: [] for k in dict(data[0]).keys()} # dict() to work with Embedding wrappers
    for elem in data:
        for k, v in elem.items():
            dic[k].append(v)
    return {k: torch.stack(v, dim) for k, v in dic.items()}


def step(
    loss: Tensor,
    optimizer: Optimizer,
    scheduler: BaseScheduler = None,
    scaler: GradScaler = None,
    clip: float = None,
    retain_graph: bool = False,
) -> None:
    """Step
    
    Optimization step.
    Zero gradients, scale loss, apply clipping, step and update states.

    Parameters
    ----------
    loss: Tensor
        loss to minimize
    optimizer: Optimizer
        model optimizer
    scheduler: BaseScheduler (default: None)
        learning rate scheduler
    scaler: GradScaler (default: None)
        gradient scaler for mixed precision
    retain_graph: bool (default: False)
        retain graph or not when computing backward pass
    """
    optimizer.zero_grad(set_to_none=True)

    if scaler is not None: scaler.scale(loss).backward(retain_graph=retain_graph)
    else: loss.backward(retain_graph=retain_graph)
    
    if clip is not None:
        if scaler is not None: scaler.unscale_(optimizer)
        params = chain((p for group in optimizer.param_groups for p in group["params"]))
        clip_grad_norm_(params, max_norm=clip)
    
    if scaler is not None: scaler.step(optimizer)
    else: optimizer.step()
    
    if scheduler is not None: scheduler.step()
    if scaler is not None: scaler.update()