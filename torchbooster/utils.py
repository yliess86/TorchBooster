"""utils.py

Training utilities.
The Module provides simple training utilities
to reduce boilerplate code when implementing
training code for PyTorch.
"""
from __future__ import annotations

from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from torch.cuda.amp.grad_scaler import GradScaler
from torchbooster.scheduler import BaseScheduler

import numpy as np
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


def step(
    loss: Tensor,
    optimizer: Optimizer,
    scheduler: BaseScheduler = None,
    scaler: GradScaler = None,
    clip: int = None,
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
    clip: int (default: None)
        gradient norm clipping value if not None
    """
    optimizer.zero_grad(set_to_none=True)
    
    if scaler is not None: scaler.scale(loss).backward()
    else: loss.backward()
    
    if clip is not None:
        if scaler is not None: scaler.unscale_(optimizer)
        for group in optimizer.param_groups:
            clip_grad_norm_(group["params"], max_norm=clip)
    
    if scaler is not None: scaler.step(optimizer)
    else: optimizer.step()
    
    if scheduler is not None: scheduler.step()
    if scaler is not None: scaler.update()