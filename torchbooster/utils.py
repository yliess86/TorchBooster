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


def step(
    loss: Tensor,
    optimizer: Optimizer,
    scheduler: BaseScheduler,
    scaler: GradScaler,
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
    scheduler: BaseScheduler
        learning rate scheduler
    scaler: GradScaler
        gradient scaler for mixed precision
    clip: int (default: None)
        gradient norm clipping value if not None
    """
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if clip is not None:
        scaler.unscale_(optimizer)
        for group in optimizer.param_groups:
            clip_grad_norm_(group["params"], max_norm=clip)
    scaler.step(optimizer)
    scheduler.step()
    scaler.update()