"""scheduler.py

Learning rate schedulers.
The module provides a base class that can be extended
to implement learning rate schedulers.
"""
from __future__ import annotations

from torch.optim import Optimizer
from typing import Any

import math


def anneal_linear(a, b, t) -> float:
    return a + t * (b - a)


def anneal_cos(a, b, t) -> float:
    return b + .5 * (a - b) * (math.cos(math.pi * t) + 1)


def anneal_exp(a, b, t) -> float:
    return a * (b / a) ** t


def anneal_flat(a, b, t) -> float:
    return a


PHASE_2_FUN = {
    "lin" : anneal_linear,
    "cos" : anneal_cos,
    "exp" : anneal_exp,
    "flat": anneal_flat,
}


class BaseScheduler:
    """Scheduler

    Base class for implementing learning rate schedulers.
    """
    
    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def state_dict(self) -> dict(str, Any):
        """State Dict"""
        raise NotImplementedError("Method 'state_dict' not implemented.")

    def load_state_dict(self, state_dict: dict(str, Any)) -> None:
        """Load State Dict"""
        raise NotImplementedError("Method 'load_state_dict' not implemented.")

    def step(self) -> float:
        """Step

        Take a step into the learning rate schedule.
        Apply learning rate and update.

        Returns
        -------
        lr: float
            new learning rate
        """
        raise NotImplementedError("Method 'step' not implemented.")


class CycleScheduler(BaseScheduler):
    """Cycle Sheduler
    
    Cycle learning rate scheduler.

    Parameters
    ----------
    optimizer: Optimizer
        optimizer used during training (holding lr parameter)
    lr: float
        base learning rate
        will be used to compute appropriate values w/ phase
    n_iter: int
        number of training iterations
    initial_multiplier, final_multiplier: float (default: 4e-2, 1e-5)
        initial learning multiplier for warmup
        final learning rate multiplier for annealing
    warmup, plateau: int (default: 500, 0)
        number of steps for warmup and plateau before annealing
    decay: tuple(int, int) (default: 'cos', 'cos')
        decay phases (warmup, anneal)

    Attributes
    ----------
    phases: list(int)
        list of all the learning rate schedule phases
        the phase should be part of the SchedulerPhase Enum
    phase, phase_step: int
        current phase and current step in the phase
    last_lr: float
        last learning rate value
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr: float,
        n_iter: int,
        initial_multiplier: float = 4e-2,
        final_multiplier: float = 1e-5,
        warmup: int = 0,
        plateau: int = 0,
        decay: tuple(str, str) = ("cos", "cos"),
    ) -> None:
        super().__init__(optimizer)
        phases = []
        if warmup > 0: phases.append((decay[0], lr * initial_multiplier, lr, warmup))
        if plateau > 0: phases.append(("linear", lr, lr, plateau))
        phases.append((decay[1], lr, lr * final_multiplier, n_iter - warmup - plateau))

        self.phases = phases
        self.phase = 0
        self.phase_step = 0

        self.last_lr = None

    def state_dict(self) -> dict(str, Any):
        """State Dict"""
        return {
            "phases"    : self.phases,
            "phase"     : self.phase,
            "phase_step": self.phase_step,
            "last_lr"   : self.last_lr,
        }

    def load_state_dict(self, state_dict: dict(str, Any)) -> None:
        """Load State Dict"""
        self.phases     = state_dict["phases"]
        self.phase      = state_dict["phase"]
        self.phase_step = state_dict["phase_step"]
        self.last_lr    = state_dict["last_lr"]

    def __repr__(self) -> str:
        name = self.__class__.__name__
        phases = [phase.upper() for phase, *_ in self.phases]
        return f"{name}(phases={phases})"

    def step(self) -> float:
        """Step

        Take a step into the learning rate schedule.
        Select the phase, phase step, apply learning rate and update.

        Returns
        -------
        lr: float
            new learning rate
        """
        phase, lr_from, lr_to, n_iter = self.phases[self.phase]
        phase_fun = PHASE_2_FUN[phase]

        lr = phase_fun(lr_from, lr_to, self.phase_step / n_iter)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.phase_step += 1
        self.last_lr = lr

        if self.phase_step > n_iter:
            self.phase += 1
            self.phase_step = 0

        return lr


__all__ = [
    BaseScheduler,
    CycleScheduler,
]
