from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from torch import (autograd, Tensor)
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import (Flatten, GELU, Linear, Module, Sequential, Sigmoid, Unflatten)
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchbooster.dataset import Split
from torchbooster.metrics import RunningAverage
from torchbooster.config import (
    BaseConfig,
    DatasetConfig,
    EnvironementConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from torchbooster.scheduler import BaseScheduler
from torchvision.utils import make_grid
from tqdm import tqdm

import torch
import torchbooster.distributed as dist
import torchbooster.utils as utils
import torchvision.transforms as T


class Generator(Sequential):
    def __init__(self, z_dim: int) -> None:
        self.z_dim = z_dim
        super().__init__(
            Linear(self.z_dim, 512), GELU(),
            Linear(512, 512), GELU(),
            Linear(512, 784), Sigmoid(),
            Unflatten(1, (1, 28, 28)),
        )


class Discriminator(Sequential):
    def __init__(self) -> None:
        super().__init__(
            Flatten(),
            Linear(784, 512), GELU(),
            Linear(512, 512), GELU(),
            Linear(512, 1),
        )


def grad_penalty(D: Module, X_real: Tensor, X_fake: Tensor) -> Tensor:
    shape = [1,] * len(X_real.shape[1:])
    alpha = torch.rand((X_real.size(0), *shape), device=X_real.device)
    
    t = alpha * X_real - (1 - alpha) * X_fake
    t.requires_grad = True

    Dt = D(t)
    grads = autograd.grad(Dt, t, torch.ones_like(Dt), True, True, True)[0]
    grads = grads.view(grads.size(0), -1)

    return torch.mean((grads.norm(2, dim=1) - 1) ** 2)


@dataclass
class Config(BaseConfig):
    epochs: int
    seed: int

    z_dim: int
    grad_penalty: float

    env: EnvironementConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig
    dataset: DatasetConfig
   

def fit(
    conf: Config,
    G: Module,
    D: Module,
    G_optim: Optimizer,
    G_scheduler: BaseScheduler,
    G_scaler: GradScaler,
    D_optim: Optimizer,
    D_scheduler: BaseScheduler,
    D_scaler: GradScaler,
    loader: DataLoader,
) -> None:
    for _ in tqdm(range(conf.epochs), desc="Epoch", disable=not dist.is_primary()):
        G.train()
        D.train()
        with tqdm(loader, desc="Train", disable=not dist.is_primary()) as pbar:
            run_G_loss, run_D_loss = RunningAverage(), RunningAverage()
            for X_real, _ in pbar:
                X_real = 1.0 - conf.env.make(X_real)
                z = torch.randn((X_real.size(0), conf.z_dim), device=X_real.device)

                with autocast(conf.env.fp16):
                    X_fake = G(z)
                    G_loss = torch.relu(1.0 - D(X_fake)).mean(dim=0)

                    X_fake = utils.detach(X_fake)
                    D_loss = torch.relu(1.0 - D(X_real)).mean(dim=0) + torch.relu(1.0 + D(X_fake)).mean(dim=0)

                    X_real, X_fake = utils.detach(X_real, X_fake)
                    D_loss += conf.grad_penalty * grad_penalty(D, X_real, X_fake)
                
                utils.step(G_loss, G_optim, scheduler=G_scheduler, scaler=G_scaler)
                utils.step(D_loss, D_optim, scheduler=D_scheduler, scaler=D_scaler)
                
                run_G_loss.update(G_loss.item())
                run_D_loss.update(D_loss.item())
                pbar.set_postfix(
                    G_loss=f"{run_G_loss.value:.2e}",
                    D_loss=f"{run_D_loss.value:.2e}",
                )


def sample(conf: Config, G: Module) -> None:
    G = G if isinstance(G, Module) else G.module.decoder
    G.eval()
    
    with autocast(conf.env.fp16):
        z = torch.randn((16 * 16, conf.z_dim))
        X = 1.0 - G(conf.env.make(z))
    
    T.ToPILImage()(make_grid(X, nrow=16)).show()


def main(conf: Config) -> None:
    set = conf.dataset.make(split=Split.TRAIN, transform=T.ToTensor())
    loader = conf.loader.make(set, shuffle=True, distributed=conf.env.distributed)

    G = conf.env.make(Generator(conf.z_dim))
    D = conf.env.make(Discriminator())

    G_optim = conf.optim.make(G.parameters())
    G_scheduler = conf.scheduler.make(G_optim)
    G_scaler = GradScaler(enabled=conf.env.fp16)

    D_optim = conf.optim.make(D.parameters())
    D_scheduler = conf.scheduler.make(D_optim)
    D_scaler = GradScaler(enabled=conf.env.fp16)

    fit(conf, G, D, G_optim, G_scheduler, G_scaler, D_optim, D_scheduler, D_scaler, loader)
    if dist.is_primary(): sample(conf, G)


if __name__ == "__main__":
    conf = Config.load(Path("gan.yml"))

    utils.seed(conf.seed)
    utils.boost(enable=True)

    dist.launch(
        main,
        conf.env.n_gpu,
        conf.env.n_machine,
        conf.env.machine_rank,
        conf.env.dist_url,
        args=(conf, )
    )