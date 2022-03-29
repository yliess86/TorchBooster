from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import (Flatten, GELU, Linear, Module, Sequential, Sigmoid, Unflatten)
from torch.nn.functional import mse_loss
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
from tqdm import tqdm

import torch
import torchbooster.distributed as dist
import torchbooster.utils as utils
import torchvision.transforms as T


class VE(Sequential):
    def __init__(self, z_dim: int) -> None:
        self.z_dim = z_dim
        super().__init__(
            Flatten(),
            Linear(784, 512), GELU(),
            Linear(512, 2 * self.z_dim),
        )

    def forward(self, x: Tensor) -> tuple(Tensor, Tensor):
        x = super().forward(x)
        mu, log_var = x[:, :self.z_dim], x[:, :self.z_dim]
        z = mu + torch.exp(log_var) * torch.randn_like(mu)
        kl = (torch.exp(log_var) ** 2 + mu ** 2 - log_var - 0.5).sum()
        return z, kl


class D(Sequential):
    def __init__(self, z_dim: int) -> None:
        self.z_dim = z_dim
        super().__init__(
            Linear(512, self.z_dim), GELU(),
            Linear(784, 512), Sigmoid(),
            Unflatten(1, (1, 28, 28)),
        )


class VAE(Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.encoder = VE(self.z_dim)
        self.decoder = D(self.z_dim)

    def forward(self, x: Tensor) -> tuple(Tensor, Tensor):
        z, kl = self.encoder(x)
        return self.decoder(z), kl


@dataclass
class Config(BaseConfig):
    epochs: int
    seed: int
    z_dim: int

    env: EnvironementConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig
    dataset: DatasetConfig


def step(
    conf: Config,
    vae: Module,
    optim: Optimizer,
    scheduler: BaseScheduler,
    scaler: GradScaler,
    loader: DataLoader,
    train: bool,
) -> None:
    vae.train(train)
    with tqdm(loader, desc="Train" if train else "Test", disable=not dist.is_primary()) as pbar:
        run_loss, run_mse, run_kl = RunningAverage(), RunningAverage(), RunningAverage()
        for X, _ in pbar:
            X = conf.env.make(X)
            
            X_, kl = vae(X)
            mse = mse_loss(X_, X)
            loss = mse + kl
            
            if train: utils.step(loss, optim, scheduler=scheduler, scaler=scaler)
            
            run_loss.update(loss.item())
            run_mse.update(mse.item())
            run_kl.update(kl.item())
            pbar.set_postfix(
                loss=f"{run_loss.value:.2e}",
                mse=f"{run_mse.value:.2e}",
                kl=f"{run_kl.value:.2e}",
            )


def fit(
    conf: Config,
    vae: Module,
    optim: Optimizer,
    scheduler: BaseScheduler,
    scaler: GradScaler,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    for _ in tqdm(range(conf.epochs), desc="Epoch", disable=not dist.is_primary()):
        step(conf, vae, optim, scheduler, scaler, train_loader, train=True)

    if dist.is_primary():
        step(conf, vae, optim, scheduler, scaler, test_loader, train=False)


def main(conf: Config) -> None:
    train_transform = T.ToTensor()
    train_set = conf.dataset.make(split=Split.TRAIN, transform=train_transform)
    train_loader = conf.loader.make(train_set, shuffle=True, distributed=conf.env.distributed)

    test_transform = T.ToTensor()
    test_set = conf.dataset.make(split=Split.TEST, transform=test_transform)
    test_loader = conf.loader.make(test_set, shuffle=False, distributed=conf.env.distributed)

    vae = conf.env.make(VAE(conf.z_dim))
    optim = conf.optim.make(vae.parameters())
    scheduler = conf.scheduler.make(optim)
    scaler = GradScaler(enabled=conf.env.fp16)

    fit(conf, vae, optim, scheduler, scaler, train_loader, test_loader)


if __name__ == "__main__":
    utils.seed(42)
    utils.boost(enable=True)

    conf = Config.load(Path("vae.yml"))

    dist.launch(
        main,
        conf.env.n_gpu,
        conf.env.n_machine,
        conf.env.machine_rank,
        conf.env.dist_url,
        args=(conf, )
    )