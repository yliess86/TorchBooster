from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from torch import Tensor
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import (Flatten, GELU, Linear, Module, Sequential, Sigmoid, Unflatten)
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits
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


class VE(Sequential):
    def __init__(self, z_dim: int) -> None:
        self.z_dim = z_dim
        super().__init__(
            Flatten(),
            Linear(784, 512), GELU(),
            Linear(512, 512), GELU(),
            Linear(512, 2 * self.z_dim),
        )

    def forward(self, x: Tensor) -> tuple(Tensor, Tensor, Tensor):
        x = super().forward(x)
        mu, log_var = torch.split(x, self.z_dim, dim=1)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        return z, mu, log_var


class D(Sequential):
    def __init__(self, z_dim: int) -> None:
        self.z_dim = z_dim
        super().__init__(
            Linear(self.z_dim, 512), GELU(),
            Linear(512, 512), GELU(),
            Linear(512, 784), Sigmoid(),
            Unflatten(1, (1, 28, 28)),
        )


class VAE(Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.encoder = VE(self.z_dim)
        self.decoder = D(self.z_dim)

    def forward(self, x: Tensor) -> tuple(Tensor, Tensor, Tensor):
        z, mu, log_var = self.encoder(x)
        return self.decoder(z), mu, log_var


def kl_divergence(mu: Tensor, log_var: Tensor) -> Tensor:
    kl = 1 + log_var - mu ** 2 - torch.exp(log_var)
    kl = -.5 * kl.sum(dim=1)
    return kl.mean(dim=0)


@dataclass
class Config(BaseConfig):
    epochs: int
    seed: int
    clip: float

    z_dim: int
    kld_weight: float

    env: EnvironementConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig
    dataset: DatasetConfig
   

def fit(
    conf: Config,
    vae: Module,
    optim: Optimizer,
    scheduler: BaseScheduler,
    scaler: GradScaler,
    loader: DataLoader,
) -> None:
    for _ in tqdm(range(conf.epochs), desc="Epoch", disable=not dist.is_primary()):
        vae.train(True)
        with tqdm(loader, desc="Train", disable=not dist.is_primary()) as pbar:
            run_loss, run_bce, run_kld = RunningAverage(), RunningAverage(), RunningAverage()
            for X, _ in pbar:
                X = 1.0 - conf.env.make(X)
                
                with autocast(conf.env.fp16):
                    X_, mu, log_var = vae(X)
                    kld = kl_divergence(mu, log_var)
                    bce = bce_with_logits(X_, X)
                    loss = bce + conf.kld_weight * kld
                
                utils.step(loss, optim, scheduler=scheduler, scaler=scaler, clip=conf.clip)
                
                run_loss.update(loss.item())
                run_bce.update(bce.item())
                run_kld.update(kld.item())
                pbar.set_postfix(
                    loss=f"{run_loss.value:.2e}",
                    bce=f"{run_bce.value:.2e}",
                    kl=f"{run_kld.value:.2e}",
                )


def sample(conf: Config, vae: Module) -> None:
    D = vae.decoder if isinstance(vae, Module) else vae.module.decoder
    D.eval()
    
    with autocast(conf.env.fp16):
        z = torch.randn((16 * 16, conf.z_dim))
        X = 1.0 - D(conf.env.make(z))
    
    T.ToPILImage()(make_grid(X, nrow=16)).show()


def main(conf: Config) -> None:
    set = conf.dataset.make(split=Split.TRAIN, transform=T.ToTensor())
    loader = conf.loader.make(set, shuffle=True, distributed=conf.env.distributed)

    vae = conf.env.make(VAE(conf.z_dim))
    optim = conf.optim.make(vae.parameters())
    scheduler = conf.scheduler.make(optim)
    scaler = GradScaler(enabled=conf.env.fp16)

    fit(conf, vae, optim, scheduler, scaler, loader)
    if dist.is_primary(): sample(conf, vae)


if __name__ == "__main__":
    conf = Config.load(Path("vae.yml"))

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