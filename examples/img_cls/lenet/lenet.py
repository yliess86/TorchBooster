from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import (BatchNorm2d, Conv2d, Flatten, GELU, Linear, MaxPool2d, Module, Sequential)
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchbooster.dataset import Split
from torchbooster.metrics import (accuracy, RunningAverage)
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

import torchbooster.distributed as dist
import torchbooster.utils as utils
import torchvision.transforms as T


LeNet = Sequential(
    Sequential(Conv2d(1,  6, 5), BatchNorm2d( 6), GELU(), MaxPool2d(2)),
    Sequential(Conv2d(6, 16, 5), BatchNorm2d(16), GELU(), MaxPool2d(2)),
    Flatten(),
    Linear(256, 120), GELU(),
    Linear(120,  84), GELU(),
    Linear( 84,  10),
)


@dataclass
class Config(BaseConfig):
    epochs: int
    seed: int

    env: EnvironementConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig
    dataset: DatasetConfig


def step(
    conf: Config,
    lenet: Module,
    optim: Optimizer,
    scheduler: BaseScheduler,
    scaler: GradScaler,
    loader: DataLoader,
    train: bool,
) -> None:
    lenet.train(train)
    with tqdm(loader, desc="Train" if train else "Test", disable=not dist.is_primary()) as pbar:
        run_loss, run_acc = RunningAverage(), RunningAverage()
        for X, labels in pbar:
            X, labels = conf.env.make(X, labels)
            
            with autocast(conf.env.fp16):
                logits = lenet(X)
                loss = cross_entropy(logits, labels)
                acc = accuracy(logits, labels)

            if train: utils.step(loss, optim, scheduler=scheduler, scaler=scaler)
            
            run_loss.update(loss.item())
            run_acc.update(acc.item())
            pbar.set_postfix(loss=f"{run_loss.value:.2e}", acc=f"{run_acc.value * 100:.2f}%")


def fit(
    conf: Config,
    lenet: Module,
    optim: Optimizer,
    scheduler: BaseScheduler,
    scaler: GradScaler,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    for _ in tqdm(range(conf.epochs), desc="Epoch", disable=not dist.is_primary()):
        step(conf, lenet, optim, scheduler, scaler, train_loader, train=True)

    if dist.is_primary():
        step(conf, lenet, optim, scheduler, scaler, test_loader, train=False)


def main(conf: Config) -> None:
    train_transform = T.Compose([T.RandAugment(), T.ToTensor()])
    train_set = conf.dataset.make(split=Split.TRAIN, transform=train_transform)
    train_loader = conf.loader.make(train_set, shuffle=True, distributed=conf.env.distributed)

    test_transform = T.ToTensor()
    test_set = conf.dataset.make(split=Split.TEST, transform=test_transform)
    test_loader = conf.loader.make(test_set, shuffle=False, distributed=conf.env.distributed)

    lenet = conf.env.make(LeNet)    
    optim = conf.optim.make(lenet.parameters())
    scheduler = conf.scheduler.make(optim)
    scaler = GradScaler(enabled=conf.env.fp16)

    fit(conf, lenet, optim, scheduler, scaler, train_loader, test_loader)


if __name__ == "__main__":
    conf = Config.load(Path("lenet.yml"))

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