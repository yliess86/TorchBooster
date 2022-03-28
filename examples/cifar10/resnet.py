from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import (Linear, Module)
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchbooster.metrics import (accuracy, RunningAverage)
from torchbooster.config import (
    BaseConfig,
    EnvironementConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from torchbooster.scheduler import BaseScheduler
from torchvision.datasets.cifar import CIFAR10
from torchvision.models.resnet import resnet18
from tqdm import tqdm

import torchbooster.distributed as dist
import torchbooster.utils as utils
import torchvision.transforms as T


@dataclass
class Config(BaseConfig):
    epochs: int
    seed: int
    clip: float

    env: EnvironementConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig


def step(
    conf: Config,
    resnet: Module,
    optim: Optimizer,
    scheduler: BaseScheduler,
    scaler: GradScaler,
    loader: DataLoader,
    train: bool,
) -> None:
    resnet.train(train)
    with tqdm(loader, desc="Train" if train else "Test", disable=not dist.is_primary()) as pbar:
        run_loss, run_acc = RunningAverage(), RunningAverage()
        for X, labels in pbar:
            X, labels = conf.env.make(X, labels)
            
            logits = resnet(X)
            loss = cross_entropy(logits, labels)
            acc = accuracy(logits, labels)

            if train: utils.step(loss, optim, scheduler=scheduler, scaler=scaler, clip=conf.clip)
            
            run_loss.update(loss.item())
            run_acc.update(acc.item())
            pbar.set_postfix(loss=f"{run_loss.value:.2e}", acc=f"{run_acc.value * 100:.2f}%")


def fit(
    conf: Config,
    resnet: Module,
    optim: Optimizer,
    scheduler: BaseScheduler,
    scaler: GradScaler,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    for _ in tqdm(range(conf.epochs), desc="Epoch", disable=not dist.is_primary()):
        step(conf, resnet, optim, scheduler, scaler, train_loader, train=True)

    if dist.is_primary():
        step(conf, resnet, optim, scheduler, scaler, test_loader, train=False)


def main(conf: Config) -> None:
    # TODO Remove when TorchVision Fix
    # https://github.com/pytorch/vision/issues/5039#issuecomment-987037760
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    download = dist.is_primary()
    normalize = T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), inplace=True)

    train_transform = T.Compose([T.RandomCrop(32, 4, padding_mode="reflect"), T.RandomHorizontalFlip(), T.RandomRotation(15), T.ToTensor(), normalize])
    train_set = CIFAR10("/tmp/cifar10/train", train=True, transform=train_transform, download=download)
    train_loader = conf.loader.make(train_set, shuffle=True, distributed=conf.env.distributed)

    test_transform = T.Compose([T.ToTensor(), normalize])
    test_set = CIFAR10("/tmp/cifar10/test", train=False, transform=test_transform, download=download)
    test_loader = conf.loader.make(test_set, shuffle=False, distributed=conf.env.distributed)

    resnet = resnet18(pretrained=True, progress=dist.is_primary())
    resnet.fc = Linear(512, 10)

    resnet = conf.env.make(resnet)
    optim = conf.optim.make(resnet.parameters())
    scheduler = conf.scheduler.make(optim)
    scaler = GradScaler(enabled=conf.env.fp16)

    fit(conf, resnet, optim, scheduler, scaler, train_loader, test_loader)


if __name__ == "__main__":
    utils.seed(42)
    utils.boost(enable=True)

    conf = Config.load(Path("resnet.yml"))

    dist.launch(
        main,
        conf.env.n_gpu,
        conf.env.n_machine,
        conf.env.machine_rank,
        conf.env.dist_url,
        args=(conf, )
    )