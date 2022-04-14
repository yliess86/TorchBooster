from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from torch.cuda.amp import (autocast, GradScaler)
from torch import Tensor
from torch.nn import (Conv2d, GELU, InstanceNorm2d, Module, ReflectionPad2d, Sequential, Upsample)
from torch.nn.functional import mse_loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchbooster.config import (
    BaseConfig,
    EnvironementConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from torchbooster.scheduler import BaseScheduler
from torchvision.datasets import ImageFolder
from torchvision.models.vgg import vgg16
from torchvision.utils import make_grid
from tqdm import tqdm
from typing import Callable

import numpy as np
import os
import pandas as pd
import torch
import torchbooster.distributed as dist
import torchbooster.utils as utils
import torchvision.transforms as T


Layer = Callable[[int, int, int, int], Sequential]
Conv      : Layer = lambda i, o, k, s: Sequential(ReflectionPad2d(k // 2), Conv2d(i, o, k, s))
ConvIN    : Layer = lambda i, o, k, s: Sequential(Conv(i, o, k, s), InstanceNorm2d(o, affine=True), GELU())
DeconvIN  : Layer = lambda i, o, k, s: Sequential(Upsample(scale_factor=2), ConvIN(i, o, k, s), GELU())


class Decoder(Sequential):
    def __init__(self) -> None:
        super().__init__(
            ConvIN  (512, 256, 3, 1),
            DeconvIN(256, 256, 3, 1),
            ConvIN  (256, 256, 3, 1),
            ConvIN  (256, 128, 3, 1),
            DeconvIN(128, 128, 3, 1),
            ConvIN  (128,  64, 3, 1),
            DeconvIN( 64,  64, 3, 1),
            Conv    ( 64,   3, 9, 1),
        )


def mu_std(feat: Module, eps: float = 1e-5) -> tuple(Tensor, Tensor):
    mu  = feat.mean(dim=[2, 3], keepdim=True)
    std = feat.var (dim=[2, 3], keepdim=True).add(eps).sqrt()
    return mu.expand_as(feat), std.expand_as(feat)


def adaIN(s_feat: Tensor, c_feat: Tensor) -> Tensor:
    (s_mu, s_std), (c_mu, c_std) = mu_std(s_feat), mu_std(c_feat)
    return s_std * (c_feat - c_mu) / c_std + s_mu


@dataclass
class CocoDatasetConfig(BaseConfig):
    name: str
    root: str
    zip: str

    def make(self, transform: Module = None) -> ImageFolder:
        if not os.path.isdir(self.root):
            os.makedirs(self.root, exist_ok=True)
            os.system(f"cd { self.root } && wget { self.zip } && unzip train2017.zip && rm train2017.zip")
        return ImageFolder(root=self.root, transform=transform)


@dataclass
class PaintingsDatasetConfig(BaseConfig):
    name: str
    root: str
    csv: str

    def make(self, transform: Module = None) -> ImageFolder:
        if not os.path.isdir(self.root):
            os.makedirs(self.root, exist_ok=True)
            os.system(f"cd {self.root} && wget {self.csv}")
            df = pd.read_csv(os.path.join(self.root, self.csv.split("/")[-1]))
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Download"):
                sub_root = os.path.join(self.root, row["Labels"][2:-1].split(" ")[0])
                os.makedirs(sub_root, exist_ok=True)
                os.system(f"cd {sub_root} && wget -o /dev/null {row['Image URL']}")
        return ImageFolder(root=self.root, transform=transform)


@dataclass
class Config(BaseConfig):
    n_iter: int
    seed: int
    size: int
    clip: float
    
    layers: list(int)

    style_weight: float
    content_weight: float

    coco: CocoDatasetConfig
    paintings: PaintingsDatasetConfig
    
    env: EnvironementConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig


def fit(
    conf: Config,
    encoder: Module,
    decoder: Module,
    s_loader: DataLoader,
    c_loader: DataLoader,
    optim: Optimizer,
    scheduler: BaseScheduler,
    scaler: GradScaler,
    utransform: Module,
) -> None:
    feats = {}
    hook = lambda module, input, output, layer: feats.update({str(layer): output})
    for l in set(conf.layers): encoder[l].register_forward_hook(partial(hook, layer=l))

    c_criterion = lambda mfs, cfs: mse_loss(mfs[-1], cfs[-1])
    s_criterion = lambda mfs, sfs: sum([mse_loss(xm, sm) + mse_loss(xs, ss) for (xm, xs), (sm, ss) in zip(map(mu_std, mfs), map(mu_std, sfs))])

    s_batches = utils.iter_loader(s_loader)
    c_batches = utils.iter_loader(c_loader)
    with tqdm(range(conf.n_iter), desc="Transfer") as pbar:
        for step in pbar:
            s_epoch, (style,   _) = next(s_batches)
            c_epoch, (content, _) = next(c_batches)
            style, content = conf.env.make(style, content)
            
            with autocast(conf.env.fp16):
                encoder(style)
                s_feats = [feats[str(l)].detach() for l in conf.layers]
                
                encoder(content)
                c_feats = [feats[str(l)].detach() for l in conf.layers]

                mixture = decoder(adaIN(s_feats[-1], c_feats[-1]))
                encoder(mixture)
                m_feats = [feats[str(l)] for l in conf.layers]
                
                loss  = conf.style_weight   * s_criterion(m_feats, s_feats)
                loss += conf.content_weight * c_criterion(m_feats, c_feats)

            utils.step(loss, optim, scheduler, scaler, clip=conf.clip)
            pbar.set_postfix(c_epoch=c_epoch, s_epoch=s_epoch, loss=f"{loss.item():.2e}")

            if step % 500 == 0 or step == conf.n_iter - 1:
                stacked = torch.cat((style[:1], content[:1], mixture[:1]), dim=0)
                utransform(make_grid(stacked, nrow=1)).show()


def main(conf: Config) -> None:
    mean, std = np.array((0.485, 0.456, 0.406)), np.array((0.229, 0.224, 0.225))
    clamp = T.Lambda(lambda x: x.clamp(0, 1))

    ctransform = T.Compose([T.Resize(conf.size), T.CenterCrop(conf.size), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    utransform = T.Compose([T.Normalize(mean=-mean / std, std=1.0 / std), clamp, T.ToPILImage()])

    paintings = conf.paintings.make(transform=ctransform)
    s_loader = conf.loader.make(paintings, shuffle=True, distributed=conf.env.distributed)

    coco = conf.coco.make(transform=ctransform)
    c_loader = conf.loader.make(coco, shuffle=True, distributed=conf.env.distributed)

    encoder = conf.env.make(utils.freeze(vgg16(pretrained=True, progress=True).features.eval()))
    decoder = conf.env.make(Decoder())

    optim = conf.optim.make(decoder.parameters())
    scheduler = conf.scheduler.make(optim)
    scaler = GradScaler(enabled=conf.env.fp16)
    
    fit(conf, encoder, decoder, s_loader, c_loader, optim, scheduler, scaler, utransform)


if __name__ == "__main__":
    conf = Config.load(Path("adain.yml"))

    utils.seed(conf.seed, deterministic=False)
    utils.boost(enable=True)
    
    dist.launch(
        main,
        conf.env.n_gpu,
        conf.env.n_machine,
        conf.env.machine_rank,
        conf.env.dist_url,
        args=(conf, )
    )