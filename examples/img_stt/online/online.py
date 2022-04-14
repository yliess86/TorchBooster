from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from PIL import Image
from torch.cuda.amp import (autocast, GradScaler)
from torch import Tensor
from torch.nn import (Conv2d, GELU, InstanceNorm2d, Module, ReflectionPad2d, Sequential, Upsample)
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
import requests
import torch
import torchbooster.distributed as dist
import torchbooster.utils as utils
import torchvision.transforms as T


class Residual(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.module = Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.module(x)


Layer = Callable[[int, int, int, int], Sequential]
Conv      : Layer = lambda i, o, k, s: Sequential(ReflectionPad2d(k // 2), Conv2d(i, o, k, s))
ConvIN    : Layer = lambda i, o, k, s: Sequential(Conv(i, o, k, s), InstanceNorm2d(o, affine=True), GELU())
DeconvIN  : Layer = lambda i, o, k, s: Sequential(Upsample(scale_factor=2), ConvIN(i, o, k, s), GELU())
Bottleneck: Layer = lambda i, o, k, s: Sequential(ConvIN(i, o, k, s), GELU(), ConvIN(o, i, k, s))


class StyleNet(Sequential):
    def __init__(self) -> None:
        modules = [Residual(Bottleneck(128, 128, 3, 1))] * 5
        modules = Sequential(ConvIN(64, 128, 3, 2), *modules, DeconvIN(128, 64, 3, 1))
        modules = Sequential(ConvIN(32,  64, 3, 2),  modules, DeconvIN( 64, 32, 3, 1))
        super().__init__    (ConvIN( 3,  32, 9, 1),  modules, Conv    ( 32,  3, 9, 1))


def gram_matrix(features: Tensor) -> Tensor:
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)
    return torch.bmm(features, features.transpose(1, 2)) / (C * H * W)


def total_variation(x: Tensor) -> Tensor:
    a = (x[:, :, :  , :-1] - x[:, :,  :, 1:]).abs().sum()
    b = (x[:, :, :-1, :  ] - x[:, :, 1:,  :]).abs().sum()
    return a + b


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
class Config(BaseConfig):
    n_iter: int
    seed: int
    size: int
    clip: float

    style: str
    content: str
    
    layers: list(int)
    content_layer: int
    
    style_weight: float
    content_weight: float
    tv_weight: float

    env: EnvironementConfig
    dataset: CocoDatasetConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig


def fit(
    conf: Config,
    stylenet: Module,
    vgg: Module,
    style: Tensor,
    loader: DataLoader,
    optim: Optimizer,
    scheduler: BaseScheduler,
    scaler: GradScaler,
    utransform: Module,
) -> None:
    feats = {}
    hook = lambda module, input, output, layer: feats.update({str(layer): output})
    for l in set(conf.layers): vgg[l].register_forward_hook(partial(hook, layer=l))
    
    with autocast(conf.env.fp16):
        vgg(style)
        s_grams = [gram_matrix(feats[str(l)]).detach() for l in conf.layers]

    batches = utils.iter_loader(loader)
    with tqdm(range(conf.n_iter), desc="Transfer") as pbar:
        for step in pbar:
            epoch, (content, _) = next(batches)
            content = conf.env.make(content)
            
            with autocast(conf.env.fp16):
                mixture = stylenet(content)

                vgg(content)
                c_feat = feats[str(conf.content_layer)].detach()
                
                vgg(mixture)
                m_grams = [gram_matrix(feats[str(l)]) for l in conf.layers]
                m_feat = feats[str(conf.content_layer)]

                s_loss = sum([(m_g - s_g).pow(2).mean() for m_g, s_g in zip(m_grams, s_grams)])
                c_loss = (m_feat - c_feat).pow(2).mean()
                
                tv_loss = total_variation(mixture)
                loss = conf.style_weight * s_loss + conf.content_weight * c_loss + conf.tv_weight * tv_loss

            utils.step(loss, optim, scheduler, scaler, clip=conf.clip)
            
            pbar.set_postfix(
                epoch=epoch,
                loss=f"{loss.item():.2e}",
                style=f"{s_loss.item():.2e}",
                content=f"{c_loss.item():.2e}",
                tv=f"{tv_loss.item():.2e}",
            )

            if step % 500 == 0:
                stacked = torch.cat((content[:1], stylenet(content[:1])), dim=0)
                utransform(make_grid(stacked, nrow=1)).show()


def main(conf: Config) -> None:
    vgg = conf.env.make(utils.freeze(vgg16(pretrained=True, progress=True).features.eval()))

    mean, std = np.array((0.485, 0.456, 0.406)), np.array((0.229, 0.224, 0.225))
    hdr = T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))

    transform  = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    stransform = T.Compose([T.Resize(conf.size), T.CenterCrop(conf.size), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    ctransform = T.Compose([T.RandomRotation(180), T.RandomResizedCrop(conf.size), T.RandAugment(), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    utransform = T.Compose([T.Normalize(mean=-mean / std, std=1.0 / std), hdr, T.ToPILImage()])

    load_img = lambda url: Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    shape = conf.loader.batch_size, 3, conf.size, conf.size
    style = conf.env.make(stransform(load_img(conf.style)).unsqueeze(0)).expand(*shape)
    content = conf.env.make(transform(load_img(conf.content)).unsqueeze(0))

    dataset = conf.dataset.make(transform=ctransform)
    loader = conf.loader.make(dataset, shuffle=True, distributed=conf.env.distributed)

    stylenet = conf.env.make(StyleNet())
    optim = conf.optim.make(stylenet.parameters())
    scheduler = conf.scheduler.make(optim)
    scaler = GradScaler(enabled=conf.env.fp16)
    
    fit(conf, stylenet, vgg, style, loader, optim, scheduler, scaler, utransform)
    with torch.inference_mode(), autocast(conf.env.fp16): utransform(stylenet(content).squeeze(0)).show()


if __name__ == "__main__":
    conf = Config.load(Path("online.yml"))

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