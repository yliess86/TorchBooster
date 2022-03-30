from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.nn import (Module, Sequential)
from torch.nn.functional import mse_loss
from torch.optim import Optimizer
from torchbooster.config import (
    BaseConfig,
    EnvironementConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from torchbooster.scheduler import BaseScheduler
from torchvision.models.vgg import vgg19
from tqdm import tqdm

import numpy as np
import requests
import torch
import torchbooster.distributed as dist
import torchbooster.utils as utils
import torchvision.transforms as T


def gram_matrix(x: Tensor) -> Tensor:
    B, C, H, W = x.size()
    features = x.view(B * C, H * W)
    G = features @ features.T
    return G / (B * C * H * W)


@dataclass
class Config(BaseConfig):
    n_iter: int
    seed: int
    size: int

    style: str
    style_layers: list(int)
    style_weight: float

    content: str
    content_layers: list(int)
    content_weight: float

    env: EnvironementConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig
   

def transfer(
    conf: Config,
    style: Tensor,
    content: Tensor,
    mixture: Tensor,
    vgg: Module,
    optim: Optimizer,
    scheduler: BaseScheduler,
) -> Tensor:
    feats = {}
    hook = lambda module, input, output, layer: feats.update({str(layer): output})
    for l in set(conf.style_layers + conf.content_layers):
        if isinstance(vgg, Module): vgg[l].register_forward_hook(partial(hook, layer=l))
        else: vgg.module[l].register_forward_hook(partial(hook, layer=l))

    vgg(style)
    s_grams = [gram_matrix(feats[str(l)]).detach() for l in conf.style_layers]

    vgg(content)
    c_feats = [feats[str(l)].detach() for l in conf.content_layers]

    with tqdm(range(conf.n_iter), desc="Transfer", disable=not dist.is_primary()) as pbar:
        for _ in pbar:
            vgg(mixture)
            m_grams = [gram_matrix(feats[str(l)]) for l in conf.style_layers]
            m_feats = [feats[str(l)] for l in conf.content_layers]

            s_loss = sum([mse_loss(m_g, s_g) for m_g, s_g in zip(m_grams, s_grams)]) / len(s_grams)
            c_loss = sum([mse_loss(m_f, c_f) for m_f, c_f in zip(m_feats, c_feats)]) / len(c_feats)
            loss = conf.style_weight * s_loss + conf.content_weight * c_loss

            utils.step(loss, optim, scheduler=scheduler)

            pbar.set_postfix(
                loss=f"{loss.item():.2e}",
                style=f"{s_loss.item():.2e}",
                content=f"{c_loss.item():.2e}",
            )

    return mixture


def main(conf: Config) -> None:
    vgg = vgg19(pretrained=True, progress=dist.is_primary()).features
    M = np.max(conf.content_layers + conf.style_layers + [len(vgg)])
    vgg = Sequential(*list(vgg.children())[:M])
    vgg = utils.freeze(conf.env.make(vgg).eval())

    norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transfrom = T.Compose([T.Resize(conf.size), T.CenterCrop(conf.size), T.ToTensor(), norm])
    
    style = Image.open(BytesIO(requests.get(conf.style).content))
    style = conf.env.make(transfrom(style).unsqueeze(0))

    content = Image.open(BytesIO(requests.get(conf.content).content))
    content = conf.env.make(transfrom(content).unsqueeze(0))

    mixture = content.clone()
    mixture.requires_grad = True

    optim = conf.optim.make([mixture])
    scheduler = conf.scheduler.make(optim)

    mixture = transfer(conf, style, content, mixture, vgg, optim, scheduler)
    with torch.no_grad(): mixture.clamp_(0., 1.)
    if dist.is_primary(): T.ToPILImage()(mixture.squeeze(0)).show()


if __name__ == "__main__":
    utils.seed(42)
    utils.boost(enable=True)

    conf = Config.load(Path("offline.yml"))

    dist.launch(
        main,
        conf.env.n_gpu,
        conf.env.n_machine,
        conf.env.machine_rank,
        conf.env.dist_url,
        args=(conf, )
    )