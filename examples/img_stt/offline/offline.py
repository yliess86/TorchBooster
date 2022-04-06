from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torchbooster.config import (
    BaseConfig,
    EnvironementConfig,
    OptimizerConfig,
)
from torchvision.models.vgg import vgg19
from tqdm import tqdm

import numpy as np
import requests
import torchbooster.utils as utils
import torchvision.transforms as T


def gram_matrix(features: Tensor) -> Tensor:
    B, C, H, W = features.size()
    features = features.view(-1, H * W)
    return features @ features.T / (B * C * H * W)


def total_variation(x: Tensor) -> Tensor:
    a = (x[:, :, :  , :-1] - x[:, :,  :, 1:]).abs().sum()
    b = (x[:, :, :-1, :  ] - x[:, :, 1:,  :]).abs().sum()
    return a + b


@dataclass
class Config(BaseConfig):
    n_iter: int
    seed: int
    size: int

    style: str
    style_layers: list(int)
    style_weights: list(float)
    style_weight: float

    content: str
    content_layers: list(int)
    content_weights: list(float)
    content_weight: float

    tv_weight: float

    env: EnvironementConfig
    optim: OptimizerConfig
   

def transfer(
    conf: Config,
    style: Tensor,
    content: Tensor,
    mixture: Tensor,
    vgg: Module,
    optim: Optimizer,
) -> Tensor:
    feats = {}
    hook = lambda module, input, output, layer: feats.update({str(layer): output})
    for l in set(conf.style_layers + conf.content_layers):
        vgg[l].register_forward_hook(partial(hook, layer=l))
        
    vgg(style)
    s_grams = [gram_matrix(feats[str(l)]).detach() for l in conf.style_layers]
    w_grams = conf.style_weights

    vgg(content)
    c_feats = [feats[str(l)].detach() for l in conf.content_layers]
    w_feats = conf.content_weights

    with tqdm(range(conf.n_iter), desc="Transfer") as pbar:
        for _ in pbar:
            vgg(mixture)
            m_grams = [gram_matrix(feats[str(l)]) for l in conf.style_layers]
            m_feats = [feats[str(l)] for l in conf.content_layers]

            s_loss = sum([w_g * (m_g - s_g).pow(2).mean() for w_g, m_g, s_g in zip(w_grams, m_grams, s_grams)])
            c_loss = sum([w_f * (m_f - c_f).pow(2).mean() for w_f, m_f, c_f in zip(w_feats, m_feats, c_feats)])
            tv_loss = total_variation(mixture)

            loss = conf.style_weight * s_loss + conf.content_weight * c_loss + conf.tv_weight * tv_loss
            utils.step(loss, optim)

            pbar.set_postfix(
                loss=f"{loss.item():.2e}",
                style=f"{s_loss.item():.2e}",
                content=f"{c_loss.item():.2e}",
                tv=f"{tv_loss.item():.2e}",
            )

    return mixture


def main(conf: Config) -> None:
    vgg = vgg19(pretrained=True, progress=True).features
    vgg = utils.freeze(conf.env.make(vgg).eval())

    mean, std = np.array((0.485, 0.456, 0.406)), np.array((0.229, 0.224, 0.225))
    hdr = T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))

    transfrom = T.Compose([T.Resize(conf.size), T.CenterCrop(conf.size), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    utransfrom = T.Compose([T.Normalize(mean=-mean / std, std=1.0 / std), hdr, T.ToPILImage()])

    load_img = lambda url: Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    content = conf.env.make(transfrom(load_img(conf.content)).unsqueeze(0))
    style = conf.env.make(transfrom(load_img(conf.style)).unsqueeze(0))

    mixture = content.clone().requires_grad_(True)
    optim = conf.optim.make([mixture])

    mixture = transfer(conf, style, content, mixture, vgg, optim)
    utransfrom(mixture.squeeze(0)).show()


if __name__ == "__main__":
    conf = Config.load(Path("offline.yml"))

    utils.seed(conf.seed)
    utils.boost(enable=True)
    
    main(conf)