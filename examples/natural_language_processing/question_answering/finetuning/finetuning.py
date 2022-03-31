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

@dataclass
class SquadConfig(BaseConfig):
    max_seq_length: int
    doc_stride: int
    max_query_length: int
    threads: int

@dataclass
class Config(BaseConfig):
    model: str
    epochs: int

    env: EnvironementConfig
    dataset: DatasetConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig

    #if dataset is squad
    squad_config: SquadConfig = None

def fit(conf: Config, model: Module, loader: DataLoader, optim: Optimizer, scheduler: BaseScheduler, train: bool = True):
    for X, Y in loader:
        print(X)
        print(Y)
        # print(model(tokenizer(X)))
        print(model(**X))
        exit()
        


def main(conf: Config):
    model = conf.env.make(torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', conf.model))
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', conf.model) 

    if "squad" in conf.dataset.name.lower():
        def collate_fn(data):
            X, Y = [], []
            
            for sample in data:
                X.append(conf.env.make(utils.to_tensor(tokenizer(sample[0], sample[1], padding='max_length', max_length=conf.squad_config.max_seq_length))))
                Y.append([sample[3][0], sample[3][0]+len(sample[2][0].split(' '))])

            return utils.stack_dictionaries(X), conf.env.make(utils.to_tensor(Y))
    
        train_set = conf.dataset.make(Split.TRAIN)
        train_loader = conf.loader.make(train_set, shuffle=True, collate_fn=collate_fn, distributed=conf.env.distributed)
    
    optim = conf.optim.make(model.parameters())
    scheduler = conf.scheduler.make(optim)
    fit(conf, model, train_loader, optim, scheduler)
   

if __name__ == "__main__":
    utils.seed(42)
    utils.boost(enable=True)

    conf = Config.load(Path("finetuning.yml"))

    dist.launch(
        main,
        conf.env.n_gpu,
        conf.env.n_machine,
        conf.env.machine_rank,
        conf.env.dist_url,
        args=(conf, )
    )