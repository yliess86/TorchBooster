from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from torch.nn import (Flatten, GELU, Linear, Module, Sequential, Sigmoid, Unflatten)
from torch.optim import Optimizer
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torchbooster.dataset import Split
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
import torch.nn.functional as F
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
    pbar = tqdm(loader)
    for X, Y in pbar:
        if len(X) == 0 or len(X['input_ids'].shape) == 0 or len(Y['start'].shape) == 0: continue # empty batch because we split a lot so it can happen
        out = model(**X)
        loss_start = cross_entropy(out.start_logits, Y['start'])
        loss_end   = cross_entropy(out.end_logits, Y['end'])
        loss       = loss_start + loss_end

        utils.step(loss, optim, scheduler)
        pbar.set_postfix(loss=f'{loss.item():.2e}')
        


def main(conf: Config):
    model = conf.env.make(torch.hub.load('huggingface/transformers', 'modelForQuestionAnswering', conf.model))#, source="local"))
    tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', conf.model)#, source="local") 

    if "squad" in conf.dataset.name.lower():
        def index_to_token(emb, idx):
            for i, yem in enumerate(emb['offset_mapping']):
                if idx >= yem[0] and idx <= yem[1]:
                    return i
            return None

        def collate_fn(data):
            X, Y = [], {'start': [], 'end': []}

            for sample in data:
                if sample[3][0] < 0:
                    continue # Note: SquadV2 supports impossible questions with no answers marked by a -1 answer start
                             #       But we got no time for that so we just skip the sample

                tok = tokenizer(sample[0], sample[1], padding='max_length', return_offsets_mapping=True, max_length=conf.squad_config.max_seq_length)
                if len(tok['input_ids']) > conf.squad_config.max_seq_length:
                    continue # Note: Traditional Squad processing required splitting contexts and getting answers from where they are located in each splits
                             #       But we got no time for that so we just skip the sample

                dico = {'input_ids': tok['input_ids'], 'attention_mask': tok['attention_mask']}
                X.append(conf.env.make(utils.to_tensor(dico)))
                Y['start'].append(index_to_token(tok, sample[3][0]))
                Y['end'  ].append(index_to_token(tok, sample[3][0]+len(sample[2][0])))
            
            Y['start'] = utils.to_tensor(Y['start'])
            Y['end']   = utils.to_tensor(Y['end'])
            return utils.stack_dictionaries(X), conf.env.make(utils.to_tensor(Y))
    
        train_set = conf.dataset.make(Split.TRAIN)
        train_loader = conf.loader.make(train_set, shuffle=True, collate_fn=collate_fn, distributed=conf.env.distributed)
    
    optim = conf.optim.make(model.parameters())
    scheduler = conf.scheduler.make(optim)
    fit(conf, model, train_loader, optim, scheduler)
   

if __name__ == "__main__":
    utils.seed(42)
    utils.boost()

    conf = Config.load(Path("finetuning.yml"))

    dist.launch(
        main,
        conf.env.n_gpu,
        conf.env.n_machine,
        conf.env.machine_rank,
        conf.env.dist_url,
        args=(conf, )
    )