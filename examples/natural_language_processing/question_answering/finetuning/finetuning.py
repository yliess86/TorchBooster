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
from transformers.data.processors.squad import SquadV2Processor, squad_convert_examples_to_features
from transformers import AutoModel, AutoTokenizer, SquadExample

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


def get_squad_examples_from_dataset(dataset, evaluate: bool):
    examples = []
    for ex in tqdm(dataset):
        if not evaluate:
            answer = ex["answers"]["text"][0] if len(ex['answers']['text']) > 0 else None
            answer_start = ex["answers"]["answer_start"][0] if len(ex["answers"]["text"]) > 0 else None
            answers = []
        else:
            answers = [
                {"answer_start": start, "text": text}
                for start, text in zip(ex["answers"]["answer_start"], ex["answers"]["text"])
            ]

            answer = None
            answer_start = None

        examples.append(SquadExample(
            qas_id=ex["id"],
            question_text=ex["question"],
            context_text=ex["context"],
            start_position_character=answer_start,
            title=ex["title"],
            answer_text=answer,
            answers=answers,
            is_impossible=answer is None
        ))

    return examples

def process_squad(conf: Config, dataset, split: Split, tokenizer) -> DataLoader:
    processor = SquadV2Processor()


    """ 
    SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )"""
    examples = get_squad_examples_from_dataset(dataset, evaluate=split != Split.TRAIN)
    print(examples[0])
    features = squad_convert_examples_to_features(examples, tokenizer, tqdm_enabled=dist.is_primary(), return_dataset='pt', is_training=split is Split.TRAIN, **conf.squad_config.__dict__)
    loader = conf.loader.make(features, shuffle=split is Split.TRAIN, distributed=conf.env.distributed)

    return loader

def main(conf: Config):

    lm =  AutoModel.from_pretrained(conf.model)
    tokenizer = AutoTokenizer.from_pretrained(conf.model)

    train_loader = None

    if conf.dataset.name == "squad_v2":
        train_set = conf.dataset.make(Split.TRAIN)
        train_loader = process_squad(conf, train_set, Split.TRAIN, tokenizer)

    for X in train_loader:
        print(X['answers'])
        exit(0)

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