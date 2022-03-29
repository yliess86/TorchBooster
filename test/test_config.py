from pathlib import Path
from torchbooster.config import BaseConfig, EnvironementConfig, LoaderConfig, OptimizerConfig, SchedulerConfig, DatasetConfig
from dataclasses import dataclass
import pytest


@dataclass
class VoidConfig(BaseConfig):
    pass


@dataclass
class FullDefaultConfig(BaseConfig):
    epochs: int
    seed: int

    env: EnvironementConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig
    dataset: DatasetConfig


def test_config_include():
    cfg = FullDefaultConfig.load(Path('./test/configs/includes/base.yaml'))
    assert cfg.dataset.name == "cifar10"

def test_config_extra_parameters():
    pass

def test_config_default_parameters():
    cfg = FullDefaultConfig.load(Path('./test/configs/full.yml'))
    assert cfg.dataset.name == "cifar10"

