from pathlib import Path
from torchbooster.config import BaseConfig, EnvironementConfig, LoaderConfig, OptimizerConfig, SchedulerConfig, DatasetConfig
from dataclasses import dataclass
import pytest
import logging

LOGGER = logging.getLogger(__name__)


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


@dataclass
class Config2(BaseConfig):
    test: str

@dataclass
class Config1(BaseConfig):
    cfg2: Config2
    test: int


def test_config_nested():
    cfg = Config1.load(Path("./test/configs/nested.yml"))
    assert cfg.cfg2.test == "test"
    assert cfg.test == 42

def test_circular_import():
    # Don't check for circular imports, if it crashes, it crashes
    with pytest.raises(RecursionError):
        VoidConfig.load(Path("./test/configs/circular/base.yml"))

def test_config_include():
    cfg = FullDefaultConfig.load(Path('./test/configs/includes/base.yaml'))
    assert cfg.dataset.name == "cifar10"

def test_config_extra_parameters(caplog):
    VoidConfig.load(Path('./test/configs/full.yml'))
    assert 'configuration problem' in caplog.text

def test_config_full_parameters():
    cfg = FullDefaultConfig.load(Path('./test/configs/full.yml'))
    assert cfg.dataset.name == "cifar10"
    assert cfg.epochs == 10
    assert cfg.seed == 42
    assert cfg.env.n_gpu == 1
    assert cfg.loader.batch_size == 1024
    assert cfg.optim.name == "adamw"

