from __future__ import annotations

from dataclasses import dataclass
from torchbooster.config import (
    BaseConfig,
    EnvironementConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
)


@dataclass
class DatasetConfig(BaseConfig):
    name: str
    path: Path
    resolution: int


@dataclass
class Config(BaseConfig):
    env: EnvironementConfig
    dataset: DatasetConfig
    loader: LoaderConfig
    optim: OptimizerConfig
    scheduler: SchedulerConfig


if __name__ == "__main__":
    from pathlib import Path


    path = Path("examples/config/base.yml")
    conf = Config.load(path)
    print(conf)