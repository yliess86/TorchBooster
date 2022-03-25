"""config.py

Configuration as yaml files.
The module provides a base class that can be extended
to generate pytorch or other objects from yaml files.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from torch import Tensor
from torch.nn import (Module, Parameter)
from torch.nn.parallel import DistributedDataParallel
from torch.optim import (AdamW, Optimizer, SGD)
from torch.utils.data import (DataLoader, Dataset)
from torchbooster.scheduler import (BaseScheduler, CycleScheduler)
from typing import (Any, Iterator)

import builtins
import torchbooster.distributed as dist
import yaml


def do_include(line: str) -> bool:
    """Do Include

    Do the line contains an include directive?
    Check if the line is of the form `#include *.yml`.

    Parameters
    ----------
    line: str
        line to process

    Returns
    -------
    condition: bool
        do the line contains an include directive
    """
    return line.startswith("#include ") and line.endswith(".yml")


def read_lines(path: Path) -> list(str):
    """Read Lines

    Recursively read lines of a yaml configuration file
    and resolve include directives.
    
    Parameters
    ----------
    path: Path
        path to the yaml file

    Returns
    -------
    lines: list(str)
        lines of the yaml file
    """
    root = path.parent
    with open(str(path.resolve()), "r") as fp:
        lines = fp.readlines()
        for include in reversed([l.strip() for l in lines if do_include(l.strip())]):
            lines = read_lines(Path(root, include[9:])) + lines
    return lines


def resolve_types(conf: BaseConfig, data: dict(str, Any)) -> dict(str, Any):
    """Resolve Types
    
    Recursively resolve types for a config given yaml loaded data.

    Parameters
    ----------
    conf: BaseConfig
        base config to look for dataclass fields
    data: dict(str, Any)
        data loaded from yaml

    Returns
    -------
    fields: dic(str, any)
        fields with resolved types to instanciate the config
    """
    fields = {}

    for field_name, field in conf.__dataclass_fields__.items():
        if not field_name in data:
            continue
        
        field_type = field.type
        field_data = data[field_name]

        if "list" in field_type or "tuple" in field_type:
            field_type, rest = field_type.split("(")
            field_subptypes = rest[:-1].split(",")
            field_data = field_data.split(",")

            if len(field_subptypes) > 1:
                assert len(field_subptypes) == len(field_data)

            field_subptypes = map(lambda t: t.strip(), cycle(field_subptypes))
            field_subptypes = map(lambda t: getattr(builtins, t), field_subptypes)
            subfields = zip(field_subptypes, field_data)
            field_data = (subptype(datum.strip()) for subptype, datum in subfields)
        
        try: field_type = builtins.__dict__[field_type]
        except KeyError:
            try: field_type = globals()[field_type]
            except KeyError:
                subclasses = BaseConfig.__subclasses__()
                idx = [cls.__name__ for cls in subclasses].index(field_type)
                field_type = subclasses[idx]
        
        if issubclass(field_type, BaseConfig):
            field_data = resolve_types(field_type, field_data)
            fields[field_name] = field_type(**field_data)
        else:
            fields[field_name] = field_type(field_data)

    return fields


def to_env(value: Any, cuda: bool, distributed: bool) -> Any:
    """To Env
    
    Put variable to environement (cuda if cuda and distributed if dsitributed).

    Parameters
    ----------
    value: Any
        if Tensor or Module will apply environement transforms else none
    cuda, distributed: bool
        put or not variable to cuda device
        wrap or not module to DistributedDataParallel

    Return
    ------
    value: Any
        if value is Tensor a Tensor
        if value is Module, a Module or DistributedDataParallel
        else input value
    """
    if isinstance(value, Tensor):
        value: Tensor = value.to("cuda" if cuda else "cpu")
    if isinstance(value, Module):
        value: Module = value.to("cuda" if cuda else "cpu")
        if distributed: value = DistributedDataParallel(value)
    return value


@dataclass
class BaseConfig:
    """Base Config
    
    Base configuration class providing basic interfaces
    for loading from yaml file and making the config a
    pytorch or other object instance.
    """

    def make(self, *args, **kwargs) -> Any:
        """Make"""
        raise NotImplementedError("Method 'make' is not implemented")

    @classmethod
    def load(cls, path: Path) -> BaseConfig:
        """Load
        
        Load configuration from a yaml file.

        Parameters
        ----------
        path: Path
            path to yaml file

        Returns
        -------
        conf: BaseConfig
            configuration intantiated from yaml file 
        """
        stream = "\n".join(read_lines(path))
        data = yaml.full_load(stream)
        fields = resolve_types(cls, data)
        return cls(**fields)


@dataclass
class EnvironementConfig(BaseConfig):
    """Environement Config

    Configuration of the compute environement.
    """
    distributed: bool = False
    fp16: bool = False
    n_gpu: int = 0
    n_machine: int = 1
    machine_rank: int = 0
    dist_url: str = "auto"

    def make(self, *args: list(Any)) -> Any:
        """Make
        
        Parameters
        ----------
        args: list(Any)
            any object to apply environement transformations
            (send to cuda device, distributed wrapper, ...)

        Returns
        -------
        args: list(Any)
            list transformed objects
            transformed object if only one is given
        """
        to = lambda x: to_env(x, self.n_gpu > 0, self.distributed)
        if len(args) == 1: return to(args[0])
        return [to(arg) for arg in args]


@dataclass
class LoaderConfig(BaseConfig):
    """Loader Config
    
    Configuration of the pytorch DataLoader.
    """
    batch_size: int
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False

    def make(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        distributed: bool = False,
    ) -> DataLoader:
        """Make

        Parameters
        ----------
        dataset: Dataset
            dataset to iterate on
        shuffle, distributed: bool (default: False, False)
            shuffle or not
            distributed or not

        Returns
        -------
        loader: DataLoader
            dataset batch loader
        """
        sampler = dist.data_sampler(dataset, shuffle, distributed)

        return DataLoader(
            dataset,
            self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )


@dataclass
class OptimizerConfig(BaseConfig):
    """Optimizer Config

    Configuration of optimizers.
    Currently on SGD and AdamW are supported.
    """
    name: str
    
    # Common Hyperparams
    lr: float
    weight_decay: float = 1e-2

    # SGD Hyperparams
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False

    # AdamW Hyperparams
    betas: tuple(float, float) = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False

    def make(self, parameters: Iterator[Parameter]) -> Optimizer:
        """Make

        Parameters
        ----------
        parameters: Iterator[Parameters]
            parameters to optimize

        Returns
        -------
        optim: Optimizer
            SGD or AdamW optimizer
        """
        if self.name == "sgd":
            return SGD(
                parameters,
                self.lr,
                self.momentum,
                self.dampening,
                self.weight_decay,
                self.nesterov,
            )

        if self.name == "adamw":
            return AdamW(
                parameters,
                self.lr,
                self.betas,
                self.eps,
                self.weight_decay,
                self.amsgrad,
            )

        raise NameError(f"Optimizer {self.name} is not supported")


@dataclass
class SchedulerConfig(BaseConfig):
    name: str

    # Cycle Scheduler HyperParams
    n_iter: int
    initial_multiplier: float = 4e-2
    final_multiplier: float = 1e-5
    warmup: int = 0
    plateau: int = 0
    decay: tuple(str, str) = ("cos", "cos")

    def make(self, optimizer: Optimizer) -> BaseScheduler:
        if self.name == "cycle":
            return CycleScheduler(
                optimizer,
                optimizer.param_groups[0]["lr"],
                self.n_iter,
                self.initial_multiplier,
                self.final_multiplier,
                self.warmup,
                self.plateau,
                self.decay,
            )
        
        raise NameError(f"Scheduler {self.name} is not supported.")


__all__ = [
    BaseConfig,
    EnvironementConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
]