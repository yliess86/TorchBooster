"""config.py

Configuration as yaml files.
The module provides a base class that can be extended
to generate pytorch or other objects from yaml files.
"""
from __future__ import annotations
from copy import deepcopy
from enum import Enum
import logging
try:
    from datasets import (DownloadMode, load_dataset)
    HUGGINGFACE_DATASETS_AVAILABLE = True
except ImportError:
    HUGGINGFACE_DATASETS_AVAILABLE = False
    logging.warning('Could not load transformers, hugging face datasets are not available')

try:
    import torchtext.datasets as ttd
    TORCHTEXT_DATASETS_AVAILABE = True
except ImportError:
    TORCHTEXT_DATASETS_AVAILABE = False
    logging.warning('Could not load torchtext, torchtext datasets are not available')

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from torch import Tensor
from torch.nn import (Module, Parameter)
from torch.nn.parallel import DistributedDataParallel
from torch.optim import (AdamW, Optimizer, SGD)
from torch.utils.data import (DataLoader, Dataset, IterableDataset)
from torchbooster.dataset import Split
from torchbooster.scheduler import (BaseScheduler, CycleScheduler)
from typing import (Any, Generator, Iterable, Iterator, Tuple, Type, TypeVar, Union)

import builtins
import inspect
import torch
import os
import torchbooster.distributed as dist
import torchvision
import traceback
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
    return line.startswith("#include ") and (line.endswith(".yml") or line.endswith(".yaml"))


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


def  resolve_types(conf: BaseConfig, data: dict(str, Any)) -> dict(str, Any):
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
        
        field_type = field.type if isinstance(field.type, str) else field.type.__name__
        field_data = data[field_name]

        if "list" in field_type or "tuple" in field_type:
            if len(field_type.split('(')) == 1:
                raise RuntimeError("Indicate the type contained by the list/tuple: e.g list(int, int)")
            field_type, rest = field_type.split("(")
            field_subptypes = rest[:-1].split(",")
            if type(field_data) == str:
                field_data = field_data.split(",")

            if len(field_subptypes) > 1:
                assert len(field_subptypes) == len(field_data)

            field_subptypes = map(lambda t: t.strip(), cycle(field_subptypes))
            field_subptypes = map(lambda t: getattr(builtins, t), field_subptypes)
            subfields = zip(field_subptypes, field_data)
            field_data = (subptype(datum.strip() if hasattr(datum, "strip") else datum) for subptype, datum in subfields)
        
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

    dataclass_fields = conf.__dataclass_fields__.keys()
    for elem in data.keys():
        if not elem in dataclass_fields:
            logging.log(level=logging.WARNING, msg=f'Extra config element {elem} for config class {conf.__name__}. This could be a configuration problem.')

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
    elif isinstance(value, Module):
        value: Module = value.to("cuda" if cuda else "cpu")
        if distributed: value = DistributedDataParallel(value)
    elif hasattr(value, "__dict__") or isinstance(value, dict):
        for k, v in value.items():
            value[k] = v.to("cuda" if cuda else "cpu")
    return value

T = TypeVar('T')

class HyperParameterConfig():

    class HyperParameterIndex():
        def __init__(self, idx) -> None:
            self.idx = idx

        def __repr__(self) -> str:
            return f'HyperParameterIndex({self.idx})'

    def __init__(self, cls: Type, content: str) -> None:
        logging.debug("Reading hyperparam config")
        #TODO docme        
        self.content = content
        self.hp_config: dict = None
        self.cls = cls
        self.parse()

    def _get_param_iterator(self, content: str, iterators: list):
        from numpy import arange
        try:
            it = eval(content)
            if hasattr(it, "__iter__"):
                it = list(it)
                iterators.append(it)
                logging.info(f'Parsed hp str: {content}')
                return HyperParameterConfig.HyperParameterIndex(len(iterators) - 1)
            return content
        except:
            return content

    
    def _find_hparams(self, d: dict, iterators: list):
        for k in d:
            if isinstance(d[k], dict):
                self._find_hparams(d[k], iterators)
            if isinstance(d[k], str):
                d[k] = self._get_param_iterator(d[k], iterators)

    def _increase_idx(self, idx: list):
        i = 0
        while idx[i]+1 >= len(self.iterators[i]):
            idx[i] = 0
            i += 1
            if i >= len(idx):
                return False
        idx[i] += 1
        return True
            
    def _gen_idx(self):
        idx = [0 for i in range(len(self.iterators))]
        yield idx
        while self._increase_idx(idx):
            yield idx

    def gen_cfg(self) -> Generator[BaseConfig]:
        def _recusrse_replace(d, idx):
            for k in d:
                if isinstance(d[k], dict):
                    _recusrse_replace(d[k], idx)
                if isinstance(d[k], HyperParameterConfig.HyperParameterIndex):
                    d[k] = self.iterators[d[k].idx][idx[d[k].idx]]

        for idx in self._gen_idx():
            cfg = deepcopy(self.hp_config)
            _recusrse_replace(cfg, idx)
            fields = resolve_types(self.cls, cfg)
            yield self.cls(**fields)

    def parse(self):
        self.hp_config = yaml.full_load(self.content)
        self.iterators = []
        self._find_hparams(self.hp_config, self.iterators)
        logging.info(f'Parsed hparam config with {len(self.iterators)} parameters')


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
    def load(cls: T, path: Path, hyperparams: bool = False) -> Union[T, Generator[T]]:
        """Load
        
        Load configuration from a yaml file.

        Parameters
        ----------
        path: Path
            path to yaml file
        hyperparams: Bool
            True if the file contains hyperparameters, 
            it will be interpreted by the HyperParameterParser
            and return an iterator of Config

        Returns
        -------
        conf: BaseConfig:
            configuration intantiated from yaml file
        Iterator[BaseConfig]:
            an iterator of BaseConfig if the hyperparam value is True 
        """
        stream = "\n".join(read_lines(path))
        if hyperparams:
            return HyperParameterConfig(cls, stream).gen_cfg()
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
        collate_fn: function = None
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
        sampler = None if isinstance(dataset, IterableDataset) else dist.data_sampler(dataset, shuffle, distributed)
        return DataLoader(
            dataset,
            self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=collate_fn
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

DEFAULT_DATASET_ACCEPTANCE_FN = lambda _: True

class IterableSizeableDataset(IterableDataset):
    def __init__(self, iterable: Iterable, size: int, acceptance_fn: Tuple[Any, bool] = DEFAULT_DATASET_ACCEPTANCE_FN,  **kwargs) -> None:
        super(IterableSizeableDataset).__init__()
        self.size = size
        self.iterable_dataset = iterable
        self.acceptance_fn = acceptance_fn

    def __iter__(self):
        for elem in self.iterable_dataset:
            if self.acceptance_fn(elem):
                yield elem
    
    def __len__(self):
        return self.size


class DistributedIterableSizeableDataset(IterableDataset):
    """
    Example implementation of an IterableDataset that handles both multiprocessing and distributed training.
    It works by skipping each num_workers * world_size element, starting at index rank + num_workers * worker_id,
    where num_workers is the number of workers as specified in the DataLoader, world_size is the number of processes
    in the distribution context, worker_id the id of worker in the DataLoader and rank the rank in the distribution
    context.
    Both rank and world_size must be retrieved outside of the dataset's context and passes in the constructor as they
    won't be available reliably when the dataset is invoked from another process for num_workers > 1.
    Note, that in this example, the data, respectively the iterator is passed in the constructor. In practise, this
    should be avoided and the iteration and skipping logic should be implemented in __iter__ according to the
    concrete needs. The reason for this is, that depending on the actual case, the given iterator may already do some
    heavy computation under the hood.
    In the case of an IterableDataset containing images for example, the images would be loaded in every iteration,
    although most of them would be skipped. The solution in such a case would be, to iterate through all image paths,
    but only load the actual image, if the iteration is not skipped.
    """
    def __init__(self, it, rank, world_size, iter_len=0, acceptance_fn: Tuple[Any, bool] = DEFAULT_DATASET_ACCEPTANCE_FN):
        self.it = it
        self.rank = rank
        self.world_size = world_size
        self.iter_len = iter_len
        self.acceptance_fn = acceptance_fn

    def __len__(self):
        return self.iter_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        mod = self.world_size
        shift = self.rank

        if worker_info:
            mod *= worker_info.num_workers
            shift = self.rank * worker_info.num_workers + worker_info.id

        for i, elem in enumerate(self.it):
            if (i + shift) % mod == 0 and self.acceptance_fn(elem):
                yield elem


@dataclass
class DatasetConfig(BaseConfig):
    """DatasetConfig
    
    Used to load common dataset by name.
    Custom datasets can use their own loading mechanism.
    """
    name: str
    root: str = './dataset'

    # Some datasets have sub tasks (eg. GLUE)
    task: str = None

    def make(self, split: Split, download: bool = True, distributed = False, acceptance_fn: Tuple[Any, bool] = DEFAULT_DATASET_ACCEPTANCE_FN, **kwargs) -> Dataset:
        """Make

        Look for the dataset name, downloads if required and returns
        the queried dataset split.

        Dataset loading strategy:
            - Look in torchvision.datasets
            - Look in huggingface datasets repository
            - #? Custom Datset loader

        Parameters
        ----------
        split: Split
            The dataset split to return
        download: bool = True
            Wether to download the dataset or not
        kwargs: dict(str, Any)
            Remaning arguments are passed to the downstream dataset loader


        Returns
        -------
        Dataset
            A torch.utils.data.Dataest object or another type of datset depending on the downstream dataset loader
        """
        root = os.path.join(self.root, split.value)
        logging.info(f'Dataset path is {root}')

        locations = ["torchvision"]
        try: # torchvision strategy
            dataset = getattr(torchvision.datasets, self.name.upper())
            if "split" in inspect.signature(dataset.__init__).parameters:
                return dataset(root=root, split=split.value, download=download, **kwargs)
            return dataset(root=root, train=split is Split.TRAIN, download=download, **kwargs)
        except AttributeError: pass

        if TORCHTEXT_DATASETS_AVAILABE:
            try: # torchtext strategy
                locations += ['torchtext datasets']
                dataset_class = getattr(ttd, self.name) #TODO: load case independent use torchtext.datasets.DATASETS
                dataset = dataset_class(self.root, split = split.value, **kwargs)
                size = getattr(getattr(ttd, self.name.lower()), "NUM_LINES")["valid" if split == Split.VALID else split.value]
                if distributed:
                    return DistributedIterableSizeableDataset(iter(dataset), dist.get_local_rank(), dist.get_world_size(), iter_len=size, acceptance_fn=acceptance_fn)
                return IterableSizeableDataset(iter(dataset), size, acceptance_fn=acceptance_fn)
            except AttributeError: pass

        if HUGGINGFACE_DATASETS_AVAILABLE:
            locations += ["huggingface datasets"]
            try: # huggingface strategy
                download = DownloadMode.REUSE_DATASET_IF_EXISTS
                split_str = split.value

                # if self.task is not None:
                dataset = load_dataset(self.name, name = self.task, download_mode=download, cache_dir=self.root, **kwargs)
                # else:
                #     dataset = load_dataset(self.name, download_mode=download, cache_dir=self.root, **kwargs)

                if Split.TEST not in dataset: # handle special case of no TEST split
                    logging.warning(f'Dataset {self.name} does not have a TEST split, splitting dataset into 80/20 for TRAIN and TEST subsets')
                    if split == Split.TRAIN:
                        split_str = 'train[:80%]'
                    elif split == Split.TEST:
                        split_str = 'train[-20%:]'
                else:
                    return dataset[split_str]
                # if self.task is not None:
                return load_dataset(self.name, name = self.task, download_mode=download, cache_dir=self.root, split = split_str, **kwargs)
                # else:
                #     dataset = load_dataset(self.name, download_mode=download, cache_dir=self.root, split = split_str, **kwargs)
                
                        
            except FileNotFoundError: pass

        logging.fatal(f"Could not find dataset {self.name}{f' with task {self.task}' if self.task else ''} in the default locations, looked in {', '.join(locations)}.")
        exit(1)


__all__ = [
    BaseConfig,
    DatasetConfig,
    EnvironementConfig,
    LoaderConfig,
    OptimizerConfig,
    SchedulerConfig,
    IterableSizeableDataset,
    DistributedIterableSizeableDataset
]