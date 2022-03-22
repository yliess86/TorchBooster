"""distributed.py

Distributed utilities.
The module provides basic utilities to help distirbuting
a training job on multiple devices using the PyTorch
DistributedDataParallel workflow.
"""
from __future__ import annotations

from torch.utils.data import (Dataset, DistributedSampler, RandomSampler, Sampler, SequentialSampler)
from torch import distributed as dist
from torch import multiprocessing as mp
from typing import (Any, Callable)

import os
import socket
import torch


LOCAL_PROCESS_GROUP = None


def get_rank() -> int:
    """Get Process Rank"""
    if not dist.is_available(): return 0
    if not dist.is_initialized(): return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Get Local Process Rank in Group"""
    global LOCAL_PROCESS_GROUP
    if not dist.is_available(): return 0
    if not dist.is_initialized(): return 0
    if not LOCAL_PROCESS_GROUP:
        raise ValueError("LOCAL_PROCESS_GROUP is None")
    return dist.get_rank(group=LOCAL_PROCESS_GROUP)


def is_primary() -> bool:
    """Is Primary Device"""
    return get_rank() == 0


def synchronize() -> None:
    """Synchronize Processes"""
    if not dist.is_available(): return
    if not dist.is_initialized(): return
    if dist.get_world_size() == 1: return
    dist.barrier()


def get_world_size() -> int:
    """Get World Size (Processes Amount)"""
    if not dist.is_available(): return 1
    if not dist.is_initialized(): return 1
    return dist.get_world_size()


def data_sampler(dataset: Dataset, shuffle: bool, distributed: bool) -> Sampler:
    """Data Sampler

    Parameters
    ----------
    dataset: Dataset
        dataset to sample from
    shuffle:
        do the sampler need to shuffle the data or not
    distributed: bool
        is the sampler distributed

    Returns
    -------
    sampler: Sampler
        dataset sampler if distributed (DistributedSampler)
        if shuffle (RandomSampler) else (SequentialSampler) 
    """
    if distributed: return DistributedSampler(dataset, shuffle=True)
    if shuffle: return RandomSampler(dataset)
    return SequentialSampler(dataset)


def find_free_port() -> int:
    """Find a Free Port"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def launch(
    fn: Callable,
    n_gpu_per_machine: int,
    n_machine: int = 1,
    machine_rank: int = 0,
    dist_url: str = None,
    args: tuple(Any) = (),
) -> None:
    """Launch
    
    Launch a distributed workload

    Parameters
    ----------
    fn: Callable
        function to be distributed
    n_gpu_per_machine, n_machine: int
        number of gpus per machine and number of machines
    machine_rank: int
        machine rank
    dist_url: str
        url used for tcp communication within the machines
    args: tuple(Any)
        arguments to pass to fn
    """
    world_size = n_machine * n_gpu_per_machine

    if world_size == 1:
        fn(*args)
        return

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"

    if dist_url == "auto":
        if n_machine > 1:
            raise ValueError("dist_url='auto' no supported in multi-machine jobs")
        
        port = find_free_port()
        dist_url = f"tcp://127.0.0.1:{port}"

    args = fn, world_size, n_gpu_per_machine, machine_rank, dist_url, args
    mp.spawn(job, nprocs=n_gpu_per_machine, args=args, daemon=False)


def job(
    local_rank: int,
    fn: Callable,
    world_size: int,
    n_gpu_per_machine: int,
    machine_rank: int = 0,
    dist_url: str = None,
    args: tuple(Any) = (),
) -> None:
    """Job"""
    global LOCAL_PROCESS_GROUP

    if not torch.cuda.is_available():
        raise OSError("CUDA is not available on this machine")

    global_rank = machine_rank * n_gpu_per_machine + local_rank

    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
        )
    
    except Exception:
        raise OSError("NCC groups failed to initialize")

    synchronize()

    if n_gpu_per_machine > torch.cuda.device_count():
        asked = n_gpu_per_machine
        available = torch.cuda.device_count()
        raise ValueError(f"Asked for {asked} gpus bu got {available} available")

    torch.cuda.set_device(local_rank)

    if LOCAL_PROCESS_GROUP is not None:
        raise ValueError("torch.distributed.LOCAL_PROCESS_GROUP is not None")

    n_machine = world_size // n_gpu_per_machine

    for i in range(n_machine):
        ranks = list(range(i * n_gpu_per_machine, (i + 1) * n_gpu_per_machine))
        process_group = dist.new_group(ranks)

        if i == machine_rank:
            LOCAL_PROCESS_GROUP = process_group

    fn(*args)