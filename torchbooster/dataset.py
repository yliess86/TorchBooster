"""dataset.py

Dataset Utilities.
The Module provides a base utility class for
implementing lmdb based datasets.
"""
from enum import Enum
from pathlib import Path
from torch.nn import Module
from torch.utils.data import Dataset
from torchbooster.lmdb import LMDBReader
from typing import Any


class Split(Enum):
    """Split
    
    Dataset split enum (train, validation, test).
    """
    TRAIN = "train"
    VALID = "validation"
    TEST  = "test"


class BaseDataset(Dataset):
    """Base Dataset
    
    Parameters
    ----------
    path: Path
        path to the lmdb dataset folder
    transform: Module
        transforms to be applied to the dataset when
        retrieving a sample
    map_size, max_readers: int (default: 1024 ** 4, 126)
        maximum allowed database size
        maximum amount of simultaneous read transactions

    Attributes
    ----------
    lmdb_reader: LMDBReader
        LMDB reader to provide context to the 'in-memory' database
    """
    
    def __init__(
        self,
        path: Path,
        transform: Module = None,
        map_size: int = 1024 * 4,
        max_readers: int = 126,
    ) -> None:
        super().__init__()
        self.path = path
        self.transform = transform
        self.map_size = map_size
        self.max_readers = max_readers
        
        self.lmdb_reader = LMDBReader(
            self.path,
            map_size=self.map_size,
            max_readers=self.max_readers,
        )

    def __len__(self) -> int:
        return len(self.lmdb_reader)

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError("Method '__getitem__' is not implemented.")

    @classmethod
    def prepare(cls, *args, **kwargs) -> None:
        """Prepare"""
        raise NotImplementedError("Method 'prepare' is not implemented.")


__all__ = [
    BaseDataset,
    Split,
]