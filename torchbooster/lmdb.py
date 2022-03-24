"""lmdb.py

LMDB utilities.
The Module provides a basic LMDB Reader helper
to provide a simple interface to create datasets
from LMDB files for fast and efficient parallel loading.
"""
from typing import Iterator

import lmdb


class LMDBReader:
    """LMDB Reader
    
    The LMDB Database must include a length key with
    associated value describing the size of the dataset.

    Parameters
    ----------
    path: str
        path to the lmdb dataset
    map_size, max_readers: int (default: 1024 ** 4, 126)
        maximum allowed database size
        maximum amount of simultaneous read transactions

    Attributes
    ----------
    env: lmdb.Environment
        lmdb environment to handle database transactions and context
    length: int
        dataset size
    """

    def __init__(
        self,
        path: str,
        map_size: int = 1024 ** 4,
        max_readers: int = 126,
    ) -> None:
        self.path = path
        self.map_size = map_size
        self.max_readers = max_readers
        
        self.env = None
        self.length = None

    def open(self) -> None:
        """Open LMDB Context Handle"""
        self.env = lmdb.open(
            self.path,
            self.map_size,
            readonly=True,
            create=False,
            readahead=False,
            lock=False,
            max_readers=self.max_readers,
        )

        if self.env is None:
            raise IOError(f"Could not open lmdb dataset {self.path}")

        try: self.length = int(self.get(b"length").decode("utf-8"))
        except KeyError: self.length = 0

    def close(self) -> None:
        """Close LMDB Context Handle"""
        if self.env is not None:
            self.env.close()
            self.env = None

    def get(self, key: str) -> bytes:
        """Get Data Bytes from Key"""
        if self.env is None:
            self.open()

        with self.env.begin(write=False) as ctx:
            value = ctx.get(key)
        
        if value is None:
            raise KeyError(f"lmdb dataset does not contain key {key}")
        
        return value

    def __len__(self) -> int:
        if not self.length:
            self.open()
            self.close()

        return self.length

    def __iter__(self) -> Iterator[bytes]:
        for i in range(self.length):
            yield self[i]

    def __getitem__(self, idx: int) -> bytes:
        return self.get(str(idx).encode("utf-8"))

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "LMDBReader":
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.close()