from typing import List, NamedTuple
from torchbooster.utils import to_tensor

import torch


class LocalNamedTuple(NamedTuple):
    field1: List[int]
    field2: List[int]

def test_to_tensor_named_tuples():
    a = LocalNamedTuple([1, 2], [3, 4])
    t = to_tensor(a)

    assert type(t.field1) == torch.Tensor
    assert type(t.field2) == torch.Tensor
    assert t.field1[0] == 1
    assert t.field2[0] == 3
