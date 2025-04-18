from typing import List

import torch
from pttp import TensorProfiler
from helpers import requires_cuda


def get_n_bytes(*tensors: List[torch.Tensor]):
    return sum(tensor.nbytes for tensor in tensors)


def test_constructor():
    with TensorProfiler() as prof:
        a = torch.Tensor([0 for _ in range(16)])

    assert prof.total_memory == get_n_bytes(a)

def test_constructor_functions():
    with TensorProfiler() as prof:
        a = torch.empty(16)
        b = torch.zeros(16)
        c = torch.ones(16)
        d = torch.full((16, ), 0)

    assert prof.total_memory == get_n_bytes(a, b, c, d)

def test_operations():
    with TensorProfiler() as prof:
        a = torch.Tensor([1 for _ in range(16)])
        b = a + a

    assert prof.total_memory == get_n_bytes(a, b)

def test_views():
    with TensorProfiler() as prof:
        a = torch.empty(16)
        b = a[:8]
        c = a[8:]
        d = a[4:12]
        del a

    assert prof.total_memory == get_n_bytes(b, c)

@requires_cuda
def test_device_movement():
    with TensorProfiler() as prof:
        a = torch.empty(16)
        b = a.to("cuda:0")

    assert prof.total_memory == get_n_bytes(a, b)
