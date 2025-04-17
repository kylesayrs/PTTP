import torch
from pttp import TensorProfiler, get_alloc_size, get_alloc_sizes


def test_constructor():
    with TensorProfiler() as prof:
        a = torch.Tensor([0 for _ in range(16)])

    assert prof.total_memory == get_alloc_size(a)

def test_constructor_functions():
    with TensorProfiler() as prof:
        a = torch.empty(16)
        b = torch.zeros(16)
        c = torch.ones(16)
        d = torch.full((16, ), 0)

    assert prof.total_memory == get_alloc_sizes(a, b, c, d)


def test_operations():
    with TensorProfiler() as prof:
        a = torch.Tensor([1 for _ in range(16)])
        b = a + a

    assert prof.total_memory == get_alloc_sizes(a, b)
