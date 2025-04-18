import gc
from typing import List

import torch
from helpers import requires_cuda

from pttp import TensorProfiler


def get_n_bytes(*tensors: List[torch.Tensor]):
    return sum(tensor.nbytes for tensor in tensors)


def test_constructor():
    with TensorProfiler() as prof:
        a = torch.Tensor([0 for _ in range(16)])

    assert prof.memory["total"] == get_n_bytes(a)


def test_constructor_functions():
    with TensorProfiler() as prof:
        a = torch.empty(16)
        b = torch.zeros(16)
        c = torch.ones(16)
        d = torch.full((16,), 0)

    assert prof.memory["total"] == get_n_bytes(a, b, c, d)


def test_operations():
    with TensorProfiler() as prof:
        a = torch.Tensor([1 for _ in range(16)])
        b = a + a

    assert prof.memory["total"] == get_n_bytes(a, b)


def test_views():
    with TensorProfiler() as prof:
        a = torch.empty(16)
        a_storage_bytes = get_n_bytes(a)

        b = a[:8]
        c = a[8:]
        d = a[4:12]

        del a
        gc.collect()
        assert prof.memory["total"] == a_storage_bytes

        del b, c
        gc.collect()
        assert prof.memory["total"] == a_storage_bytes


@requires_cuda
def test_device_movement():
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0")
    meta_device = torch.device("meta")

    with TensorProfiler() as prof:
        a = torch.empty(16, device=cpu_device)
        b = a.to(device=gpu_device)
        c = a.to(device=meta_device)

    assert prof.memory["total"] == get_n_bytes(a, b, c)
    assert prof.memory[cpu_device] == get_n_bytes(a)
    assert prof.memory[gpu_device] == get_n_bytes(b)
    assert prof.memory[meta_device] == get_n_bytes(c)


@requires_cuda
def test_dtype_movement():
    with TensorProfiler() as prof:
        a = torch.empty(16, dtype=torch.float32)
        b = a.to(dtype=torch.bfloat16)
        c = a.to(dtype=torch.float8_e4m3fn)

    assert prof.memory["total"] == get_n_bytes(a, b, c)
