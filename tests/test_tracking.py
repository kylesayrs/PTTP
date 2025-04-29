import torch
from helpers import get_n_bytes

from pttp import TensorProfiler


def test_use_after_exit():
    with TensorProfiler() as prof:
        a = torch.empty(16)
        b = torch.empty(16)

        del a

    b_bytes = get_n_bytes(b)
    del b
    c = torch.empty(16)

    assert prof.memory["total"] == b_bytes


def test_nested_profilers():
    with TensorProfiler() as outer_prof:
        a = torch.randn(8)
        b = torch.randn(16)

        with TensorProfiler() as inner_prof:
            c = torch.randn(32)
            d = torch.randn(64)
            d_size = get_n_bytes(d)
            assert inner_prof.memory["total"] == get_n_bytes(c, d)

        assert outer_prof.memory["total"] == get_n_bytes(a, b, c, d)

        del a, d
        assert outer_prof.memory["total"] == get_n_bytes(b, c)
        assert inner_prof.memory["total"] == get_n_bytes(c) + d_size
