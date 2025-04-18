import gc
import torch

from pttp import TensorProfiler
from helpers import get_n_bytes

def test_use_after_exit():
    with TensorProfiler() as prof:
        a = torch.empty(16)
        b = torch.empty(16)

        del a; gc.collect()

    b_bytes = get_n_bytes(b)
    del b; gc.collect()
    c = torch.empty(16)

    assert prof.memory["total"] == b_bytes
