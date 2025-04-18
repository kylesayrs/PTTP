from typing import List

import pytest
import torch

__all__ = ["requires_cuda", "get_n_bytes"]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")


def get_n_bytes(*tensors: List[torch.Tensor]):
    return sum(tensor.nbytes for tensor in tensors)
