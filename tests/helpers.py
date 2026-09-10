import functools
import inspect
from typing import List

import pytest
import torch

__all__ = ["requires_cuda", "get_n_bytes", "device_parametrize"]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")


def get_n_bytes(*tensors: List[torch.Tensor]):
    return sum(tensor.nbytes for tensor in tensors)


def device_parametrize(test):
    signature = inspect.signature(test)

    @functools.wraps(test)
    def wrapper(*args, _device, **kwargs):
        if _device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")

        with torch.device(_device):
            return test(*args, **kwargs)

    # Tell pytest that this function accepts all of the original test's
    # arguments, plus our hidden `_device` parameter.
    wrapper.__signature__ = signature.replace(
        parameters=[
            *signature.parameters.values(),
            inspect.Parameter(
                "_device",
                inspect.Parameter.KEYWORD_ONLY,
            ),
        ]
    )

    return pytest.mark.parametrize(
        "_device",
        ["meta", "cpu", "cuda"],
    )(wrapper)
