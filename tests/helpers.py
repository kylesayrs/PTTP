import pytest

import torch

__all__ = ["requires_cuda"]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
