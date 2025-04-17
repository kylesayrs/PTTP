import torch

__all__ = ["get_alloc_size", "get_alloc_sizes"]


def get_alloc_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def get_alloc_sizes(*tensors):
    return sum(get_alloc_size(tensor) for tensor in tensors)
