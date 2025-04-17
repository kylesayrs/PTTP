from typing import List, Set
from functools import partial
import matplotlib.pyplot as plt
from torch.utils._python_dispatch import TorchDispatchMode

import gc
import torch
import weakref

from .helpers import get_alloc_size

__all__ = ["TensorProfiler"]


class TensorProfiler(TorchDispatchMode):
    total_tensor_memory: int
    memory_timeline: List[int]
    
    _tracked_tensors: Set[int]

    def __init__(self):
        self.total_tensor_memory = 0
        self.memory_timeline = []

        self._tracked_tensors = set()

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        ret = func(*args, **(kwargs or {}))
        if isinstance(ret, torch.Tensor):
            self.track_tensor(ret)

        return ret
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        gc.collect()

    def track_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_hash = hash(tensor)
        tensor_memory = get_alloc_size(tensor)

        # warn when init is called twice
        if tensor_hash in self._tracked_tensors:
            print("double init")
            return

        # add memory
        self.total_tensor_memory += tensor_memory
        self._add_to_timeline()
        self._tracked_tensors.add(tensor_hash)

        # register hook to subtract memory
        weakref.finalize(tensor, partial(self._on_tensor_deallocated, tensor_memory, tensor_hash))

    def _on_tensor_deallocated(self, tensor_memory, tensor_hash):
        self.total_tensor_memory -= tensor_memory
        self._add_to_timeline()
        self._tracked_tensors.remove(tensor_hash)
    
    @property
    def total_tensor_memory_mib(self):
        return self.total_tensor_memory / (1024 * 1024)
    
    def _add_to_timeline(self):
        self.memory_timeline.append(self.total_tensor_memory)

    def plot_values_over_time(self, dpi=300):
        values = self.memory_timeline
        """
        Plots a list of float values over time using matplotlib.

        Parameters:
            values (list of float): The values to plot.
        """
        if not values:
            print("The list of values is empty.")
            return

        plt.figure(figsize=(10, 4))
        plt.plot(range(len(values)), values, marker='o', linestyle='-')
        plt.title("Values Over Time")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("file.png", dpi=dpi)
