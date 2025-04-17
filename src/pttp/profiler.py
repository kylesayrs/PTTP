from typing import List, Set, Dict, Tuple

import gc
import weakref
import warnings
from functools import partial
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch.utils._python_dispatch import TorchDispatchMode
import matplotlib.pyplot as plt

from .helpers import get_alloc_size
from .global_access import GlobalAccess

__all__ = ["TensorProfiler"]

aten = torch._ops.ops.aten


@dataclass
class MemoryProfile:
    current: int = 0
    timeline: List[int] = field(default_factory=list)


class TensorProfiler(TorchDispatchMode, GlobalAccess):
    _memory = Dict[torch.device, MemoryProfile]
    _tracked_tensors: Set[int]
    _events = List[Tuple[int, str]]

    def __init__(self):
        self._memory: Dict[torch.device, MemoryProfile] = defaultdict(MemoryProfile)
        self._tracked: Set[int] = set()
        _events: List[Tuple[int, str]] = list()

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        ret = func(*args, **(kwargs or {}))
        if isinstance(ret, torch.Tensor):
            tensor_hash = hash(ret)

            # TODO: do not warn on these functions
            # aten.set_.source_Storage
            # aten.copy_.default
            # aten.add_.Tensor
            if tensor_hash in self._tracked:
                warnings.warn(f"Attempted to track tensor twice from dispatch {func}")
            
            self.track_tensor(ret)

        return ret
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def track_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        _hash = hash(tensor)
        size = get_alloc_size(tensor)
        device = tensor.device

        # skip if already tracked
        if _hash in self._tracked:
            return

        # add tensor
        self._memory[device].current += size
        self._snapshot_timeline()
        self._tracked.add(_hash)

        # register hook to subtract memory
        finalizer = partial(self._on_tensor_deallocated, _hash, size, device)
        weakref.finalize(tensor, finalizer)

    def _on_tensor_deallocated(self, hash: int, size: int, device: torch.device):
        # subtract tensor
        self._memory[device].current -= size
        self._snapshot_timeline()
        self._tracked.remove(hash)

    def _snapshot_timeline(self):
        for mem in self._memory.values():
            mem.timeline.append(mem.current)

    
    ## Public functions


    @property
    def total_memory(self) -> int:
        return sum((mem.current for mem in self._memory.values()), 0)

    @property
    def total_memory_mib(self) -> float:
        return self.total_memory / (1024 * 1024)
    
    def get_device_memory(self, device: torch.device) -> int:
        return self._memory[device].current
    
    def get_device_memory_mib(self, device: torch.device) -> int:
        return self._memory[device].current / (1024 * 1024)
    

    ## Plotting

    def mark_event(self, name: str):
        self._snapshot_timeline()
        timeline_lens = set(len(profile.timeline) for profile in self._memory.values())
        assert len(timeline_lens) == 1
        self._events.append(timeline_lens[0], name)

    def save_memory_profile(self, save_path: str):
        plt.figure()

        for device, mem in self._memory.items():
            plt.plot(mem.timeline, label=str(device))

        for index, name in self._events:
            plt.axvline(x=index, color="gray", linestyle="--", linewidth=1)
            plt.text(index, plt.ylim()[1] * 0.95, name, rotation=90, verticalalignment="top", fontsize=8)

        plt.xlabel("Operation Index")
        plt.ylabel("Memory Usage (bytes)")
        plt.title("Tensor Memory Usage Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
