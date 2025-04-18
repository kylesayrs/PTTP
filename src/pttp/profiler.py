from typing import List, Set, Dict, Tuple

import gc
import weakref
from functools import partial
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .global_access import GlobalAccess

__all__ = ["TensorProfiler"]


class MemoryProfile:
    _timelines: Dict[torch.device, List[int]]

    def __init__(self):
        self._timelines: Dict[torch.device, List[int]] = defaultdict(list)

    def add(self, device: torch.device, size: int):
        if device not in self._timelines:
            timeline = [0 for _ in range(max(len(self), 1))]
            self._timelines[device] = timeline

        for dev in self._timelines:
            if dev == device:
                diff = size
            else:
                diff = 0

            self._timelines[dev].append(self._timelines[dev][-1] + diff)
            
    def subtract(self, device: torch.device, size: int):
        self.add(device, -size)

    @property
    def timeline(self) -> Dict[torch.device, List[int]]:
        return self._timelines
    
    @property
    def current(self) -> Dict[torch.device, int]:
        return {
            device: self._timelines[device][-1]
            for device in self._timelines
        }

    def __len__(self) -> int:
        return max((len(timeline) for timeline in self._timelines.values()), default=0)


class TensorProfiler(TorchDispatchMode, GlobalAccess):
    _memory = MemoryProfile
    _tracked: Set[int]
    _events = List[Tuple[int, str]]

    def __init__(self):
        self._memory: MemoryProfile = MemoryProfile()
        self._tracked: Set[int] = set()
        self._events: List[Tuple[int, str]] = list()

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        ret = func(*args, **(kwargs or {}))
        if isinstance(ret, torch.Tensor):
            storage = ret.untyped_storage()
            self._track(storage)

        return ret

    def _track(self, storage: torch.UntypedStorage):
        _hash = storage.data_ptr()
        size = storage.nbytes()
        device = storage.device

        # skip if already tracked
        if _hash in self._tracked:
            return

        # track
        self._memory.add(device, size)
        self._tracked.add(_hash)

        # register hook to subtract memory
        finalizer = partial(self._untrack, _hash, size, device)
        weakref.finalize(storage, finalizer)

    def _untrack(self, hash: int, size: int, device: torch.device):
        # untrack
        self._memory.subtract(device, size)
        self._tracked.remove(hash)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        return super().__exit__(exc_type, exc_val, exc_tb)

    
    ## Public functions


    @property
    def total_memory(self) -> int:
        return sum((mem for mem in self._memory.current.values()), 0)

    @property
    def total_memory_mib(self) -> float:
        return self.total_memory / (1024 * 1024)
    
    def get_device_memory(self, device: torch.device) -> int:
        return self._memory.current[device]
    
    def get_device_memory_mib(self, device: torch.device) -> int:
        return self._memory.current[device] / (1024 * 1024)
    
    @property
    def memory_timeline(self):
        return self._memory.timeline
    

    ## Plotting

    def mark_event(self, name: str):
        index = max(len(self._memory), 1) - 1
        self._events.append((index, name))

    def save_memory_profile(self, save_path: str):
        plt.figure()

        for device, mem in self.memory_timeline.items():
            plt.plot(mem, label=str(device))

        for index, name in self._events:
            plt.axvline(x=index, color="gray", linestyle="--", linewidth=1)
            plt.text(index, plt.ylim()[1] * 0.95, name, rotation=90, verticalalignment="top", fontsize=8)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.xlabel("Operation Index")
        plt.ylabel("Memory Usage (bytes)")
        plt.title("Tensor Memory Usage Over Time")
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
