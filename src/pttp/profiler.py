import sys
import traceback as tb
import weakref
from functools import partial
from types import TracebackType
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
from torch.utils._python_dispatch import TorchDispatchMode

from .global_access import GlobalAccess
from .memory import MemoryProfile

__all__ = ["TensorProfiler"]


class TensorProfiler(TorchDispatchMode, GlobalAccess):
    _tracked: Set[int]
    _memory: MemoryProfile
    _events: List[Tuple[int, str]]

    def __init__(self, catch_errors: bool = True):
        self._tracked = set()
        self._memory = MemoryProfile()
        self._events = list()
        self._catch_errors = catch_errors

    # ::::::::::::::::::::::::::::::::::::::::::::::::
    # 📤 Public API — user-facing methods
    # ::::::::::::::::::::::::::::::::::::::::::::::::

    @property
    def memory(self) -> Dict[Union[torch.device, str], int]:
        ret = self._memory.current.copy()
        total = sum(ret.values(), start=0)
        ret.update({"total": total})
        return ret

    @property
    def memory_mib(self) -> Dict[Union[torch.device, str], float]:
        return {device: value / (1024 * 1024) for device, value in self.memory.items()}

    @property
    def memory_peak(self) -> Dict[Union[torch.device, str], int]:
        ret = self._memory.peak.copy()
        all = max(ret.values(), default=0)
        ret.update({"all": all})
        return ret

    @property
    def memory_peak_mib(self) -> Dict[Union[torch.device, str], float]:
        return {
            device: value / (1024 * 1024) for device, value in self.memory_peak.items()
        }

    @property
    def memory_timeline(self) -> Dict[torch.device, List[int]]:
        return self._memory.timeline

    @property
    def memory_timeline_mib(self) -> Dict[torch.device, List[float]]:
        return {
            device: [value / (1024 * 1024) for value in timeline]
            for device, timeline in self._memory.timeline
        }

    # ::::::::::::::::::::::::::::::::::::::::::::::::
    # 📊 Plotting — visualization utilities
    # ::::::::::::::::::::::::::::::::::::::::::::::::

    def mark_event(self, name: str):
        """
        Mark an event which will be drawn in any saved plots

        :param name: event label
        """
        index = max(len(self._memory), 1) - 1
        self._events.append((index, name))

    def save_memory_timeline(self, save_path: str):
        """
        Save a plot representing the captured memory profile

        :param save_path: path to file with image-like extension (png, jpg, ect.)
        """
        plt.figure()

        for device, mem in self.memory_timeline.items():
            plt.plot(mem, label=str(device))

        for index, name in self._events:
            plt.axvline(x=index, color="gray", linestyle="--", linewidth=1)
            plt.text(
                index,
                plt.ylim()[1] * 0.95,
                name,
                rotation=90,
                verticalalignment="top",
                fontsize=8,
            )

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

    # ::::::::::::::::::::::::::::::::::::::::::::::::
    # ⚙️ Tracking - Dispatch overload and finalizers
    # ::::::::::::::::::::::::::::::::::::::::::::::::

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        ret = func(*args, **(kwargs or {}))
        if isinstance(ret, torch.Tensor):
            storage = ret.untyped_storage()
            self._track(storage)

        return ret

    def _track(self, storage: torch.UntypedStorage):
        hash = storage.data_ptr()
        size = storage.nbytes()
        device = storage.device

        # skip if already tracked
        if hash in self._tracked:
            return

        # track
        self._memory.add(device, size)
        self._tracked.add(hash)

        # register finalizer to subtract memory
        finalizer = partial(self._untrack, hash, size, device)
        weakref.finalize(storage, finalizer)  # triggers regardless of gc

    def _untrack(self, hash: int, size: int, device: torch.device):
        # skip if no longer tracking
        if hash not in self._tracked:
            return

        # untrack
        self._memory.subtract(device, size)
        self._tracked.remove(hash)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        if self._catch_errors and exc_type is not None:
            tb.print_exception(exc_type, exc_value, traceback, file=sys.stderr)

        self._tracked = set()
        return super().__exit__(exc_type, exc_value, traceback) or self._catch_errors
