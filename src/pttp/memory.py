from typing import Dict, List

import torch

__all__ = ["MemoryProfile"]


class MemoryProfile:
    _timelines: Dict[torch.device, List[int]]

    def __init__(self):
        self._timelines: Dict[torch.device, List[int]] = dict()

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
    def current(self) -> Dict[torch.device, int]:
        return {device: self._timelines[device][-1] for device in self._timelines}

    @property
    def peak(self) -> Dict[torch.device, int]:
        return {device: max(self._timelines[device]) for device in self._timelines}

    @property
    def timeline(self) -> Dict[torch.device, List[int]]:
        return self._timelines

    def __len__(self) -> int:
        return max((len(timeline) for timeline in self._timelines.values()), default=0)
