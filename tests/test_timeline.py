import torch

from pttp import TensorProfiler
from helpers import requires_cuda

@requires_cuda
def test_timeline_with_events():
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0")

    with TensorProfiler() as prof:
        prof.mark_event("Start event")

        prof._memory.add(cpu_device, 1)
        prof.mark_event("After cpu add")

        prof._memory.add(gpu_device, 2)
        prof.mark_event("After cuda add")

        assert prof.memory_timeline[cpu_device] == [0, 1, 1]
        assert prof.memory_timeline[gpu_device] == [0, 0, 2]
        assert prof._events == [
            (0, "Start event"),
            (1, "After cpu add"),
            (2, "After cuda add")
        ]
