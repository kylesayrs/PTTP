import torch

from pttp import TensorProfiler


def test_timeline_with_events():
    with TensorProfiler() as prof:
        prof.mark_event("Start event")

        prof._memory.add(torch.device("cpu"), 1)
        prof.mark_event("After cpu add")

        prof._memory.add(torch.device("cuda:0"), 2)
        prof.mark_event("After cuda add")

        assert prof.memory_timeline[torch.device("cpu")] == [0, 1, 1]
        assert prof.memory_timeline[torch.device("cuda:0")] == [0, 0, 2]
        assert prof._events == [
            (1, "Start event"),
            (2, "After cpu add"),
            (3, "After cuda add")
        ]
