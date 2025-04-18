import gc

import torch

from pttp import TensorProfiler

with TensorProfiler() as prof:
    a = torch.randn(10)
    b = torch.randn(10)
    prof.mark_event("A and B allocated")

    c = a + b
    prof.mark_event("C allocated")

    del a, b
    gc.collect()
    prof.mark_event("A and B collected")

prof.save_memory_timeline("memory.png")
remaining_memory = prof.memory  # 40 bytes
