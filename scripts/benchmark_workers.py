import os, sys, time, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared import get_dataloader

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "something-something-v2")
device = torch.device("cuda")

for nw in [24, 32, 48]:
    loader = get_dataloader(split="train", batch_size=64, num_frames=16, num_workers=nw, root=DATA_ROOT)
    it = iter(loader)
    for _ in range(3):
        next(it)
    t = time.perf_counter()
    s = 0
    for i, b in enumerate(it):
        if i >= 50:
            break
        if b:
            s += b[1].size(0)
    sps = s / (time.perf_counter() - t)
    print(f"workers={nw}  {sps:.1f} samples/sec  (~{168913/sps/3600:.1f}h/epoch)")
