"""
Benchmark original PointOdyssey with multi-threading.
"""

import time
from torch.utils.data import DataLoader

from datasets.collate import d4rt_collate_fn
from datasets.mixture import MixtureDataset
from datasets.registry import create_adapter


print("Creating PointOdyssey adapter...")
adapter = create_adapter(
    "pointodyssey",
    root="/data2/d4rt/datasets/PointOdyssey",
    split="train"
)
print(f"Sequences: {len(adapter)}")

print("\nCreating dataset...")
dataset = MixtureDataset(
    adapters=[adapter],
    clip_len=8,
    img_size=256,
    num_queries=2048,
)

print("\nCreating DataLoader with 16 workers...")
loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=16,
    collate_fn=d4rt_collate_fn,
    shuffle=False,
    pin_memory=False,
    persistent_workers=True,
)

print("\nWarming up (5 batches)...")
warmup_start = time.time()
for i, batch in enumerate(loader):
    if i >= 5:
        break
warmup_time = time.time() - warmup_start
print(f"Warmup: {warmup_time:.2f}s")

print("\nBenchmarking (30 batches)...")
batch_times = []
start = time.time()

for i, batch in enumerate(loader):
    batch_end = time.time()

    if i > 0:
        batch_times.append(batch_end - batch_start)

    batch_start = batch_end

    if i >= 30:
        break

    if (i + 1) % 10 == 0:
        recent = batch_times[-10:]
        avg = sum(recent) / len(recent)
        print(f"  Batch {i+1}: {avg:.3f}s/batch ({1/avg:.2f} batch/s)")

total = time.time() - start
avg = sum(batch_times) / len(batch_times)
throughput = 1.0 / avg

print(f"\n{'='*60}")
print(f"Results:")
print(f"  Total: {total:.2f}s")
print(f"  Average: {avg:.3f}s/batch")
print(f"  Throughput: {throughput:.2f} batches/s")
print(f"  Samples/s: {throughput * 4:.2f}")

if throughput >= 2.0:
    print(f"\n✅ PASS: {throughput:.2f} >= 2.0 batches/s")
else:
    print(f"\n⚠️  WARN: {throughput:.2f} < 2.0 batches/s")
