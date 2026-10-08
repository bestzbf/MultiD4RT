"""Test PointOdyssey_fast speed."""

import time
from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset

print("Creating PointOdyssey_fast adapter...")
adapter = create_adapter(
    "pointodyssey_fast",
    root="/data2/d4rt/datasets/PointOdyssey_fast",
    rgb_root="/data2/d4rt/datasets/PointOdyssey",
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

print("\nTesting 10 samples...")
times = []
for i in range(10):
    start = time.time()
    sample = dataset[i]
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Sample {i}: {elapsed:.3f}s")

avg = sum(times) / len(times)
print(f"\nAverage: {avg:.3f}s/sample")
print(f"Throughput: {1/avg:.2f} samples/s")
print(f"With batch_size=4: {4/avg:.2f} samples/s = {1/(avg/4):.2f} batches/s")
