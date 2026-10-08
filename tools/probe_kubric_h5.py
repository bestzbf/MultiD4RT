"""Probe a single Kubric .h5 file's chunk layout via s3fs-mounted /data_cos.

This is application-layer h5py open on a single file — NOT a directory scan.
Does NOT use ls/find/du on /data_cos.
"""
from pathlib import Path
import sys
import time

import h5py

# Caller passes seq name(s), or default to 0661 from the training log.
SEQ = sys.argv[1] if len(sys.argv) > 1 else "0661"
KUBRIC_ROOT = Path("/data_cos/hdu_datasets/Kubric")

# Kubric records store h5 as either <root>/<seq>/<seq>.h5 (nested) or <root>/<seq>.h5.
candidates = [
    KUBRIC_ROOT / SEQ / f"{SEQ}.h5",
    KUBRIC_ROOT / f"{SEQ}.h5",
]
h5_path = next((p for p in candidates if p.exists()), None)
if h5_path is None:
    print(f"[ERROR] No .h5 found for seq={SEQ} under {KUBRIC_ROOT}")
    print(f"  tried: {[str(c) for c in candidates]}")
    sys.exit(1)

size_bytes = h5_path.stat().st_size
print(f"h5_path      = {h5_path}")
print(f"size         = {size_bytes / 1024 / 1024:.2f} MB")

t0 = time.perf_counter()
with h5py.File(h5_path, "r") as hf:
    print(f"open_elapsed = {(time.perf_counter() - t0)*1000:.1f} ms")
    print(f"keys         = {list(hf.keys())}")
    for key in ("trajs_2d", "coords_depth", "visibility"):
        if key not in hf:
            print(f"[missing] {key}")
            continue
        d = hf[key]
        print(
            f"  {key:15s} shape={d.shape} dtype={d.dtype} "
            f"chunks={d.chunks} compression={d.compression} "
            f"compression_opts={d.compression_opts}"
        )
    t1 = time.perf_counter()
    frame_idx = list(range(0, min(48, d.shape[0])))
    slice_data = hf["trajs_2d"][frame_idx]
    slice_elapsed = time.perf_counter() - t1
    print(
        f"slice[:48] of trajs_2d shape={slice_data.shape} "
        f"elapsed={slice_elapsed*1000:.1f} ms"
    )
