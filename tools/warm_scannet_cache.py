#!/usr/bin/env python3
"""One-time script to build ScanNet index cache via coscli + precomputed.h5 metadata."""
import subprocess, hashlib, json, pickle, tempfile, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BUCKET = "hd-ai-data-1251882982"
COS_PREFIX = "hdu_datasets/scannet/scans"
CACHE_DIR = Path("/data/zbf/openclaw/d4rt/.index_cache_5datasets_local")
CACHE_DIR.mkdir(exist_ok=True)

cache_key = {"dataset": "scannet", "cache_schema": 1}
suffix = hashlib.sha1(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:12]
cache_path = CACHE_DIR / f"scannet_train_{suffix}.pkl"

if cache_path.exists():
    data = pickle.loads(cache_path.read_bytes())
    print(f"Cache already exists: {cache_path}")
    print(f"  {len(data)} scenes")
    raise SystemExit(0)

def coscli_ls(prefix):
    r = subprocess.run(
        ["coscli", "ls", f"cos://{BUCKET}/{prefix}/"],
        capture_output=True, text=True, timeout=60
    )
    stripped = prefix.rstrip("/") + "/"
    dirs = []
    for line in r.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        key = parts[0].strip()
        typ = parts[1].strip()
        if typ == "DIR" and key.endswith("/"):
            rel = key[len(stripped):].rstrip("/")
            if rel and "/" not in rel and rel.startswith("scene"):
                dirs.append(rel)
    return dirs

def get_num_frames(scene_id):
    """Download precomputed.h5 metadata to get num_frames."""
    import h5py, numpy as np
    cos_path = f"cos://{BUCKET}/{COS_PREFIX}/{scene_id}/processed/precomputed.h5"
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        tmp = f.name
    try:
        # Only download the first few KB (h5 metadata is at the start)
        # But coscli doesn't support range downloads, so we'll use a different approach
        # Read num_frames from the h5 file - need to download it
        # Actually h5 files can be very large. Let's check if there's a lighter way.
        # The precomputed.h5 stores num_frames as a scalar dataset.
        # We can't partially download h5. Instead, use npz if available.
        # Check for npz first
        npz_cos = f"cos://{BUCKET}/{COS_PREFIX}/{scene_id}/processed/precomputed.npz"
        # Actually, looking at the coscli output, scenes only have .h5
        # h5 files are 8GB+, can't download just for num_frames
        # Alternative: count files in processed/images/
        images_prefix = f"{COS_PREFIX}/{scene_id}/processed/images"
        r = subprocess.run(
            ["coscli", "ls", f"cos://{BUCKET}/{images_prefix}/"],
            capture_output=True, text=True, timeout=60
        )
        count = 0
        for line in r.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 2:
                continue
            typ = parts[1].strip()
            if typ == "STANDARD":
                count += 1
        return count
    except Exception:
        return 0
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

print("Listing scenes...")
scenes = coscli_ls(COS_PREFIX)
print(f"  {len(scenes)} scenes")

print(f"Getting frame counts for {len(scenes)} scenes...")
results = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(get_num_frames, s): s for s in scenes}
    for i, fut in enumerate(as_completed(futures)):
        s = futures[fut]
        nf = fut.result()
        if nf > 0:
            results.append((s, nf))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(scenes)} done", flush=True)

results.sort()
cache_path.write_bytes(pickle.dumps(results))
print(f"Saved {len(results)} scenes to {cache_path}")
