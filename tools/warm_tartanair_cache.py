#!/usr/bin/env python3
"""
One-time script to build TartanAir index cache via coscli (avoids slow COS iterdir).
Usage: python tools/warm_tartanair_cache.py
"""
import subprocess, hashlib, json, pickle, tempfile, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BUCKET = "hd-ai-data-1251882982"
COS_PREFIX = "hdu_datasets/tartanairv1"
CAMERA = "left"
CACHE_DIR = Path("/data/zbf/openclaw/d4rt/.index_cache_5datasets_local")
CACHE_DIR.mkdir(exist_ok=True)

cache_key = {"dataset": "tartanair", "split": "train", "camera": CAMERA, "cache_schema": 1}
suffix = hashlib.sha1(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:12]
cache_path = CACHE_DIR / f"tartanair_train_{suffix}.pkl"

if cache_path.exists():
    print(f"Cache already exists: {cache_path}")
    data = pickle.loads(cache_path.read_bytes())
    print(f"  {len(data)} sequences")
    raise SystemExit(0)

def coscli_ls(prefix):
    r = subprocess.run(
        ["coscli", "ls", f"cos://{BUCKET}/{prefix}/"],
        capture_output=True, text=True, timeout=30
    )
    dirs = []
    stripped_prefix = prefix.rstrip("/") + "/"
    for line in r.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        key = parts[0].strip()
        typ = parts[1].strip()
        if typ == "DIR" and key.endswith("/"):
            # key is full path like "hdu_datasets/tartanairv1/scene/"
            rel = key[len(stripped_prefix):].rstrip("/")
            if rel and "/" not in rel and rel != ".cache":
                dirs.append(rel)
    return dirs

def get_num_frames(seq_rel):
    """Download pose file and count lines."""
    cos_path = f"cos://{BUCKET}/{COS_PREFIX}/{seq_rel}/pose_{CAMERA}.txt"
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        tmp = f.name
    try:
        r = subprocess.run(
            ["coscli", "cp", cos_path, tmp],
            capture_output=True, timeout=30
        )
        if r.returncode != 0:
            return 0
        with open(tmp) as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0
    finally:
        os.unlink(tmp)

print("Listing scenes...")
scenes = coscli_ls(COS_PREFIX)
print(f"  {len(scenes)} scenes: {scenes}")

sequences = []  # (seq_rel, nf)
for scene in scenes:
    difficulties = coscli_ls(f"{COS_PREFIX}/{scene}")
    for diff in difficulties:
        p00xs = coscli_ls(f"{COS_PREFIX}/{scene}/{diff}")
        for p in p00xs:
            sequences.append(f"{scene}/{diff}/{p}")

print(f"Found {len(sequences)} sequences. Fetching frame counts...")

results = []
with ThreadPoolExecutor(max_workers=16) as ex:
    futures = {ex.submit(get_num_frames, s): s for s in sequences}
    for i, fut in enumerate(as_completed(futures)):
        s = futures[fut]
        nf = fut.result()
        if nf > 0:
            results.append((s, nf))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(sequences)} done", flush=True)

results.sort()
cache_path.write_bytes(pickle.dumps(results))
print(f"Saved {len(results)} sequences to {cache_path}")
