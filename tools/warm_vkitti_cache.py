#!/usr/bin/env python3
"""One-time script to build VirtualKitti index cache via coscli."""
import subprocess, pickle
from pathlib import Path

BUCKET = "hd-ai-data-1251882982"
COS_PREFIX = "hdu_datasets/VirtualKitti"
CACHE_DIR = Path("/data/zbf/openclaw/d4rt/.index_cache_5datasets_local")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_PATH = CACHE_DIR / "vkitti2_train_sequences.pkl"

if CACHE_PATH.exists():
    data = pickle.loads(CACHE_PATH.read_bytes())
    print(f"Cache exists: {len(data)} sequences"); raise SystemExit(0)

def coscli_ls(prefix):
    r = subprocess.run(["coscli", "ls", f"cos://{BUCKET}/{prefix}/"],
                       capture_output=True, text=True, timeout=30)
    stripped = prefix.rstrip("/") + "/"
    dirs = []
    for line in r.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 2: continue
        key = parts[0].strip(); typ = parts[1].strip()
        if typ == "DIR" and key.endswith("/"):
            rel = key[len(stripped):].rstrip("/")
            if rel and "/" not in rel:
                dirs.append(rel)
    return dirs

print("Listing scenes...")
scenes = [s for s in coscli_ls(COS_PREFIX) if "Scene" in s]
print(f"  {len(scenes)} scenes")

sequences = []  # (seq_name, seq_dir_rel)
for scene in scenes:
    variations = coscli_ls(f"{COS_PREFIX}/{scene}")
    for var in variations:
        seq_name = f"{scene}_{var}"
        sequences.append(seq_name)

print(f"Found {len(sequences)} sequences")
CACHE_PATH.write_bytes(pickle.dumps(sequences))
print(f"Saved to {CACHE_PATH}")
