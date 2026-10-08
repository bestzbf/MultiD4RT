#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
MVSSYNTH_ROOT="${MVSSYNTH_ROOT:-/data2/d4rt/datasets/MVS-Synth/GTAV_1080}"
SAMPLE_STAGE_ROOT="${SAMPLE_STAGE_ROOT:-/data1/zbf/d4rt_sample_stage}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/data/zbf/openclaw/d4rt/.index_cache_5datasets_local}"

export D4RT_BUILD_TIMEOUT="${D4RT_BUILD_TIMEOUT:-30}"
export D4RT_PROFILE_BUILDER=1
export D4RT_PROFILE_BUILDER_ALL=1

echo "MVSSYNTH_ROOT=$MVSSYNTH_ROOT"
echo "D4RT_BUILD_TIMEOUT=$D4RT_BUILD_TIMEOUT"

"$PYTHON_BIN" - "$MVSSYNTH_ROOT" "$SAMPLE_STAGE_ROOT" "$INDEX_CACHE_DIR" << 'PY'
import sys, os, random, time
sys.path.insert(0, '.')

mvssynth_root = sys.argv[1]
stage_root = sys.argv[2]
cache_dir = sys.argv[3]

from datasets.registry import create_adapter
from datasets.sample_stage import build_sample_stager
from datasets.transforms import GeometryTransformPipeline
from datasets.query_builder import D4RTQueryBuilder
from datasets.sample_builder import SampleBuilder
from datasets.planning import SampleSpec
from datasets.sample_spool import SampleSpool

print(f"[test] creating adapter root={mvssynth_root}", flush=True)
adapter = create_adapter(name='mvssynth', root=mvssynth_root, split='train',
                         load_precomputed=True, cache_dir=cache_dir)
print(f"[test] adapter ok seqs={len(adapter.list_sequences())}", flush=True)

sample_stage_config = {
    'backend': 'cos_sdk',
    'stage_root': stage_root,
    'sdk_workers': 32,
    'passwd_file': '/etc/passwd-s3fs-data_cos',
    'enabled_datasets': ['pointodyssey','kubric','dynamic_replica','co3dv2','scannetpp'],
    'mount_root': '/data_cos',
    'bucket': 'hd-ai-data-1251882982',
    'region': 'ap-beijing',
    'cache_max_bytes': 100 * 1024**3,
}

print("[test] building stager...", flush=True)
t0 = time.perf_counter()
stager = build_sample_stager(sample_stage_config)
print(f"[test] stager built in {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

transform = GeometryTransformPipeline(img_size=256, use_augs=True)
qb = D4RTQueryBuilder(num_queries=2048)

spool = SampleSpool(spool_dir='/tmp/test_mvssynth_spool', rank=0)

builder = SampleBuilder(
    builder_id=0,
    adapters=[adapter],
    transform=transform,
    query_builder=qb,
    spool=spool,
    clip_len=48,
    sample_stage_config=sample_stage_config,
)

seq = random.Random(42).choice(adapter.list_sequences())
n = adapter.get_num_frames(seq)
fi = list(range(0, min(48, n)))
spec = SampleSpec(
    dataset_idx=0,
    sequence_name=seq,
    frame_indices=fi,
    local_index=0,
    global_index=0,
    generation=0,
    rng_state=random.Random(42).getstate(),
)

print(f"[test] building sample seq={seq} frames={len(fi)}", flush=True)
t0 = time.perf_counter()
ok = builder._build_sample_with_retry(spec)
print(f"[test] done ok={ok} total={time.perf_counter()-t0:.2f}s", flush=True)
PY
