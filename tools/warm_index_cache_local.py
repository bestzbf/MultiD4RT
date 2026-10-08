#!/usr/bin/env python3
"""Warm up index cache for local datasets to avoid COS access during training."""

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.adapters.pointodyssey import PointOdysseyAdapter
from datasets.adapters.kubric import KubricAdapter
from datasets.adapters.dynamic_replica import DynamicReplicaAdapter
from datasets.adapters.co3dv2 import Co3Dv2Adapter
from datasets.adapters.blendedmvs import BlendedMVSAdapter


def warm_dataset(name, adapter_class, root, cache_dir, split="train", **kwargs):
    """Warm up cache for a single dataset."""
    print(f"[warm_index_cache_local] warming {name} split={split} root={root}")
    t0 = time.time()

    adapter = adapter_class(
        root=root,
        split=split,
        cache_dir=cache_dir,
        **kwargs
    )

    t1 = time.time()
    num_seq = len(adapter)
    print(f"[warm_index_cache_local] ready {name} split={split}: {num_seq} sequences in {t1-t0:.1f}s")
    return adapter


def main():
    # Configuration from mixture_5datasets_local.yaml
    cache_dir = "/data/zbf/openclaw/d4rt/.index_cache_5datasets_local"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[warm_index_cache_local] CACHE_DIR={cache_dir}")
    print(f"[warm_index_cache_local] INDEX_WORKERS=16")

    # PointOdyssey (local)
    if Path("/data2/d4rt/datasets/PointOdyssey").exists():
        warm_dataset(
            "pointodyssey",
            PointOdysseyAdapter,
            root="/data2/d4rt/datasets/PointOdyssey",
            cache_dir=cache_dir,
            split="train",
            require_tracks=True,
            index_workers=16,
        )
    else:
        print("[warm_index_cache_local] SKIP pointodyssey: /data2/d4rt/datasets/PointOdyssey not found")

    # Kubric (local)
    if Path("/data/d4rt/kubric").exists():
        warm_dataset(
            "kubric",
            KubricAdapter,
            root="/data/d4rt/kubric",
            cache_dir=cache_dir,
            split="train",
            index_workers=16,
        )
    else:
        print("[warm_index_cache_local] SKIP kubric: /data/d4rt/kubric not found")

    # Dynamic Replica (local)
    if Path("/data1/d4rt/datasets/Dynamic_Replica").exists():
        warm_dataset(
            "dynamic_replica",
            DynamicReplicaAdapter,
            root="/data1/d4rt/datasets/Dynamic_Replica",
            cache_dir=cache_dir,
            split="train",
            # DynamicReplica doesn't have index_workers parameter
        )
    else:
        print("[warm_index_cache_local] SKIP dynamic_replica: /data1/d4rt/datasets/Dynamic_Replica not found")

    # Co3Dv2 (local)
    if Path("/data2/d4rt/datasets/Co3Dv2").exists():
        warm_dataset(
            "co3dv2",
            Co3Dv2Adapter,
            root="/data2/d4rt/datasets/Co3Dv2",
            cache_dir=cache_dir,
            split="train",
            subset_name="fewview_train",
            categories=None,
            require_precomputed=False,
            require_pointcloud=True,
            sequence_denylist="/data/zbf/openclaw/d4rt/configs/co3dv2_denylist_degenerate_clips_20260422.txt",
            # Co3Dv2 doesn't have index_workers parameter
        )
    else:
        print("[warm_index_cache_local] SKIP co3dv2: /data2/d4rt/datasets/Co3Dv2 not found")

    # BlendedMVS (local)
    if Path("/data/d4rt/data/BlendedMVS").exists():
        warm_dataset(
            "blendedmvs",
            BlendedMVSAdapter,
            root="/data/d4rt/data/BlendedMVS",
            cache_dir=cache_dir,
            split="train",
            use_masked=False,
            index_workers=16,
        )
    else:
        print("[warm_index_cache_local] SKIP blendedmvs: /data/d4rt/data/BlendedMVS not found")

    print("[warm_index_cache_local] All done!")


if __name__ == "__main__":
    main()
