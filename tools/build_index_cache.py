#!/usr/bin/env python3
"""One-off script: build all adapter index caches defined in a YAML config."""
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.registry import create_adapter

def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "configs/mixture_5datasets_blendedmvs_hdu.yaml"
    if not config_path.is_absolute() and not config_path.exists():
        config_path = REPO_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    t0 = time.time()
    print(f"[build_index_cache] config={config_path}")
    print(f"[build_index_cache] index_workers={config.get('index_workers', 'default')}")
    print(f"[build_index_cache] cache_dir={config.get('index_cache_dir', 'none')}")
    print()

    index_cache_dir = config.get('index_cache_dir')
    index_workers = config.get('index_workers')

    for ds_config in config['datasets']:
        split = ds_config.get('split', 'train')
        extra = {}
        if index_cache_dir:
            extra['cache_dir'] = index_cache_dir
        if index_workers is not None:
            extra['index_workers'] = index_workers

        print(f"[{ds_config['name']}] Building index for split={split}...")
        t_start = time.time()
        adapter = create_adapter(
            name=ds_config['name'],
            root=ds_config['root'],
            split=split,
            **ds_config.get('adapter_kwargs', {}),
            **extra,
        )
        print(f"[{ds_config['name']}] Done in {time.time()-t_start:.1f}s (sequences={len(adapter)})")

    print(f"\n[build_index_cache] Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
