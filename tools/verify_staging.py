import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np, h5py
from datasets.sample_stage import SampleLocalStager, SampleStageConfig
from datasets.adapters.base import load_precomputed_fast


class FakeAdapter:
    dataset_name = 'scannetpp'
    def __init__(self, root):
        self.root = Path(root)
        self.data_root = self.root
        self._scene_cache = {}
        self._staged_depth_map_tmp = None
        self._staged_h5_frame_map = None
        self._depth_chunk_cache = {}
        self.precomputed_name = 'precomputed.npz'

    def _get_scene_data(self, scene_name):
        if scene_name in self._scene_cache:
            return self._scene_cache[scene_name]
        data = {
            'scene_dir': self.data_root / scene_name,
            'full_indices': np.arange(960, dtype=np.int32),
            'frame_stems': [f'{i:06d}' for i in range(960)],
        }
        self._scene_cache[scene_name] = data
        return data


config = SampleStageConfig(
    backend='cos_sdk', stage_root='/data1/zbf/d4rt_sample_stage', sdk_workers=32,
    mount_root='/data_cos', bucket='hd-ai-data-1251882982', region='ap-beijing',
    enabled_datasets=('scannetpp',),
)
stager = SampleLocalStager(config)
adapter = FakeAdapter('/data_cos/hdu_datasets/scannetpp/data')
scene = '7e7d2e8640'
frame_indices = list(range(100, 148))

with stager.stage_sample(adapter, scene, frame_indices, sample_tag='verify') as staged:
    sd = staged._scene_cache[scene]
    h5_frame_map = sd.get('_staged_h5_frame_map')
    npz_path = sd.get('_precomputed_dir', sd['scene_dir']) / 'precomputed.npz'
    mapped = [h5_frame_map[fi] for fi in frame_indices]
    cache = load_precomputed_fast(npz_path, mapped)
    print(f'trajs_2d: {cache["trajs_2d"].shape}')
    print(f'extrinsics: {cache["extrinsics"].shape}')
    # Verify frame 100 data matches original
    with h5py.File('/data_cos/hdu_datasets/scannetpp/data/7e7d2e8640/precomputed.h5', 'r') as f:
        orig = f['trajs_2d'][100]
    match = np.allclose(orig, cache['trajs_2d'][0])
    print(f'frame 100 data matches original: {match}')
    print('ALL OK' if match else 'MISMATCH!')
