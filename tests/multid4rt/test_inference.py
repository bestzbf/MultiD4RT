"""Model-only distribution, image IO and legacy checkpoint compatibility."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from multid4rt import MultiD4RT, MultiD4RTConfig
from multid4rt.checkpoint import read_checkpoint, save_checkpoint
from multid4rt.demo import make_demo_checkpoint, make_demo_input
from multid4rt.infer import main as infer_main


def test_inference_cli_imports_without_optional_project_components():
    root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc
import sys
class ModelOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'multicamera4rc':
            raise ImportError('old namespace: ' + fullname)
        if fullname.startswith(('multid4rt.datasets', 'multid4rt.train', 'multid4rt.losses',
                                'multid4rt.benchmarks', 'multid4rt.evaluate')):
            raise ImportError('removed dependency: ' + fullname)
sys.meta_path.insert(0, ModelOnly())
from multid4rt.__main__ import main
for arguments in (['--help'], ['infer', '--help'], ['demo', '--help']):
    try:
        main(arguments)
    except SystemExit as exc:
        assert exc.code == 0, (arguments, exc.code)
"""
    result = subprocess.run([sys.executable, '-c', script], cwd=root,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_manifest_inference_keeps_shared_frame_gaps_and_ignores_label_paths(tmp_path):
    torch.set_num_threads(1)
    manifest = make_demo_input(tmp_path / 'images')
    data = json.loads(manifest.read_text())
    data['clips'][0]['targets'] = 'not_available.pt'
    manifest.write_text(json.dumps(data))
    checkpoint = make_demo_checkpoint(tmp_path / 'model.pt')
    output = tmp_path / 'prediction'
    infer_main(['--manifest', str(manifest), '--checkpoint', str(checkpoint), '--output', str(output),
                '--source-obs', '0', '3', '--query-times', '0', '10', '--device', 'cpu', '--cpu-threads', '1'])
    result = torch.load(output / 'clip_000000.pt', weights_only=True)
    assert result['clip_id'] == 'demo'
    assert result['time_idx'].tolist() == [[0, 5, 10, 0, 5, 10]]
    assert result['query_time_idx'].tolist() == [[0, 10]]
    assert result['trajectory'].shape == (1, 2, 2, 28, 42, 3)
    assert result['visibility'].shape == (1, 2, 2, 2, 28, 42)
    assert not result['displacement'][:, :, 0].any()
    assert torch.isfinite(result['trajectory']).all()
    assert not (manifest.parent / 'targets.pt').exists()


def test_old_native_checkpoint_retains_strict_coverage_and_numerical_outputs(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(6)
    config = MultiD4RTConfig.tiny(gradient_checkpointing=False, decoder_memory='target_time', time_origin='absolute')
    model = MultiD4RT(config).eval()
    path = tmp_path / 'old.pt'
    save_checkpoint(path, model)
    data = read_checkpoint(path)
    assert data['architecture'] == 'multid4rt'
    data['architecture'] = 'multicamera4rc'
    torch.save(data, path)
    restored = MultiD4RT.from_checkpoint(path).eval()
    assert restored.config == config
    image = torch.rand(1, 4, 3, 28, 42)
    cameras = torch.tensor([[0, 0, 1, 1]])
    times = torch.tensor([[101, 106, 101, 106]])
    with torch.inference_mode():
        expected = model(image, cameras, times, source_obs_idx=[0, 2])
        actual = restored(image, cameras, times, source_obs_idx=[0, 2])
    for key in ('depth', 'K', 'c2w', 'base_points', 'displacement', 'trajectory', 'visibility'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    data['model'].pop('target_projection.weight')
    torch.save(data, path)
    with pytest.raises(ValueError, match='native checkpoint must cover the complete model'):
        MultiD4RT.from_checkpoint(path)


def test_readme_python_example_runs_with_documented_shapes():
    import ast

    root = Path(__file__).resolve().parents[2]
    readme = (root / 'README.md').read_text()
    snippet = readme.split('```python\n', 1)[1].split('```', 1)[0]
    torch.set_num_threads(1)
    namespace = {}
    exec(compile(ast.parse(snippet), 'README.md example', 'exec'), namespace)
    assert namespace['pred']['trajectory'].shape == (1, 2, 3, 28, 42, 3)
    assert namespace['pred']['visibility'].shape == (1, 2, 3, 2, 28, 42)
