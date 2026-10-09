"""End-to-end inference and physical evaluation without training modules."""

import json
from pathlib import Path
import subprocess
import sys

import h5py
import torch

from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc.checkpoint import save_checkpoint
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples
from multicamera4rc.datasets.partial_dense import attach_partial_dense_recipe
from multicamera4rc.demo import make_demo_checkpoint, make_demo_dataset
from multicamera4rc.evaluation import evaluate_batch
from multicamera4rc.infer import main as infer_main
from test_dense_supervision import _scene
from test_native_motion_stereo import particles
from test_static_multiview import configuration


def test_all_cli_commands_import_with_training_modules_forbidden():
    root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc
import sys
class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'multicamera4rc.train', 'multicamera4rc.launch',
                        'multicamera4rc.losses', 'multicamera4rc.distributed',
                        'multicamera4rc.schedule', 'multicamera4rc.quality'}:
            raise ImportError('training dependency: ' + fullname)
sys.meta_path.insert(0, NoTraining())
from multicamera4rc.__main__ import COMMANDS, main
assert 'train' not in COMMANDS and 'launch' not in COMMANDS
for command in COMMANDS:
    try:
        main([command, '--help'])
    except SystemExit as exc:
        assert exc.code == 0, (command, exc.code)
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=root,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr


def test_manifest_inference_needs_no_gt_and_preserves_saved_semantics(tmp_path):
    manifest = make_demo_dataset(tmp_path / "images")
    record = json.loads(manifest.read_text())
    record['clips'][0]['targets'] = 'not_available.pt'
    manifest.write_text(json.dumps(record))
    (manifest.parent / 'targets.pt').unlink()
    config = MultiCamera4RCConfig.tiny(gradient_checkpointing=False,
                                     decoder_memory='target_time', time_origin='absolute')
    checkpoint = tmp_path / 'model.pt'
    save_checkpoint(checkpoint, MultiCamera4RC(config).eval())
    output = tmp_path / 'inference'
    infer_main(['--manifest', str(manifest), '--checkpoint', str(checkpoint),
                '--output', str(output), '--source-obs', '0', '3', '--query-times', '0', '10',
                '--device', 'cpu', '--cpu-threads', '1'])
    prediction = torch.load(output / 'clip_000000.pt', weights_only=True)
    settings = json.loads((output / 'inference.json').read_text())['model_config']
    assert settings['decoder_memory'] == 'target_time' and settings['time_origin'] == 'absolute'
    assert prediction['clip_id'] == 'synthetic_smoke'
    assert prediction['trajectory'].shape == (1, 2, 2, 28, 42, 3)
    assert prediction['visibility'].shape == (1, 2, 2, 2, 28, 42)
    assert not prediction['displacement'][:, :, 0].any()
    assert prediction['trajectory'].device.type == 'cpu'
    assert torch.isfinite(prediction['trajectory']).all()


def test_static_evaluation_uses_geometry_without_motion_decoder(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    batch = collate_adapter_samples([create_dataset(configuration(tmp_path, count=4, choices=4))[0]])
    source = torch.arange(4)[None]
    target = materialize_dense_targets(batch, source)
    model = MultiCamera4RC(MultiCamera4RCConfig.tiny(gradient_checkpointing=False)).eval()

    def forbidden(*args, **kwargs):
        raise AssertionError('geometry-only evaluation called the motion decoder')

    monkeypatch.setattr(model, 'forward', forbidden)
    metrics = evaluate_batch(model, batch, source, target)
    assert target['trajectory_valid'].sum() == 4 * 27 * 39
    assert 'depth_abs_rel' in metrics and 'trajectory_epe_normalized' in metrics
    assert 'loss' not in metrics and all(torch.isfinite(value) for value in metrics.values())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_renderer_evaluation_keeps_all_dense_valid_point_times(tmp_path):
    torch.set_num_threads(1)
    _, _, batch = _scene(tmp_path, include_visibility=True)
    source = torch.tensor([[0, 3]])
    target = materialize_dense_targets(batch, source)
    model = MultiCamera4RC(MultiCamera4RCConfig.tiny(gradient_checkpointing=False)).eval()
    metrics = evaluate_batch(model, batch, source, target, need_visibility=True)
    assert metrics['supervised_trajectory_pixels'] == target['trajectory_valid'].sum()
    assert metrics['supervised_source_pixels'] == 2 * 27 * 39
    assert 'displacement_epe_normalized' in metrics
    assert all(torch.isfinite(value) for value in metrics.values())


def test_native_evaluation_includes_occluded_particle_coordinates(tmp_path):
    torch.set_num_threads(1)
    clip = particles()
    sample = prepare_clip(clip, image_mode='native', num_queries=0, include_visibility=True,
                          all_track_queries=True)
    attach_partial_dense_recipe(clip, sample, kind='pointodyssey_tracks', image_mode='native')
    batch = collate_adapter_samples([sample])
    source = torch.tensor([[0, 1, 2]])
    target = materialize_dense_targets(batch, source)
    model = MultiCamera4RC(MultiCamera4RCConfig.tiny(gradient_checkpointing=False)).eval()
    metrics = evaluate_batch(model, batch, source, target, need_visibility=True)
    assert metrics['native_track_queries'] == len(target['sparse_queries'][0])
    assert metrics['native_track_point_times'] == target['sparse_trajectory_valid'].sum()
    assert metrics['native_temporal_point_times'] > 0
    hidden = (target['sparse_track_id'][0] == 0) & (target['sparse_queries'][0, :, 0] == 0)
    assert target['sparse_trajectory_valid'][0, hidden, 1].all()
    assert not target['sparse_visibility'][0, hidden, 1].any()


def test_static_benchmark_cli_runs_from_native_data_to_report(tmp_path):
    from multicamera4rc.benchmarks.static_multiview import main

    config = configuration(tmp_path, count=4, choices=4)
    with h5py.File(config['sources'][0]['root'], 'r+') as file:
        file.copy('train/scene', 'train/held')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(root=config['sources'][0]['root'],
        sequences={'scene': dict(length=4, physical_scene='observed_room')},
        validation_sequences={'held': dict(length=4, physical_scene='heldout_room')})))
    config['sources'][0]['manifest'] = str(manifest)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    checkpoint = make_demo_checkpoint(tmp_path / 'tiny.pt')
    output = tmp_path / 'benchmark'
    main(['--data-config', str(path), '--plan', str(tmp_path / 'plan.json'),
          '--output', str(output), '--checkpoints', 'tiny=' + str(checkpoint), '--device', 'cpu'])
    report = json.loads((output / 'evaluation.json').read_text())
    assert report['stage'] == 'passed' and report['splits']['blended']['scene_overlap'] == 0
    result = report['checkpoints']['tiny']
    assert result['metric_scope'] == 'physical_errors_without_loss'
    assert result['records'][0]['sequence'] == 'held'
    assert result['records'][0]['actual_cameras'] == 4
    assert 'loss' not in result['records'][0]['metrics']
