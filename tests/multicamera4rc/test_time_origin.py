"""Frame-origin invariance is explicit and cannot silently migrate checkpoints."""

from dataclasses import replace

import pytest
import torch

from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc.checkpoint import config_from_checkpoint, save_checkpoint


@pytest.mark.parametrize('scope', ['target_time', 'all_observations'])
def test_relative_time_preserves_gaps_padding_batch_origins_and_target_subsets(scope):
    torch.manual_seed(921)
    torch.set_num_threads(1)
    model = MultiCamera4RC(MultiCamera4RCConfig.tiny(
        decoder_memory=scope, time_origin='clip_start', gradient_checkpointing=False)).eval()
    with torch.no_grad():
        # Exercise learned temporal inputs; zero-initialized projections could
        # otherwise make an incorrect absolute implementation pass this test.
        model.backbone.frame_projection.weight.normal_(std=.01)
        model.target_projection.weight.normal_(std=.01)
        model.delta_projection.weight.normal_(std=.01)
        for block in model.motion_decoder.self_blocks:
            block.adaLN_modulation[-1].weight.normal_(std=.01)
    images = torch.rand(2, 4, 3, 28, 28)
    cameras = torch.tensor([[0, 0, 1, -1], [5, 5, 9, -1]])
    times = torch.tensor([[10, 17, 10, -999], [44, 51, 44, -999]])
    valid = cameras >= 0
    offsets = torch.tensor([[1000], [-77]])
    origin = model._time_origin(times, valid)
    assert origin.tolist() == [[10], [44]]
    assert (times-origin)[:, :3].tolist() == [[0, 7, 0], [0, 7, 0]]
    with torch.no_grad():
        first = model(images, cameras, times, times[:, 1:2], [0], observation_valid=valid, return_visibility=False)
        shifted = model(images, cameras, times+offsets, times[:, 1:2]+offsets, [0], observation_valid=valid, return_visibility=False)
        both = model(images, cameras, times, times[:, :2], [0], observation_valid=valid, return_visibility=False)
    for name in ('depth', 'ray', 'K', 'c2w', 'trajectory', 'displacement'):
        torch.testing.assert_close(first[name], shifted[name], atol=0, rtol=0)
    torch.testing.assert_close(first['trajectory'], both['trajectory'][:, :, 1:], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(shifted['query_time_idx'], first['query_time_idx']+offsets)


def test_legacy_absolute_time_roundtrip_and_explicit_migration(tmp_path):
    torch.manual_seed(192)
    model = MultiCamera4RC(MultiCamera4RCConfig.tiny(time_origin='absolute')).eval()
    with torch.no_grad():
        model.backbone.frame_projection.weight.normal_(std=.01)
        model.target_projection.weight.normal_(std=.01)
    path = tmp_path/'old.pt'
    save_checkpoint(path, model)
    data = torch.load(path, weights_only=True)
    del data['config']['time_origin']
    restored = MultiCamera4RC.from_checkpoint(data).eval()
    assert restored.config.time_origin == 'absolute'
    inputs = (torch.rand(1, 3, 3, 28, 28), torch.zeros(1, 3, dtype=torch.long), torch.tensor([[51, 53, 57]]))
    with torch.no_grad():
        before = model(*inputs, source_obs_idx=[0], return_visibility=False)
        after = restored(*inputs, source_obs_idx=[0], return_visibility=False)
    torch.testing.assert_close(before['trajectory'], after['trajectory'], atol=0, rtol=0)
    migrated = MultiCamera4RC.from_checkpoint(data, config=replace(config_from_checkpoint(data), time_origin='clip_start'))
    save_checkpoint(path, migrated)
    assert MultiCamera4RC.from_checkpoint(path).config.time_origin == 'clip_start'
    assert set(migrated.state_dict()) == set(model.state_dict())
    assert MultiCamera4RCConfig.tiny().time_origin == 'clip_start'
    with pytest.raises(ValueError, match='time_origin'):
        replace(model.config, time_origin='per_camera')
