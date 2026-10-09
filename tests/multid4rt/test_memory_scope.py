"""All-time decoder access, shared-KV gradients, and legacy resume semantics."""

from dataclasses import replace
import copy

import pytest
import torch

from multid4rt import MultiD4RT, MultiD4RTConfig
from multid4rt.checkpoint import config_from_checkpoint, save_checkpoint
from multid4rt.motion import MultiViewMotionDecoder


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_shared_kv_matches_repeated_memory_outputs_and_every_gradient(checkpointed):
    torch.manual_seed(815)
    shared = MultiViewMotionDecoder(64, num_heads=4, attention_chunk_size=None,
                                   checkpoint=checkpointed).train()
    with torch.no_grad():
        shared.camera_bias.uniform_(-.4, .4)
        for block in shared.self_blocks:
            block.adaLN_modulation[-1].weight.normal_(std=.01)
    repeated = copy.deepcopy(shared)
    source = torch.randn(2, 6, 64, requires_grad=True)
    memory = torch.randn(2, 18, 64, requires_grad=True)
    condition = torch.randn(2, 3, 64, requires_grad=True)
    reference = [value.detach().clone().requires_grad_() for value in (source, memory, condition)]
    cameras = torch.tensor([[0]*6+[1]*6+[0]*6]*2)
    valid = torch.ones(2, 18, dtype=torch.bool)
    valid[1, 12:] = False
    src_camera = torch.tensor([0, 1])
    output = shared(source, memory, condition, src_camera, cameras, valid, (2, 3))
    expected = repeated(reference[0], reference[1][:, None].expand(-1, 3, -1, -1),
                        reference[2], src_camera, cameras[:, None].expand(-1, 3, -1),
                        valid[:, None].expand(-1, 3, -1), (2, 3))
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
    output.square().mean().backward()
    expected.square().mean().backward()
    for value, other in zip((source, memory, condition), reference):
        torch.testing.assert_close(value.grad, other.grad, atol=1e-7, rtol=2e-4)
    for (name, value), (other_name, other) in zip(shared.named_parameters(), repeated.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(value.grad, other.grad, atol=1e-7, rtol=2e-4, msg=name)
    assert memory.grad[valid].abs().sum() > 0
    assert memory.grad[~valid].count_nonzero() == 0


@pytest.mark.parametrize("scope,expected_shape", [("all_observations", (1, 24, 64)),
                                                ("target_time", (1, 2, 8, 64))])
def test_full_model_passes_unqueried_observations_only_in_all_time_scope(scope, expected_shape):
    torch.manual_seed(22)
    model = MultiD4RT(MultiD4RTConfig.tiny(decoder_memory=scope)).eval()
    with torch.no_grad():
        model.target_projection.weight.normal_(std=.02)
        model.delta_projection.weight.normal_(std=.02)
        for block in model.motion_decoder.self_blocks:
            block.adaLN_modulation[-1].weight.normal_(std=.02)
            block.adaLN_modulation[-1].bias.normal_(std=.02)
    clip = dict(images=torch.rand(1, 6, 3, 28, 28),
                camera_id=torch.tensor([[0, 0, 0, 1, 1, 1]]),
                time_idx=torch.tensor([[0, 5, 10, 0, 5, 10]]))
    with torch.no_grad():
        scene = model.encode(**clip)
    # Request only t10 from a t0 source. t5 belongs to neither target nor anchor;
    # checking decoder memory directly excludes indirect encoder-global effects.
    captured = []
    hook = model.motion_decoder.register_forward_pre_hook(lambda _, args: captured.append(args))
    with torch.no_grad():
        model.decode(scene, query_time_idx=[10], source_obs_idx=[0], return_visibility=False)
    hook.remove()
    assert len(captured) == 4
    for tap, args in zip(scene.features, captured):
        assert args[1].shape == expected_shape
        if scope == "all_observations":
            torch.testing.assert_close(args[1], tap[0][..., 64:].reshape(1, 24, 64))
            assert args[4].tolist() == [[0]*12+[1]*12]
            assert args[5].all()
    # Only an unqueried time changes, with geometry and time conditions frozen.
    altered = replace(scene, features=[(p.clone(), c, t) for p, c, t in scene.features])
    for patch, _, _ in altered.features:
        patch[:, [1, 4], :, 64:] += torch.randn_like(patch[:, [1, 4], :, 64:])
    with torch.no_grad():
        original = model.decode(scene, [10], [0], return_visibility=False)["trajectory"]
        changed = model.decode(altered, [10], [0], return_visibility=False)["trajectory"]
    if scope == "target_time":
        torch.testing.assert_close(changed, original, atol=0, rtol=0)
    else:
        assert (changed-original).abs().max() > 1e-6


def test_legacy_native_checkpoint_preserves_exact_behavior_and_explicit_migration(tmp_path):
    model = MultiD4RT(MultiD4RTConfig.tiny(decoder_memory="target_time")).eval()
    path = tmp_path/"legacy.pt"
    save_checkpoint(path, model)
    data = torch.load(path, weights_only=True)
    data["format_version"] = 1
    del data["config"]["decoder_memory"]
    del data["config"]["geometry_chunk_size"]
    restored = MultiD4RT.from_checkpoint(data).eval()
    assert restored.config.decoder_memory == "target_time"
    assert restored.config.geometry_chunk_size == 8
    inputs = (torch.rand(1, 3, 3, 28, 28), torch.zeros(1, 3, dtype=torch.long),
              torch.tensor([[0, 5, 10]]))
    with torch.no_grad():
        before = model(*inputs, source_obs_idx=[0], return_visibility=False)
        after = restored(*inputs, source_obs_idx=[0], return_visibility=False)
    torch.testing.assert_close(before["trajectory"], after["trajectory"], atol=0, rtol=0)
    migrated = MultiD4RT.from_checkpoint(data, config=replace(
        config_from_checkpoint(data), decoder_memory="all_observations"))
    assert migrated.config.decoder_memory == "all_observations"
    assert set(migrated.state_dict()) == set(model.state_dict())
    new_path = tmp_path/"new.pt"
    save_checkpoint(new_path, migrated)
    assert MultiD4RT.from_checkpoint(new_path).config.decoder_memory == "all_observations"


def test_new_configs_use_all_time_memory_but_checkpoint_migration_is_isolated():
    config = MultiD4RTConfig.tiny()
    old_values = config.to_dict()
    old_values.pop("decoder_memory")
    assert MultiD4RTConfig.from_dict(old_values).decoder_memory == "all_observations"
    assert config_from_checkpoint({"config": old_values}).decoder_memory == "target_time"
    assert config_from_checkpoint({}, fallback=config) is config
    assert "decoder_memory" not in old_values
    with pytest.raises(ValueError, match="decoder_memory"):
        replace(config, decoder_memory="typo")
