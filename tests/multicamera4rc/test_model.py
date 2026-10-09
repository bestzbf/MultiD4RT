"""Complete multi-camera geometry, trajectory, task and checkpoint contracts."""

import pytest
import torch

from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc.checkpoint import load_4rc_weights, save_checkpoint
from multicamera4rc.tasks import sample_tracks, scene_flow


@pytest.fixture(scope="module", autouse=True)
def _limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model(activate_conditions=True):
    torch.manual_seed(41)
    model = MultiCamera4RC(MultiCamera4RCConfig.tiny(
        attention_chunk_size=5, visibility_chunk_size=1024,
    ))
    if activate_conditions:
        with torch.no_grad():
            # Exercise camera/time conditioning after its zero-initialized
            # warm start and avoid a random FoV ReLU disabling camera grads.
            model.backbone.frame_projection.weight.normal_(std=0.01)
            model.target_projection.weight.normal_(std=0.01)
            model.delta_projection.weight.normal_(std=0.01)
            model.cam_dec.fc_fov[0].bias.fill_(1.2)
            model.motion_decoder.camera_bias.uniform_(-0.3, 0.3)
            for block in model.backbone.pretrained.blocks:
                if getattr(block.attn, "camera_relation_weight", None) is not None:
                    block.attn.camera_relation_weight.uniform_(-0.3, 0.3)
            for block in model.motion_decoder.self_blocks:
                block.adaLN_modulation[-1].weight.normal_(std=0.01)
                block.adaLN_modulation[-1].bias.normal_(std=0.01)
    return model


def _clip():
    generator = torch.Generator().manual_seed(43)
    return dict(
        images=torch.rand(1, 6, 3, 28, 28, generator=generator),
        camera_id=torch.tensor([[0, 0, 0, 1, 1, 1]]),
        time_idx=torch.tensor([[0, 5, 10, 0, 5, 10]]),
    )


def _assert_finite(value, name):
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        assert torch.isfinite(value).all(), name
    elif isinstance(value, (list, tuple)):
        for slot, item in enumerate(value):
            _assert_finite(item, f"{name}[{slot}]")


@pytest.fixture(scope="module")
def complete_scene():
    model = _model().eval()
    clip = _clip()
    with torch.no_grad():
        encoded = model.encode(**clip)
        output = model.decode(encoded)
    return model, clip, encoded, output


def test_full_two_camera_three_time_geometry_trajectory_and_visibility(complete_scene):
    _, _, _, output = complete_scene
    assert output["depth"].shape == (1, 6, 28, 28)
    assert output["K"].shape == (1, 6, 3, 3)
    assert output["c2w"].shape == (1, 6, 4, 4)
    assert output["camera_encoding"].shape == (1, 6, 11)
    assert output["base_points"].shape == (1, 6, 28, 28, 3)
    assert output["trajectory"].shape == (1, 6, 3, 28, 28, 3)
    assert output["visibility"].shape == (1, 6, 3, 2, 28, 28)
    assert output["query_time_idx"].tolist() == [[0, 5, 10]]
    assert output["source_time_idx"].tolist() == [[0, 5, 10, 0, 5, 10]]
    assert output["output_camera_id"].tolist() == [[0, 1]]
    assert output["target_observation_valid"].all()
    for key, value in output.items():
        _assert_finite(value, key)
    assert (output["depth"] > 0).all()
    assert ((output["visibility"] >= 0) & (output["visibility"] <= 1)).all()
    for source, target_slot in enumerate([0, 1, 2, 0, 1, 2]):
        torch.testing.assert_close(
            output["displacement"][0, source, target_slot],
            torch.zeros(28, 28, 3), atol=0, rtol=0,
        )
        torch.testing.assert_close(
            output["trajectory"][0, source, target_slot],
            output["base_points"][0, source], atol=0, rtol=0,
        )


def test_subsetting_target_queries_preserves_full_result_and_hidden_source_anchor(complete_scene):
    model, _, scene, full = complete_scene
    with torch.no_grad():
        subset = model.decode(scene, query_time_idx=[10], source_obs_idx=[1])
    assert subset["trajectory"].shape == (1, 1, 1, 28, 28, 3)
    assert subset["query_time_idx"].tolist() == [[10]]
    assert subset["source_time_idx"].tolist() == [[5]]
    for key in ("trajectory", "displacement", "motion_confidence", "visibility", "visibility_logits"):
        torch.testing.assert_close(subset[key][:, 0, 0], full[key][:, 1, 2], atol=5e-6, rtol=5e-5, msg=key)


def test_cached_encode_decode_is_equivalent_to_full_forward(complete_scene):
    model, clip, scene, _ = complete_scene
    with torch.no_grad():
        cached = model.decode(scene, query_time_idx=[0, 10], source_obs_idx=[1, 4])
        direct = model(**clip, query_time_idx=[0, 10], source_obs_idx=[1, 4])
    for key in cached:
        torch.testing.assert_close(cached[key], direct[key], atol=1e-6, rtol=1e-5, msg=key)


def test_per_batch_source_selection_missing_views_and_different_camera_counts():
    model = _model().eval()
    generator = torch.Generator().manual_seed(47)
    cameras = torch.tensor([[7, 7, 7, 9, 9, -1], [3, 3, 8, 8, 12, 12]])
    frames = torch.tensor([[0, 5, 10, 0, 10, 99], [100, 120, 100, 120, 100, 120]])
    valid = torch.tensor([[True, True, True, True, True, False], [True] * 6])
    images = torch.rand(2, 6, 3, 28, 28, generator=generator)
    with torch.no_grad():
        output = model(
            images, cameras, frames, query_time_idx=[[0, 10], [100, 120]],
            source_obs_idx=[[1, 4], [0, 3]], observation_valid=valid,
        )
    assert output["trajectory"].shape == (2, 2, 2, 28, 28, 3)
    assert output["visibility"].shape == (2, 2, 2, 3, 28, 28)
    assert output["source_obs_idx"].tolist() == [[1, 4], [0, 3]]
    assert output["source_time_idx"].tolist() == [[5, 10], [100, 120]]
    assert output["output_camera_id"].tolist() == [[7, 9, -1], [3, 8, 12]]
    assert output["target_observation_valid"].tolist() == [
        [[True, True, False], [True, True, False]], [[True] * 3, [True] * 3],
    ]
    assert not output["projection_valid"][0, :, :, 2].any()
    assert not output["visibility"][0, :, :, 2].any()
    torch.testing.assert_close(output["displacement"][0, 1, 1], torch.zeros(28, 28, 3), atol=0, rtol=0)
    torch.testing.assert_close(output["displacement"][1, 0, 0], torch.zeros(28, 28, 3), atol=0, rtol=0)
    torch.testing.assert_close(output["displacement"][1, 1, 1], torch.zeros(28, 28, 3), atol=0, rtol=0)
    assert torch.isfinite(output["trajectory"]).all()
    assert torch.isfinite(output["visibility_logits"]).all()

    # Invalid observation contents must not enter the geometry or motion of
    # valid sources, even while the padded source image contains valid RGB.
    changed = images.clone()
    changed[0, 5] = 1 - changed[0, 5]
    with torch.no_grad():
        changed_output = model(
            changed, cameras, frames, query_time_idx=[[0, 10], [100, 120]],
            source_obs_idx=[[1, 4], [0, 3]], observation_valid=valid,
        )
    torch.testing.assert_close(changed_output["trajectory"], output["trajectory"], atol=5e-6, rtol=5e-5)

    with torch.no_grad():
        padded_source = model(
            images, cameras, frames, query_time_idx=[[0, 10], [100, 120]],
            source_obs_idx=[[5], [3]], observation_valid=valid,
        )
    assert padded_source["source_valid"].tolist() == [[False], [True]]
    for key in ("trajectory", "displacement", "motion_confidence", "visibility"):
        assert not padded_source[key][0].any(), key
    assert not padded_source["projection_valid"][0].any()


def test_camera_id_renaming_preserves_world_trajectory_and_reorders_visibility(complete_scene):
    model, clip, _, full = complete_scene
    renamed = dict(clip)
    # Reverse numeric order, ensuring visibility's sorted output-camera axis
    # changes while every physical observation keeps its identity relation.
    renamed["camera_id"] = torch.where(clip["camera_id"] == 0, 17, 5)
    with torch.no_grad():
        output = model(**renamed)
    assert output["output_camera_id"].tolist() == [[5, 17]]
    for key in ("depth", "K", "c2w", "trajectory", "displacement"):
        torch.testing.assert_close(output[key], full[key], atol=5e-6, rtol=5e-5, msg=key)
    for key in ("visibility", "visibility_logits"):
        torch.testing.assert_close(output[key][:, :, :, [1, 0]], full[key], atol=5e-6, rtol=5e-5, msg=key)
    torch.testing.assert_close(output["projection_valid"][:, :, :, [1, 0]], full["projection_valid"])


def test_nonreference_observation_permutation_preserves_geometry_and_source_tracks(complete_scene):
    model, clip, _, full = complete_scene
    # Observation zero remains the same reference camera/world gauge.
    order = torch.tensor([0, 4, 2, 5, 1, 3])
    restore = torch.argsort(order)
    shuffled = {key: value[:, order] for key, value in clip.items()}
    with torch.no_grad():
        output = model(**shuffled)
    for key in ("depth", "K", "c2w", "base_points", "trajectory", "displacement", "visibility"):
        torch.testing.assert_close(output[key][:, restore], full[key], atol=7e-6, rtol=7e-5, msg=key)
    torch.testing.assert_close(output["projection_valid"][:, restore], full["projection_valid"])
    # Invalid projections can amplify a few ulps in camera math before the
    # head masks them. Compare logits where the prediction is meaningful.
    valid = full["projection_valid"]
    torch.testing.assert_close(
        output["visibility_logits"][:, restore][valid], full["visibility_logits"][valid],
        atol=7e-6, rtol=7e-5,
    )


def _extension_key(key):
    # Explicit list of new branches, rather than consulting the loader under
    # test to decide which entries an original Arc checkpoint may omit.
    return (key.startswith(("backbone.frame_projection.", "target_projection.", "delta_projection.",
                            "visibility_head.", "cam_dec.fc_principal."))
            or key == "motion_decoder.camera_bias"
            or key.endswith(".attn.camera_relation_weight"))


def _legacy_state(model):
    return {f"_orig_mod.module.{key}": value.clone()
            for key, value in model.state_dict().items() if not _extension_key(key)}


def test_legacy_four_rc_weights_load_all_base_branches_and_preserve_zero_extensions():
    original = _model(activate_conditions=False)
    recipient = _model(activate_conditions=False)
    with torch.no_grad():
        original.backbone.pretrained.patch_embed.proj.weight.add_(0.2)
    report = load_4rc_weights(recipient, {"model": _legacy_state(original)}, strict_base=True)
    assert not report["missing_base"]
    assert report["matched"] > 100
    original_state = original.state_dict()
    for key, value in recipient.state_dict().items():
        if not _extension_key(key):
            torch.testing.assert_close(value, original_state[key], atol=0, rtol=0, msg=key)
    zero_parameters = [
        recipient.backbone.frame_projection.weight, recipient.target_projection.weight,
        recipient.delta_projection.weight, recipient.cam_dec.fc_principal.weight,
        recipient.cam_dec.fc_principal.bias, recipient.motion_decoder.camera_bias,
    ] + [block.attn.camera_relation_weight for block in recipient.backbone.pretrained.blocks
         if getattr(block.attn, "camera_relation_weight", None) is not None]
    assert all(torch.count_nonzero(value) == 0 for value in zero_parameters)


def test_native_checkpoint_cannot_silently_reinitialize_a_missing_new_branch():
    model = _model(activate_conditions=False)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    del state["delta_projection.weight"]
    with pytest.raises(ValueError, match="native checkpoint must cover the complete model"):
        load_4rc_weights(model, {"architecture": "multicamera4rc", "model": state})


@pytest.mark.parametrize("strict_base", [False, True])
def test_weight_shape_mismatch_is_rejected_before_mutation(strict_base):
    model = _model(activate_conditions=False)
    legacy = _legacy_state(model)
    key = "_orig_mod.module.backbone.pretrained.patch_embed.proj.weight"
    legacy[key] = legacy[key][:1]
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="shape mismatch"):
        load_4rc_weights(model, {"model": legacy}, strict_base=strict_base)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_wrong_d4rt_checkpoint_is_not_silently_accepted():
    model = _model(activate_conditions=False)
    with pytest.raises(ValueError, match="no weights match.*D4RT"):
        load_4rc_weights(model, {"model": {"encoder.encoder.layers.0.attn.qkv.weight": torch.ones(3, 2)}})


def test_native_checkpoint_preserves_config_weights_and_prediction(tmp_path, complete_scene):
    model, clip, _, _ = complete_scene
    path = tmp_path / "multicamera.pt"
    save_checkpoint(path, model, step=17, experiment="tiny-correctness")
    assert path.exists() and not path.with_name(path.name + ".tmp").exists()
    restored = MultiCamera4RC.from_checkpoint(path).eval()
    assert restored.config == model.config
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0, msg=name)
    with torch.no_grad():
        expected = model(**clip, source_obs_idx=[1], query_time_idx=[10])
        actual = restored(**clip, source_obs_idx=[1], query_time_idx=[10])
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0, msg=key)


def _linear_prediction():
    frames = torch.tensor([0.0, 5.0, 10.0])
    rows, columns = torch.meshgrid(torch.arange(4).float(), torch.arange(5).float(), indexing="ij")
    sources = [2, 7]
    trajectory = torch.stack([
        torch.stack([torch.stack((columns + 10 * source, 2 * rows + frame,
                                 3 * columns - rows + 0.5 * frame), -1)
                     for frame in frames]) for source in sources
    ])[None]
    displacement = torch.zeros_like(trajectory)
    source_times = torch.tensor([[5, 0]])
    displacement[..., 1] = (frames[None, None, :] - source_times[:, :, None])[:, :, :, None, None]
    displacement[..., 2] = displacement[..., 1] * 0.5
    confidence = (2 + columns * 0.1 + rows * 0.1)[None, None, None].expand(1, 2, 3, 4, 5).clone()
    visibility = torch.stack([
        torch.stack([torch.stack([0.1 + camera * 0.1 + columns * 0.01 + rows * 0.02 + frame * 0.005
                                 for camera in range(2)]) for frame in frames])
        for _ in sources
    ])[None]
    return dict(
        trajectory=trajectory, displacement=displacement, motion_confidence=confidence,
        visibility=visibility, visibility_logits=visibility * 2 - 1,
        projection_valid=torch.ones_like(visibility, dtype=torch.bool),
        source_obs_idx=torch.tensor([sources]), source_valid=torch.ones(1, 2, dtype=torch.bool),
        source_time_idx=source_times, query_time_idx=frames.long()[None],
    )


def test_sample_tracks_bilinearly_reads_known_linear_world_field():
    prediction = _linear_prediction()
    queries = torch.tensor([[[7, 1.5, 2.0], [2, 4.0, 3.0]]])
    sampled = sample_tracks(prediction, queries)
    assert sampled["trajectory"].shape == (1, 2, 3, 3)
    assert sampled["visibility"].shape == (1, 2, 3, 2)
    assert sampled["source_time_idx"].tolist() == [[0, 5]]
    frames = [0, 5, 10]
    expected = torch.tensor([[
        [[71.5, 4.0 + frame, 2.5 + 0.5 * frame] for frame in frames],
        [[24.0, 6.0 + frame, 9.0 + 0.5 * frame] for frame in frames],
    ]])
    torch.testing.assert_close(sampled["trajectory"], expected, atol=1e-6, rtol=1e-6)
    expected_visibility = torch.tensor([[
        [[0.155 + frame * 0.005, 0.255 + frame * 0.005] for frame in frames],
        [[0.20 + frame * 0.005, 0.30 + frame * 0.005] for frame in frames],
    ]])
    torch.testing.assert_close(sampled["visibility"], expected_visibility, atol=1e-6, rtol=1e-6)
    assert sampled["query_valid"].all() and sampled["projection_valid"].all()


def test_sample_tracks_matches_actual_dense_output_at_integer_source_pixels(complete_scene):
    _, _, _, dense = complete_scene
    sampled = sample_tracks(dense, [[[1, 10, 7], [5, 4, 5]]])
    for query, (source, u, v) in enumerate([(1, 10, 7), (5, 4, 5)]):
        for key in ("trajectory", "displacement", "motion_confidence"):
            torch.testing.assert_close(sampled[key][0, query], dense[key][0, source, :, v, u], atol=2e-6, rtol=2e-5, msg=key)
        torch.testing.assert_close(sampled["visibility"][0, query], dense["visibility"][0, source, :, :, v, u], atol=2e-6, rtol=2e-5)


def test_scene_flow_uses_frame_labels_not_output_slots():
    prediction = _linear_prediction()
    flow = scene_flow(prediction, start_time=5, end_time=10)
    expected = torch.tensor([0.0, 5.0, 2.5]).expand(1, 2, 4, 5, 3)
    torch.testing.assert_close(flow, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="frame label 1 was not queried"):
        scene_flow(prediction, start_time=1, end_time=2)


def test_unselected_or_outside_sparse_queries_can_be_masked():
    prediction = _linear_prediction()
    sampled = sample_tracks(prediction, [[[7, 1, 2], [1, 2, 2], [2, -1, 0]]], strict=False)
    assert sampled["query_valid"].tolist() == [[True, False, False]]
    assert not sampled["trajectory"][:, 1:].any()
    with pytest.raises(ValueError, match="outside.*unselected sources"):
        sample_tracks(prediction, [[[1, 2, 2]]])
